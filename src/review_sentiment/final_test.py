import json
import os
import sys

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from review_sentiment.baseline import PARAMS
from review_sentiment.data import SEED
from review_sentiment.predict import LABELS, Predictor
from review_sentiment.text import encode_text

REPORT = "reports/final_test.json"
BATCH = 16


def predict_labels(p, texts):
    encoded = [encode_text(t, p.tok, p.max_len, p.strategy) for t in texts]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    out = [None] * len(encoded)
    pad = p.tok.pad_token_id
    for s in range(0, len(order), BATCH):
        idx = order[s : s + BATCH]
        longest = max(len(encoded[i]) for i in idx)
        ids = torch.tensor(
            [encoded[i] + [pad] * (longest - len(encoded[i])) for i in idx]
        )
        mask = torch.tensor(
            [[1] * len(encoded[i]) + [0] * (longest - len(encoded[i])) for i in idx]
        )
        with torch.no_grad():
            pred = p.model(input_ids=ids, attention_mask=mask).logits.argmax(-1)
        for i, k in zip(idx, pred.tolist()):
            out[i] = LABELS[k]
    return out


def summarize(name, y, pred, rng):
    y_arr, p_arr = np.array(y), np.array(pred)
    macro = f1_score(y_arr, p_arr, average="macro", labels=LABELS)
    per_class = f1_score(y_arr, p_arr, average=None, labels=LABELS)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        boots.append(f1_score(y_arr[idx], p_arr[idx], average="macro", labels=LABELS))
    low, high = np.percentile(boots, [2.5, 97.5])
    print(f"\n=== {name}: n={len(y_arr)} ===")
    print(f"macro-F1 {macro:.3f} (95% интервал {low:.3f}-{high:.3f})")
    print(classification_report(y_arr, p_arr, labels=LABELS, digits=3))
    print(pd.crosstab(pd.Series(y_arr, name="truth"), pd.Series(p_arr, name="pred")))
    return {
        "macro_f1": float(macro),
        "ci95": [float(low), float(high)],
        "f1": {lab: float(s) for lab, s in zip(LABELS, per_class)},
    }


def main():
    if os.path.exists(REPORT) and "--force" not in sys.argv:
        sys.exit(f"Финальная оценка уже выполнена: {REPORT}. Повторный запуск запрещён.")

    train = pd.read_parquet("data/train.parquet")
    test = pd.read_parquet("data/test.parquet")
    y = test["grade3"].tolist()
    rng = np.random.default_rng(SEED)

    bert = Predictor()
    bert_pred = predict_labels(bert, test["content"].tolist())

    tfidf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=PARAMS["max_features"],
                    ngram_range=PARAMS["ngram_range"],
                    min_df=PARAMS["min_df"],
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=PARAMS["C"], max_iter=1000, class_weight="balanced", random_state=SEED
                ),
            ),
        ]
    )
    tfidf.fit(train["content"], train["grade3"])
    tfidf_pred = list(tfidf.predict(test["content"]))

    results = {
        "n": len(test),
        "rubert-256": summarize("ruBERT-256 head+tail", y, bert_pred, rng),
        "tfidf": summarize("TF-IDF + LogReg", y, tfidf_pred, rng),
    }
    os.makedirs("reports", exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("kinopoisk-sentiment")
    for key in ("rubert-256", "tfidf"):
        with mlflow.start_run(run_name=f"final-test-{key}"):
            mlflow.log_metric("test_macro_f1", results[key]["macro_f1"])
            for lab, score in results[key]["f1"].items():
                mlflow.log_metric(f"test_f1_{lab.lower()}", score)
            mlflow.set_tag("environment", "local-cpu")
    print(f"\nРезультаты сохранены в {REPORT}")


if __name__ == "__main__":
    main()