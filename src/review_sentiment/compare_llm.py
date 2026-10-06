import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from review_sentiment.baseline import PARAMS
from review_sentiment.data import SEED
from review_sentiment.predict import LABELS, Predictor

LLM_DIR = sys.argv[1] if len(sys.argv) > 1 else "kaggle_artifacts/llm_v1"
rng = np.random.default_rng(SEED)


def macro(y, p):
    return f1_score(y, p, average="macro", labels=LABELS, zero_division=0)


def per_class(y, p):
    return f1_score(y, p, average=None, labels=LABELS, zero_division=0)


def bootstrap(y, preds, n=1000):
    y = np.array(y)
    preds = {k: np.array(v) for k, v in preds.items()}
    boot = {k: [] for k in preds}
    diffs = {k: [] for k in preds if k != "llm"}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        scores = {k: macro(y[idx], v[idx]) for k, v in preds.items()}
        for k in boot:
            boot[k].append(scores[k])
        for k in diffs:
            diffs[k].append(scores["llm"] - scores[k])
    return boot, diffs


def report(title, y, preds):
    print(f"\n=== {title}: n={len(y)} ===")
    boot, diffs = bootstrap(y, preds)
    for name, p in preds.items():
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        classes = ", ".join(f"{lab} {s:.3f}" for lab, s in zip(LABELS, per_class(y, p)))
        print(f"{name:11} macro-F1 {macro(y, p):.3f} ({lo:.3f}-{hi:.3f}) | {classes}")
    for name, d in diffs.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        diff = macro(y, preds["llm"]) - macro(y, preds[name])
        print(f"разница llm - {name}: {diff:+.3f} (95% интервал {lo:+.3f}..{hi:+.3f})")
    print("ответов llm вне формата:", list(preds["llm"]).count("invalid"))


llm_val = pd.read_csv(f"{LLM_DIR}/valsub_predictions.csv", dtype={"id": str})
llm_gold = pd.read_csv(f"{LLM_DIR}/gold_predictions.csv", dtype={"id": str})

val = (
    pd.read_parquet("data/val.parquet")
    .sample(len(llm_val), random_state=SEED)
    .reset_index(drop=True)
)
assert val["review_id"].astype(str).tolist() == llm_val["id"].tolist(), "выборки val не совпали"
assert val["grade3"].tolist() == llm_val["label"].tolist(), "метки val не совпали"
val["llm"] = llm_val["pred"]

train = pd.read_parquet("data/train.parquet")
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
bert = Predictor()


def bert_label(text):
    probs = bert.predict(text)
    return max(probs, key=probs.get)


val_preds = {
    "llm": val["llm"].tolist(),
    "rubert-256": [bert_label(t) for t in val["content"]],
    "tfidf": list(tfidf.predict(val["content"])),
}
report("val Кинопоиска (выборка)", val["grade3"].tolist(), val_preds)

gold = pd.read_csv("data/gold_predictions.csv", dtype={"id": str})
gold = gold.merge(llm_gold[["id", "pred"]].rename(columns={"pred": "llm"}), on="id")
gold_preds = {
    "llm": gold["llm"].tolist(),
    "rubert-256": gold["pred_rubert-256"].tolist(),
    "tfidf": gold["pred_tfidf"].tolist(),
}
report("gold: отзывы на товары (ваша разметка)", gold["label"].tolist(), gold_preds)