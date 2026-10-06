import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.pipeline import Pipeline

from review_sentiment.baseline import PARAMS
from review_sentiment.data import SEED
from review_sentiment.predict import LABELS, Predictor

cand = pd.read_csv("data/gold_candidates.csv", dtype={"id": str})
src = pd.read_csv("data/gold_source.csv", dtype={"id": str})
mine = pd.read_csv("gold/gold_labels.csv", dtype={"id": str})
mine = mine.drop_duplicates("id", keep="last")
gold = cand.merge(src, on="id").merge(mine, on="id")
skipped = int((gold["label"] == "Skip").sum())
gold = gold[gold["label"].isin(LABELS)].reset_index(drop=True)
print(f"размечено: {len(mine)}, пропущено: {skipped}, в оценке: {len(gold)}")
print("ваши классы:", gold["label"].value_counts().to_dict())

print("\n=== Согласие вашей разметки с меткой датасета ===")
print("доля совпадений:", round(float((gold["label"] == gold["src_label"]).mean()), 3))
kappa = cohen_kappa_score(gold["label"], gold["src_label"], labels=LABELS)
print("каппа Коэна:", round(float(kappa), 3))
print("строки: метка датасета, столбцы: ваша метка")
print(
    pd.crosstab(gold["src_label"], gold["label"]).reindex(
        index=LABELS, columns=LABELS, fill_value=0
    )
)

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
                C=PARAMS["C"],
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    ]
)
tfidf.fit(train["content"], train["grade3"])

bert = Predictor()


def bert_label(text):
    probs = bert.predict(text)
    return max(probs, key=probs.get)


preds = {
    "tfidf": list(tfidf.predict(gold["text"])),
    "rubert-256": [bert_label(t) for t in gold["text"]],
}

rng = np.random.default_rng(SEED)


def macro(y, p):
    return f1_score(y, p, average="macro", labels=LABELS, zero_division=0)


def per_class(y, p):
    return f1_score(y, p, average=None, labels=LABELS, zero_division=0)


def bootstrap_ci(y, p, n=1000):
    y, p = np.array(y), np.array(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals.append(macro(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


print("\n=== Качество на gold-наборе (отзывы на товары) ===")
for name, pred in preds.items():
    for ref_name, ref in [("ваша разметка", gold["label"]), ("метка датасета", gold["src_label"])]:
        f1s = per_class(ref, pred)
        line = f"{name:11} | {ref_name:14} | macro-F1 {macro(ref, pred):.3f}"
        if ref_name == "ваша разметка":
            lo, hi = bootstrap_ci(ref, pred)
            line += f" (95% интервал {lo:.3f}-{hi:.3f})"
        line += " | F1 " + ", ".join(f"{lab} {s:.3f}" for lab, s in zip(LABELS, f1s))
        print(line)
    print(f"Ошибки {name} (строки: ваша метка, столбцы: ответ модели)")
    print(
        pd.crosstab(gold["label"], pd.Series(pred, name="pred")).reindex(
            index=LABELS, columns=LABELS, fill_value=0
        )
    )

out = gold.assign(**{f"pred_{k}": v for k, v in preds.items()})
out.to_csv("data/gold_predictions.csv", index=False, encoding="utf-8")