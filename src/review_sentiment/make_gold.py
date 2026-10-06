import pandas as pd
from datasets import load_dataset

SEED = 42
PER_CLASS = 70
MIN_WORDS, MAX_WORDS = 8, 300
LABEL_MAP = {"negative": "Bad", "neutral": "Neutral", "positive": "Good"}


def cyrillic_share(text):
    letters = [c.lower() for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("а" <= c <= "я" or c == "ё" for c in letters) / len(letters)


df = load_dataset("ai-forever/ru-reviews-classification", split="test").to_pandas()
print(df["label_text"].value_counts())
df["src_label"] = df["label_text"].map(LABEL_MAP)
assert df["src_label"].notna().all(), "неожиданные названия классов"

df["words"] = df["text"].str.split().str.len()
df = df.drop_duplicates("text").drop_duplicates("id")
df = df[df["text"].map(cyrillic_share) >= 0.8]
df = df[(df["words"] >= MIN_WORDS) & (df["words"] <= MAX_WORDS)]

parts = [g.sample(min(PER_CLASS, len(g)), random_state=SEED) for _, g in df.groupby("src_label")]
gold = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

gold[["id", "text"]].to_csv("data/gold_candidates.csv", index=False, encoding="utf-8")
gold[["id", "src_label", "words"]].to_csv("data/gold_source.csv", index=False, encoding="utf-8")
print(gold["src_label"].value_counts())
print("слов в отзыве: медиана", int(gold["words"].median()), "макс", int(gold["words"].max()))