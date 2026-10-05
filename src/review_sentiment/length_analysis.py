import sys

import pandas as pd
import torch
from sklearn.metrics import f1_score

from review_sentiment.predict import LABELS, Predictor
from review_sentiment.text import encode_text

PER_BUCKET = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BUCKETS = [(0, 50), (50, 150), (150, 300), (300, 600), (600, 10**6)]
BATCH = 16

val = pd.read_parquet("data/val.parquet")
val["words"] = val["content"].str.split().str.len()
p = Predictor()


def predict_many(texts):
    encoded = [encode_text(t, p.tok, p.max_len, p.strategy) for t in texts]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    out = [None] * len(encoded)
    for s in range(0, len(order), BATCH):
        idx = order[s : s + BATCH]
        longest = max(len(encoded[i]) for i in idx)
        pad = p.tok.pad_token_id
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


print(f"{'слов':>8} {'всего':>6} {'взято':>6} {'macro-F1':>9} {'Neutral-F1':>11}")
for lo, hi in BUCKETS:
    part = val[(val.words >= lo) & (val.words < hi)]
    total = len(part)
    if total == 0:
        continue
    part = part.sample(min(PER_BUCKET, total), random_state=42)
    preds = predict_many(part["content"].tolist())
    macro = f1_score(part["grade3"], preds, average="macro", labels=LABELS, zero_division=0)
    per_class = f1_score(part["grade3"], preds, average=None, labels=LABELS, zero_division=0)
    name = f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
    print(f"{name:>8} {total:>6} {len(part):>6} {macro:>9.3f} {per_class[1]:>11.3f}")