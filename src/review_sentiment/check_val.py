import sys

import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer

from review_sentiment.predict import LABELS, Predictor

_p = Predictor()
tok, model, MAX_LEN, STRATEGY = _p.tok, _p.model, _p.max_len, _p.strategy
from review_sentiment.text import encode_text

# 1. Токенизатор из репозитория совпадает с оригинальным?
orig = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
t = "Скучно, сюжет рассыпается, зря потратил время."
print("токенизаторы совпадают:", orig(t)["input_ids"] == tok(t)["input_ids"])

# 2. Метрики на части val
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
val = pd.read_parquet("data/val.parquet").sample(n, random_state=42)
preds = []
for text in val["content"]:
    ids = encode_text(text, tok, MAX_LEN, STRATEGY)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits
    preds.append(LABELS[int(logits.argmax(-1))])
print(classification_report(val["grade3"], preds, labels=LABELS, digits=3))
print("macro-F1:", round(f1_score(val["grade3"], preds, average="macro", labels=LABELS), 3))