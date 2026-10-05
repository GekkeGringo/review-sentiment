import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from review_sentiment.text import encode_text

LABELS = ["Bad", "Neutral", "Good"]


class Predictor:
    def __init__(self, repo=None, subfolder=None, max_len=256, strategy="head_tail"):
        self.repo = repo or os.getenv("MODEL_REPO", "GekkeGringo/rubert-kinopoisk-sentiment")
        self.subfolder = subfolder or os.getenv("MODEL_SUBFOLDER", "rubert-256-head_tail")
        self.max_len, self.strategy = max_len, strategy
        self.tok = AutoTokenizer.from_pretrained(self.repo, subfolder=self.subfolder)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.repo, subfolder=self.subfolder
        ).eval()

    def predict(self, text: str) -> dict:
        ids = encode_text(text, self.tok, self.max_len, self.strategy)
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([ids])).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        return {label: round(p, 3) for label, p in zip(LABELS, probs)}


if __name__ == "__main__":
    p = Predictor()
    print(p.predict("Нормальное кино, ничего особенного."))