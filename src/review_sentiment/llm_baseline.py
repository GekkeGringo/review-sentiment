import json
import re
import sys
import time
from typing import Literal

import mlflow
import pandas as pd
import torch
from pydantic import BaseModel, ValidationError
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT_VERSION = "v1"
LABELS = ["Bad", "Neutral", "Good"]
SEED = 42
MAX_CHARS = 6000
BATCH = 8

SYSTEM = (
    "Ты классификатор тональности отзывов. Определи отношение автора отзыва "
    "к предмету отзыва (фильму, товару) в целом.\n"
    "Классы:\n"
    "- Bad: негатив преобладает или есть серьёзный недостаток.\n"
    "- Good: позитив преобладает, автор хвалит или рекомендует.\n"
    "- Neutral: смешанное мнение (плюсы и минусы примерно равны), "
    "«нормально, ничего особенного» или только факты без оценки.\n"
    'Ответь строго одной строкой JSON: {"label": "Bad"}, {"label": "Neutral"} '
    'или {"label": "Good"}. Ничего кроме JSON.'
)


class Verdict(BaseModel):
    label: Literal["Bad", "Neutral", "Good"]


def clip(text):
    if len(text) <= MAX_CHARS:
        return text
    head = MAX_CHARS // 4
    return text[:head] + " [...] " + text[-(MAX_CHARS - head) :]


def wrap(text):
    return "Отзыв:\n" + clip(text)


def parse(raw):
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return Verdict.model_validate_json(match.group(0)).label
    except ValidationError:
        return None


def few_shot_messages(train):
    words = train["content"].str.split().str.len()
    short = train[words.between(30, 80)]
    parts = [short[short["grade3"] == lab].sample(2, random_state=SEED) for lab in LABELS]
    shots = pd.concat(parts).sample(frac=1, random_state=SEED)
    messages = []
    for _, row in shots.iterrows():
        messages.append({"role": "user", "content": wrap(row["content"])})
        messages.append({"role": "assistant", "content": json.dumps({"label": row["grade3"]})})
    return messages


class LLM:
    def __init__(self, shots):
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        self.base = [{"role": "system", "content": SYSTEM}] + shots

    def classify(self, texts):
        prompts = [
            self.tok.apply_chat_template(
                self.base + [{"role": "user", "content": wrap(t)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in texts
        ]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outs = [None] * len(prompts)
        for s in range(0, len(order), BATCH):
            idx = order[s : s + BATCH]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True)
            enc = enc.to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=16, do_sample=False)
            decoded = self.tok.batch_decode(
                gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for i, text in zip(idx, decoded):
                outs[i] = text
        return outs


def evaluate(llm, df, name):
    start = time.time()
    raws = llm.classify(df["text"].tolist())
    secs = (time.time() - start) / len(df)
    preds = [parse(r) for r in raws]
    fail = sum(p is None for p in preds) / len(preds)
    y = df["label"].tolist()
    shown = [p if p is not None else "invalid" for p in preds]
    macro = f1_score(y, shown, average="macro", labels=LABELS, zero_division=0)
    per_class = f1_score(y, shown, average=None, labels=LABELS, zero_division=0)
    print(f"\n=== {name}: n={len(df)} ===")
    print(f"macro-F1 {macro:.3f} | " + ", ".join(f"{lab} {s:.3f}" for lab, s in zip(LABELS, per_class)))
    print(f"неразобранных ответов: {fail:.3f} | сек/отзыв: {secs:.2f}")
    print(pd.crosstab(pd.Series(y, name="truth"), pd.Series(shown, name="pred")))
    out = df[["id", "label"]].assign(pred=shown, raw=raws)
    out.to_csv(f"outputs_llm/{name}_predictions.csv", index=False, encoding="utf-8")
    metrics = {f"{name}_macro_f1": macro, f"{name}_parse_fail": fail, f"{name}_sec_per_review": secs}
    metrics.update({f"{name}_f1_{lab.lower()}": s for lab, s in zip(LABELS, per_class)})
    return metrics


def main():
    smoke = "--smoke" in sys.argv
    n_val = 16 if smoke else 500
    train = pd.read_parquet("data/train.parquet")
    val = pd.read_parquet("data/val.parquet").sample(n_val, random_state=SEED)
    val = val.rename(columns={"review_id": "id", "content": "text", "grade3": "label"})
    val["id"] = val["id"].astype(str)

    cand = pd.read_csv("data/gold_candidates.csv", dtype={"id": str})
    mine = pd.read_csv("gold/gold_labels.csv", dtype={"id": str}).drop_duplicates("id", keep="last")
    gold = cand.merge(mine, on="id")
    gold = gold[gold["label"].isin(LABELS)].reset_index(drop=True)
    if smoke:
        gold = gold.head(16)
    print("gold отзывов в оценке:", len(gold))

    import os

    os.makedirs("outputs_llm", exist_ok=True)
    llm = LLM(few_shot_messages(train))
    metrics = {}
    metrics.update(evaluate(llm, val[["id", "text", "label"]].reset_index(drop=True), "valsub"))
    metrics.update(evaluate(llm, gold[["id", "text", "label"]], "gold"))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("kinopoisk-sentiment")
    name = "llm-qwen2.5-7b-fewshot-" + PROMPT_VERSION + ("-smoke" if smoke else "")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {"model": MODEL, "prompt_version": PROMPT_VERSION, "n_shots": 6,
             "max_chars": MAX_CHARS, "n_val_sample": n_val, "seed": SEED}
        )
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()