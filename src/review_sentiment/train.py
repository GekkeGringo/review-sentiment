import argparse

from review_sentiment.text import truncate
import math
import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

LABELS = ["Bad", "Neutral", "Good"]
L2I = {label: i for i, label in enumerate(LABELS)}
SEED = 42



def encode(df, tok, max_len, strategy):
    raw = tok(df["content"].tolist(), add_special_tokens=False)["input_ids"]
    ids = [[tok.cls_token_id] + truncate(x, max_len, strategy) + [tok.sep_token_id] for x in raw]
    return Dataset.from_dict({
        "input_ids": ids,
        "attention_mask": [[1] * len(x) for x in ids],
        "labels": [L2I[g] for g in df["grade3"]],
    })


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    per_class = f1_score(p.label_ids, preds, average=None, labels=[0, 1, 2])
    out = {"macro_f1": f1_score(p.label_ids, preds, average="macro")}
    out.update({f"f1_{l.lower()}": s for l, s in zip(LABELS, per_class)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DeepPavlov/rubert-base-cased")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--strategy", choices=["head", "head_tail"], default="head_tail")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=0, help="подвыборка для быстрой проверки")
    ap.add_argument("--run-name", default="rubert")
    args = ap.parse_args()

    set_seed(SEED)
    train = pd.read_parquet("data/train.parquet")
    val = pd.read_parquet("data/val.parquet")  # test не трогаем
    if args.limit:
        train, val = train.sample(args.limit, random_state=SEED), val.sample(args.limit // 4, random_state=SEED)

    tok = AutoTokenizer.from_pretrained(args.model)
    train_ds = encode(train, tok, args.max_len, args.strategy)
    val_ds = encode(val, tok, args.max_len, args.strategy)

    weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_ds["labels"])
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)

    n_dev = max(1, torch.cuda.device_count())
    total_steps = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum * n_dev)) * args.epochs
    warmup_steps = int(0.1 * total_steps)


    targs = TrainingArguments(
        output_dir=f"outputs/{args.run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
        seed=SEED,
    )
    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
        class_weights=torch.tensor(weights, dtype=torch.float),
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("kinopoisk-sentiment")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({k: v for k, v in vars(args).items()})
        mlflow.log_metric("val_macro_f1", metrics["eval_macro_f1"])
        for label in LABELS:
            mlflow.log_metric(f"val_f1_{label.lower()}", metrics[f"eval_f1_{label.lower()}"])
    trainer.save_model(f"outputs/{args.run_name}/best")


if __name__ == "__main__":
    main()