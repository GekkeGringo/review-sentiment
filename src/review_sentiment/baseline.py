import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from review_sentiment.data import SEED

LABELS = ["Bad", "Neutral", "Good"]
PARAMS = {"max_features": 100_000, "ngram_range": (1, 2), "min_df": 3, "C": 3.0}


def main():
    train = pd.read_parquet("data/train.parquet")
    val = pd.read_parquet("data/val.parquet")  # test пока не трогаем

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=PARAMS["max_features"],
            ngram_range=PARAMS["ngram_range"],
            min_df=PARAMS["min_df"],
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=PARAMS["C"], max_iter=1000, class_weight="balanced", random_state=SEED,
        )),
    ])
    model.fit(train["content"], train["grade3"])
    pred = model.predict(val["content"])

    macro_f1 = f1_score(val["grade3"], pred, average="macro", labels=LABELS)
    per_class = f1_score(val["grade3"], pred, average=None, labels=LABELS)
    report = classification_report(val["grade3"], pred, labels=LABELS, digits=3)
    print(report)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("kinopoisk-sentiment")
    with mlflow.start_run(run_name="tfidf-logreg"):
        mlflow.log_params({k: str(v) for k, v in PARAMS.items()})
        mlflow.log_metric("val_macro_f1", macro_f1)
        for label, score in zip(LABELS, per_class):
            mlflow.log_metric(f"val_f1_{label.lower()}", score)
        mlflow.log_text(report, "val_report.txt")


if __name__ == "__main__":
    main()