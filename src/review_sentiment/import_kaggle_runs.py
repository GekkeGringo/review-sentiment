import mlflow
from mlflow.tracking import MlflowClient

SRC = "sqlite:///kaggle_artifacts/mlflow_kaggle.db"
DST = "sqlite:///mlflow.db"
EXPERIMENT = "kinopoisk-sentiment"
SKIP = {"smoke-test"}

src = MlflowClient(tracking_uri=SRC)
exp = src.get_experiment_by_name(EXPERIMENT)
runs = src.search_runs([exp.experiment_id])

mlflow.set_tracking_uri(DST)
mlflow.set_experiment(EXPERIMENT)
for r in runs:
    name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
    if name in SKIP:
        continue
    with mlflow.start_run(run_name=name):
        mlflow.log_params(r.data.params)
        mlflow.log_metrics(r.data.metrics)
        mlflow.set_tag("environment", "kaggle-2xT4")
    print("перенесён:", name)