import sys

import mlflow
from mlflow.tracking import MlflowClient

SRC_FILE = sys.argv[1] if len(sys.argv) > 1 else "kaggle_artifacts/mlflow_kaggle.db"
SRC = f"sqlite:///{SRC_FILE}"
DST = "sqlite:///mlflow.db"
EXPERIMENT = "kinopoisk-sentiment"
SKIP = {"smoke-test"}

src = MlflowClient(tracking_uri=SRC)
exp = src.get_experiment_by_name(EXPERIMENT)
runs = src.search_runs([exp.experiment_id])

mlflow.set_tracking_uri(DST)
mlflow.set_experiment(EXPERIMENT)
dst = MlflowClient(tracking_uri=DST)
dst_exp = dst.get_experiment_by_name(EXPERIMENT)
existing = {
    r.data.tags.get("mlflow.runName")
    for r in dst.search_runs([dst_exp.experiment_id])
}

for r in runs:
    name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
    if name in SKIP or name in existing or "smoke" in name:
        continue
    with mlflow.start_run(run_name=name):
        mlflow.log_params(r.data.params)
        mlflow.log_metrics(r.data.metrics)
        mlflow.set_tag("environment", "kaggle-2xT4")
    print("перенесён:", name)