#!/usr/bin/env python3
import argparse
import os
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
from mlflow.tracking import MlflowClient

PRINT_RUN_ID_FILE = "/tmp/student_offer_last_run_id.txt"

class StudentOfferLabelModel(mlflow.pyfunc.PythonModel):
    def __init__(self, threshold: float = 80.0, epochs: int = 1):
        self.pipeline = None
        self.threshold = threshold
        self.epochs = max(1, int(epochs))

    def fit(self):
        data = pd.DataFrame({
            "student": [f"S{i+1}" for i in range(10)],
            "marks":   [55, 62, 71, 79, 80, 81, 85, 90, 95, 67],
        })
        data["placed"] = (data["marks"] > self.threshold).astype(int)
        X = data[["marks"]].astype(float)
        y = data["placed"].astype(int)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression())
        ])
        for _ in range(self.epochs):
            self.pipeline.fit(X, y)

        preds = self.pipeline.predict(X)
        acc = accuracy_score(y, preds)
        return acc

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        preds = self.pipeline.predict(model_input[["marks"]].astype(float))
        return np.where(preds == 1, "got offer", "Not Placed").astype(str)


def _canonicalize_number_str(x: float) -> str:
    s = str(x)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=80.0, help="Marks cutoff for placement")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="(optional) MLflow experiment id to log into")
    args = parser.parse_args()

    # ensure MLflow points to your tracking server (env var override supported)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://10.0.11.179:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # Determine experiment: prefer explicit experiment-id arg; otherwise ensure by name
    if args.experiment_id:
        exp = mlflow.get_experiment(args.experiment_id)
        if exp is None:
            raise RuntimeError(f"Experiment with id '{args.experiment_id}' not found on server {tracking_uri}")
        experiment_id = args.experiment_id
        experiment_name = exp.name
    else:
        experiment_name = "sixdee_experiments"
        mlflow.set_experiment(experiment_name)
        exp = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = exp.experiment_id if exp else None

    print(f"Using experiment: {experiment_name} (id={experiment_id})")

    # Train model
    model = StudentOfferLabelModel(threshold=args.threshold, epochs=args.epochs)
    acc = model.fit()

    run_name = "sixdee_logistic_regression_placement_prediction"
    signature = ModelSignature(inputs=Schema([ColSpec("double", "marks")]),
                               outputs=Schema([ColSpec("string")]))

    # Start run using explicit experiment_id if available
    if experiment_id is not None:
        run_ctx = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    else:
        run_ctx = mlflow.start_run(run_name=run_name)

    with run_ctx:
        run = mlflow.active_run()
        print("RUN_ID:", run.info.run_id)
        print("Run name:", run_name)
        exp_info = mlflow.get_experiment(run.info.experiment_id)
        print("Experiment (final):", exp_info.name if exp_info else run.info.experiment_id)
        print("MLflow UI:", f"{mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

        # Save run id to file for debugging / DAG readability
        try:
            with open(PRINT_RUN_ID_FILE, "w") as fh:
                fh.write(run.info.run_id)
        except Exception as e:
            print("Warning: failed to write run id file:", e)

        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_metric("train_accuracy", acc)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            signature=signature,
            pip_requirements=[
                "mlflow==3.3.2",
                "scikit-learn",
                "pandas",
                "numpy",
                "joblib",
                "cloudpickle",
            ],
        )

        print(f"SUCCESS: Model logged with accuracy {acc}")
