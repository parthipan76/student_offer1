#!/usr/bin/env python3
import argparse
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
from mlflow.tracking import MlflowClient


class StudentOfferLabelModel(mlflow.pyfunc.PythonModel):
    def __init__(self, threshold: float = 80.0, epochs: int = 1):
        self.pipeline = None
        self.threshold = threshold
        self.epochs = max(1, int(epochs))

    # ------- Train pipeline -------
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
        # epochs is mostly for API compatibility; scikit-learn refits each loop
        for _ in range(self.epochs):
            self.pipeline.fit(X, y)

        preds = self.pipeline.predict(X)
        acc = accuracy_score(y, preds)
        return acc

    # ------- PyFunc predict -------
    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        preds = self.pipeline.predict(model_input[["marks"]].astype(float))
        return np.where(preds == 1, "Placed", "Not Placed").astype(str)


def _canonicalize_number_str(x: float) -> str:
    # Make "85.0" -> "85" so it won't collide with MLflow Projects' stringy params
    s = str(x)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=80.0,
                        help="Marks cutoff for placement")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs (compatibility for mlflow -P epochs=...)")
    args = parser.parse_args()

    # Set tracking and experiment explicitly BEFORE any run is started
    mlflow.set_tracking_uri("http://10.0.11.179:5000")
    mlflow.set_experiment("sixdee_experiments")

    # helpful debug prints
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    exp = mlflow.get_experiment_by_name("sixdee_experiments")
    print("Using experiment:", exp)

    model = StudentOfferLabelModel(threshold=args.threshold, epochs=args.epochs)
    acc = model.fit()

    run_name = "sixdee_logistic_regression_placement_prediction"

    signature = ModelSignature(
        inputs=Schema([ColSpec("double", "marks")]),
        outputs=Schema([ColSpec("string")]),
    )

    # 🚨 Now start the run after setting the experiment
    run_ctx = mlflow.start_run(run_name=run_name)

    with run_ctx:
        run = mlflow.active_run()
        ...
        print("RUN_ID:", run.info.run_id)
        print("Run name:", run_name)
        exp = mlflow.get_experiment(run.info.experiment_id)
        print("Experiment:", exp.name if exp else run.info.experiment_id)

