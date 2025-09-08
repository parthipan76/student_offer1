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
        return np.where(preds == 1, "Placed", "Not Placed").astype(str)

def canonicalize_number_str(x: float) -> str:
    s = str(x)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=80.0,
                        help="Marks cutoff for placement")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs")
    args = parser.parse_args()
    
    print("DEBUG: Running updated student_offer.py script")
    
    # 🔥 FIXED: Set tracking URI first
    mlflow.set_tracking_uri("http://10.0.11.179:5000")
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    
    # 🔥 FIXED: Create experiment if it doesn't exist
    experiment_name = "sixdee_experiments"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Creating experiment: {experiment_name}")
            experiment_id = mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment(experiment_id)
        else:
            print(f"Using existing experiment: {experiment_name}")
        
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Experiment Name: {experiment.name}")
        
    except Exception as e:
        print(f"Error with experiment setup: {e}")
        raise
    
    # 🔥 FIXED: Set experiment AFTER confirming it exists
    mlflow.set_experiment(experiment_name)
    
    # Verify the experiment is set correctly
    current_exp = mlflow.get_experiment_by_name(experiment_name)
    print(f"Current experiment confirmed: {current_exp.name} (ID: {current_exp.experiment_id})")
    
    # Train the model
    model = StudentOfferLabelModel(threshold=args.threshold, epochs=args.epochs)
    acc = model.fit()
    
    run_name = "sixdee_logistic_regression_placement_prediction"
    signature = ModelSignature(
        inputs=Schema([ColSpec("double", "marks")]),
        outputs=Schema([ColSpec("string")]),
    )
    
    # 🔥 FIXED: Start run AFTER setting experiment
    with mlflow.start_run(run_name=run_name) as run:
        print("RUN_ID:", run.info.run_id)
        print("Run name:", run_name)
        
        # Verify we're in the correct experiment
        exp = mlflow.get_experiment(run.info.experiment_id)
        print("Experiment:", exp.name if exp else run.info.experiment_id)
        print("MLflow UI:", f"http://10.0.11.179:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        
        # Log parameters
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("epochs", args.epochs)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", acc)
        
        # Log model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            signature=signature,
        )
        
        print(f"✅ Model logged successfully with accuracy: {acc}")
        print(f"🧪 View experiment at: http://10.0.11.179:5000/#/experiments/{run.info.experiment_id}")
