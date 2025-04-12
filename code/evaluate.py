#!/usr/bin/env python
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tarfile
import glob


def evaluate_model(model_dir, test_data_dir, output_dir):
    """Evaluate the model on test data"""
    print(f"Evaluating model from {model_dir} on test data from {test_data_dir}")

    # Find and extract model artifact
    model_tar_path = glob.glob(os.path.join(model_dir, "*.tar.gz"))[0]

    with tarfile.open(model_tar_path) as tar:
        tar.extractall(path=model_dir)

    # Load the model
    model_path = os.path.join(model_dir, "xgboost-model")
    model = pickle.load(open(model_path, "rb"))

    # Load test data
    test_data_path = os.path.join(test_data_dir, "test.csv")
    test_df = pd.read_csv(test_data_path)

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    # Make predictions
    y_pred = model.predict(xgb.DMatrix(X_test))
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate metrics
    accuracy = np.mean(y_pred_binary == y_test)
    auc = roc_auc_score(y_test, y_pred)

    # Create classification report
    report = classification_report(y_test, y_pred_binary, output_dict=True)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)

    # Save results
    results = {
        "accuracy": float(accuracy),
        "auc": float(auc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Model evaluated with accuracy: {accuracy:.4f} and AUC: {auc:.4f}")
    print(f"Results saved to {output_dir}/evaluation.json")


if __name__ == "__main__":
    # SageMaker Processing paths
    model_dir = "/opt/ml/processing/input/model"
    test_data_dir = "/opt/ml/processing/input/test"
    output_dir = "/opt/ml/processing/output/evaluation"

    evaluate_model(model_dir, test_data_dir, output_dir)
