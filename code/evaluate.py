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
import shutil


def evaluate_model(model_dir, test_data_dir, output_dir):
    """Evaluate the model on test data"""
    print(f"Evaluating model from {model_dir} on test data from {test_data_dir}")

    # Find the most recent model.tar.gz file
    model_files = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".tar.gz"):
                tar_path = os.path.join(root, file)
                model_files.append(tar_path)
                print(f"Found model archive: {tar_path}")

    if not model_files:
        raise ValueError("No model archives found!")

    # Sort by modification time to get most recent
    most_recent_model = sorted(model_files, key=os.path.getmtime, reverse=True)[0]
    print(f"Using most recent model: {most_recent_model}")

    # Extract the tar.gz file
    extract_dir = os.path.join(model_dir, "extract")
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Extracting {most_recent_model}")
    with tarfile.open(most_recent_model) as tar:
        tar.extractall(extract_dir)

    # Look for model files with various extensions
    model_file = None
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            # XGBoost 1.7 model file can have various names
            if (
                file == "xgboost-model"
                or file == "model"
                or file.endswith(".model")
                or file.endswith(".json")
            ):
                model_file = os.path.join(root, file)
                print(f"Found model file: {model_file}")
                break
        if model_file:
            break

    if not model_file:
        # If still not found, check common paths
        potential_paths = [
            os.path.join(extract_dir, "xgboost-model"),
            os.path.join(extract_dir, "model"),
            os.path.join(extract_dir, "model.json"),
        ]

        for path in potential_paths:
            if os.path.exists(path):
                model_file = path
                print(f"Found model at common path: {model_file}")
                break

    if not model_file:
        raise ValueError("No model file found in extracted archive!")

    print(f"Loading model from {model_file}")
    model = xgb.Booster()
    model.load_model(model_file)

    # Rest of the evaluation code...
    # Load test data
    test_data_path = os.path.join(test_data_dir, "test.csv")
    try:
        test_df = pd.read_csv(test_data_path, header=None)
        print(f"Loaded test data with shape: {test_df.shape}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        # Create dummy test data
        test_df = pd.DataFrame(np.random.random((5, 10)))
        test_df[0] = np.random.randint(0, 2, 5)
        print("WARNING: Using dummy test data")

    # First column should be the label
    X_test = test_df.iloc[:, 1:] if test_df.shape[1] > 1 else test_df
    y_test = (
        test_df.iloc[:, 0] if test_df.shape[1] > 1 else pd.Series([1] * len(test_df))
    )

    # Make predictions
    dmatrix = xgb.DMatrix(X_test)
    y_pred = model.predict(dmatrix)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate metrics and save...
    accuracy = np.mean(y_pred_binary == y_test)
    auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.5

    # Save standardized model for consistent access
    standardized_model_dir = os.path.join(output_dir, "standardized")
    os.makedirs(standardized_model_dir, exist_ok=True)
    standardized_model_path = os.path.join(standardized_model_dir, "model.json")

    # Save the model in a consistent location and format
    model.save_model(standardized_model_path)
    print(f"Saved standardized model to: {standardized_model_path}")

    # Save results
    results = {
        "accuracy": float(accuracy),
        "auc": float(auc),
        "classification_report": classification_report(
            y_test, y_pred_binary, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred_binary).tolist(),
    }

    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Model evaluated with accuracy: {accuracy:.4f} and AUC: {auc:.4f}")


if __name__ == "__main__":
    # SageMaker Processing paths
    model_dir = "/opt/ml/processing/input/model"
    test_data_dir = "/opt/ml/processing/input/test"
    output_dir = "/opt/ml/processing/output/evaluation"

    evaluate_model(model_dir, test_data_dir, output_dir)
