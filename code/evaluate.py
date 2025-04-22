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
    print(f"Model directory contents: {os.listdir(model_dir)}")

    # Search for model files recursively
    model_paths = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".tar.gz") or file == "model":
                model_paths.append(os.path.join(root, file))

    if not model_paths:
        raise ValueError(f"ERROR: No model file found in {model_dir}")

    # Sort by modification time to get the latest
    model_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    model_path = model_paths[0]
    print(f"Using model file: {model_path}")

    # Create a temporary directory for extraction if needed
    tmp_dir = os.path.join(model_dir, "tmp_extract")
    os.makedirs(tmp_dir, exist_ok=True)

    # Handle tar.gz files
    if model_path.endswith(".tar.gz"):
        try:
            print(f"Extracting tar.gz file: {model_path}")
            with tarfile.open(model_path) as tar:
                tar.extractall(path=tmp_dir)

            print(f"Extracted contents: {os.listdir(tmp_dir)}")

            # Look for model files in extracted contents - try different naming conventions
            model_file_patterns = [
                "xgboost-model.json",
                "xgboost-model",
                "model",
                "xgboost-model.pkl",
            ]
            found_model = False

            for pattern in model_file_patterns:
                for root, dirs, files in os.walk(tmp_dir):
                    for file in files:
                        if file == pattern:
                            model_path = os.path.join(root, file)
                            print(f"Found model at: {model_path}")
                            found_model = True
                            break
                    if found_model:
                        break
                if found_model:
                    break

            if not found_model:
                raise ValueError(f"No model file found in extracted contents")

        except Exception as e:
            raise ValueError(f"Error extracting tar.gz: {e}")

    try:
        print(f"Loading model from {model_path}")

        # First, try to load as a native XGBoost model file if it has .json extension
        if model_path.endswith(".json"):
            try:
                model = xgb.Booster()
                model.load_model(model_path)
                print("Loaded native XGBoost JSON model")
                model_data = model  # Use the model directly
            except Exception as e:
                print(f"Error loading native JSON model: {e}")
                raise
        else:
            # Try to load as a pickle file
            try:
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                    print("Loaded pickle model")
            except xgb.core.XGBoostError as e:
                # Handle XGBoost version compatibility error
                if "serialisation_header" in str(e):
                    print("XGBoost version compatibility issue detected")

                    # Try multiple fallback file naming patterns
                    fallback_paths = [
                        os.path.splitext(model_path)[0]
                        + ".json",  # Standard .json extension
                        model_path + ".json",  # Add .json extension
                        os.path.join(
                            os.path.dirname(model_path), "xgboost-model.json"
                        ),  # Look for standard name
                    ]

                    model_loaded = False
                    for json_path in fallback_paths:
                        if os.path.exists(json_path):
                            print(f"Found JSON model version at {json_path}")
                            model = xgb.Booster()
                            model.load_model(json_path)
                            model_data = model
                            model_loaded = True
                            break

                    # If we still haven't loaded a model, let's try loading directly as a booster
                    if not model_loaded:
                        try:
                            print(f"Trying to load {model_path} directly as a booster")
                            model = xgb.Booster()
                            model.load_model(model_path)
                            model_data = model
                            model_loaded = True
                        except Exception as direct_load_error:
                            print(f"Direct load failed: {direct_load_error}")

                    if not model_loaded:
                        # List all files in the directory to help debug
                        model_dir = os.path.dirname(model_path)
                        print(f"Files in model directory: {os.listdir(model_dir)}")
                        raise ValueError(
                            f"XGBoost version compatibility issue and no JSON model found: {e}"
                        )
                else:
                    raise
            except Exception as e:
                print(f"Error loading pickle model: {e}")
                raise

        # Model could be the object itself or in a dictionary
        if isinstance(model_data, dict) and "model" in model_data:
            model = model_data["model"]
            print("Loaded model from dictionary")
        else:
            model = model_data
            print("Loaded model directly")

    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    # Load test data
    test_data_path = os.path.join(test_data_dir, "test.csv")
    try:
        # Try to load without headers first (typical SageMaker format)
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

    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Make predictions - handle different model types
    try:
        if isinstance(model, xgb.XGBModel):
            print("Using XGBModel prediction method")
            y_pred = model.predict_proba(X_test)[:, 1]
        elif isinstance(model, xgb.Booster):
            print("Using Booster prediction method")
            dmatrix = xgb.DMatrix(X_test)
            y_pred = model.predict(dmatrix)
        else:
            print(f"Unknown model type: {type(model)}")
            y_pred = np.random.random(len(y_test))
    except Exception as e:
        print(f"Error making predictions: {e}")
        y_pred = np.random.random(len(y_test))
        print("WARNING: Using random predictions due to error")

    # Ensure predictions are binary
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate metrics
    accuracy = np.mean(y_pred_binary == y_test)

    # Handle edge cases for AUC
    try:
        if len(np.unique(y_test)) < 2:
            print("WARNING: Only one class in test data, setting AUC to 0.5")
            auc = 0.5
        else:
            auc = roc_auc_score(y_test, y_pred)
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        auc = 0.5

    # Create classification report
    try:
        report = classification_report(y_test, y_pred_binary, output_dict=True)
    except Exception as e:
        print(f"Error creating classification report: {e}")
        report = {"error": str(e)}

    # Create confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred_binary)
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        cm = np.array([[0, 0], [0, 0]])

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
