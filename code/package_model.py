#!/usr/bin/env python
import os
import tarfile
import shutil
import xgboost as xgb
import sys


def package_model(model_path, output_dir):
    """Package the model for deployment using XGBoost 1.7.4"""
    print(f"Packaging model from {model_path}")
    print(f"XGBoost version: {xgb.__version__}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find model tarball
    model_tarball = None
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith(".tar.gz"):
                model_tarball = os.path.join(root, file)
                break
        if model_tarball:
            break

    if not model_tarball:
        print("No model tarball found!")
        return

    print(f"Found model tarball: {model_tarball}")

    # Create temporary directory for extraction
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Extract the tarball
        with tarfile.open(model_tarball, "r:gz") as tar:
            tar.extractall(temp_dir)

        # Look for the model file
        model_file = None
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".json") or file == "xgboost-model":
                    model_file = os.path.join(root, file)
                    break
            if model_file:
                break

        if not model_file:
            print("No model file found in the archive!")
            return

        print(f"Found model file: {model_file}")

        # If it's already a JSON file, just copy it
        if model_file.endswith(".json"):
            shutil.copy(model_file, os.path.join(output_dir, "xgboost-model.json"))
            print(
                f"Copied JSON model to: {os.path.join(output_dir, 'xgboost-model.json')}"
            )
        else:
            # Try to load and save as JSON
            try:
                # Load model
                bst = xgb.Booster()
                bst.load_model(model_file)

                # Save as JSON
                json_path = os.path.join(output_dir, "xgboost-model.json")
                bst.save_model(json_path)
                print(f"Saved model as JSON to: {json_path}")
            except Exception as e:
                print(f"Error converting model to JSON: {e}")
                # Fallback to copying original
                shutil.copy(model_file, os.path.join(output_dir, "xgboost-model"))
                print(
                    f"Copied original model to: {os.path.join(output_dir, 'xgboost-model')}"
                )

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    package_model(args.model, args.output_dir)
