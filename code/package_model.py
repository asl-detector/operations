#!/usr/bin/env python
import os
import tarfile
import pickle


def package_model(model_path, output_dir):
    """Package the model for deployment"""
    print(f"Packaging model from {model_path}")

    # Create temporary package directory
    package_dir = os.path.join(output_dir, "package")
    os.makedirs(package_dir, exist_ok=True)

    # Find the actual model file
    model_data = None
    if os.path.isdir(model_path):
        # Search recursively for model files
        model_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file == "model" or file == "xgboost-model":
                    model_files.append(os.path.join(root, file))

        if not model_files:
            print("No model files found!")
            return

        # Use the most recent model file
        model_path = model_files[0]

    # Load the model
    try:
        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract the XGBoost model and other components
    if isinstance(model_data, dict) and "model" in model_data:
        xgb_model = model_data["model"]
        # Save other model components (scaler, feature names, etc.)
        metadata = {k: v for k, v in model_data.items() if k != "model"}
        metadata_path = os.path.join(package_dir, "model_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
    else:
        xgb_model = model_data

    xgb_model_path = os.path.join(package_dir, "xgboost-model.json")
    xgb_model.save_model(xgb_model_path)
    print(f"Saved XGBoost model to {xgb_model_path}")

    # Also save the pickle version for backward compatibility
    pkl_model_path = os.path.join(package_dir, "xgboost-model.pkl")
    with open(pkl_model_path, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"Saved pickle model to {pkl_model_path}")

    # Create tarball
    tar_path = os.path.join(output_dir, "model.tar.gz")
    print(f"Creating tarball at {tar_path}")

    with tarfile.open(tar_path, "w:gz") as tar:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                archive_name = os.path.relpath(file_path, package_dir)
                print(f"Adding {file_path} as {archive_name}")
                tar.add(file_path, arcname=archive_name)

    print(f"Model packaged successfully to {tar_path}")
