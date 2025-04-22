#!/usr/bin/env python
import os
import tarfile
import shutil


def package_model(model_path, output_dir):
    """Package the model for deployment"""
    print(f"Packaging model from {model_path}")

    # Create temporary package directory
    package_dir = os.path.join(output_dir, "package")
    os.makedirs(package_dir, exist_ok=True)

    # Find JSON model files
    model_file = None
    if os.path.isdir(model_path):
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(".json"):
                    model_file = os.path.join(root, file)
                    break
            if model_file:
                break
    else:
        model_file = model_path

    if not model_file:
        print("No model file found!")
        return

    # Just copy the model file to the package directory
    shutil.copy(model_file, os.path.join(package_dir, "xgboost-model.json"))

    # Create tarball
    tar_path = os.path.join(output_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for file in os.listdir(package_dir):
            tar.add(os.path.join(package_dir, file), arcname=file)

    print(f"Model packaged successfully to {tar_path}")
