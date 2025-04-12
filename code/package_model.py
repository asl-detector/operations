#!/usr/bin/env python
import os
import shutil
import tarfile
import argparse


def package_model(model_path, output_dir):
    """Package the model for SageMaker deployment"""
    print(f"Packaging model from {model_path}")

    # Create temporary package directory
    package_dir = os.path.join(output_dir, "package")
    os.makedirs(package_dir, exist_ok=True)
    os.makedirs(os.path.join(package_dir, "code"), exist_ok=True)

    # Copy model file
    shutil.copy(model_path, os.path.join(package_dir, "xgboost-model"))

    # Copy inference script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    shutil.copy(
        os.path.join(script_dir, "inference.py"),
        os.path.join(package_dir, "code", "inference.py"),
    )

    # Create tarball
    with tarfile.open(os.path.join(output_dir, "initial-model.tar.gz"), "w:gz") as tar:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                archive_name = os.path.relpath(file_path, package_dir)
                tar.add(file_path, arcname=archive_name)

    print(
        f"Model packaged successfully to {os.path.join(output_dir, 'initial-model.tar.gz')}"
    )

    # Clean up
    shutil.rmtree(package_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Package a model for SageMaker deployment"
    )
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--output-dir", default="model", help="Output directory")

    args = parser.parse_args()
    package_model(args.model, args.output_dir)
