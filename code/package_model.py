#!/usr/bin/env python
import os
import shutil
import tarfile
import argparse
import glob


def package_model(model_path, output_dir):
    """Package the model for SageMaker deployment"""
    print(f"Packaging model from {model_path}")

    # Check if model_path is a directory or file
    if os.path.isdir(model_path):
        print(f"Model path is a directory, searching for model files...")

        # List contents of the directory
        print(f"Contents of {model_path}: {os.listdir(model_path)}")

        # Search for model.tar.gz files in subdirectories
        tar_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(".tar.gz"):
                    file_path = os.path.join(root, file)
                    print(f"Found tar.gz file: {file_path}")
                    tar_files.append(file_path)

        if not tar_files:
            print("No model tar.gz files found in the directory structure!")
            return

        # Use the most recent tar file (assuming it's the last in alphabetical order)
        model_tar_path = tar_files[-1]
        print(f"Using model tar file: {model_tar_path}")

        # Extract the model file to a temporary directory
        tmp_dir = os.path.join(model_path, "tmp_extract")
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            with tarfile.open(model_tar_path) as tar:
                tar.extractall(path=tmp_dir)

            print(f"Extracted files: {os.listdir(tmp_dir)}")

            # Look for the model file
            model_files = glob.glob(os.path.join(tmp_dir, "*model*"))
            if not model_files:
                print("No model files found in the extracted tar file!")
                return

            # Use the first model file
            extracted_model_path = model_files[0]
            print(f"Using extracted model file: {extracted_model_path}")

            # Update model_path to the extracted file
            model_path = extracted_model_path

        except Exception as e:
            print(f"Error extracting model: {e}")
            return
    else:
        print(f"Model path is a file: {model_path}")

    # Create temporary package directory
    package_dir = os.path.join(output_dir, "package")
    os.makedirs(package_dir, exist_ok=True)
    os.makedirs(os.path.join(package_dir, "code"), exist_ok=True)

    # Copy model file
    print(
        f"Copying model from {model_path} to {os.path.join(package_dir, 'xgboost-model')}"
    )
    shutil.copy(model_path, os.path.join(package_dir, "xgboost-model"))

    # Copy inference script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    inference_path = os.path.join(script_dir, "inference.py")
    print(f"Copying inference script from {inference_path}")

    if not os.path.exists(inference_path):
        print(f"Warning: inference.py not found at {inference_path}")
        print(f"Current directory contents: {os.listdir(script_dir)}")

        # Try to find inference.py in the code directory
        possible_paths = glob.glob(
            os.path.join(script_dir, "..", "**", "inference.py"), recursive=True
        )
        if possible_paths:
            inference_path = possible_paths[0]
            print(f"Found inference.py at {inference_path}")
        else:
            print("Could not find inference.py anywhere!")
            # Create a basic inference.py file
            inference_content = """
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import io

# Load the model at startup
model = None

def model_fn(model_dir):
    \"\"\"Load the model from the model_dir\"\"\"
    global model
    model_path = os.path.join(model_dir, "xgboost-model")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, request_content_type):
    \"\"\"Parse input data\"\"\"
    if request_content_type == "text/csv":
        # Parse CSV data
        df = pd.read_csv(io.StringIO(request_body))
        return xgb.DMatrix(df.values)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    \"\"\"Make predictions using the model\"\"\"
    return model.predict(input_data)

def output_fn(prediction, response_content_type):
    \"\"\"Format the prediction output\"\"\"
    if response_content_type == "text/csv":
        # Convert predictions to binary class
        binary_preds = (prediction > 0.5).astype(int)
        # Combine probability and class prediction
        results = np.column_stack((prediction, binary_preds))
        return ",".join([str(x) for x in results.flatten()])
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
"""
            with open(os.path.join(package_dir, "code", "inference.py"), "w") as f:
                f.write(inference_content)
            print("Created basic inference.py file")

    if os.path.exists(inference_path):
        shutil.copy(
            inference_path,
            os.path.join(package_dir, "code", "inference.py"),
        )

    # Create tarball
    tar_path = os.path.join(output_dir, "initial-model.tar.gz")
    print(f"Creating tarball at {tar_path}")

    with tarfile.open(tar_path, "w:gz") as tar:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                archive_name = os.path.relpath(file_path, package_dir)
                print(f"Adding {file_path} as {archive_name}")
                tar.add(file_path, arcname=archive_name)

    print(f"Model packaged successfully to {tar_path}")

    # List the contents of the output directory
    print(f"Output directory contents: {os.listdir(output_dir)}")

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
