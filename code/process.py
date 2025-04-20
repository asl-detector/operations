#!/usr/bin/env python
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import sys
import traceback

# Add debug statements to see what's happening
print("Starting process.py script")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Add the directory containing train.py to the Python path
sys.path.append("/opt/ml/processing/code")
print(f"Python path: {sys.path}")

try:
    # Import from your existing train.py
    print("Attempting to import extract_features from train")
    from train import extract_features

    print("Successfully imported extract_features")
except ImportError as e:
    print(f"Failed to import extract_features: {e}")
    print(
        f"Directory contents of /opt/ml/processing/code: {os.listdir('/opt/ml/processing/code')}"
    )
    sys.exit(1)


def process_json_files(input_dir, output_dir, window_size_sec=3.0, overlap_ratio=0.5):
    """Process all JSON files in the input directory and extract features"""
    print(f"Processing files from {input_dir} to {output_dir}")

    # Check if the input directory exists and what's in it
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory {input_dir} does not exist!")
        sys.exit(1)

    print(f"Input directory contents: {os.listdir(input_dir)}")

    # Create output directories
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/validation", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    # Find all JSON files
    asl_dir = os.path.join(input_dir, "asl")
    no_asl_dir = os.path.join(input_dir, "no_asl")

    # Check if subdirectories exist
    asl_dir_exists = os.path.exists(asl_dir)
    no_asl_dir_exists = os.path.exists(no_asl_dir)

    print(f"ASL directory exists: {asl_dir_exists}")
    print(f"Non-ASL directory exists: {no_asl_dir_exists}")

    asl_files = []
    no_asl_files = []

    # Look for files in subdirectories if they exist
    if asl_dir_exists:
        asl_files = glob.glob(os.path.join(asl_dir, "*.json"))

    if no_asl_dir_exists:
        no_asl_files = glob.glob(os.path.join(no_asl_dir, "*.json"))

    # If no files found in subdirectories, look directly in input_dir
    if len(asl_files) == 0 and len(no_asl_files) == 0:
        print(
            "No files found in asl/no_asl subdirectories. Looking for files directly in input directory."
        )
        all_json_files = glob.glob(os.path.join(input_dir, "*.json"))
        print(
            f"Found {len(all_json_files)} JSON files in input directory: {all_json_files}"
        )

        # Classify files based on filename
        for file_path in all_json_files:
            filename = os.path.basename(file_path).lower()
            if "asl" in filename or "ash" in filename:
                print(f"Classifying {filename} as ASL file")
                asl_files.append(file_path)
            else:
                print(f"Classifying {filename} as non-ASL file")
                no_asl_files.append(file_path)

    print(f"Found {len(asl_files)} ASL files and {len(no_asl_files)} non-ASL files")

    if len(asl_files) == 0 and len(no_asl_files) == 0:
        print("ERROR: No JSON files found to process!")
        sys.exit(1)
    asl_features = []
    for file_path in asl_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            fps = data.get("video_info", {}).get("fps", 30.0)
            frames = data.get("frames", [])

            if not frames:
                continue

            window_frames = int(window_size_sec * fps)
            step_size = int(window_frames * (1 - overlap_ratio))
            step_size = max(1, step_size)

            for i in range(0, len(frames), step_size):
                end_idx = min(i + window_frames, len(frames))
                if end_idx - i < window_frames // 2:
                    continue

                window_data = frames[i:end_idx]
                features = extract_features(window_data, fps)

                features["file_name"] = os.path.basename(file_path)
                features["window_start_frame"] = i
                features["window_end_frame"] = end_idx
                features["window_duration"] = (end_idx - i) / fps
                features["label"] = 1

                asl_features.append(features)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Process non-ASL files
    no_asl_features = []
    for file_path in no_asl_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            fps = data.get("video_info", {}).get("fps", 30.0)
            frames = data.get("frames", [])

            if not frames:
                continue

            window_frames = int(window_size_sec * fps)
            step_size = int(window_frames * (1 - overlap_ratio))
            step_size = max(1, step_size)

            for i in range(0, len(frames), step_size):
                end_idx = min(i + window_frames, len(frames))
                if end_idx - i < window_frames // 2:
                    continue

                window_data = frames[i:end_idx]
                try:
                    features = extract_features(window_data, fps)

                    features["file_name"] = os.path.basename(file_path)
                    features["window_start_frame"] = i
                    features["window_end_frame"] = end_idx
                    features["window_duration"] = (end_idx - i) / fps
                    features["label"] = 0

                    no_asl_features.append(features)
                except Exception as e:
                    print(f"Error extracting features from {file_path}: {e}")
                    continue
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Combine features
    all_features = asl_features + no_asl_features

    if not all_features:
        print("No features extracted.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_features)
    df = df.dropna(axis=1, how="all")
    df = df.fillna(0)

    # Split data
    X = df.drop(
        [
            "label",
            "file_name",
            "window_start_frame",
            "window_end_frame",
            "window_duration",
        ],
        axis=1,
        errors="ignore",
    )
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Save datasets
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(f"{output_dir}/train/training.csv", index=False)
    val_df.to_csv(f"{output_dir}/validation/validation.csv", index=False)
    test_df.to_csv(f"{output_dir}/test/test.csv", index=False)

    print(
        f"Saved {len(train_df)} training samples, {len(val_df)} validation samples, and {len(test_df)} test samples"
    )


if __name__ == "__main__":
    # SageMaker Processing paths
    input_dir = "/opt/ml/processing/input/pose"
    output_dir = "/opt/ml/processing/output/features"
    process_json_files(input_dir, output_dir)
