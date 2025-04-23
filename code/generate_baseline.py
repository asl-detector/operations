#!/usr/bin/env python
import os
import json
import numpy as np
import pandas as pd
import argparse
from local_train import extract_features


def generate_baseline_files(input_csv=None, input_json_dir=None, output_dir="model"):
    """
    Generate baseline constraints and statistics from either:
    - An existing CSV file with features
    - A directory of JSON files that will be processed to extract features
    """
    print("Generating baseline files for monitoring...")

    # Ensure output directories exist
    os.makedirs(os.path.join(output_dir, "constraints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "statistics"), exist_ok=True)

    # Load or create feature data
    if input_csv:
        print(f"Loading features from CSV: {input_csv}")
        df = pd.read_csv(input_csv)
        # Remove label and metadata columns if present
        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                "label",
                "file_name",
                "window_start_frame",
                "window_end_frame",
                "window_duration",
            ]
        ]
        features_df = df[feature_cols]
    elif input_json_dir:
        print(f"Extracting features from JSON files in: {input_json_dir}")
        features = []

        # Find all JSON files
        import glob

        json_files = glob.glob(os.path.join(input_json_dir, "*.json"))
        print(f"Found {len(json_files)} JSON files")

        # Process files
        for file_path in json_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                fps = data.get("video_info", {}).get("fps", 30.0)
                frames = data.get("frames", [])

                if not frames:
                    print(f"Warning: No frames found in {file_path}")
                    continue

                # Instead of just using first window, process multiple windows with overlap
                window_size_sec = 3.0
                window_frames = int(window_size_sec * fps)
                overlap_ratio = 0.5
                step_size = int(window_frames * (1 - overlap_ratio))
                step_size = max(1, step_size)

                # Process up to 5 windows per file to keep baseline generation efficient
                window_count = 0
                for i in range(0, min(len(frames), 5 * step_size), step_size):
                    end_idx = min(i + window_frames, len(frames))
                    if end_idx - i < window_frames // 2:
                        continue

                    window_data = frames[i:end_idx]

                    # Extract features from this window
                    feature_set = extract_features(window_data, fps)

                    # Only add if we got meaningful features
                    if feature_set.get("hand_presence_ratio", 0) > 0.1:
                        features.append(feature_set)
                        window_count += 1

                    # Limit number of windows per file
                    if window_count >= 5:
                        break

                print(f"Extracted {window_count} feature windows from {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        if not features:
            raise ValueError("No features extracted from JSON files")

        features_df = pd.DataFrame(features)
        print(f"Created feature dataframe with {len(features_df)} samples")
    else:
        raise ValueError("Either input_csv or input_json_dir must be provided")

    print(f"Loaded {len(features_df)} samples with {len(features_df.columns)} features")

    # Generate data constraints
    constraints = {"version": 0.1, "features": []}

    # Generate statistics
    statistics = {"version": 0.1, "features": []}

    # Process each feature to create constraints and statistics
    for column in features_df.columns:
        feature_values = features_df[column].dropna()

        # Skip if no valid data
        if len(feature_values) == 0:
            continue

        # Determine the type
        if np.issubdtype(feature_values.dtype, np.integer):
            inferred_type = "Integral"
        elif np.issubdtype(feature_values.dtype, np.floating):
            inferred_type = "Fractional"
        else:
            inferred_type = "String"

        # Calculate statistics
        completeness = len(feature_values) / len(features_df)

        # Create feature constraint
        feature_constraint = {
            "name": column,
            "inferred_type": inferred_type,
            "completeness": completeness,
        }

        # Add min/max for numeric types
        if inferred_type in ["Integral", "Fractional"]:
            feature_constraint["min"] = float(feature_values.min())
            feature_constraint["max"] = float(feature_values.max())

            # Add a baseline margin to avoid false positives
            # Widen the min/max by 20% to account for normal variation
            range_width = feature_constraint["max"] - feature_constraint["min"]
            feature_constraint["min"] -= range_width * 0.1
            feature_constraint["max"] += range_width * 0.1

        constraints["features"].append(feature_constraint)

        # Create feature statistics
        feature_statistics = {
            "name": column,
            "inferred_type": inferred_type,
            "count": int(len(feature_values)),
            "completeness": completeness,
        }

        # Add statistics for numeric types
        if inferred_type in ["Integral", "Fractional"]:
            feature_statistics["min"] = float(feature_values.min())
            feature_statistics["max"] = float(feature_values.max())
            feature_statistics["mean"] = float(feature_values.mean())
            feature_statistics["stddev"] = float(feature_values.std())

            # Add quantiles
            quantiles = feature_values.quantile([0.25, 0.5, 0.75]).to_dict()
            feature_statistics["distribution"] = {
                "quantiles": {
                    "quantile_25": float(quantiles[0.25]),
                    "quantile_50": float(quantiles[0.5]),
                    "quantile_75": float(quantiles[0.75]),
                }
            }

        statistics["features"].append(feature_statistics)

    # Save constraints file
    constraints_path = os.path.join(output_dir, "constraints", "data_constraints.json")
    with open(constraints_path, "w") as f:
        json.dump(constraints, f, indent=2)

    # Save statistics file
    statistics_path = os.path.join(output_dir, "statistics", "data_statistics.json")
    with open(statistics_path, "w") as f:
        json.dump(statistics, f, indent=2)

    # Create simple quality constraints and statistics
    quality_constraints = {
        "version": 0.1,
        "regression_constraints": {
            "accuracy": {"threshold": 0.8},
            "auc": {"threshold": 0.8},
        },
    }

    quality_statistics = {
        "version": 0.1,
        "metrics": {"accuracy": {"value": 0.85}, "auc": {"value": 0.9}},
    }

    # Save quality constraints and statistics
    quality_constraints_path = os.path.join(
        output_dir, "constraints", "quality_constraints.json"
    )
    with open(quality_constraints_path, "w") as f:
        json.dump(quality_constraints, f, indent=2)

    quality_statistics_path = os.path.join(
        output_dir, "statistics", "quality_statistics.json"
    )
    with open(quality_statistics_path, "w") as f:
        json.dump(quality_statistics, f, indent=2)

    print(f"Successfully created baseline files:")
    print(f"- {constraints_path}")
    print(f"- {statistics_path}")
    print(f"- {quality_constraints_path}")
    print(f"- {quality_statistics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate baseline files for SageMaker model monitoring"
    )
    parser.add_argument("--csv", type=str, help="Path to input CSV file with features")
    parser.add_argument(
        "--json-dir", type=str, help="Path to directory containing JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model",
        help="Output directory for baseline files",
    )

    args = parser.parse_args()

    if not args.csv and not args.json_dir:
        parser.error("Either --csv or --json-dir must be provided")

    generate_baseline_files(args.csv, args.json_dir, args.output_dir)
