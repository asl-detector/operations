import argparse
import os
import json
import xgboost as xgb
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # Find the model file
    model_dir = args.model
    model_files = [
        f
        for f in os.listdir(model_dir)
        if f.endswith(".model") or f.endswith(".tar.gz")
    ]

    if not model_files:
        raise ValueError(f"No model files found in {model_dir}")

    model_path = os.path.join(model_dir, model_files[0])

    # Load the model
    if model_path.endswith(".tar.gz"):
        # Extract model from tarball if needed
        import tarfile

        with tarfile.open(model_path) as tar:
            tar.extractall(path=os.path.join(args.output_dir, "extracted"))
        # Find the extracted model file
        extracted_dir = os.path.join(args.output_dir, "extracted")
        for root, _, files in os.walk(extracted_dir):
            for file in files:
                if file.endswith(".model"):
                    model_path = os.path.join(root, file)
                    break

    # Load the XGBoost model
    try:
        # First try loading as a pickle file
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except:
        # If that fails, try loading as an XGBoost model
        model = xgb.Booster()
        model.load_model(model_path)

    # Convert model to JSON
    json_model = model.save_config()

    # Save JSON model
    output_path = os.path.join(args.output_dir, "model.json")
    with open(output_path, "w") as f:
        f.write(json_model)

    print(f"Model saved as JSON to {output_path}")


if __name__ == "__main__":
    main()
