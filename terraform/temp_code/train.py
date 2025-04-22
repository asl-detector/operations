import argparse
import os
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import json
from sklearn.metrics import accuracy_score, roc_auc_score


def parse_args():
    """Parse the arguments passed to the script."""
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )

    # Hyperparameters
    parser.add_argument("--num-round", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--min-child-weight", type=float, default=2.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--eval-metric", type=str, default="auc")

    return parser.parse_args()


def load_data(train_path, validation_path):
    """Load training and validation data."""
    train_files = [
        os.path.join(train_path, file)
        for file in os.listdir(train_path)
        if file.endswith(".csv")
    ]
    train_df = pd.concat([pd.read_csv(file) for file in train_files])

    if validation_path:
        validation_files = [
            os.path.join(validation_path, file)
            for file in os.listdir(validation_path)
            if file.endswith(".csv")
        ]
        validation_df = pd.concat([pd.read_csv(file) for file in validation_files])
    else:
        validation_df = None

    return train_df, validation_df


def train_model(args, train_df, validation_df):
    """Train the XGBoost model."""
    # Prepare data
    y_train = train_df.iloc[:, 0].values
    X_train = train_df.iloc[:, 1:].values

    dtrain = xgb.DMatrix(X_train, label=y_train)

    if validation_df is not None:
        y_val = validation_df.iloc[:, 0].values
        X_val = validation_df.iloc[:, 1:].values
        dval = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(dtrain, "train"), (dval, "validation")]
    else:
        dval = None
        watchlist = [(dtrain, "train")]

    # Set up hyperparameters
    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "objective": args.objective,
        "eval_metric": args.eval_metric,
    }

    # Train model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=watchlist,
        verbose_eval=10,
    )

    # Calculate metrics
    if dval is not None:
        val_pred = model.predict(dval)
        auc = roc_auc_score(y_val, val_pred)
        accuracy = accuracy_score(y_val, val_pred > 0.5)
        print(f"Validation AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

    return model


def save_model(model, model_dir):
    """Save the model in multiple formats."""
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # 1. Save in native XGBoost format
    model_path = os.path.join(model_dir, "xgboost-model")
    model.save_model(model_path)
    print(f"Model saved in native format: {model_path}")

    # 2. Save in JSON format
    json_path = os.path.join(model_dir, "xgboost-model.json")
    model.save_model(json_path)
    print(f"Model saved in JSON format: {json_path}")

    # 3. Save in pickle format
    pkl_path = os.path.join(model_dir, "xgboost-model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved in pickle format: {pkl_path}")

    # 4. Create and save inference.py script for model serving
    inference_script = '''
import os
import json
import pickle
import numpy as np
import xgboost as xgb

def model_fn(model_dir):
    """Load a model. For XGBoost models we support loading the model in various formats."""
    model_files = os.listdir(model_dir)
    
    # Try loading different formats
    if 'xgboost-model.pkl' in model_files:
        with open(os.path.join(model_dir, 'xgboost-model.pkl'), 'rb') as f:
            model = pickle.load(f)
        print("Loaded model from pickle format")
    elif 'xgboost-model.json' in model_files:
        model = xgb.Booster()
        model.load_model(os.path.join(model_dir, 'xgboost-model.json'))
        print("Loaded model from JSON format")
    elif 'xgboost-model' in model_files:
        model = xgb.Booster()
        model.load_model(os.path.join(model_dir, 'xgboost-model'))
        print("Loaded model from native format")
    else:
        raise ValueError("No model file found in model directory")
    
    return model

def input_fn(request_body, request_content_type):
    """Parse input data payload"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        # Handle both array and dictionary inputs
        if isinstance(data, dict):
            data = data.get('instances', data.get('inputs', data))
        return xgb.DMatrix(np.array(data))
    elif request_content_type == 'text/csv':
        return xgb.DMatrix(np.array([float(i) for i in request_body.decode('utf-8').split(',')]).reshape(1, -1))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction with the input data"""
    return model.predict(input_data)

def output_fn(prediction, accept):
    """Format prediction output"""
    if accept == 'application/json':
        return json.dumps({'predictions': prediction.tolist()}), accept
    else:
        return ','.join(map(str, prediction.tolist())), 'text/csv'
'''
    with open(os.path.join(model_dir, "inference.py"), "w") as f:
        f.write(inference_script)
    print(f"Inference script saved: {os.path.join(model_dir, 'inference.py')}")


def main():
    """Main training function."""
    args = parse_args()

    # Load data
    print("Loading data...")
    train_df, validation_df = load_data(args.train, args.validation)
    print(f"Loaded training data: {train_df.shape}")
    if validation_df is not None:
        print(f"Loaded validation data: {validation_df.shape}")

    # Train model
    print("Training model...")
    model = train_model(args, train_df, validation_df)

    # Save model
    print("Saving model...")
    save_model(model, args.model_dir)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
