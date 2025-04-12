import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import io

# Load the model at startup
model = None


def model_fn(model_dir):
    """Load the model from the model_dir"""
    global model
    model_path = os.path.join(model_dir, "xgboost-model")
    model = pickle.load(open(model_path, "rb"))
    return model


def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == "text/csv":
        # Parse CSV data
        df = pd.read_csv(io.StringIO(request_body))
        return xgb.DMatrix(df.values)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions using the model"""
    return model.predict(input_data)


def output_fn(prediction, response_content_type):
    """Format the prediction output"""
    if response_content_type == "text/csv":
        # Convert predictions to binary class
        binary_preds = (prediction > 0.5).astype(int)
        # Combine probability and class prediction
        results = np.column_stack((prediction, binary_preds))
        return ",".join([str(x) for x in results.flatten()])
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
