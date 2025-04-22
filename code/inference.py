import pickle
import xgboost as xgb


# Load the model
def load_model(model_path="xgboost-model"):
    """Load the model file"""
    with open(model_path, "rb") as f:
        return pickle.load(f)


# Make predictions
def predict(model_data, input_data):
    """Make predictions using the model"""
    # Extract model and preprocessing components
    if isinstance(model_data, dict) and "model" in model_data:
        model = model_data["model"]
        scaler = model_data.get("scaler")
    else:
        model = model_data
        scaler = None

    # Preprocess if needed
    if scaler is not None:
        input_data = scaler.transform(input_data)

    # Handle different model types
    if isinstance(model, xgb.XGBModel):
        return model.predict_proba(input_data)[:, 1]
    elif isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(input_data)
        return model.predict(dmatrix)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


# Example usage
if __name__ == "__main__":
    # This code runs when the script is executed directly
    model_data = load_model()
    print("Model loaded successfully")
    print(
        f"Model type: {type(model_data['model'] if isinstance(model_data, dict) else model_data)}"
    )
