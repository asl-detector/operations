import xgboost as xgb


# Load the model
def load_model(model_path="xgboost-model.json"):
    """Load the model file"""
    model = xgb.Booster()
    model.load_model(model_path)
    return model


# Make predictions
def predict(model, input_data):
    """Make predictions using the model"""
    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(input_data)

    # Return predictions
    return model.predict(dmatrix)
