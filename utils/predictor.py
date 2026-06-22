import os
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache
from tensorflow.keras.models import load_model


MODEL_PATH = "model/model.h5"
SCALER_PATH = "model/scaler.pkl"
FEATURE_COLUMNS_PATH = "model/feature_columns.pkl"


@lru_cache(maxsize=1)
def load_prediction_artifacts():
    """
    Loads trained model, scaler, and feature columns.
    Uses joblib because the scaler and feature column files were saved using joblib.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    if not os.path.exists(FEATURE_COLUMNS_PATH):
        raise FileNotFoundError(f"Feature columns file not found: {FEATURE_COLUMNS_PATH}")

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, scaler, feature_columns


def get_scaler_columns(scaler, input_df):
    """
    Finds which columns the scaler was originally fitted on.
    Newer sklearn scalers store this in feature_names_in_.
    """

    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)

    numeric_candidates = [
        "amount",
        "balance_diff_org",
        "balance_diff_dest"
    ]

    return [col for col in numeric_candidates if col in input_df.columns]


def prepare_transaction_input(
    step,
    tx_type,
    amount,
    balance_diff_org,
    balance_diff_dest
):
    """
    Converts raw transaction input into the exact model feature format.
    Only scales the columns that the saved scaler was trained on.
    """

    _, scaler, feature_columns = load_prediction_artifacts()

    input_data = {
        "step": float(step),
        "amount": float(amount),
        "balance_diff_org": float(balance_diff_org),
        "balance_diff_dest": float(balance_diff_dest),
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 0,
    }

    encoded_type_col = f"type_{tx_type}"

    if encoded_type_col in input_data:
        input_data[encoded_type_col] = 1

    input_df = pd.DataFrame([input_data])

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    model_input = input_df.copy()

    scaler_columns = get_scaler_columns(scaler, input_df)

    valid_scaler_columns = [
        col for col in scaler_columns
        if col in model_input.columns
    ]

    if valid_scaler_columns:
        model_input[valid_scaler_columns] = scaler.transform(
            model_input[valid_scaler_columns]
        )

    return model_input, input_df, valid_scaler_columns


def predict_fraud_risk(
    step,
    tx_type,
    amount,
    balance_diff_org,
    balance_diff_dest,
    threshold=0.5
):
    """
    Predicts fraud risk using the trained DNN model.
    """

    model, _, _ = load_prediction_artifacts()

    model_input, input_df, scaled_columns = prepare_transaction_input(
        step=step,
        tx_type=tx_type,
        amount=amount,
        balance_diff_org=balance_diff_org,
        balance_diff_dest=balance_diff_dest
    )

    probability = float(model.predict(model_input, verbose=0)[0][0])
    prediction = 1 if probability >= threshold else 0

    return {
        "probability": probability,
        "risk_percent": probability * 100,
        "prediction": prediction,
        "label": "Fraud" if prediction == 1 else "Non-Fraud",
        "threshold": threshold,
        "features_used": input_df,
        "model_input": model_input,
        "scaled_columns": scaled_columns
    }