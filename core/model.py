"""Model artifacts and the exact preprocessing used in the training notebook.

Training pipeline (notebooks/fraud-risk-analytics.ipynb):
  1. cap amount and the four balance columns to their 10th/90th percentiles  (cell 50)
  2. balance_diff_org  = newbalanceOrig - oldbalanceOrg
     balance_diff_dest = newbalanceDest - oldbalanceDest                       (cell 128)
  3. one-hot encode `type` with drop_first=True (CASH_IN is the baseline)      (cell 132)
  4. MinMax-scale amount and both balance differences; `step` stays unscaled  (cell 134)

`engineer_features` covers steps 1-2 for raw transaction details.
`model_input` covers steps 3-4 for rows that are already engineered, such as
the bundled sample, which was exported after step 2.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
PREPROCESSING = json.loads((MODEL_DIR / "preprocessing.json").read_text(encoding="utf-8"))
CAPS: dict[str, tuple[float, float]] = {k: tuple(v) for k, v in PREPROCESSING["capping"].items()}
TRANSACTION_TYPES: list[str] = PREPROCESSING["transaction_types"]
SCALED_COLUMNS: list[str] = PREPROCESSING["scaled_columns"]
DEFAULT_THRESHOLD = 0.5
FEATURE_FRAME_COLUMNS = ["step", "type", "amount", "balance_diff_org", "balance_diff_dest"]


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the trained network, the fitted scaler and the feature order."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf  # imported lazily: slow, and only needed for scoring

    missing = [
        name
        for name in ("model.h5", "scaler.pkl", "feature_columns.pkl")
        if not (MODEL_DIR / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing model files in {MODEL_DIR}: {', '.join(missing)}")

    model = tf.keras.models.load_model(MODEL_DIR / "model.h5", compile=False)
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_columns = list(joblib.load(MODEL_DIR / "feature_columns.pkl"))
    return model, scaler, feature_columns


def engineer_features(
    *,
    step: int,
    tx_type: str,
    amount: float,
    old_balance_origin: float,
    new_balance_origin: float,
    old_balance_dest: float,
    new_balance_dest: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Turn raw transaction details into the engineered feature row.

    Returns the one-row feature frame and the names of inputs that were capped.
    """
    if tx_type not in TRANSACTION_TYPES:
        raise ValueError(f"Unknown transaction type: {tx_type}")

    raw = {
        "amount": amount,
        "oldbalanceOrg": old_balance_origin,
        "newbalanceOrig": new_balance_origin,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
    }
    capped, adjusted = {}, []
    for name, value in raw.items():
        low, high = CAPS[name]
        clipped = float(min(max(float(value), low), high))
        if not np.isclose(clipped, float(value)):
            adjusted.append(name)
        capped[name] = clipped

    row = pd.DataFrame(
        [
            {
                "step": float(step),
                "type": tx_type,
                "amount": capped["amount"],
                "balance_diff_org": capped["newbalanceOrig"] - capped["oldbalanceOrg"],
                "balance_diff_dest": capped["newbalanceDest"] - capped["oldbalanceDest"],
            }
        ]
    )
    return row, adjusted


def model_input(features: pd.DataFrame) -> pd.DataFrame:
    """Encode and scale engineered rows exactly as in notebook cells 132-134."""
    _, scaler, feature_columns = load_artifacts()
    frame = features[FEATURE_FRAME_COLUMNS].copy()
    frame["type"] = pd.Categorical(frame["type"], categories=TRANSACTION_TYPES)
    encoded = pd.get_dummies(frame, columns=["type"], drop_first=True, dtype=float)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0.0).astype(float)
    encoded[SCALED_COLUMNS] = scaler.transform(encoded[SCALED_COLUMNS])
    return encoded


def score(features: pd.DataFrame) -> np.ndarray:
    """Model scores (sigmoid outputs) for engineered rows."""
    model, _, _ = load_artifacts()
    if features.empty:
        return np.array([], dtype=float)
    values = model_input(features).to_numpy(dtype=np.float32)
    return model.predict(values, verbose=0, batch_size=8192).ravel().astype(float)


def is_flagged(scores, threshold: float = DEFAULT_THRESHOLD):
    """Notebook decision rule: fraud when the score is above the threshold."""
    return np.asarray(scores) > threshold
