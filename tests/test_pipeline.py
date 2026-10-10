"""Checks that the app reproduces the notebook's pipeline and reports consistent numbers.

    python tests/test_pipeline.py      # or: python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data import COLUMNS, load_dataset  # noqa: E402
from core.metrics import METRICS, precision_at_base_rate, full_dataset_fraud_rate, threshold_table  # noqa: E402
from core.model import CAPS, engineer_features, load_artifacts, model_input, score  # noqa: E402


def test_sample_loads_with_expected_shape():
    data = load_dataset()
    assert data.source == "sample"
    assert list(data.frame.columns[: len(COLUMNS)]) == COLUMNS
    assert len(data.frame) == 50_000
    assert int(data.frame["isFraud"].sum()) == 70


def test_capping_and_balance_changes_follow_the_notebook():
    row, adjusted = engineer_features(
        step=10, tx_type="TRANSFER", amount=2_500_000,
        old_balance_origin=2_500_000, new_balance_origin=0,
        old_balance_dest=0, new_balance_dest=0,
    )
    r = row.iloc[0]
    assert r["amount"] == CAPS["amount"][1]
    assert r["balance_diff_org"] == 0 - CAPS["oldbalanceOrg"][1]
    assert r["balance_diff_dest"] == 0
    assert set(adjusted) == {"amount", "oldbalanceOrg"}

    small, adjusted = engineer_features(
        step=1, tx_type="PAYMENT", amount=100, old_balance_origin=500, new_balance_origin=400,
        old_balance_dest=0, new_balance_dest=0,
    )
    assert small.iloc[0]["amount"] == CAPS["amount"][0]
    assert adjusted == ["amount"]


def test_scaler_matches_the_capping_constants():
    _, scaler, _ = load_artifacts()
    assert np.isclose(scaler.data_min_[0], CAPS["amount"][0])
    assert np.isclose(scaler.data_max_[0], CAPS["amount"][1])
    assert np.isclose(scaler.data_min_[1], -CAPS["oldbalanceOrg"][1])
    assert np.isclose(scaler.data_max_[2], CAPS["newbalanceDest"][1])


def test_encoding_order_baseline_and_unscaled_step():
    _, _, feature_columns = load_artifacts()
    rows = pd.DataFrame(
        [
            {"step": 5, "type": "CASH_IN", "amount": 10_000, "balance_diff_org": 0, "balance_diff_dest": 0},
            {"step": 7, "type": "TRANSFER", "amount": 10_000, "balance_diff_org": 0, "balance_diff_dest": 0},
        ]
    )
    x = model_input(rows)
    assert list(x.columns) == feature_columns
    assert x.loc[0, [c for c in feature_columns if c.startswith("type_")]].sum() == 0  # CASH_IN baseline
    assert x.loc[1, "type_TRANSFER"] == 1
    assert list(x["step"]) == [5.0, 7.0]


def test_metrics_file_is_internally_consistent():
    cm = METRICS["confusion_matrix"]
    total = sum(cm.values())
    assert total == METRICS["evaluation_set"]["rows"]
    assert np.isclose((cm["true_negative"] + cm["true_positive"]) / total, METRICS["accuracy"])
    assert np.isclose(cm["true_positive"] / (cm["true_positive"] + cm["false_negative"]), METRICS["fraud"]["recall"])
    assert np.isclose(cm["true_positive"] / (cm["true_positive"] + cm["false_positive"]), METRICS["fraud"]["precision"])
    assert 0.010 < precision_at_base_rate(full_dataset_fraud_rate()) < 0.020


def test_shipped_model_behaves_like_the_notebook_model():
    """On real class balance the shipped model flags ~8.4% of legitimate rows,
    the notebook's false-positive rate, which ties model.h5 to metrics.json."""
    df = load_dataset().frame
    scores = score(df)
    assert scores.shape == (len(df),)
    assert np.all((scores >= 0) & (scores <= 1))
    legit_flag_rate = float((scores[df["isFraud"].to_numpy() == 0] > 0.5).mean())
    notebook_fpr = METRICS["confusion_matrix"]["false_positive"] / METRICS["evaluation_set"]["legitimate"]
    assert abs(legit_flag_rate - notebook_fpr) < 0.02

    table = threshold_table(df["isFraud"].to_numpy(), scores, [0.5, 0.9])
    assert table["flagged"].is_monotonic_decreasing

    one = score(df.iloc[[0]])[0]
    assert np.isclose(one, scores[0], atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
