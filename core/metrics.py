"""Reported notebook metrics plus figures derived from them or from the sample."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "model" / "metrics.json").read_text(encoding="utf-8"))


def notebook_rates() -> dict:
    cm = METRICS["confusion_matrix"]
    tpr = cm["true_positive"] / (cm["true_positive"] + cm["false_negative"])
    fpr = cm["false_positive"] / (cm["false_positive"] + cm["true_negative"])
    return {"recall": tpr, "false_positive_rate": fpr}


def precision_at_base_rate(base_rate: float) -> float:
    """Precision the notebook's recall and false-positive rate imply at a given fraud rate.

    The notebook's test set is 50% fraud, so its precision does not carry over
    to real traffic, where fraud is about 0.13% of transactions.
    """
    rates = notebook_rates()
    caught = rates["recall"] * base_rate
    false_alarms = rates["false_positive_rate"] * (1 - base_rate)
    return caught / (caught + false_alarms)


def full_dataset_fraud_rate() -> float:
    training = METRICS["training"]
    return training["dataset_fraud_rows"] / training["dataset_rows"]


def threshold_table(labels: np.ndarray, scores: np.ndarray, thresholds=None) -> pd.DataFrame:
    """Precision, recall and alert volume on labelled rows for a range of thresholds."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2).tolist() + [0.97, 0.99]
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)
    total_fraud = int(labels.sum())
    rows = []
    for t in sorted(set(thresholds)):
        flagged = scores > t
        tp = int((flagged & (labels == 1)).sum())
        n_flagged = int(flagged.sum())
        rows.append(
            {
                "threshold": float(t),
                "flagged": n_flagged,
                "flagged_share": n_flagged / len(labels) if len(labels) else 0.0,
                "fraud_caught": tp,
                "false_alarms": n_flagged - tp,
                "precision": tp / n_flagged if n_flagged else np.nan,
                "recall": tp / total_fraud if total_fraud else np.nan,
            }
        )
    return pd.DataFrame(rows)
