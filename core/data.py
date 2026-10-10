"""Transaction data: an optional MySQL `transactions` table, else the bundled sample.

The bundled CSV is the 50,000-row sample exported by notebook cell 130, after
outlier capping and balance-difference engineering. The MySQL table described in
the README holds the same rows, so both sources have identical columns.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "transactions_sample.csv"
COLUMNS = ["step", "type", "amount", "isFraud", "balance_diff_org", "balance_diff_dest"]
QUERY = "SELECT step, type, amount, isFraud, balance_diff_org, balance_diff_dest FROM transactions"


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    source: str          # "mysql" or "sample"
    note: str | None     # why the sample is in use when MySQL was configured


def _secret(key: str, env_key: str, default=None):
    """Read [mysql] values from Streamlit secrets, then environment variables."""
    try:
        import streamlit as st

        section = st.secrets.get("mysql", {})
        if key in section:
            return section[key]
    except Exception:  # no secrets.toml: fall through to the environment
        pass
    return os.getenv(env_key, default)


def mysql_configured() -> bool:
    return bool(_secret("host", "MYSQL_HOST"))


def _load_mysql() -> pd.DataFrame:
    import mysql.connector

    config = {
        "host": _secret("host", "MYSQL_HOST"),
        "user": _secret("user", "MYSQL_USER"),
        "password": _secret("password", "MYSQL_PASSWORD", ""),
        "database": _secret("database", "MYSQL_DATABASE", "fraud_risk_platform"),
        "port": int(_secret("port", "MYSQL_PORT", 3306)),
        "connection_timeout": 10,
    }
    ssl_ca = _secret("ssl_ca", "MYSQL_SSL_CA")
    ca_content = _secret("ssl_ca_content", "MYSQL_SSL_CA_CONTENT")
    if ca_content:
        handle = tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False)
        handle.write(ca_content)
        handle.close()
        ssl_ca = handle.name
    ssl_disabled = str(_secret("ssl_disabled", "MYSQL_SSL_DISABLED", "false")).lower() in {"1", "true", "yes"}
    if ssl_ca and not ssl_disabled:
        config.update(ssl_ca=ssl_ca, ssl_verify_cert=True)

    connection = mysql.connector.connect(**config)
    try:
        cursor = connection.cursor()
        cursor.execute(QUERY)
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=COLUMNS)
    finally:
        connection.close()


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    frame = frame[COLUMNS].copy()
    for column in ("step", "amount", "isFraud", "balance_diff_org", "balance_diff_dest"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    frame["step"] = frame["step"].astype(int)
    frame["isFraud"] = frame["isFraud"].astype(int)
    frame["type"] = frame["type"].astype(str)
    frame["day"] = (frame["step"] - 1) // 24 + 1
    frame["hour"] = (frame["step"] - 1) % 24
    return frame.reset_index(drop=True)


def load_dataset() -> Dataset:
    note = None
    if mysql_configured():
        try:
            return Dataset(_clean(_load_mysql()), "mysql", None)
        except Exception as exc:  # reported in the UI, not swallowed
            note = f"MySQL is configured but could not be read ({type(exc).__name__}: {exc})."
    return Dataset(_clean(pd.read_csv(SAMPLE_CSV)), "sample", note)
