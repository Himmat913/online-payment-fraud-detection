"""Streamlit-cached access to data, the model and per-row scores."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core import model
from core.data import Dataset, load_dataset


@st.cache_data(show_spinner="Loading transactions…", ttl=3600)
def dataset() -> Dataset:
    return load_dataset()


@st.cache_resource(show_spinner="Loading the fraud model…")
def warm_model():
    return model.load_artifacts()


@st.cache_data(show_spinner="Scoring transactions…")
def scored(_data: Dataset, source: str, rows: int) -> pd.DataFrame:
    """The dataset with a model score per row. `source` and `rows` key the cache."""
    warm_model()
    frame = _data.frame.copy()
    frame["score"] = model.score(frame)
    return frame
