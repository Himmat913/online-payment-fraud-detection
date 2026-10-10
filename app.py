"""Fraud Risk Analytics — Streamlit entry point.

Pages live in views/. Data loading is in core/data.py, the model and its
notebook-identical preprocessing in core/model.py, and presentation in ui/.
"""

from pathlib import Path

import streamlit as st

from core import cache
from ui import components as ui

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="Fraud Risk Analytics",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(ui.load_css(ROOT / "ui" / "styles.css"), unsafe_allow_html=True)

pages = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/transactions.py", title="Transactions", url_path="transactions"),
    st.Page("views/score.py", title="Score a transaction", url_path="score"),
    st.Page("views/model.py", title="Model", url_path="model"),
]
current = st.navigation(pages, position="hidden")

with st.container(horizontal=True, key="fr-nav", horizontal_alignment="distribute", vertical_alignment="center"):
    st.html('<p class="fr-wordmark">fraud risk analytics</p>')
    with st.container(horizontal=True, key="fr-nav-links", gap="small", width="content"):
        for page in pages:
            if page.url_path == current.url_path:
                st.html(ui.nav_current(page.title))
            else:
                st.page_link(page)

current.run()

st.html(ui.footer(cache.dataset().source))
