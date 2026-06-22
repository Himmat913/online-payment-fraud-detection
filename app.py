import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from utils.predictor import predict_fraud_risk

try:
    from database.mysql_connection import run_query
except Exception:
    run_query = None

try:
    from sql.fraud_queries import TOP_FRAUD_TRANSACTIONS_QUERY
except Exception:
    TOP_FRAUD_TRANSACTIONS_QUERY = """
    SELECT
        transaction_id,
        step,
        type,
        amount,
        balance_diff_org,
        balance_diff_dest
    FROM transactions
    WHERE isFraud = 1
    ORDER BY amount DESC
    LIMIT 10;
    """

st.set_page_config(page_title="Fraud Risk Analytics Platform", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

BG = "#1a1a1a"
CARD = "#242424"
CARD_ALT = "#2a2a2a"
BORDER = "#3a3a3a"
RED = "#ef3b36"
RED_DARK = "#d32f2c"
GREEN = "#6aaa3b"
AMBER = "#f5a524"
ORANGE = "#ff7043"
BLUE = "#5b8def"
MUTED = "#9ca3af"
WHITE = "#f5f5f5"

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #1a1a1a 0%, #151515 100%);
}

.main .block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1440px;
}

#MainMenu, footer {visibility: hidden;}

[data-testid="stSidebar"] {
    background-color: #131313;
    border-right: 1px solid #2c2c2c;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.6rem;
}

.sidebar-brand {
    font-size: 1.25rem;
    font-weight: 800;
    color: #f5f5f5;
    margin-bottom: 2px;
}

.sidebar-sub {
    color: #9ca3af;
    font-size: 0.78rem;
    margin-bottom: 1.4rem;
}

div[role="radiogroup"] > label {
    background-color: #1c1c1c;
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    padding: 10px 14px !important;
    margin-bottom: 8px;
    width: 100%;
    transition: all 0.2s ease;
}

div[role="radiogroup"] > label:hover {
    border-color: #ef3b36;
    background-color: #221c1c;
}

h1 { color: #f5f5f5 !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2, h3, h4 { color: #f5f5f5 !important; font-weight: 700 !important; }
p, label, span { color: #c9ccd1; }
.stCaption, [data-testid="stCaptionContainer"] { color: #9ca3af !important; }

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
    gap: 16px;
    margin: 10px 0 26px 0;
}

.kpi-card {
    background: linear-gradient(150deg, #242424 0%, #1c1c1c 100%);
    border: 1px solid #3a3a3a;
    border-radius: 16px;
    padding: 20px 22px;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.kpi-card:hover {
    transform: translateY(-3px);
    border-color: #ef3b36;
    box-shadow: 0 10px 26px rgba(239,59,54,0.18);
}

.kpi-top { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.kpi-icon { font-size: 0.85rem; }
.kpi-label { color: #9ca3af; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; }
.kpi-value { color: #f5f5f5; font-size: 1.85rem; font-weight: 800; line-height: 1.15; }
.kpi-sub { font-size: 0.78rem; margin-top: 6px; font-weight: 600; }

.section-title { color: #f5f5f5; font-size: 1.05rem; font-weight: 700; margin-bottom: 4px; }
.section-sub { color: #9ca3af; font-size: 0.82rem; margin-bottom: 16px; }

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge-red { background: rgba(239,59,54,0.15); color: #ef3b36; border: 1px solid rgba(239,59,54,0.4); }
.badge-green { background: rgba(106,170,59,0.15); color: #6aaa3b; border: 1px solid rgba(106,170,59,0.4); }
.badge-amber { background: rgba(245,165,36,0.15); color: #f5a524; border: 1px solid rgba(245,165,36,0.4); }
.badge-orange { background: rgba(255,112,67,0.15); color: #ff7043; border: 1px solid rgba(255,112,67,0.4); }
.badge-blue { background: rgba(91,141,239,0.15); color: #5b8def; border: 1px solid rgba(91,141,239,0.4); }

.insight-card {
    background-color: #1a1a1a;
    border: 1px solid #3a3a3a;
    border-left: 3px solid #ef3b36;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}

.insight-title { color: #f5f5f5; font-weight: 700; font-size: 0.88rem; margin-bottom: 4px; }
.insight-text { color: #9ca3af; font-size: 0.84rem; line-height: 1.45; }

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #3a3a3a !important;
    border-radius: 16px !important;
    background-color: #1e1e1e;
}

[data-testid="stMetric"] {
    background: linear-gradient(150deg, #242424 0%, #1c1c1c 100%);
    border: 1px solid #3a3a3a;
    border-radius: 14px;
    padding: 16px 18px;
}

[data-testid="stMetricLabel"] { color: #9ca3af !important; }
[data-testid="stMetricValue"] { color: #f5f5f5 !important; }

.stButton button, .stDownloadButton button, .stFormSubmitButton button {
    background-color: #ef3b36 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.3rem !important;
    font-weight: 700 !important;
    transition: all 0.2s ease;
}

.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
    background-color: #d32f2c !important;
    box-shadow: 0 6px 18px rgba(239,59,54,0.35);
    transform: translateY(-1px);
}

div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border: 1px solid #3a3a3a;
    border-radius: 14px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #1a1a1a;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #3a3a3a;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #9ca3af;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #ef3b36 !important;
    color: white !important;
}

.streamlit-expanderHeader {
    background-color: #1c1c1c;
    border-radius: 10px;
    border: 1px solid #3a3a3a;
    color: #f5f5f5 !important;
    font-weight: 600;
}

div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="base-input"] {
    background-color: #1c1c1c !important;
    border-color: #3a3a3a !important;
    border-radius: 10px !important;
}

hr { border-color: #3a3a3a !important; }

.risk-result { text-align: center; padding: 18px; }
.risk-score-text { color: #9ca3af; font-size: 0.9rem; margin-top: 10px; }

::-webkit-scrollbar { height: 8px; width: 8px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 8px; }

@media (max-width: 768px) {
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def fmt_int(n):
    try:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            return "N/A"
        return f"{int(n):,}"
    except Exception:
        return "N/A"


def fmt_money(n, decimals=0):
    try:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            return "N/A"
        return f"${n:,.{decimals}f}"
    except Exception:
        return "N/A"


def fmt_pct(n, decimals=2):
    try:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            return "N/A"
        return f"{n:.{decimals}f}%"
    except Exception:
        return "N/A"


def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def first_existing(frame, candidates):
    if frame is None or frame.empty:
        return None
    for c in candidates:
        if c in frame.columns:
            return c
    return None


def style_fig(fig, height=400, showlegend=True):
    fig.update_layout(
        paper_bgcolor=CARD_ALT,
        plot_bgcolor=CARD_ALT,
        font_color=WHITE,
        font_family="Inter, sans-serif",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=showlegend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=CARD, font_color=WHITE, bordercolor=BORDER),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED)
    return fig


def kpi_card(label, value, sub=None, sub_color=GREEN, icon="◆"):
    sub_html = f'<div class="kpi-sub" style="color:{sub_color};">{sub}</div>' if sub else ""
    return f"""<div class="kpi-card">
<div class="kpi-top"><span class="kpi-icon" style="color:{RED};">{icon}</span><span class="kpi-label">{label}</span></div>
<div class="kpi-value">{value}</div>
{sub_html}
</div>"""


def render_kpis(cards):
    st.markdown('<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def section_header(title, sub=None):
    sub_html = f'<div class="section-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="section-title">{title}</div>{sub_html}', unsafe_allow_html=True)


def insight_card(title, text):
    st.markdown(f'<div class="insight-card"><div class="insight-title">{title}</div><div class="insight-text">{text}</div></div>', unsafe_allow_html=True)


def segmented_control_safe(key, options, default=None):
    default = default or options[0]
    try:
        val = st.segmented_control("", options, default=default, key=key, label_visibility="collapsed")
        return val if val else default
    except Exception:
        return st.radio("", options, index=options.index(default), horizontal=True, key=key + "_radio", label_visibility="collapsed")


@st.cache_data
def load_dashboard_data():
    """
    Primary source: MySQL transactions table.
    Fallback source: local CSV, so the app still runs if MySQL is unavailable.
    """

    query = """
    SELECT
        transaction_id,
        step,
        type,
        amount,
        isFraud,
        balance_diff_org,
        balance_diff_dest,
        created_at
    FROM transactions;
    """

    if run_query is not None:
        mysql_df = run_query(query)
        if mysql_df is not None and not mysql_df.empty:
            return mysql_df

    try:
        return pd.read_csv("data/dashboard_data.csv")
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_fraud_data():
    """
    Primary source: MySQL query for top fraud transactions.
    Fallback source: local fraud_transactions.csv.
    """

    if run_query is not None:
        mysql_df = run_query(TOP_FRAUD_TRANSACTIONS_QUERY)
        if mysql_df is not None and not mysql_df.empty:
            return mysql_df

    try:
        return pd.read_csv("data/fraud_transactions.csv")
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_metrics():
    try:
        with open("model/metrics.json", "r") as f:
            return json.load(f)
    except Exception:
        return {}


df = load_dashboard_data()
fraud_df = load_fraud_data()
metrics = load_metrics()

TYPE_COL = first_existing(df, ["type", "transaction_type", "Type"])
AMOUNT_COL = first_existing(df, ["amount", "Amount", "transaction_amount"])
FRAUD_COL = first_existing(df, ["isFraud", "is_fraud", "Fraud", "fraud"])
STEP_COL = first_existing(df, ["step", "time", "Time", "timestamp"])
ORIG_OLD_COL = first_existing(df, ["oldbalanceOrg", "oldbalanceOrig", "old_balance_orig"])
ORIG_NEW_COL = first_existing(df, ["newbalanceOrig", "newbalanceOrg", "new_balance_orig"])
DEST_OLD_COL = first_existing(df, ["oldbalanceDest", "old_balance_dest"])
DEST_NEW_COL = first_existing(df, ["newbalanceDest", "new_balance_dest"])
NAME_ORIG_COL = first_existing(df, ["nameOrig", "name_orig", "origin_account"])
NAME_DEST_COL = first_existing(df, ["nameDest", "name_dest", "dest_account"])
FLAGGED_COL = first_existing(df, ["isFlaggedFraud", "is_flagged_fraud"])


def render_overview():
    st.title("Fraud Risk Analytics & Detection Platform")
    st.caption("Transaction Monitoring • Fraud Detection • Risk Intelligence")

    if df.empty:
        st.warning("Dashboard data could not be loaded. Please verify data/dashboard_data.csv exists.")
        return

    total_transactions = len(df)
    fraud_transactions = int(df[FRAUD_COL].sum()) if FRAUD_COL else 0
    fraud_rate = (fraud_transactions / total_transactions * 100) if total_transactions else 0.0
    total_value = df[AMOUNT_COL].sum() if AMOUNT_COL else 0
    avg_amount = df[AMOUNT_COL].mean() if AMOUNT_COL else 0
    high_risk_threshold = df[AMOUNT_COL].quantile(0.95) if AMOUNT_COL else None
    high_risk_count = int((df[AMOUNT_COL] >= high_risk_threshold).sum()) if AMOUNT_COL else 0

    accuracy = safe_get(metrics, ["accuracy"])
    accuracy = accuracy * 100 if isinstance(accuracy, (int, float)) else None
    fraud_block = safe_get(metrics, ["fraud"]) or safe_get(metrics, ["1"]) or safe_get(metrics, ["Fraud"]) or {}
    precision = fraud_block.get("precision") if isinstance(fraud_block, dict) else None
    recall = fraud_block.get("recall") if isinstance(fraud_block, dict) else None
    precision = precision * 100 if isinstance(precision, (int, float)) else None
    recall = recall * 100 if isinstance(recall, (int, float)) else None

    cards = [
        kpi_card("Transactions", fmt_int(total_transactions)),
        kpi_card("Fraud Cases", fmt_int(fraud_transactions), sub_color=RED),
        kpi_card("Fraud Rate", fmt_pct(fraud_rate, 3), sub_color=RED),
        kpi_card("Model Accuracy", fmt_pct(accuracy) if accuracy is not None else "N/A", sub_color=GREEN),
        kpi_card("Precision", fmt_pct(precision) if precision is not None else "N/A", sub_color=GREEN),
        kpi_card("Recall", fmt_pct(recall) if recall is not None else "N/A", sub_color=GREEN),
        kpi_card("Total Value", fmt_money(total_value)),
        kpi_card("Avg Transaction", fmt_money(avg_amount, 2)),
        kpi_card("High-Risk Transactions", fmt_int(high_risk_count), sub="Top 5% by amount", sub_color=AMBER),
    ]
    render_kpis(cards)

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            section_header("Fraud Distribution", "Share of fraudulent vs legitimate transactions")
            if FRAUD_COL:
                counts = df[FRAUD_COL].value_counts().rename({0: "Legitimate", 1: "Fraud"})
                fig = px.pie(values=counts.values, names=counts.index, hole=0.62, color=counts.index,
                             color_discrete_map={"Legitimate": BLUE, "Fraud": RED})
                fig.update_traces(textfont_color=WHITE)
                st.plotly_chart(style_fig(fig, 380), use_container_width=True)
            else:
                st.info("No fraud label column found in the dataset.")

    with right:
        with st.container(border=True):
            section_header("Transaction Types", "Volume by transaction category")
            if TYPE_COL:
                tcounts = df[TYPE_COL].value_counts()
                fig = px.bar(x=tcounts.index, y=tcounts.values, color=tcounts.index,
                             color_discrete_sequence=[BLUE, RED, GREEN, AMBER, "#9575cd", "#26c6da"])
                fig.update_layout(xaxis_title="", yaxis_title="Transactions")
                st.plotly_chart(style_fig(fig, 380, showlegend=False), use_container_width=True)
            else:
                st.info("No transaction type column found in the dataset.")

    if STEP_COL and FRAUD_COL:
        with st.container(border=True):
            section_header("Fraud Trend Over Time", "Daily transaction and fraud volume")
            work = df.copy()
            work[STEP_COL] = pd.to_numeric(work[STEP_COL], errors="coerce")
            work = work.dropna(subset=[STEP_COL])
            if not work.empty:
                work["_day"] = (work[STEP_COL] // 24) + 1
                trend = work.groupby("_day").agg(transactions=(STEP_COL, "count"), fraud=(FRAUD_COL, "sum")).reset_index()
                trend["fraud_rate"] = trend["fraud"] / trend["transactions"] * 100
                tab1, tab2 = st.tabs(["Fraud Count", "Fraud Rate"])
                with tab1:
                    fig = px.line(trend, x="_day", y="fraud", markers=True)
                    fig.update_traces(line_color=RED)
                    fig.update_layout(xaxis_title="Day", yaxis_title="Fraud Cases")
                    st.plotly_chart(style_fig(fig, 360), use_container_width=True)
                with tab2:
                    fig = px.line(trend, x="_day", y="fraud_rate", markers=True)
                    fig.update_traces(line_color=AMBER)
                    fig.update_layout(xaxis_title="Day", yaxis_title="Fraud Rate (%)")
                    st.plotly_chart(style_fig(fig, 360), use_container_width=True)
            else:
                st.info("Step/time column could not be parsed for trend analysis.")

    if AMOUNT_COL:
        with st.container(border=True):
            section_header("Transaction Amount Distribution")
            c1, c2 = st.columns([2, 1])
            with c1:
                view = segmented_control_safe("amount_view", ["All Transactions", "Fraud Only"])
            with c2:
                log_scale = st.checkbox("Logarithmic scale", value=False, key="overview_log_scale")
            plot_df = df if view == "All Transactions" or not FRAUD_COL else df[df[FRAUD_COL] == 1]
            fig = px.histogram(plot_df, x=AMOUNT_COL, nbins=60, color_discrete_sequence=[RED])
            if log_scale:
                fig.update_yaxes(type="log")
            fig.update_layout(xaxis_title="Amount", yaxis_title="Count")
            st.plotly_chart(style_fig(fig, 380, showlegend=False), use_container_width=True)

    with st.container(border=True):
        section_header("Recent Fraud Transactions")
        if not fraud_df.empty:
            sort_col = "amount" if "amount" in fraud_df.columns else (AMOUNT_COL if AMOUNT_COL in fraud_df.columns else None)
            display_df = fraud_df.sort_values(sort_col, ascending=False) if sort_col else fraud_df
            st.dataframe(display_df, use_container_width=True, height=360)
        else:
            st.info("No fraud transaction records available.")


def render_transaction_monitor():
    st.title("Transaction Monitor")
    st.caption("Search, filter, and audit transaction activity in real time")

    if df.empty:
        st.warning("Dashboard data could not be loaded.")
        return

    with st.container(border=True):
        section_header("Filters")
        c1, c2, c3, c4 = st.columns([1.2, 1.4, 1, 1.4])

        with c1:
            if TYPE_COL:
                type_options = sorted(df[TYPE_COL].dropna().unique().tolist())
                selected_types = st.multiselect("Transaction Type", type_options, default=type_options)
            else:
                selected_types = None

        with c2:
            if AMOUNT_COL:
                amin, amax = float(df[AMOUNT_COL].min()), float(df[AMOUNT_COL].max())
                if amin == amax:
                    amax = amin + 1.0
                amount_range = st.slider("Amount Range", min_value=amin, max_value=amax, value=(amin, amax))
            else:
                amount_range = None

        with c3:
            if FRAUD_COL:
                fraud_status = st.selectbox("Fraud Status", ["All", "Fraud Only", "Legitimate Only"])
            else:
                fraud_status = "All"

        with c4:
            search_col = NAME_ORIG_COL or NAME_DEST_COL
            if search_col:
                search_term = st.text_input("Search Account ID", "")
            else:
                search_term = ""

    filtered = df.copy()
    if selected_types is not None and TYPE_COL:
        filtered = filtered[filtered[TYPE_COL].isin(selected_types)]
    if amount_range is not None and AMOUNT_COL:
        filtered = filtered[(filtered[AMOUNT_COL] >= amount_range[0]) & (filtered[AMOUNT_COL] <= amount_range[1])]
    if FRAUD_COL and fraud_status != "All":
        if fraud_status == "Fraud Only":
            filtered = filtered[filtered[FRAUD_COL] == 1]
        else:
            filtered = filtered[filtered[FRAUD_COL] == 0]
    if search_term:
        mask = pd.Series(False, index=filtered.index)
        if NAME_ORIG_COL:
            mask = mask | filtered[NAME_ORIG_COL].astype(str).str.contains(search_term, case=False, na=False)
        if NAME_DEST_COL:
            mask = mask | filtered[NAME_DEST_COL].astype(str).str.contains(search_term, case=False, na=False)
        filtered = filtered[mask]

    filtered_count = len(filtered)
    filtered_value = filtered[AMOUNT_COL].sum() if AMOUNT_COL else 0
    filtered_fraud = int(filtered[FRAUD_COL].sum()) if FRAUD_COL else 0
    filtered_avg = filtered[AMOUNT_COL].mean() if AMOUNT_COL and filtered_count else 0

    cards = [
        kpi_card("Filtered Transactions", fmt_int(filtered_count)),
        kpi_card("Filtered Total Value", fmt_money(filtered_value)),
        kpi_card("Filtered Fraud Cases", fmt_int(filtered_fraud), sub_color=RED),
        kpi_card("Filtered Avg Amount", fmt_money(filtered_avg, 2)),
    ]
    render_kpis(cards)

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            section_header("Transaction Records", f"{filtered_count:,} matching transactions")
        with c2:
            csv_data = filtered.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_data, file_name="filtered_transactions.csv", mime="text/csv", use_container_width=True)

        if filtered.empty:
            st.info("No transactions match the selected filters.")
        else:
            display_df = filtered.copy()
            threshold = df[AMOUNT_COL].quantile(0.95) if AMOUNT_COL else None

            def highlight_rows(row):
                if FRAUD_COL and row.get(FRAUD_COL, 0) == 1:
                    return ["background-color: rgba(239,59,54,0.18)"] * len(row)
                if threshold is not None and AMOUNT_COL and row.get(AMOUNT_COL, 0) >= threshold:
                    return ["background-color: rgba(245,165,36,0.12)"] * len(row)
                return [""] * len(row)

            try:
                styled = display_df.style.apply(highlight_rows, axis=1)
                st.dataframe(styled, use_container_width=True, height=420)
            except Exception:
                st.dataframe(display_df, use_container_width=True, height=420)

            st.markdown('<span class="badge badge-red">Fraud</span>&nbsp;&nbsp;<span class="badge badge-amber">High Amount</span>', unsafe_allow_html=True)


def render_fraud_analytics():
    st.title("Fraud Analytics")
    st.caption("Deep-dive analysis into fraud patterns and behavior")

    if df.empty or not FRAUD_COL:
        st.warning("Fraud analytics requires transaction data with a fraud label column.")
        return

    fraud_data = df[df[FRAUD_COL] == 1]
    legit_data = df[df[FRAUD_COL] == 0]

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            section_header("Fraud Count by Transaction Type")
            if TYPE_COL:
                counts = fraud_data[TYPE_COL].value_counts()
                fig = px.bar(x=counts.index, y=counts.values, color_discrete_sequence=[RED])
                fig.update_layout(xaxis_title="", yaxis_title="Fraud Cases")
                st.plotly_chart(style_fig(fig, 360, showlegend=False), use_container_width=True)
            else:
                st.info("No transaction type column available.")

    with right:
        with st.container(border=True):
            section_header("Fraud Rate by Transaction Type")
            if TYPE_COL:
                total_by_type = df[TYPE_COL].value_counts()
                fraud_by_type = fraud_data[TYPE_COL].value_counts()
                rate = (fraud_by_type / total_by_type * 100).fillna(0).sort_values(ascending=True)
                fig = px.bar(x=rate.values, y=rate.index, orientation="h", color_discrete_sequence=[AMBER])
                fig.update_layout(xaxis_title="Fraud Rate (%)", yaxis_title="")
                st.plotly_chart(style_fig(fig, 360, showlegend=False), use_container_width=True)
            else:
                st.info("No transaction type column available.")

    with st.container(border=True):
        section_header("Fraud Amount Analysis")
        if AMOUNT_COL:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Avg Fraud Amount", fmt_money(fraud_data[AMOUNT_COL].mean(), 2))
            with c2:
                st.metric("Median Fraud Amount", fmt_money(fraud_data[AMOUNT_COL].median(), 2))
            with c3:
                st.metric("Max Fraud Amount", fmt_money(fraud_data[AMOUNT_COL].max(), 2))
            plot_df = df.copy()
            plot_df["_label"] = plot_df[FRAUD_COL].map({0: "Legitimate", 1: "Fraud"})
            fig = px.box(plot_df, x="_label", y=AMOUNT_COL, color="_label", color_discrete_map={"Legitimate": BLUE, "Fraud": RED})
            fig.update_layout(xaxis_title="", yaxis_title="Amount")
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        else:
            st.info("No amount column available.")

    if NAME_ORIG_COL or NAME_DEST_COL:
        with st.container(border=True):
            section_header("Top Suspicious Accounts")
            tab_labels = []
            if NAME_ORIG_COL:
                tab_labels.append("Origin Accounts")
            if NAME_DEST_COL:
                tab_labels.append("Destination Accounts")
            tab_objs = st.tabs(tab_labels)
            idx = 0
            if NAME_ORIG_COL:
                with tab_objs[idx]:
                    agg = fraud_data.groupby(NAME_ORIG_COL).agg(
                        fraud_cases=(NAME_ORIG_COL, "count"),
                        total_amount=(AMOUNT_COL, "sum") if AMOUNT_COL else (NAME_ORIG_COL, "count"),
                    ).sort_values("fraud_cases", ascending=False).head(10).reset_index()
                    st.dataframe(agg, use_container_width=True, height=320)
                idx += 1
            if NAME_DEST_COL:
                with tab_objs[idx]:
                    agg = fraud_data.groupby(NAME_DEST_COL).agg(
                        fraud_cases=(NAME_DEST_COL, "count"),
                        total_amount=(AMOUNT_COL, "sum") if AMOUNT_COL else (NAME_DEST_COL, "count"),
                    ).sort_values("fraud_cases", ascending=False).head(10).reset_index()
                    st.dataframe(agg, use_container_width=True, height=320)

    balance_candidates = [c for c in [ORIG_OLD_COL, ORIG_NEW_COL, DEST_OLD_COL, DEST_NEW_COL] if c]
    if AMOUNT_COL and balance_candidates:
        with st.container(border=True):
            section_header("Amount vs Balance Scatter")
            y_axis = st.selectbox("Compare amount against", balance_candidates)
            sample_df = df if len(df) <= 5000 else df.sample(5000, random_state=42)
            color_vals = sample_df[FRAUD_COL].map({0: "Legitimate", 1: "Fraud"}) if FRAUD_COL else None
            fig = px.scatter(sample_df, x=AMOUNT_COL, y=y_axis, color=color_vals,
                              color_discrete_map={"Legitimate": BLUE, "Fraud": RED}, opacity=0.6)
            fig.update_layout(xaxis_title="Amount", yaxis_title=y_axis, legend_title="")
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    with st.container(border=True):
        section_header("Key Insights")
        any_insight = False
        if TYPE_COL and not fraud_data.empty:
            top_type = fraud_data[TYPE_COL].value_counts().idxmax()
            share = fraud_data[TYPE_COL].value_counts(normalize=True).max() * 100
            insight_card("Dominant Fraud Channel", f"{top_type} transactions account for {share:.1f}% of all fraud cases.")
            any_insight = True
        if AMOUNT_COL and not fraud_data.empty and not legit_data.empty:
            diff = fraud_data[AMOUNT_COL].mean() - legit_data[AMOUNT_COL].mean()
            direction = "higher" if diff > 0 else "lower"
            insight_card("Amount Behavior", f"Fraudulent transactions average {fmt_money(abs(diff), 2)} {direction} than legitimate transactions.")
            any_insight = True
        if ORIG_OLD_COL and ORIG_NEW_COL and not fraud_data.empty:
            drained = fraud_data[(fraud_data[ORIG_OLD_COL] > 0) & (fraud_data[ORIG_NEW_COL] == 0)]
            pct = len(drained) / len(fraud_data) * 100
            insight_card("Account Draining Pattern", f"{pct:.1f}% of fraud cases fully emptied the origin account balance.")
            any_insight = True
        if FLAGGED_COL:
            flagged = int(df[FLAGGED_COL].sum())
            insight_card("System-Flagged Transactions", f"{fmt_int(flagged)} transactions were automatically flagged by existing fraud rules.")
            any_insight = True
        if not any_insight:
            st.info("Not enough data available to generate insights.")


def compute_risk_score(tx_type, amount, old_orig, new_orig, old_dest, new_dest, avg_amount, max_amount):
    score = 0.0
    factors = []

    type_weights = {"TRANSFER": 25, "CASH_OUT": 20, "CASH_IN": 5, "PAYMENT": 4, "DEBIT": 8}
    w = type_weights.get(tx_type, 12)
    score += w
    factors.append(("Transaction Type Risk", tx_type or "Unknown", w, f"{tx_type or 'This type of'} transactions historically carry elevated fraud likelihood."))

    if avg_amount and avg_amount > 0 and amount is not None:
        ratio = amount / avg_amount
        if ratio > 10:
            pts = 25
        elif ratio > 5:
            pts = 16
        elif ratio > 2:
            pts = 8
        else:
            pts = 0
        if pts:
            score += pts
            factors.append(("Amount Anomaly", f"{ratio:.1f}x the average transaction", pts, "The transaction amount significantly exceeds typical activity."))

    if old_orig is not None and new_orig is not None and amount is not None:
        expected = old_orig - amount
        diff = abs(new_orig - expected)
        if old_orig > 0 and diff > 0.01:
            pts = min(20, int((diff / max(old_orig, 1)) * 100))
            if pts:
                score += pts
                factors.append(("Origin Balance Mismatch", fmt_money(diff, 2), pts, "The reported origin balance does not reconcile with the transaction amount."))
        if old_orig > 0 and new_orig == 0 and amount >= old_orig * 0.95:
            score += 15
            factors.append(("Account Drained", "Origin balance emptied", 15, "The origin account was fully or nearly fully emptied by this transaction."))

    if old_dest is not None and new_dest is not None and amount is not None and amount > 0:
        if old_dest == 0 and new_dest == 0:
            score += 10
            factors.append(("Destination Anomaly", "Static zero balance", 10, "The destination balance shows no change despite an incoming transfer."))

    if max_amount and amount is not None and max_amount > 0 and amount >= max_amount * 0.8:
        score += 10
        factors.append(("Extreme Amount", "Near historical maximum", 10, "This amount approaches the highest transactions on record."))

    score = max(0.0, min(100.0, score))
    return score, factors


def classify_risk(score):
    if score < 25:
        return "Low Risk", GREEN, "badge-green"
    elif score < 50:
        return "Medium Risk", AMBER, "badge-amber"
    elif score < 75:
        return "High Risk", ORANGE, "badge-orange"
    else:
        return "Critical Risk", RED, "badge-red"


def risk_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"color": WHITE, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": CARD_ALT,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(106,170,59,0.25)"},
                {"range": [25, 50], "color": "rgba(245,165,36,0.25)"},
                {"range": [50, 75], "color": "rgba(255,112,67,0.25)"},
                {"range": [75, 100], "color": "rgba(239,59,54,0.25)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.85, "value": score},
        },
    ))
    fig.update_layout(paper_bgcolor=CARD_ALT, font_color=WHITE, height=300, margin=dict(l=24, r=24, t=20, b=10))
    return fig


def render_risk_prediction():
    st.title("Risk Prediction")
    st.caption("Real-time fraud-risk prediction using the trained Deep Neural Network model")

    type_options = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]

    with st.container(border=True):
        section_header(
            "Transaction Details",
            "Enter transaction parameters to generate a model-based fraud prediction"
        )

        with st.form("model_prediction_form"):
            c1, c2 = st.columns(2)

            with c1:
                step = st.number_input(
                    "Step",
                    min_value=1,
                    value=278,
                    step=1,
                    help="Time step of the transaction"
                )

                tx_type = st.selectbox(
                    "Transaction Type",
                    type_options,
                    index=type_options.index("CASH_OUT")
                )

                amount = st.number_input(
                    "Transaction Amount",
                    min_value=0.0,
                    value=330218.42,
                    step=100.0
                )

            with c2:
                balance_diff_org = st.number_input(
                    "Origin Balance Difference",
                    value=-330218.42,
                    step=100.0,
                    help="Difference in origin account balance after the transaction"
                )

                balance_diff_dest = st.number_input(
                    "Destination Balance Difference",
                    value=330218.42,
                    step=100.0,
                    help="Difference in destination account balance after the transaction"
                )

                threshold = st.slider(
                    "Fraud Decision Threshold",
                    min_value=0.10,
                    max_value=0.90,
                    value=0.50,
                    step=0.05
                )

            submitted = st.form_submit_button(
                "Predict Fraud Risk",
                use_container_width=True
            )

    if submitted:
        try:
            result = predict_fraud_risk(
                step=step,
                tx_type=tx_type,
                amount=amount,
                balance_diff_org=balance_diff_org,
                balance_diff_dest=balance_diff_dest,
                threshold=threshold
            )

            risk_percent = result["risk_percent"]
            probability = result["probability"]
            prediction = result["prediction"]
            label = result["label"]

            if prediction == 1:
                badge_class = "badge-red"
                color = RED
                result_text = "Fraud Detected"
                explanation = "The trained model classifies this transaction as fraudulent based on the learned transaction patterns."
            else:
                badge_class = "badge-green"
                color = GREEN
                result_text = "Non-Fraud"
                explanation = "The trained model classifies this transaction as legitimate based on the current threshold."

            c1, c2 = st.columns([1, 1.3])

            with c1:
                with st.container(border=True):
                    st.plotly_chart(
                        risk_gauge(risk_percent, color),
                        use_container_width=True
                    )

                    st.markdown(
                        f"""
                        <div class="risk-result">
                            <span class="badge {badge_class}">{result_text}</span>
                            <div class="risk-score-text">
                                Model fraud probability: {risk_percent:.2f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with c2:
                with st.container(border=True):
                    section_header("Prediction Summary")

                    insight_card(
                        "Model Decision",
                        explanation
                    )

                    insight_card(
                        "Fraud Probability",
                        f"The model returned a probability score of {probability:.4f}. With the threshold set to {threshold:.2f}, the final class is {label}."
                    )

                    insight_card(
                        "Features Used",
                        "The prediction uses step, amount, origin balance difference, destination balance difference, and one-hot encoded transaction type."
                    )

            with st.container(border=True):
                section_header("Model Input Features")

                features_display = result["features_used"].copy()
                st.dataframe(features_display, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.info("Fill in the transaction details above and click Predict Fraud Risk.")

def build_classification_table(metrics_dict):
    rows = []
    class_candidates = {
        "Legitimate": ["legit", "0", "Legitimate", "non_fraud", "not_fraud"],
        "Fraud": ["fraud", "1", "Fraud"],
    }
    for label, candidates in class_candidates.items():
        block = None
        for c in candidates:
            if isinstance(metrics_dict.get(c), dict):
                block = metrics_dict[c]
                break
        if block:
            rows.append({
                "Class": label,
                "Precision": block.get("precision"),
                "Recall": block.get("recall"),
                "F1 Score": block.get("f1-score", block.get("f1_score")),
                "Support": block.get("support"),
            })
    aggregate_candidates = {
        "Macro Avg": ["macro avg", "macro_avg"],
        "Weighted Avg": ["weighted avg", "weighted_avg"],
    }
    for display_name, keys in aggregate_candidates.items():
        block = None
        for key in keys:
            if isinstance(metrics_dict.get(key), dict):
                block = metrics_dict[key]
                break

        if isinstance(block, dict):
            rows.append({
                "Class": display_name,
                "Precision": block.get("precision"),
                "Recall": block.get("recall"),
                "F1 Score": block.get("f1-score", block.get("f1_score")),
                "Support": block.get("support"),
            })
    return pd.DataFrame(rows)


def render_model_performance():
    st.title("Model Performance")
    st.caption("Evaluation metrics for the underlying fraud detection model")

    if not metrics:
        st.warning("Model metrics could not be loaded from model/metrics.json.")
        return

    accuracy = metrics.get("accuracy")
    fraud_block = None
    for c in ["fraud", "1", "Fraud"]:
        if isinstance(metrics.get(c), dict):
            fraud_block = metrics[c]
            break
    fraud_block = fraud_block or {}

    precision = fraud_block.get("precision")
    recall = fraud_block.get("recall")
    f1 = fraud_block.get("f1-score", fraud_block.get("f1_score"))

    cards = [
        kpi_card("Accuracy", fmt_pct(accuracy * 100) if isinstance(accuracy, (int, float)) else "N/A"),
        kpi_card("Precision (Fraud)", fmt_pct(precision * 100) if isinstance(precision, (int, float)) else "N/A"),
        kpi_card("Recall (Fraud)", fmt_pct(recall * 100) if isinstance(recall, (int, float)) else "N/A"),
        kpi_card("F1 Score (Fraud)", fmt_pct(f1 * 100) if isinstance(f1, (int, float)) else "N/A"),
    ]
    render_kpis(cards)

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            section_header("Confusion Matrix")
            cm = metrics.get("confusion_matrix")

            if isinstance(cm, dict):
                cm = [
                    [cm.get("true_negative", 0), cm.get("false_positive", 0)],
                    [cm.get("false_negative", 0), cm.get("true_positive", 0)],
                ]

            if cm and isinstance(cm, list) and len(cm) == 2:
                labels = ["Legitimate", "Fraud"]
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale=[[0, CARD_ALT], [1, RED]],
                    showscale=False,
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"color": WHITE, "size": 16},
                ))
                fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                st.plotly_chart(style_fig(fig, 360, showlegend=False), use_container_width=True)
            else:
                st.info("No confusion matrix found in model/metrics.json.")

    with right:
        with st.container(border=True):
            section_header("Precision vs Recall vs F1")
            table = build_classification_table(metrics)
            chart_rows = table[table["Class"].isin(["Legitimate", "Fraud"])] if not table.empty else table
            if not chart_rows.empty:
                melted = chart_rows.melt(id_vars="Class", value_vars=["Precision", "Recall", "F1 Score"], var_name="Metric", value_name="Score")
                melted["Score"] = melted["Score"] * 100
                fig = px.bar(melted, x="Metric", y="Score", color="Class", barmode="group",
                             color_discrete_map={"Legitimate": BLUE, "Fraud": RED})
                fig.update_layout(yaxis_title="Score (%)", xaxis_title="")
                st.plotly_chart(style_fig(fig, 360), use_container_width=True)
            else:
                st.info("Insufficient class-level metrics to compare.")

    with st.container(border=True):
        section_header("Classification Report")
        table = build_classification_table(metrics)
        if not table.empty:
            display_table = table.copy()
            for col in ["Precision", "Recall", "F1 Score"]:
                display_table[col] = display_table[col].apply(lambda v: fmt_pct(v * 100) if isinstance(v, (int, float)) else "N/A")
            display_table["Support"] = display_table["Support"].apply(lambda v: fmt_int(v) if v is not None else "N/A")
            st.dataframe(display_table, use_container_width=True, hide_index=True)
        else:
            st.info("No classification report data found in model/metrics.json.")

    with st.container(border=True):
        section_header("Model Quality Insights")
        any_insight = False
        if isinstance(recall, (int, float)):
            if recall < 0.6:
                insight_card("Low Fraud Recall", f"The model only captures {recall*100:.1f}% of actual fraud cases, meaning a significant share of fraud may go undetected. Consider rebalancing training data or lowering the classification threshold.")
            elif recall > 0.9:
                insight_card("Strong Fraud Recall", f"The model successfully identifies {recall*100:.1f}% of actual fraud cases, indicating strong sensitivity to fraudulent behavior.")
            any_insight = True
        if isinstance(precision, (int, float)):
            if precision < 0.6:
                insight_card("Elevated False Positive Rate", f"Precision of {precision*100:.1f}% suggests a meaningful share of flagged transactions are false alarms, which can increase manual review workload.")
            elif precision > 0.9:
                insight_card("High Precision", f"Precision of {precision*100:.1f}% indicates flagged transactions are highly likely to be genuine fraud.")
            any_insight = True
        if isinstance(accuracy, (int, float)):
            insight_card("Overall Accuracy", f"The model correctly classifies {accuracy*100:.2f}% of all transactions in the evaluation set.")
            any_insight = True
        if not any_insight:
            st.info("No sufficient metrics available to generate insights.")


with st.sidebar:
    st.markdown('<div class="sidebar-brand">🛡️ Fraud Risk Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Real-time monitoring &amp; detection</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Overview", "Transaction Monitor", "Fraud Analytics", "Risk Prediction", "Model Performance"], label_visibility="collapsed")
    st.markdown("---")
    if not df.empty:
        st.caption(f"Dataset loaded: {len(df):,} transactions")
    if not fraud_df.empty:
        st.caption(f"Fraud records: {len(fraud_df):,}")
    st.caption(datetime.now().strftime("%B %d, %Y"))

if page == "Overview":
    render_overview()
elif page == "Transaction Monitor":
    render_transaction_monitor()
elif page == "Fraud Analytics":
    render_fraud_analytics()
elif page == "Risk Prediction":
    render_risk_prediction()
elif page == "Model Performance":
    render_model_performance()