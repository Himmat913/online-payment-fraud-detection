import streamlit as st

from core import cache
from core.model import DEFAULT_THRESHOLD, TRANSACTION_TYPES, is_flagged
from ui import components as ui

data = cache.dataset()
try:
    df = cache.scored(data, data.source, len(data.frame))
    scoring_error = None
except Exception as exc:  # the table still works without scores
    df = data.frame.copy()
    df["score"] = float("nan")
    scoring_error = exc

st.html(
    ui.page_header(
        "Transactions",
        "Triage the <em>riskiest</em> transactions first",
        "Filter the dataset, sort by model score and open any row in the scorer. "
        "Rows are sorted by score, highest first.",
    )
)
if scoring_error is not None:
    st.error(f"Model scores are unavailable, so the score column is empty: {scoring_error}")

state = st.session_state

# ---------------------------------------------------------------- filters
with st.container(key="fr-card-filters"):
    c1, c2, c3 = st.columns([1.5, 1, 1.2], gap="medium")
    with c1:
        types = st.multiselect("Transaction type", TRANSACTION_TYPES, default=TRANSACTION_TYPES)
    with c2:
        status = st.segmented_control(
            "Known outcome", ["All", "Fraud", "Legitimate"], default="All", key="status_filter"
        ) or "All"
    with c3:
        alerts_only = st.toggle("Only transactions the model flags", value=False)
    c4, c5, c6 = st.columns([1.5, 1, 1.2], gap="medium")
    with c4:
        lo, hi = float(df["amount"].min()), float(df["amount"].max())
        amount_range = st.slider("Amount", lo, hi, (lo, hi), format="%.0f")
    with c5:
        day_lo, day_hi = int(df["day"].min()), int(df["day"].max())
        days = st.slider("Simulation day", day_lo, day_hi, (day_lo, day_hi))
    with c6:
        threshold = st.slider(
            "Alert threshold",
            0.05, 0.99, value=float(state.get("threshold_value", DEFAULT_THRESHOLD)), step=0.01,
            help="The model flags a transaction when its score is above this value. "
            "The notebook used 0.50.",
        )

state["threshold_value"] = threshold

view = df[
    df["type"].isin(types)
    & df["amount"].between(*amount_range)
    & df["day"].between(*days)
]
if status == "Fraud":
    view = view[view["isFraud"] == 1]
elif status == "Legitimate":
    view = view[view["isFraud"] == 0]
flags = is_flagged(view["score"].fillna(-1), threshold)
if alerts_only:
    view, flags = view[flags], flags[flags]
view = view.assign(alert=flags).sort_values("score", ascending=False, na_position="last")

caught = int((view["alert"] & (view["isFraud"] == 1)).sum())
st.html(
    ui.numbers_row(
        [
            (ui.fmt_int(len(view)), "transactions match"),
            (ui.fmt_int(int(view["isFraud"].sum())), "of them are known fraud"),
            (ui.fmt_int(int(view["alert"].sum())), f"flagged above {threshold:.2f}"),
            (ui.fmt_int(caught), "known fraud among the flagged"),
            (
                ui.fmt_pct(caught / view["alert"].sum(), 1) if view["alert"].sum() else "–",
                "of flagged transactions are fraud",
            ),
        ]
    )
)

# ---------------------------------------------------------------- table
table = view.assign(
    outcome=view["isFraud"].map({1: "Fraud", 0: "Legitimate"}),
    alert_text=view["alert"].map({True: "Flagged", False: ""}),
)[["day", "hour", "type", "amount", "balance_diff_org", "balance_diff_dest", "outcome", "score", "alert_text"]]

with st.container(key="fr-card-table"):
    head_l, head_r = st.columns([3, 1], vertical_alignment="bottom")
    with head_l:
        st.html(
            ui.card_title(
                "Transactions",
                "Select a row to score it in detail. Balance changes are new minus old balance, after capping.",
            )
        )
    with head_r:
        st.download_button(
            "Download these rows (CSV)",
            data=view.drop(columns=["alert"]).to_csv(index=False).encode("utf-8"),
            file_name="transactions_filtered.csv",
            mime="text/csv",
            width="stretch",
            disabled=view.empty,
        )
    if view.empty:
        st.html(ui.note("No transactions match these filters. Widen the amount or day range, or clear a filter.", "box"))
        selection = None
    else:
        selection = st.dataframe(
            table,
            hide_index=True,
            height=460,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "day": st.column_config.NumberColumn("Day", format="%d", width="small"),
                "hour": st.column_config.NumberColumn("Hour", format="%02d:00", width="small"),
                "type": st.column_config.TextColumn("Type"),
                "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                "balance_diff_org": st.column_config.NumberColumn("Origin change", format="%.2f"),
                "balance_diff_dest": st.column_config.NumberColumn("Destination change", format="%.2f"),
                "outcome": st.column_config.TextColumn("Known outcome"),
                "score": st.column_config.ProgressColumn("Model score", min_value=0.0, max_value=1.0, format="%.3f"),
                "alert_text": st.column_config.TextColumn("Alert"),
            },
        )

    rows = selection.selection.rows if selection is not None else []
    if rows:
        picked = view.iloc[rows[0]]
        state["picked_row"] = {
            k: (picked[k].item() if hasattr(picked[k], "item") else picked[k])
            for k in ["step", "type", "amount", "balance_diff_org", "balance_diff_dest", "isFraud"]
        }
        if st.button("Score the selected transaction", type="primary"):
            state["score_mode"] = "From the dataset"
            st.switch_page("views/score.py")
