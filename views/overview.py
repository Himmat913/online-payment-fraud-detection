import numpy as np
import streamlit as st

from core import cache
from core.metrics import METRICS, full_dataset_fraud_rate, precision_at_base_rate
from core.model import CAPS
from ui import charts
from ui import components as ui

data = cache.dataset()
df = data.frame
fraud = df[df["isFraud"] == 1]
legit = df[df["isFraud"] == 0]
stats = {"rows": len(df), "fraud": len(fraud), "fraud_rate": len(fraud) / len(df) if len(df) else 0.0}

# ---------------------------------------------------------------- hero
with st.container(key="fr-hero"):
    copy_col, art_col = st.columns([1.25, 1], gap="large", vertical_alignment="center")
    with copy_col:
        st.html(ui.hero_copy())
        with st.container(horizontal=True, key="fr-hero-actions"):
            st.page_link("views/score.py", label="Score a transaction")
            st.page_link("views/transactions.py", label="Browse transactions")
    with art_col:
        st.html(ui.hero_art(stats))

# ---------------------------------------------------------------- key figures
fraud_types = sorted(fraud["type"].unique().tolist())
cap_high = CAPS["amount"][1]
at_cap_fraud = float((fraud["amount"] >= cap_high - 0.01).mean()) if len(fraud) else 0.0
st.html(
    ui.numbers_row(
        [
            (ui.fmt_int(stats["rows"]), "transactions in this dataset"),
            (ui.fmt_int(stats["fraud"]), "of them are confirmed fraud"),
            (str(len(fraud_types)), f"of {df['type'].nunique()} transaction types ever carry fraud"),
            (ui.fmt_pct(METRICS["fraud"]["recall"]), "of fraud caught on the notebook’s test set"),
            (
                ui.fmt_pct(precision_at_base_rate(full_dataset_fraud_rate())),
                "of alerts would be fraud at the real fraud rate",
            ),
        ]
    )
)
st.html(ui.source_notice(data.source, len(df), data.note))

# ---------------------------------------------------------------- where fraud happens
st.html(
    ui.section_heading(
        "Where fraud happens",
        "Fraud hides in <em>two</em> kinds of payment",
        f"Every confirmed fraud case in this data is a {' or '.join(t.replace('_', '-').lower() for t in fraud_types)}. "
        "The other types never carry fraud, so a transaction’s type is the first filter an analyst can apply.",
    )
)
left, right = st.columns([1.35, 1], gap="large")
with left:
    with st.container(key="fr-card-type"):
        st.html(ui.card_title("Fraud rate by transaction type", "Labels show fraud cases out of all transactions of that type."))
        st.plotly_chart(charts.fraud_rate_by_type(df), config=charts.CONFIG, width="stretch")
with right:
    insights = []
    if len(fraud):
        by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
        top_type, top_rate = by_type.index[0], by_type.iloc[0]
        insights.append(
            (
                f"{ui.fmt_pct(top_rate, 2)} of {top_type.replace('_', '-').lower()}s are fraud",
                f"That is {top_rate / stats['fraud_rate']:.1f}× the overall rate of {ui.fmt_pct(stats['fraud_rate'], 2)}.",
            )
        )
        drained = float(np.isclose(fraud["balance_diff_org"], -fraud["amount"]).mean())
        drained_legit = float(np.isclose(legit["balance_diff_org"], -legit["amount"]).mean())
        insights.append(
            (
                f"{ui.fmt_pct(drained, 0)} of fraud moves money out by exactly the amount",
                f"Compared with {ui.fmt_pct(drained_legit, 0)} of legitimate payments: the origin "
                "balance falls by the full transaction value.",
            )
        )
        insights.append(
            (
                f"{ui.fmt_pct(at_cap_fraud, 0)} of fraud sits at the amount cap",
                f"Amounts were capped at {ui.fmt_amount(cap_high)} in preprocessing; "
                f"only {ui.fmt_pct(float((legit['amount'] >= cap_high - 0.01).mean()), 0)} of legitimate "
                "payments reach it. Fraudsters move large sums.",
            )
        )
    with st.container(key="fr-card-outline-insights"):
        st.html(ui.card_title("What stands out"))
        st.html(ui.insight_list(insights) if insights else ui.note("No fraud cases in the current data."))

# ---------------------------------------------------------------- patterns
st.html(
    ui.section_heading(
        "Patterns",
        "How fraudulent payments <em>behave</em>",
        "Amounts, balance changes and timing, compared between fraud and legitimate transactions.",
    )
)
a, b = st.columns(2, gap="large")
with a:
    with st.container(key="fr-card-amount"):
        st.html(ui.card_title("Amount", "Each group shown as a share of itself, so the rare fraud cases stay visible."))
        st.plotly_chart(charts.amount_distribution(df), config=charts.CONFIG, width="stretch")
        st.html(
            ui.note(
                f"Amounts below {ui.fmt_amount(CAPS['amount'][0])} and above {ui.fmt_amount(cap_high)} were "
                "clipped to those values during preprocessing, which is why bars pile up at the edges."
            )
        )
with b:
    with st.container(key="fr-card-balance"):
        st.html(ui.card_title("Balance changes", "How origin and destination balances moved."))
        st.plotly_chart(charts.balance_behaviour(df), config=charts.CONFIG, width="stretch")
        st.html(
            ui.note(
                "Balance changes are computed from capped balances, exactly as the model sees them."
            )
        )

night = df["hour"].between(0, 6)
night_rate = float(df.loc[night, "isFraud"].mean()) if night.any() else 0.0
day_rate = float(df.loc[~night, "isFraud"].mean()) if (~night).any() else 0.0
with st.container(key="fr-card-hour"):
    st.html(
        ui.card_title(
            "Time of day",
            "Transaction volume (bars) against fraud cases (line) by hour on the simulation clock. "
            f"Between midnight and 6 a.m. normal activity almost stops but fraud does not: "
            f"{ui.fmt_pct(night_rate, 1)} of night-time transactions are fraud, against "
            f"{ui.fmt_pct(day_rate, 2)} during the rest of the day "
            f"({ui.fmt_int(int(df.loc[night, 'isFraud'].sum()))} of {ui.fmt_int(int(night.sum()))} night transactions).",
        )
    )
    st.plotly_chart(charts.fraud_by_hour(df), config=charts.CONFIG, width="stretch")

# ---------------------------------------------------------------- model reality check
with st.container(key="fr-callout"):
    st.html(
        ui.callout(
            "A high test score is not the same as <em>few false alarms</em>",
            "The notebook reports "
            f"<strong>{ui.fmt_pct(METRICS['accuracy'], 2)} accuracy</strong> on a test set that is half fraud. "
            "Real traffic is about 0.13% fraud, and at the default threshold the same model would flag "
            f"roughly <strong>{ui.fmt_pct(1 - METRICS['non_fraud']['recall'], 1)} of all legitimate payments</strong>. "
            "The Model page shows the trade-off and how raising the threshold changes it.",
        )
    )
    st.page_link("views/model.py", label="See how the model performs")
