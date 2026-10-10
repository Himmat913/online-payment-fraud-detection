import numpy as np
import streamlit as st

from core import cache
from core.metrics import METRICS, full_dataset_fraud_rate, notebook_rates, precision_at_base_rate, threshold_table
from core.model import CAPS, DEFAULT_THRESHOLD
from ui import charts
from ui import components as ui

state = st.session_state
data = cache.dataset()
ev = METRICS["evaluation_set"]
rates = notebook_rates()
real_rate = full_dataset_fraud_rate()

st.html(
    ui.page_header(
        "Model",
        "How good is the model, <em>really</em>?",
        "A dense neural network trained in the project notebook. Its headline scores are strong, "
        "but they come from a test set that is half fraud. Here is what they mean for real traffic.",
    )
)

st.html(
    ui.numbers_row(
        [
            (ui.fmt_pct(METRICS["accuracy"], 2), "accuracy on the notebook test set"),
            (ui.fmt_pct(METRICS["fraud"]["recall"], 1), "of fraud caught (recall)"),
            (ui.fmt_pct(METRICS["fraud"]["precision"], 1), "of alerts were fraud on that test set"),
            (ui.fmt_pct(METRICS["fraud"]["f1_score"], 1), "fraud F1 score"),
            (ui.fmt_pct(rates["false_positive_rate"], 1), "of legitimate payments wrongly flagged"),
        ]
    )
)
st.html(
    ui.note(
        f"Test set: {ui.fmt_int(ev['rows'])} rows held out from the SMOTE-rebalanced data "
        f"({ui.fmt_int(ev['legitimate'])} real legitimate transactions and {ui.fmt_int(ev['fraud'])} fraud rows, "
        "most of them synthetic). Threshold 0.50. Source: notebook cell 146."
    )
)

left, right = st.columns([1, 1.05], gap="large")
with left:
    with st.container(key="fr-card-cm"):
        st.html(ui.card_title("Confusion matrix", "Notebook test set, threshold 0.50."))
        st.html(ui.confusion_matrix(METRICS["confusion_matrix"]))
with right:
    with st.container(key="fr-card-yellow-reality"):
        st.html(
            ui.card_title(
                "At a real fraud rate",
                f"In the full dataset only {ui.fmt_pct(real_rate, 2)} of transactions are fraud "
                f"({ui.fmt_int(METRICS['training']['dataset_fraud_rows'])} of "
                f"{ui.fmt_int(METRICS['training']['dataset_rows'])}).",
            )
        )
        implied = precision_at_base_rate(real_rate)
        st.html(
            '<p class="fr-number__value">' + ui.fmt_pct(implied, 1) + "</p>"
            '<p class="fr-muted">of alerts would be fraud at that rate, using the notebook’s own recall '
            f"({ui.fmt_pct(rates['recall'], 1)}) and false-positive rate "
            f"({ui.fmt_pct(rates['false_positive_rate'], 2)}). Roughly "
            f"{int(round((1 - implied) / implied))} false alarms for every real fraud case at threshold 0.50.</p>"
        )
        st.html(
            ui.note(
                "Precision depends on how common fraud is. A model that looks balanced on a 50/50 test set "
                "can still bury analysts in alerts when fraud is rare, which is why the threshold matters.",
                "box",
            )
        )

# ---------------------------------------------------------------- threshold explorer
st.html(
    ui.section_heading(
        "Choose a threshold",
        "Trade <em>missed fraud</em> against false alarms",
        "Scores for every transaction in the dataset, with known outcomes. Raising the threshold "
        "sends fewer, more reliable alerts but lets more fraud through.",
    )
)
try:
    scored = cache.scored(data, data.source, len(data.frame))
except Exception as exc:
    st.error("The model could not score the dataset, so the threshold explorer is unavailable.")
    with st.expander("Technical details"):
        st.exception(exc)
else:
    labels, scores = scored["isFraud"].to_numpy(), scored["score"].to_numpy()
    with st.container(key="fr-card-threshold"):
        threshold = st.slider(
            "Alert threshold", 0.05, 0.99,
            value=float(state.get("threshold_value", DEFAULT_THRESHOLD)), step=0.01,
            help="The same threshold is used on the Transactions and Score pages.",
        )
        state["threshold_value"] = threshold
        curve = threshold_table(labels, scores, sorted(set(np.round(np.arange(0.05, 1.0, 0.01), 2))))
        here = threshold_table(labels, scores, [threshold]).iloc[0]
        st.html(
            ui.numbers_row(
                [
                    (ui.fmt_int(here["flagged"]), f"alerts ({ui.fmt_pct(here['flagged_share'], 1)} of transactions)"),
                    (f"{int(here['fraud_caught'])} of {int(labels.sum())}", "known fraud cases caught"),
                    (ui.fmt_int(here["false_alarms"]), "false alarms"),
                    (ui.fmt_pct(here["precision"], 1) if here["flagged"] else "–", "of alerts are fraud"),
                    (ui.fmt_pct(here["recall"], 1), "of fraud caught"),
                ]
            )
        )
        st.plotly_chart(charts.threshold_curve(curve, threshold), config=charts.CONFIG, width="stretch")
        st.html(
            ui.note(
                f"Computed on the {ui.fmt_int(len(labels))}-row dataset at its real fraud rate. "
                "Most of these rows were also part of the training data (the notebook split after "
                "rebalancing), so recall here is likely optimistic; the drop in precision is the robust finding."
            )
        )

# ---------------------------------------------------------------- how it was built
st.html(
    ui.section_heading(
        "How it was built",
        "From raw payments to a <em>score</em>",
        "The training pipeline in notebooks/fraud-risk-analytics.ipynb. The app repeats steps 1 to 3 for every transaction it scores.",
    )
)
st.html(
    ui.steps(
        [
            ("01", "Cap outliers",
             f"Amount and the four balance columns are clipped to their 10th and 90th percentiles "
             f"(amount: {ui.fmt_amount(CAPS['amount'][0])} to {ui.fmt_amount(CAPS['amount'][1])})."),
            ("02", "Engineer features",
             "Old and new balances become two balance changes. Account IDs and the rule-based "
             "isFlaggedFraud column are dropped."),
            ("03", "Encode and scale",
             "Type is one-hot encoded (cash-in is the baseline). Amount and both balance changes are min-max "
             "scaled; the step stays as it is."),
            ("04", "Rebalance",
             f"SMOTE adds synthetic fraud rows until both classes have {ui.fmt_int(6354407)} rows, "
             "then the data is split 80 / 10 / 10."),
            ("05", "Train",
             "A 128-64-32 dense network with 20% dropout, Adam and binary cross-entropy. Early stopping "
             "kept the weights from epoch 1 of 3."),
        ]
    )
)

with st.container(key="fr-card-specs"):
    st.html(ui.card_title("Model details"))
    st.html(
        ui.spec_list(
            [
                ("Inputs", "step, amount, sender balance change, recipient balance change, and four transaction-type flags"),
                ("Architecture", "Dense 128 → 64 → 32 (ReLU, dropout 0.2) → 1 sigmoid output; 11,521 parameters"),
                ("Output", "A score between 0 and 1. Fraud when the score is above the threshold (0.50 in the notebook)."),
                ("Training data", f"{ui.fmt_int(METRICS['training']['train_rows'])} rows: the 80% training split of the "
                                  f"SMOTE-rebalanced PaySim data ({ui.fmt_int(METRICS['training']['dataset_rows'])} original transactions)"),
                ("Files", "<code>model/model.h5</code>, <code>scaler.pkl</code>, <code>feature_columns.pkl</code>; "
                          "metrics in <code>model/metrics.json</code>"),
            ]
        )
    )
