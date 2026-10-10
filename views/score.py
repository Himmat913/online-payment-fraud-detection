import numpy as np
import pandas as pd
import streamlit as st

from core import cache
from core.metrics import threshold_table
from core.model import CAPS, DEFAULT_THRESHOLD, TRANSACTION_TYPES, engineer_features, is_flagged, score
from ui import components as ui

state = st.session_state
data = cache.dataset()

st.html(
    ui.page_header(
        "Score a transaction",
        "Ask the model about <em>one</em> payment",
        "Enter a transaction’s details, or pick one from the dataset, to see the model’s score and "
        "whether it would raise an alert at your threshold.",
    )
)

MODES = ["Enter details", "From the dataset"]
mode = st.segmented_control("Input", MODES, default=state.get("score_mode", MODES[0]), key="score_mode_widget")
mode = mode or MODES[0]
state["score_mode"] = mode

form_col, result_col = st.columns([1.1, 1], gap="large")
features, adjusted, known_outcome, source_text = None, [], None, ""

with form_col:
    with st.container(key="fr-card-form"):
        if mode == "Enter details":
            st.html(
                ui.card_title(
                    "Transaction details",
                    "Prefilled with the first transaction of the original dataset (a legitimate payment).",
                )
            )
            with st.form("score_form", border=False):
                c1, c2 = st.columns(2)
                tx_type = c1.selectbox("Type", TRANSACTION_TYPES, index=TRANSACTION_TYPES.index("PAYMENT"))
                amount = c2.number_input("Amount", min_value=0.0, value=9839.64, step=1000.0, format="%.2f")
                c3, c4 = st.columns(2)
                day = c3.number_input("Simulation day", min_value=1, max_value=31, value=1, step=1)
                hour = c4.number_input("Hour (0–23)", min_value=0, max_value=23, value=0, step=1)
                st.html('<p class="fr-label" style="margin:0.5rem 0 0">Sender account</p>')
                c5, c6 = st.columns(2)
                old_org = c5.number_input("Balance before", min_value=0.0, value=170136.0, step=1000.0, format="%.2f", key="old_org")
                new_org = c6.number_input("Balance after", min_value=0.0, value=160296.36, step=1000.0, format="%.2f", key="new_org")
                st.html('<p class="fr-label" style="margin:0.5rem 0 0">Recipient account</p>')
                c7, c8 = st.columns(2)
                old_dest = c7.number_input("Balance before", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key="old_dest")
                new_dest = c8.number_input("Balance after", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key="new_dest")
                threshold = st.slider(
                    "Alert threshold", 0.05, 0.99,
                    value=float(state.get("threshold_value", DEFAULT_THRESHOLD)), step=0.01,
                    help="Flag when the score is above this value. The notebook used 0.50.",
                )
                submitted = st.form_submit_button("Score transaction", type="primary", width="stretch")
            state["threshold_value"] = threshold
            if submitted:
                step = (int(day) - 1) * 24 + int(hour) + 1
                features, adjusted = engineer_features(
                    step=step, tx_type=tx_type, amount=amount,
                    old_balance_origin=old_org, new_balance_origin=new_org,
                    old_balance_dest=old_dest, new_balance_dest=new_dest,
                )
                state["manual_result"] = {"features": features.to_dict("records")[0], "adjusted": adjusted}
            elif state.get("manual_result"):
                kept = state["manual_result"]
                features, adjusted = pd.DataFrame([kept["features"]]), kept["adjusted"]
            source_text = "Your input"
        else:
            st.html(
                ui.card_title(
                    "A transaction from the dataset",
                    "Pick a random case, or select a row on the Transactions page. "
                    "Rows are already preprocessed, so they go straight to the model.",
                )
            )
            df = data.frame
            b1, b2 = st.columns(2)
            if b1.button("Random fraud case", width="stretch", disabled=not (df["isFraud"] == 1).any()):
                row = df[df["isFraud"] == 1].sample(1).iloc[0]
                state["picked_row"] = {k: row[k].item() if hasattr(row[k], "item") else row[k]
                                       for k in ["step", "type", "amount", "balance_diff_org", "balance_diff_dest", "isFraud"]}
            if b2.button("Random legitimate case", width="stretch"):
                row = df[df["isFraud"] == 0].sample(1).iloc[0]
                state["picked_row"] = {k: row[k].item() if hasattr(row[k], "item") else row[k]
                                       for k in ["step", "type", "amount", "balance_diff_org", "balance_diff_dest", "isFraud"]}
            threshold = st.slider(
                "Alert threshold", 0.05, 0.99,
                value=float(state.get("threshold_value", DEFAULT_THRESHOLD)), step=0.01,
                help="Flag when the score is above this value. The notebook used 0.50.",
            )
            state["threshold_value"] = threshold
            picked = state.get("picked_row")
            if picked:
                features = pd.DataFrame([{k: picked[k] for k in ["step", "type", "amount", "balance_diff_org", "balance_diff_dest"]}])
                known_outcome = "Fraud" if int(picked["isFraud"]) == 1 else "Legitimate"
                source_text = "Dataset row"
            else:
                st.html(ui.note("No transaction picked yet. Choose a random case above.", "box"))

        if features is not None:
            row = features.iloc[0]
            step = int(row["step"])
            amount_note = ""
            if mode == "Enter details" and "amount" in adjusted:
                amount_note = '<span class="fr-muted">Clipped to the training range '
                amount_note += f'{ui.fmt_amount(CAPS["amount"][0])}–{ui.fmt_amount(CAPS["amount"][1])}.</span>'
            balance_note = ""
            balance_adjusted = [a for a in adjusted if a != "amount"]
            if mode == "Enter details" and balance_adjusted:
                balance_note = ('<span class="fr-muted">Balances above the training caps were clipped before '
                                "the change was computed, as in the notebook.</span>")
            rows = [
                ("Source", source_text),
                ("Type", row["type"]),
                ("Day and hour", f"Day {(step - 1) // 24 + 1}, {((step - 1) % 24):02d}:00 (step {step})"),
                ("Amount", f"{ui.fmt_amount(row['amount'])}{amount_note}"),
                ("Sender balance change", f"{ui.fmt_amount(row['balance_diff_org'])}{balance_note}"),
                ("Recipient balance change", ui.fmt_amount(row["balance_diff_dest"])),
            ]
            if known_outcome:
                rows.append(("Known outcome", known_outcome))
            st.html('<div class="fr-receives">' + ui.card_title("What the model receives") + ui.spec_list(rows) + "</div>")

with result_col:
    if features is None:
        st.html(ui.empty_result())
    else:
        try:
            value = float(score(features)[0])
        except Exception as exc:
            st.error("The model could not score this transaction. Check that the files in the model folder are present.")
            with st.expander("Technical details"):
                st.exception(exc)
        else:
            flagged = bool(is_flagged([value], threshold)[0])
            try:
                scored = cache.scored(data, data.source, len(data.frame))
                t = threshold_table(scored["isFraud"].to_numpy(), scored["score"].to_numpy(), [threshold]).iloc[0]
                context = {
                    "flagged_share": t["flagged_share"],
                    "fraud_caught": int(t["fraud_caught"]),
                    "total_fraud": int(scored["isFraud"].sum()),
                    "precision_text": ui.fmt_pct(t["precision"], 1) if not np.isnan(t["precision"]) else "None",
                }
            except Exception:
                context = None
            st.html(ui.score_result(value, threshold, flagged, context))
            if known_outcome:
                correct = (known_outcome == "Fraud") == flagged
                st.html(
                    ui.note(
                        f"This transaction is known to be <strong>{known_outcome.lower()}</strong>. "
                        + ("The model’s decision matches." if correct else "The model’s decision does not match.")
                        + " Most dataset rows were also used in training, so expect agreement more often than on new data.",
                        "box",
                    )
                )
            st.html(
                ui.note(
                    "The score is the network’s sigmoid output. It was trained on data rebalanced to 50% fraud, "
                    "so it is a ranking signal, not the probability that this payment is fraudulent."
                )
            )
