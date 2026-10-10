# Fraud Risk Analytics

A Streamlit app for exploring online-payment fraud in the PaySim dataset and scoring transactions with a trained neural network. It shows where fraud concentrates, lets you triage transactions by model score, and is upfront about what the model's headline accuracy does and doesn't mean when fraud is rare.

Research and educational use only. PaySim is simulated mobile-money data.

## Run locally (Windows, VS Code terminal set to Command Prompt)

Requires Python 3.11 or 3.12. TensorFlow 2.16 does not support 3.13 or newer.

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

The app opens at http://localhost:8501. The first page load takes a few seconds while TensorFlow loads and the 50,000 transactions are scored. After that, results are cached.

No database or secrets are needed. By default the app reads `data/transactions_sample.csv`.

## Optional: read from MySQL

To read the same columns from a MySQL `transactions` table (for example, an Aiven instance), copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in. On Streamlit Cloud, paste the same content into the app's Secrets settings. Environment variables (`MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`, `MYSQL_PORT`, `MYSQL_SSL_CA`, `MYSQL_SSL_CA_CONTENT`) also work.

If MySQL is configured but can't be reached, the app falls back to the bundled sample and says why at the top of the Overview page.

```sql
CREATE TABLE transactions (
    transaction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    step INT NOT NULL,
    type VARCHAR(30) NOT NULL,
    amount DECIMAL(18, 2) NOT NULL,
    isFraud TINYINT NOT NULL,
    balance_diff_org DECIMAL(18, 2),
    balance_diff_dest DECIMAL(18, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (isFraud IN (0, 1))
);
```

The table should hold preprocessed rows, the same as the CSV. If you load raw PaySim rows, apply the notebook's capping first; otherwise the model's scores won't be meaningful.

## Deploy to Streamlit Community Cloud

1. Push this folder to GitHub.
2. Create a new app with `app.py` as the entry point.
3. Choose Python 3.12 in the advanced settings.
4. Add MySQL secrets only if you want the database.

All model files are small and are committed with the code.

## Pages

- **Overview.** Key figures, where fraud happens (by type), how fraudulent payments behave (amount, balance changes, time of day), and a reality check on the model.
- **Transactions.** Filter by type, outcome, amount, day and model alert. Rows are sorted by model score, and you can download the filtered rows. Select a row to open it in the scorer.
- **Score a transaction.** Enter raw transaction details, or pick a dataset row. You get the decision at your threshold, the raw score, what the model actually received (including any capping), and how that threshold performs across the dataset.
- **Model.** The notebook's test results, the precision you'd get at the real fraud rate, an interactive threshold explorer, the training pipeline, and model details.

The alert threshold is shared across pages.

## Model

| | |
|---|---|
| Notebook | `notebooks/fraud-risk-analytics.ipynb` (the run that produced the shipped model files) |
| Inputs | `step`, `amount`, `balance_diff_org`, `balance_diff_dest`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER` |
| Network | Dense 128 → 64 → 32 (ReLU, dropout 0.2) → 1 sigmoid unit, Adam, binary cross-entropy |
| Training data | PaySim, 6,362,620 transactions (8,213 fraud), rebalanced with SMOTE, split 80/10/10 |
| Decision | Fraud when the score is above the threshold (0.50 in the notebook) |

Notebook test set results (half fraud, mostly synthetic):

| Accuracy | Fraud recall | Fraud precision | Fraud F1 |
|---:|---:|---:|---:|
| 93.98% | 96.41% | 91.94% | 94.12% |

At the real fraud rate (0.13%), the same recall and false-positive rate imply that only about **1.5% of alerts are fraud** at threshold 0.50. The Model page lets you explore higher thresholds.

### Preprocessing, as the app applies it

1. Cap `amount` and the four balance columns to their 10th and 90th percentiles. The constants are in `model/preprocessing.json`, taken from notebook cell 48.
2. `balance_diff_org = newbalanceOrig − oldbalanceOrg`, and `balance_diff_dest = newbalanceDest − oldbalanceDest`.
3. One-hot encode `type` with `CASH_IN` as the baseline.
4. Min-max scale `amount` and both balance changes with `model/scaler.pkl`. `step` is not scaled.

## Project structure

```
app.py                       Navigation, page frame, footer
views/                       overview.py, transactions.py, score.py, model.py
core/
  data.py                    Bundled CSV or optional MySQL; computes day/hour
  model.py                   Model files and notebook-identical preprocessing
  metrics.py                 Notebook metrics, real-rate precision, threshold table
  cache.py                   Streamlit caching for data, model and scores
ui/
  components.py              HTML builders (all data escaped)
  charts.py                  Plotly charts in the template style
  styles.css                 Layout and custom blocks
.streamlit/config.toml       Theme (colours, fonts, radii)
model/                       model.h5, scaler.pkl, feature_columns.pkl, metrics.json, preprocessing.json
data/transactions_sample.csv 50,000 preprocessed rows (notebook cell 130)
notebooks/                   Training notebook
tests/test_pipeline.py       Preprocessing, metrics and model checks
REDESIGN_NOTES.md            What changed and why, and how it was tested
```

## Tests

```cmd
python tests\test_pipeline.py
```

You can also run `python -m pytest tests`.
