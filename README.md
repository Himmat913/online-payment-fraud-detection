# Fraud Risk Analytics & Detection Platform

An end-to-end fraud-risk analytics platform for detecting suspicious online payment transactions using a trained Deep Neural Network, SQL-based fraud analysis, a cloud MySQL transaction store, and an interactive Streamlit dashboard.

The platform analyzes transaction behavior, highlights fraud patterns, monitors transaction records, and provides real-time fraud-risk prediction using deployed model artifacts.

---

## Live Demo

**Streamlit App:** Add your live Streamlit link here
**GitHub Repository:** Add your GitHub repository link here

---

## Project Overview

Online payment fraud datasets are highly imbalanced, with fraudulent transactions forming only a very small fraction of total records. This project focuses on building a complete fraud-risk analytics system that combines:

* Machine learning-based fraud detection
* SQL-based transaction analysis
* Cloud-hosted MySQL data storage
* Streamlit dashboard for monitoring and insights
* Real-time model-based fraud prediction

The final deployed application connects to a cloud MySQL database hosted on Aiven and uses trained model artifacts for live fraud-risk prediction.

---

## Key Features

### Fraud Analytics Dashboard

The dashboard provides a visual overview of transaction activity, fraud distribution, fraud rates, transaction type patterns, and high-risk transaction behavior.

### Transaction Monitor

A searchable and filterable transaction monitoring interface that allows users to inspect transaction records by type, amount range, and fraud status.

### SQL-Based Fraud Analysis

The platform uses SQL queries to uncover fraud patterns across:

* Transaction types
* Transaction amounts
* Fraud rate by category
* Balance difference behavior
* High-value transaction risk

### Real-Time Fraud Risk Prediction

The Risk Prediction page uses the trained Deep Neural Network model to classify a transaction as fraud or non-fraud based on transaction features.

### Model Performance Dashboard

The model performance page displays evaluation metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

## Tech Stack

| Area                | Tools / Technologies           |
| ------------------- | ------------------------------ |
| Programming         | Python                         |
| Machine Learning    | TensorFlow/Keras, Scikit-learn |
| Data Processing     | Pandas, NumPy                  |
| Imbalanced Learning | SMOTE, Class Weighting         |
| Database            | MySQL, Aiven Cloud MySQL       |
| SQL Analysis        | MySQL Queries                  |
| Dashboard           | Streamlit, Plotly              |
| Deployment          | Streamlit Cloud                |
| Version Control     | Git, GitHub                    |

---

## Project Architecture

```text
GitHub Repository
        │
        ▼
Streamlit Cloud Deployment
        │
        ├── Loads trained model artifacts
        │       ├── model.h5
        │       ├── scaler.pkl
        │       └── feature_columns.pkl
        │
        ▼
Aiven Cloud MySQL
        │
        └── transactions table with 50,000 records
                │
                ▼
SQL Analytics + Dashboard + Real-Time Prediction
```

---

## Dataset Summary

The dashboard uses a transaction dataset sample containing:

```text
Total transactions: 50,000
Fraud transactions: 70
Fraud rate: ~0.14%
```

The original fraud problem is highly imbalanced, with fraud cases forming only about 0.13% of the data.

---

## Model Performance

The trained Deep Neural Network achieved the following fraud-class performance:

| Metric    | Fraud Class Score |
| --------- | ----------------: |
| Precision |            93.67% |
| Recall    |            97.66% |
| F1-score  |            95.62% |

Overall model accuracy:

```text
95.53%
```

---

## Machine Learning Pipeline

The ML pipeline includes:

1. Data cleaning
2. Feature engineering
3. Transaction type encoding
4. Feature scaling
5. SMOTE oversampling
6. Class weighting
7. Deep Neural Network training
8. Early stopping
9. Model evaluation
10. Artifact export for deployment

The trained model uses engineered transaction-risk features such as:

```text
step
amount
balance_diff_org
balance_diff_dest
type_CASH_OUT
type_DEBIT
type_PAYMENT
type_TRANSFER
```

---

## Database Design

The cloud MySQL database contains a `transactions` table with the following structure:

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

The deployed Streamlit app connects to this cloud MySQL database using secure environment secrets.

---

## SQL Analytics

The project includes SQL queries for:

* Fraud distribution
* Fraud rate by transaction type
* Average transaction amount by fraud status
* Fraud amount behavior by transaction type
* Balance difference behavior in fraud transactions
* High-value transaction fraud risk
* Top fraud transactions

Example query:

```sql
SELECT 
    type,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_cases,
    ROUND(SUM(isFraud) / COUNT(*) * 100, 4) AS fraud_rate_percent
FROM transactions
GROUP BY type
ORDER BY fraud_cases DESC;
```

---

## Dashboard Pages

### 1. Overview

Shows key metrics, fraud distribution, transaction type breakdown, fraud trends, amount distribution, and recent fraud transactions.

### 2. Transaction Monitor

Allows filtering and monitoring of transactions by transaction type, amount range, and fraud status.

### 3. Fraud Analytics

Provides deeper fraud analysis across transaction types, fraud amount behavior, balance patterns, and key insights.

### 4. Risk Prediction

Uses the trained neural network model to predict whether a transaction is fraudulent.

### 5. Model Performance

Displays model metrics, class-level performance, confusion matrix, and model quality insights.

---

## Local Setup

### 1. Clone the repository

```bash
git clone <your-github-repo-link>
cd fraud-risk-analytics-platform
```

### 2. Create a virtual environment

```bash
py -3.11 -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Streamlit secrets

Create a local file:

```text
.streamlit/secrets.toml
```

Add your MySQL credentials:

```toml
[mysql]
host = "YOUR_MYSQL_HOST"
user = "YOUR_MYSQL_USER"
password = "YOUR_MYSQL_PASSWORD"
database = "defaultdb"
port = 27249
ssl_ca = "PATH_TO_CA_CERTIFICATE"
```

Do not commit this file to GitHub.

### 5. Run the app

```bash
streamlit run app.py
```

---

## Deployment

The application is deployed on Streamlit Cloud.

Deployment uses:

* GitHub for app code
* Streamlit Cloud for hosting
* Aiven Cloud MySQL for the transaction database
* Streamlit secrets for database credentials and SSL certificate content

The following secrets are configured in Streamlit Cloud:

```toml
[mysql]
host = "AIVEN_HOST"
user = "avnadmin"
password = "AIVEN_PASSWORD"
database = "defaultdb"
port = 27249
ssl_ca_content = """
-----BEGIN CERTIFICATE-----
PASTE_CA_CERTIFICATE_CONTENT_HERE
-----END CERTIFICATE-----
"""
```

Secrets are not stored in the GitHub repository.

---

## Repository Structure

```text
fraud-risk-analytics-platform/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dashboard/
│   ├── overview.py
│   ├── transaction_monitor.py
│   ├── fraud_analytics.py
│   ├── risk_prediction.py
│   └── model_performance.py
│
├── database/
│   └── mysql_connection.py
│
├── sql/
│   └── fraud_queries.py
│
├── utils/
│   └── predictor.py
│
├── model/
│   ├── model.h5
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── metrics.json
│   └── confusion_matrix.png
│
└── data/
    └── .gitkeep
```

---

## Security Notes

Sensitive files are excluded from GitHub using `.gitignore`, including:

```text
.streamlit/secrets.toml
.env
.venv/
data/*.csv
```

Database passwords, SSL certificates, and cloud credentials are stored only in deployment secrets.

---

## Resume Summary

This project supports the following resume claim:

> Developed an end-to-end fraud-risk analytics platform using Python, TensorFlow/Keras, SQL, MySQL, and Streamlit to detect suspicious online payment transactions from a highly imbalanced fraud dataset. Designed a cloud MySQL-backed transaction store, performed SQL-based fraud analysis, trained a dropout-regularized Deep Neural Network, and deployed a Streamlit dashboard for fraud analytics, transaction monitoring, model performance reporting, and real-time fraud-risk prediction.

---

## Future Improvements

Potential future enhancements include:

* Add user authentication for dashboard access
* Add transaction upload support
* Store live prediction results back into MySQL
* Add alerting for high-risk transactions
* Add model retraining pipeline
* Add explainability using SHAP or feature importance methods
* Add Docker-based deployment
