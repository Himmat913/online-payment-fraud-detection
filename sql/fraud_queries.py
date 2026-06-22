FRAUD_DISTRIBUTION_QUERY = """
SELECT
    isFraud,
    COUNT(*) AS transaction_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 4) AS percentage
FROM transactions
GROUP BY isFraud;
"""


FRAUD_BY_TRANSACTION_TYPE_QUERY = """
SELECT 
    type,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_cases,
    ROUND(SUM(isFraud) / COUNT(*) * 100, 4) AS fraud_rate_percent
FROM transactions
GROUP BY type
ORDER BY fraud_cases DESC;
"""


AMOUNT_ANALYSIS_QUERY = """
SELECT
    isFraud,
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(MIN(amount), 2) AS min_amount,
    ROUND(MAX(amount), 2) AS max_amount
FROM transactions
GROUP BY isFraud;
"""


FRAUD_AMOUNT_BY_TYPE_QUERY = """
SELECT
    type,
    COUNT(*) AS fraud_cases,
    ROUND(AVG(amount), 2) AS avg_fraud_amount,
    ROUND(MIN(amount), 2) AS min_fraud_amount,
    ROUND(MAX(amount), 2) AS max_fraud_amount
FROM transactions
WHERE isFraud = 1
GROUP BY type
ORDER BY fraud_cases DESC;
"""


BALANCE_BEHAVIOR_QUERY = """
SELECT
    type,
    COUNT(*) AS fraud_cases,
    ROUND(AVG(balance_diff_org), 2) AS avg_origin_balance_difference,
    ROUND(AVG(balance_diff_dest), 2) AS avg_destination_balance_difference
FROM transactions
WHERE isFraud = 1
GROUP BY type
ORDER BY fraud_cases DESC;
"""


HIGH_VALUE_RISK_QUERY = """
SELECT
    CASE
        WHEN amount < 10000 THEN 'Below 10K'
        WHEN amount BETWEEN 10000 AND 50000 THEN '10K - 50K'
        WHEN amount BETWEEN 50000 AND 100000 THEN '50K - 100K'
        WHEN amount BETWEEN 100000 AND 500000 THEN '100K - 500K'
        ELSE 'Above 500K'
    END AS amount_bucket,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_cases,
    ROUND(SUM(isFraud) / COUNT(*) * 100, 4) AS fraud_rate_percent
FROM transactions
GROUP BY amount_bucket
ORDER BY fraud_rate_percent DESC;
"""


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