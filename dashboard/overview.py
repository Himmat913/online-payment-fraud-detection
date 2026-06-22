import pandas as pd
import json

df = pd.read_csv("data/sample_transactions.csv")

with open("model/metrics.json") as f:
    metrics = json.load(f)