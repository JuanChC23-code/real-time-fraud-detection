import pandas as pd
import json
import joblib
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Load features
with open("model_registry/feature_cols_v1.json", "r") as f:
    feature_cols = json.load(f)

X = df[feature_cols]
y = df["Class"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model (overwrite)
joblib.dump(model, "model_registry/logistic_v1.pkl")

print("Logistic model retrained and saved")