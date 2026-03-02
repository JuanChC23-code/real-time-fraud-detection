import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import joblib
import os
import time
import json

print("Loading dataset...")

# Cargar dataset real
df = pd.read_csv("data/creditcard.csv")

# Separar features y target
X = df.drop("Class", axis=1)
y = df["Class"]

# Guardar esquema de columnas (CRÍTICO para inferencia correcta)
if not os.path.exists("model_registry"):
    os.makedirs("model_registry")

feature_cols = list(X.columns)

with open("model_registry/feature_cols_v1.json", "w") as f:
    json.dump(feature_cols, f)

print("Feature schema saved.")

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============================
# Logistic Regression
# =============================

print("\nTraining Logistic Regression...")
start = time.time()

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_time = time.time() - start

log_probs = log_model.predict_proba(X_test)[:, 1]

log_auc = roc_auc_score(y_test, log_probs)
log_pr = average_precision_score(y_test, log_probs)

joblib.dump(log_model, "model_registry/logistic_v1.pkl")

print("\nLogistic Regression Results:")
print("Training time:", round(log_time, 2), "seconds")
print("ROC-AUC:", round(log_auc, 4))
print("PR-AUC:", round(log_pr, 4))

# =============================
# XGBoost
# =============================

print("\nTraining XGBoost...")
start = time.time()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

xgb_time = time.time() - start

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

xgb_auc = roc_auc_score(y_test, xgb_probs)
xgb_pr = average_precision_score(y_test, xgb_probs)

joblib.dump(xgb_model, "model_registry/xgboost_v1.pkl")

print("\nXGBoost Results:")
print("Training time:", round(xgb_time, 2), "seconds")
print("ROC-AUC:", round(xgb_auc, 4))
print("PR-AUC:", round(xgb_pr, 4))

print("\nModels saved in model_registry/")
