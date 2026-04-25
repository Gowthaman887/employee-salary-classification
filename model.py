import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import json
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("Employee_Salary_Dataset.csv")

# ── 2. Salary class labels ────────────────────────────────────────────────────
# Low    : salary < 100,000
# Medium : 100,000 – 999,999
# High   : >= 1,000,000
def classify_salary(s):
    if s < 100_000:
        return "Low"
    elif s < 1_000_000:
        return "Medium"
    else:
        return "High"

df["Salary_Class"] = df["Salary"].apply(classify_salary)

# ── 3. Encode Gender ──────────────────────────────────────────────────────────
le = LabelEncoder()
df["Gender_Enc"] = le.fit_transform(df["Gender"])   # Female=0, Male=1

# ── 4. Feature engineering ────────────────────────────────────────────────────
# Add interaction & ratio features that help separate the classes
df["Exp_x_Age"]   = df["Experience_Years"] * df["Age"]
df["Exp_per_Age"] = df["Experience_Years"] / df["Age"]

feature_cols = ["Experience_Years", "Age", "Gender_Enc", "Exp_x_Age", "Exp_per_Age"]
X = df[feature_cols]
y = df["Salary_Class"]

# ── 5. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ── 6. Pipeline: scale → polynomial features → Logistic Regression ───────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("clf",    LogisticRegression(
                   C=0.5,
                   max_iter=5000,
                   solver="lbfgs",
                   multi_class="multinomial",
                   random_state=42
               )),
])

pipeline.fit(X_train, y_train)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred   = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred, output_dict=True)
cm       = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])

# Cross-validation on full dataset (gives a fairer picture with 35 rows)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

print(f"\nTest Accuracy  : {accuracy:.2%}")
print(f"CV Accuracy    : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 8. Predict on full dataset for display ────────────────────────────────────
df["Predicted_Class"] = pipeline.predict(X)

# ── 9. Build JSON payload ─────────────────────────────────────────────────────
class_counts = df["Salary_Class"].value_counts().to_dict()
pred_counts  = df["Predicted_Class"].value_counts().to_dict()

metrics_per_class = {}
for cls in ["Low", "Medium", "High"]:
    if cls in report:
        metrics_per_class[cls] = {
            "precision": round(report[cls]["precision"], 3),
            "recall":    round(report[cls]["recall"],    3),
            "f1":        round(report[cls]["f1-score"],  3),
            "support":   int(report[cls]["support"]),
        }

cm_data = {
    "labels": ["Low", "Medium", "High"],
    "matrix": cm.tolist()
}

table_rows = []
for _, row in df.iterrows():
    table_rows.append({
        "id":         int(row["ID"]),
        "experience": int(row["Experience_Years"]),
        "age":        int(row["Age"]),
        "gender":     row["Gender"],
        "salary":     int(row["Salary"]),
        "actual":     row["Salary_Class"],
        "predicted":  row["Predicted_Class"],
    })

payload = {
    "accuracy":          round(accuracy * 100, 2),
    "cv_accuracy":       round(cv_scores.mean() * 100, 2),
    "cv_std":            round(cv_scores.std()  * 100, 2),
    "class_counts":      class_counts,
    "pred_counts":       pred_counts,
    "metrics_per_class": metrics_per_class,
    "confusion_matrix":  cm_data,
    "table_rows":        table_rows,
    "train_size":        len(X_train),
    "test_size":         len(X_test),
}

with open("results.json", "w") as f:
    json.dump(payload, f, indent=2)

# ── 11. Embed JSON directly into index.html (no fetch / CORS issues) ──────────
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

import re
new_data_line = f"const DATA = {json.dumps(payload, separators=(',', ':'))};"
html = re.sub(r"const DATA = \{.*?\};", new_data_line, html, flags=re.DOTALL)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("\nresults.json + index.html updated — open index.html in a browser to view the dashboard.")
