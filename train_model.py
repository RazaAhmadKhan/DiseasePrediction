"""
CodeAlpha_DiseasePrediction — Task 4: Disease Prediction from Medical Data
============================================================================
Predicts the likelihood of Diabetes using the Pima Indians Diabetes Dataset
(UCI Machine Learning Repository).

This script:
  1. Downloads the dataset directly from a public URL (no manual download needed)
  2. Cleans data (handles biologically impossible 0-values as missing)
  3. Trains & compares multiple classification algorithms
  4. Evaluates each with Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
  5. Selects the best model automatically (by ROC-AUC) and saves it to disk
     (model.pkl + scaler.pkl) for use by the Streamlit GUI (app.py)

Author: Raza Ahmad Khan
Repo:   CodeAlpha_DiseasePrediction
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD DATASET DIRECTLY FROM URL
# ---------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

print("=" * 70)
print("STEP 1: Downloading Pima Indians Diabetes Dataset from UCI (via URL)")
print("=" * 70)
df = pd.read_csv(DATA_URL, names=COLUMN_NAMES)
print(f"Dataset shape: {df.shape}")
print(df.head(), "\n")

# ---------------------------------------------------------------------------
# 2. DATA CLEANING / PREPROCESSING
# ---------------------------------------------------------------------------
# In this dataset, a 0 in these columns is medically impossible and really
# means "missing data" -> we replace with NaN then impute with the median.
print("=" * 70)
print("STEP 2: Cleaning data (handling invalid zero values as missing)")
print("=" * 70)
cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_invalid_zero] = df[cols_with_invalid_zero].replace(0, np.nan)

missing_before = df.isnull().sum()
print("Missing values introduced:\n", missing_before[missing_before > 0], "\n")

for col in cols_with_invalid_zero:
    df[col] = df[col].fillna(df[col].median())

print("Missing values remaining:", df.isnull().sum().sum(), "\n")

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT + FEATURE SCALING
# ---------------------------------------------------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 70)
print(f"STEP 3: Train/Test split -> Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print("=" * 70, "\n")

# ---------------------------------------------------------------------------
# 4. TRAIN & COMPARE MULTIPLE MODELS
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
}
if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE
    )

print("=" * 70)
print("STEP 4: Training & evaluating models")
print("=" * 70)

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="roc_auc").mean()

    trained_models[name] = model
    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": roc_auc, "CV ROC-AUC": cv_score
    })

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}  (5-fold CV: {cv_score:.4f})")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("STEP 5: Model comparison summary (ranked by ROC-AUC)")
print("=" * 70)
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. SELECT BEST MODEL & SAVE
# ---------------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_metrics = results_df.iloc[0].to_dict()

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_model_name}  (ROC-AUC = {best_metrics['ROC-AUC']:.4f})")
print("=" * 70)
print(classification_report(y_test, best_model.predict(X_test_scaled)))

joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

metadata = {
    "best_model_name": best_model_name,
    "feature_names": feature_names,
    "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in best_metrics.items() if k != "Model"},
    "feature_medians": {col: float(df[col].median()) for col in feature_names},
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

results_df.to_csv("model_comparison_results.csv", index=False)

print("\nSaved: model.pkl, scaler.pkl, model_metadata.json, model_comparison_results.csv")
print("Ready to run the GUI: streamlit run app.py")
