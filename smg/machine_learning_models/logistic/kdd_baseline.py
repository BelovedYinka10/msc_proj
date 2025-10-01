#!/usr/bin/env python3
# logisti.py
# Train & evaluate NSL-KDD Logistic Regression (5 classes, class5).
# Loads best_lr_params_5class.json if present; otherwise uses multinomial L2 defaults.

import json
import time
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support, roc_auc_score,
    roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, label_binarize
from sklearn.linear_model import LogisticRegression

# ---------- Config ----------
FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

DEFAULT_TRAIN = "/Users/mac/Desktop/machine learning/KDD/KDDTrain+_5class.csv"
DEFAULT_TEST  = "/Users/mac/Desktop/machine learning/KDD/KDDTest+_5class.csv"
DEFAULT_JSON  = "best_lr_params_5class.json"

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def get_label_column(df: pd.DataFrame) -> str:
    for cand in ["class5", "Class5", "class", "label"]:
        if cand in df.columns:
            return cand
    raise ValueError("Could not find a class column (expected 'class5').")

def build_preprocessor(available_features, categorical_features):
    numeric = [f for f in available_features if f not in categorical_features]
    cats   = [f for f in categorical_features if f in available_features]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", MinMaxScaler())])
    cat_pipe = Pipeline([("onehot", ohe)])
    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, cats)],
        remainder="drop", verbose_feature_names_out=False
    )

def load_best_params(json_path: Path):
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text())
        best = payload.get("best_params")
        fixed = payload.get("fixed", {})
        if best is None:
            return None
        # ensure required fixed settings
        best.setdefault("classifier__multi_class", fixed.get("classifier__multi_class", "multinomial"))
        best.setdefault("classifier__penalty", fixed.get("classifier__penalty", "l2"))
        return best
    except Exception:
        return None

def per_class_specificity(cm: np.ndarray) -> np.ndarray:
    # cm shape (K,K). For class i, treat as binary OvR.
    K = cm.shape[0]
    spec = np.zeros(K)
    for i in range(K):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        spec[i] = TN / (TN + FP) if (TN + FP) else 0.0
    return spec

def per_class_accuracy_ovr(cm: np.ndarray) -> np.ndarray:
    K = cm.shape[0]
    acc = np.zeros(K)
    N = cm.sum()
    for i in range(K):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = N - (TP + FN + FP)
        acc[i] = (TP + TN) / N if N else 0.0
    return acc

def main():
    ap = argparse.ArgumentParser(description="NSL-KDD 5-class Logistic Regression trainer/evaluator")
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--test",  default=DEFAULT_TEST)
    ap.add_argument("--params", default=DEFAULT_JSON)
    args = ap.parse_args()

    print("="*70)
    print("NSL-KDD Logistic Regression — 5-Class (multinomial)")
    print("="*70)

    print(f"\nLoading training data from: {args.train}")
    train_df = load_csv(args.train)
    print(f"Loading test data from: {args.test}")
    test_df  = load_csv(args.test)

    y_col = get_label_column(train_df)
    print(f"Label column: {y_col}")

    # classes (string labels for stability)
    classes = sorted(train_df[y_col].astype(str).unique().tolist())
    print(f"Classes: {classes}")

    feats = [f for f in FEATURES if f in train_df.columns]
    print(f"Features used: {len(feats)}")

    X_train = train_df[feats].copy()
    X_test  = test_df[feats].copy()
    y_train = train_df[y_col].astype(str)
    y_test  = test_df[y_col].astype(str)

    # dtype coercion
    for c in feats:
        if c in CATEGORICAL_FEATURES:
            X_train[c] = X_train[c].astype(str)
            X_test[c]  = X_test[c].astype(str)
        else:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_test[c]  = pd.to_numeric(X_test[c], errors="coerce")

    preproc = build_preprocessor(feats, CATEGORICAL_FEATURES)

    # Load best params if available
    best_params = load_best_params(Path(args.params))
    if best_params:
        print("\nUsing BEST params from JSON:")
        for k, v in best_params.items():
            print(f"  {k} = {v}")
    else:
        print("\nNo JSON found — using multinomial paper-style defaults.")
        best_params = {
            "classifier__multi_class": "multinomial",
            "classifier__penalty": "l2",
            "classifier__C": 10.0,
            "classifier__solver": "lbfgs",
            "classifier__max_iter": 3000,
            "classifier__class_weight": None
        }

    model = LogisticRegression(
        multi_class="multinomial",
        penalty="l2",
        C=10.0,
        solver="lbfgs",
        max_iter=3000,
        n_jobs=-1,
        random_state=42
    )
    pipe = Pipeline([("preprocessor", preproc), ("classifier", model)])
    pipe.set_params(**best_params)

    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"✓ Training completed in {time.time()-t0:.2f}s")

    # Predict
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    y_pred = pipe.predict(X_test)

    # Overall accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Per-class metrics
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, zero_division=0
    )

    # Confusion matrix and derived per-class specificity & OvR accuracy
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    spec = per_class_specificity(cm)
    acc_ovr = per_class_accuracy_ovr(cm)

    # Pretty table
    header = f"{'class':>8}  {'support':>7}  {'precision':>9}  {'recall':>6}  {'f1':>6}  {'accuracy_ovr':>13}  {'specificity':>11}"
    print("\n=== Per-class metrics ===")
    print(header)
    for i, cls in enumerate(classes):
        print(f"{cls:>8}  {int(support[i]):7d}  {prec[i]*100:9.2f}%  {rec[i]*100:6.2f}%  {f1[i]*100:6.2f}%  {acc_ovr[i]*100:13.2f}%  {spec[i]*100:11.2f}%")

    print("\n--- Classification Report (macro/weighted) ---")
    print(classification_report(y_test, y_pred, labels=classes, digits=4))

    # Confusion Matrix plot
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix — NSL-KDD (5-class) Logistic Regression")
    plt.tight_layout()
    plt.show()

    # Optional: macro-averaged ROC-AUC (OvR) if probabilities available
    try:
        y_proba = pipe.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        print(f"\nMacro ROC-AUC (OvR): {auc_macro:.4f}")
    except Exception:
        pass

    print("\nDONE.")

if __name__ == "__main__":
    main()
