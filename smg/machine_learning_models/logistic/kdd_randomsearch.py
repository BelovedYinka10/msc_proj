#!/usr/bin/env python3
# randomsear.py
# Multiclass (5-class) NSL-KDD Logistic Regression hyperparameter tuning
# Saves best params to best_lr_params_5class.json

import json
import time
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import loguniform

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from joblib import Memory

# ---------------- Speed/Noise controls ----------------
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic"
)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
DEFAULT_OUT   = "best_lr_params_5class.json"

np.random.seed(42)

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

    # Use sparse OHE; collapse rare categories if supported
    try:
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,            # keep sparse for speed/memory
            min_frequency=10        # collapse very-rare levels; remove if not desired
            # alternatively: max_categories=50
        )
    except TypeError:
        # Older sklearn without min_frequency
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])
    cat_pipe = Pipeline([
        ("onehot", ohe)
    ])

    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, cats)],
        remainder="drop",
        verbose_feature_names_out=False
    )

def main():
    ap = argparse.ArgumentParser(description="RandomizedSearchCV for NSL-KDD 5-class LogisticRegression")
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--test",  default=DEFAULT_TEST)
    ap.add_argument("--out",   default=DEFAULT_OUT)
    ap.add_argument("--n_iter", type=int, default=10)   # smaller/faster while iterating
    ap.add_argument("--cv",     type=int, default=3)    # smaller/faster while iterating
    args = ap.parse_args()

    print("="*70)
    print("NSL-KDD (5-class) — Logistic Regression Hyperparameter Search")
    print("="*70)
    print(f"Train: {args.train}")
    print(f"Test : {args.test}")

    train_df = load_csv(args.train)
    test_df  = load_csv(args.test)

    y_col = get_label_column(train_df)
    classes = sorted(train_df[y_col].astype(str).unique().tolist())
    print(f"Detected label column: {y_col}")
    print(f"Classes: {classes}")

    feats = [f for f in FEATURES if f in train_df.columns]
    if not feats:
        raise ValueError("No expected features found in training CSV.")
    X_train = train_df[feats].copy()
    y_train = train_df[y_col].astype(str)

    # dtype coercion
    for c in feats:
        if c in CATEGORICAL_FEATURES:
            X_train[c] = X_train[c].astype(str)
        else:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")

    preproc = build_preprocessor(feats, CATEGORICAL_FEATURES)

    # Cache pipeline steps that support caching (helps a bit)
    memory = Memory(location="./skcache", verbose=0)

    base = LogisticRegression(
        penalty="l2",
        C=10.0,
        solver="lbfgs",
        max_iter=1000,   # lower during search
        tol=1e-3,        # slightly looser for speed
        n_jobs=-1,
        random_state=42
    )

    pipeline = Pipeline(
        [("preprocessor", preproc), ("classifier", base)],
        memory=memory
    )

    # Search space – focus on fast solvers and practical ranges
    param_distributions = {
        "classifier__C": loguniform(1e-3, 1e3),
        "classifier__solver": ["saga", "lbfgs"],  # good with (multi)logit + sparse
        "classifier__class_weight": [None, "balanced"],
        "classifier__max_iter": [500, 1000, 2000],
        "classifier__tol": [1e-4, 1e-3],
    }

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring="f1_weighted",
        n_jobs=-1,
        pre_dispatch="2*n_jobs",
        cv=cv,
        verbose=1,
        refit=True,
        random_state=42
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    t1 = time.time() - t0

    print("\n" + "-"*70)
    print(f"✓ Randomized search completed in {t1:.2f}s")
    print("Best CV f1_weighted:", f"{search.best_score_:.4f}")
    print("Best Params:")
    for k, v in search.best_params_.items():
        print(f"  {k} = {v}")

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_cv_f1_weighted": float(search.best_score_),
        "best_params": search.best_params_,
        "fixed": {"classifier__penalty": "l2"},  # keep penalty fixed per paper style
        "features_used": feats,
        "classes": classes,
        "label_column": y_col
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved best params to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
