# nsl_kdd_gb_randomsearch.py
# Hyperparameter search for Gradient Boosting on NSL-KDD (5-class pre-grouped CSVs)
# OPTIMIZED FOR SPEED: Reduced iterations and CV folds

import time
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# ----------- config -----------
TARGET_COL = "class5"  # your grouped target column
CATS = ["protocol_type", "service", "flag"]

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

# ---- paths (edit if needed) ----
TRAIN_PATH = "/Users/mac/Desktop/machine learning/KDD/KDDTrain+_5class.csv"
TEST_PATH = "/Users/mac/Desktop/machine learning/KDD/KDDTest+_5class.csv"


# ----------- loading -----------
def load_df(p: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    if TARGET_COL not in df.columns:
        raise ValueError(f"{Path(p).name} missing '{TARGET_COL}'.")
    return df


print(f"Using train file: {TRAIN_PATH}")
print(f"Using test  file: {TEST_PATH}")

train_df = load_df(TRAIN_PATH)
test_df = load_df(TEST_PATH)

# Keep only columns we know how to process (features + target)
keep_cols = [c for c in FEATURES if c in train_df.columns] + [TARGET_COL]
train_df = train_df[keep_cols].copy()
test_df = test_df[keep_cols].copy()

# ----------- target encoding -----------
y_train_str = train_df[TARGET_COL].astype(str).str.strip()
y_test_str = test_df[TARGET_COL].astype(str).str.strip()

le = LabelEncoder()
le.fit(pd.concat([y_train_str, y_test_str], axis=0))
y_train = le.transform(y_train_str)
y_test = le.transform(y_test_str)
class_names = le.classes_
print(f"Classes ({len(class_names)}): {list(class_names)}")

# ----------- features -----------
X_train = train_df.drop(columns=[TARGET_COL]).copy()
X_test = test_df.drop(columns=[TARGET_COL]).copy()

# numeric/categorical split actually present
num_feats = [c for c in FEATURES if c not in CATS and c in X_train.columns]
cat_feats = [c for c in CATS if c in X_train.columns]

# Coerce numerics; ensure cats are strings
for df in (X_train, X_test):
    for col in num_feats:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in cat_feats:
        df[col] = df[col].astype(str)

# ----------- preprocessing -----------
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", MinMaxScaler()),
])

pre = ColumnTransformer(
    transformers=[
        ("cat", ohe, cat_feats),
        ("num", num_pipe, num_feats),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# ----------- search space (minimal for ultra speed) -----------
param_dist = {
    "clf__n_estimators": [50, 100, 150],  # Only 3 options
    "clf__learning_rate": [0.05, 0.1, 0.2],  # Only 3 options
    "clf__max_depth": [3, 5, 7],  # Only 3 options
    "clf__min_samples_split": [2, 10],  # Only 2 options
    "clf__min_samples_leaf": [1, 4],  # Only 2 options
    "clf__max_features": ["sqrt", "log2"],  # Only 2 options
    "clf__subsample": [0.8, 1.0],  # Only 2 options
}

pipe = Pipeline([
    ("prep", pre),
    ("clf", GradientBoostingClassifier(random_state=42))
])

# ULTRA SPEED OPTIMIZATION: Only 2 folds
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scorer = make_scorer(f1_score, average="macro", zero_division=0)

# ----------- randomized search (ULTRA FAST) -----------
print("\n⏳ Starting RandomizedSearchCV (Gradient Boosting on NSL-KDD 5-class)...")
print("⚡ ULTRA SPEED MODE: 10 iterations × 2 folds = 20 fits total")
t0 = time.time()
rand = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=10,  # REDUCED to 10 for ultra speed
    scoring=scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2,  # Show progress
    random_state=42,
    refit=True
)
rand.fit(X_train, y_train)
elapsed = time.time() - t0

print("\n=== Randomized Search Results (Gradient Boosting) ===")
print("Best Params:", rand.best_params_)
print(f"Best CV f1_macro: {rand.best_score_:.4f}")
print(f"Search time: {elapsed:.2f}s")

# ----------- save best params -----------
# Strip the "clf__" prefix for cleanliness
best_params = {k.replace("clf__", ""): v for k, v in rand.best_params_.items()}


def py_to_jsonable(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


json_params = {k: py_to_jsonable(v) for k, v in best_params.items()}

out = {
    "dataset": "NSL-KDD (5-class)",
    "model_type": "GradientBoostingClassifier",
    "best_parameters": json_params,
    "best_cv_score_f1_macro": float(rand.best_score_),
    "class_names": list(map(str, class_names)),
    "cv_folds": 2,
    "n_iter": 5,
    "search_seconds": float(elapsed),
    "notes": "ULTRA SPEED: 2-fold CV, 10 iterations. Preprocessing: OHE for categorical (protocol_type, service, flag), MinMax+median impute for numerics."
}

with open("kdd_best_gb_hyperparameters.json", "w") as f:
    json.dump(out, f, indent=2)

print("✅ Saved best params to 'kdd_best_gb_hyperparameters.json'")