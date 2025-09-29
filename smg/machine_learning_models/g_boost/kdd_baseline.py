# kdd_benchmark_5class_gb.py
# Uses pre-made 5-class CSVs and PRINTS metrics only (no file saves other than best-hyperparams json).

import time
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# -------- schema --------
FEATURES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]
CATS = ["protocol_type", "service", "flag"]
TARGET_COL = "class5"   # your grouped label

# -------- paths --------
ROOT = Path.cwd()
TRAIN_PATH =  "/Users/mac/Desktop/machine learning/KDD/KDDTrain+_5class.csv"
TEST_PATH  = "/Users/mac/Desktop/machine learning//KDD/KDDTest+_5class.csv"

BEST_JSON = "best_gb_hyperparameters.json"  # changed for GradientBoosting

def load_df(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    if TARGET_COL not in df.columns:
        raise ValueError(f"{p.name} missing '{TARGET_COL}'. Columns: {df.columns.tolist()}")
    return df

print(f"Using train file: {TRAIN_PATH}")
print(f"Using test  file: {TEST_PATH}")

train_df = load_df(TRAIN_PATH)
test_df  = load_df(TEST_PATH)

# -------- build X/y --------
y_train_str = train_df[TARGET_COL].astype(str).str.strip()
y_test_str  = test_df[TARGET_COL].astype(str).str.strip()

le = LabelEncoder()
le.fit(pd.concat([y_train_str, y_test_str], axis=0))
y_train = le.transform(y_train_str)
y_test  = le.transform(y_test_str)
class_names = le.classes_
print(f"Classes ({len(class_names)}): {list(class_names)}")

DROP = [TARGET_COL, "label", "difficulty", "binary"]
X_train = train_df.drop(columns=[c for c in DROP if c in train_df.columns], errors="ignore").copy()
X_test  = test_df.drop(columns=[c for c in DROP if c in test_df.columns], errors="ignore").copy()

# Coerce numerics and ensure categoricals are strings
num_feats = [c for c in FEATURES if c not in CATS and c in X_train.columns]
for df in (X_train, X_test):
    if "duration" in df.columns and df["duration"].astype(str).str.lower().eq("duration").any():
        df.drop(df[df["duration"].astype(str).str.lower().eq("duration")].index, inplace=True)
    for col in num_feats:
        df[col] = pd.to_numeric(df[col], errors="coerce")
for c in CATS:
    if c in X_train.columns: X_train[c] = X_train[c].astype(str)
    if c in X_test.columns:  X_test[c]  = X_test[c].astype(str)

# -------- preprocessing --------
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:  # older sklearn
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", MinMaxScaler()),
])

pre = ColumnTransformer(
    transformers=[
        ("cat", ohe, [c for c in CATS if c in X_train.columns]),
        ("num", num_pipe, num_feats),
    ],
    remainder="drop"
)

# -------- RandomizedSearchCV for hyperparameter tuning (Gradient Boosting) --------
print("\n⏳ Starting RandomizedSearchCV for GradientBoosting hyperparameter tuning...")

param_grid = {
    'clf__n_estimators': [100, 200, 300, 400],
    'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'clf__max_depth': [3, 5, 7, 9, None],
    'clf__min_samples_split': [2, 5, 10, 15],
    'clf__min_samples_leaf': [1, 2, 4, 6],
    'clf__subsample': [0.6, 0.8, 1.0],
    'clf__max_features': [None, 'sqrt', 'log2']
}

search_pipe = Pipeline([
    ("prep", pre),
    ("clf", GradientBoostingClassifier(random_state=42))
])

random_search = RandomizedSearchCV(
    search_pipe,
    param_distributions=param_grid,
    n_iter=50,      # number of configs to try
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Fit RandomizedSearchCV
rs_start = time.time()
random_search.fit(X_train, y_train)
rs_time = time.time() - rs_start

print(f"✅ RandomizedSearchCV completed in {rs_time:.2f}s")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")
print(f"Best parameters (raw): {random_search.best_params_}")

# Extract best parameters (remove 'clf__' prefix)
best_params = {k.replace('clf__', ''): v for k, v in random_search.best_params_.items()}

# Save best parameters to JSON file
best_hyperparams = {
    "best_parameters": best_params,
    "best_cv_score": random_search.best_score_,
    "search_time_seconds": rs_time,
    "n_iterations": 50,
    "cv_folds": 3
}

with open(BEST_JSON, "w") as f:
    json.dump(best_hyperparams, f, indent=2)

print(f"✅ Best hyperparameters saved to '{BEST_JSON}'")

# -------- classifier using best params --------
try:
    loaded = json.load(open(BEST_JSON))
    best_params_loaded = loaded["best_parameters"]
    print("\nUsing saved hyperparameters:", best_params_loaded)
except Exception:
    # fallback defaults for GradientBoosting
    best_params_loaded = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 7,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "subsample": 1.0,
        "max_features": None
    }
    print("\nUsing default hyperparameters:", best_params_loaded)

# Filter params to valid keys for GradientBoostingClassifier
valid_keys = GradientBoostingClassifier().get_params().keys()
gb_params = {k: v for k, v in best_params_loaded.items() if k in valid_keys}

# Build pipeline
pipe = Pipeline([
    ("prep", pre),
    ("clf", GradientBoostingClassifier(random_state=42, **gb_params)),
])

# -------- train & predict --------
print("\n⏳ Training (5-class with GradientBoosting)...")
t0 = time.time(); pipe.fit(X_train, y_train); train_time = time.time() - t0
print(f"✅ Training time: {train_time:.3f}s")

print("⏳ Predicting (5-class)...")
t1 = time.time(); y_pred = pipe.predict(X_test); test_time = time.time() - t1
print(f"✅ Testing time: {test_time:.5f}s")

# -------- overall metrics --------
acc  = accuracy_score(y_test, y_pred)
f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n=== Overall Metrics (5-class) ===")
print(f"Accuracy:      {acc*100:.2f}%")
print(f"F1 (macro):    {f1_m*100:.2f}%")
print(f"F1 (weighted): {f1_w*100:.2f}%")

# -------- per-class metrics (printed) --------
labels_idx = np.arange(len(class_names))
prec, rec, f1, supp = precision_recall_fscore_support(
    y_test, y_pred, labels=labels_idx, zero_division=0
)
cm = confusion_matrix(y_test, y_pred, labels=labels_idx)
N = cm.sum()

rows = []
for i, name in enumerate(class_names):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = N - TP - FN - FP
    acc_ovr = (TP + TN) / N if N else 0.0
    spec = TN / (TN + FP) if (TN + FP) else 0.0
    rows.append([name, int(supp[i]), prec[i], rec[i], f1[i], acc_ovr, spec])

per_class_df = pd.DataFrame(rows, columns=[
    "class", "support", "precision", "recall", "f1", "accuracy_ovr", "specificity"
]).sort_values("class")

pd.options.display.float_format = lambda v: f"{v:.4f}"
print("\n=== Per-class metrics ===")
print(per_class_df.to_string(index=False))

# -------- confusion matrix plot (optional) --------
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Gradient Boosting — NSL-KDD 5-class (pre-grouped CSVs)")
plt.tight_layout()
plt.show()
