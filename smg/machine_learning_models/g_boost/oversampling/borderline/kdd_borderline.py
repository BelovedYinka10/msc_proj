# kdd_benchmark_5class_gb.py
# Uses pre-made 5-class CSVs and PRINTS metrics only (loads saved hyperparameters).
# Now with Borderline-SMOTE oversampling inserted into the pipeline.

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
# Use imblearn's Pipeline so we can insert the sampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# NEW: Borderline-SMOTE sampler
from imblearn.over_sampling import BorderlineSMOTE

# -------- schema --------
FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]
CATS = ["protocol_type", "service", "flag"]
TARGET_COL = "class5"  # your grouped label

# -------- paths --------
ROOT = Path.cwd()
TRAIN_PATH = "/Users/mac/Desktop/machine learning/KDD/KDDTrain+_5class.csv"
TEST_PATH = "/Users/mac/Desktop/machine learning//KDD/KDDTest+_5class.csv"

BEST_JSON = "/Users/mac/Desktop/machine learning/smg/machine_learning_models/g_boost/kdd_best_gb_hyperparameters.json"  # from nsl_kdd_gb_randomsearch.py


def load_df(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    if TARGET_COL not in df.columns:
        raise ValueError(f"{p.name} missing '{TARGET_COL}'. Columns: {df.columns.tolist()}")
    return df


print(f"Using train file: {TRAIN_PATH}")
print(f"Using test  file: {TEST_PATH}")

train_df = load_df(TRAIN_PATH)
test_df = load_df(TEST_PATH)

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
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # dense so sampler can work
except TypeError:  # older sklearn
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)         # dense so sampler can work

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

# -------- load saved hyperparameters --------
try:
    loaded = json.load(open(BEST_JSON))
    best_params_loaded = loaded["best_parameters"]
    print("\n✅ Loaded saved hyperparameters from:", BEST_JSON)
    print("Parameters:", best_params_loaded)
except Exception as e:
    print(f"\n⚠️  Could not load {BEST_JSON}: {e}")
    best_params_loaded = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 7,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "subsample": 1.0,
        "max_features": None
    }
    print("Using default hyperparameters:", best_params_loaded)

valid_keys = GradientBoostingClassifier().get_params().keys()
gb_params = {k: v for k, v in best_params_loaded.items() if k in valid_keys}

# -------- Borderline-SMOTE sampler --------
# borderline-1 focuses generation near the decision boundary (harder examples)
bsmote = BorderlineSMOTE(
    random_state=42,
    k_neighbors=5,     # neighbors for minority class
    m_neighbors=10,    # neighbors for majority to identify borderline samples
    kind="borderline-1"  # or "borderline-2"
)

# -------- build pipeline (prep -> BorderlineSMOTE -> clf) --------
pipe = Pipeline(steps=[
    ("prep", pre),
    ("bsmote", bsmote),
    ("clf", GradientBoostingClassifier(random_state=42, **gb_params)),
])

# -------- train & predict --------
print("\n⏳ Training (5-class with GradientBoosting + Borderline-SMOTE)...")
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
plt.title("Gradient Boosting + Borderline-SMOTE — NSL-KDD 5-class (pre-grouped CSVs)")
plt.tight_layout()
plt.show()
