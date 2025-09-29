# kdd_benchmark_5class_print.py
# Uses pre-made 5-class CSVs, NearMiss sampling, and PRINTS metrics only (no file saves).

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import NearMiss
import json

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

# -------- Preprocessing and NearMiss Undersampling --------
print("\n⏳ Applying preprocessing...")

# Fit preprocessing on training data
X_train_prep = pre.fit_transform(X_train)
X_test_prep = pre.transform(X_test)

print(f"Original training set shape: {X_train.shape}")
print(f"Original class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for i, (cls, cnt) in enumerate(zip(unique, counts)):
    print(f"  Class {class_names[cls]}: {cnt} samples")

# Apply NearMiss undersampling
print("\n⏳ Applying NearMiss undersampling...")
nm_start = time.time()
nm = NearMiss(version=1, n_jobs=-1)
X_train_resampled, y_train_resampled = nm.fit_resample(X_train_prep, y_train)
nm_time = time.time() - nm_start

print(f"✅ NearMiss undersampling completed in {nm_time:.3f}s")
print(f"Resampled training set shape: {X_train_resampled.shape}")
print(f"Resampled class distribution:")
unique, counts = np.unique(y_train_resampled, return_counts=True)
for i, (cls, cnt) in enumerate(zip(unique, counts)):
    print(f"  Class {class_names[cls]}: {cnt} samples")

# -------- classifier --------
try:
    best_params = json.load(open("best_dt_hyperparameters.json"))["best_parameters"]
    print(f"\n✅ Using saved hyperparameters from JSON: {best_params}")
except Exception as e:
    best_params = {"criterion":"entropy","max_depth":25,"min_samples_split":15,
                   "min_samples_leaf":8,"max_leaf_nodes":200}
    print(f"\n⚠️  Could not load saved hyperparameters ({e}). Using default: {best_params}")

# Since we're using NearMiss, we don't need class_weight balancing
if 'class_weight' in best_params:
    best_params.pop('class_weight')
    print("   Removed 'class_weight' parameter since NearMiss handles class imbalance")

clf = DecisionTreeClassifier(random_state=42, **best_params)

# -------- train & predict --------
print("\n⏳ Training (5-class with NearMiss)...")
t0 = time.time(); clf.fit(X_train_resampled, y_train_resampled); train_time = time.time() - t0
print(f"✅ Training time: {train_time:.3f}s")

print("⏳ Predicting (5-class)...")
t1 = time.time(); y_pred = clf.predict(X_test_prep); test_time = time.time() - t1
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

# Convert percentage columns to percentage format (multiply by 100)
percentage_cols = ["precision", "recall", "f1", "accuracy_ovr", "specificity"]
for col in percentage_cols:
    per_class_df[col] = per_class_df[col] * 100

pd.options.display.float_format = lambda v: f"{v:.2f}"
print("\n=== Per-class metrics ===")
print(per_class_df.to_string(index=False))

# -------- confusion matrix plot (optional) --------
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Decision Tree — NSL-KDD 5-class (NearMiss + Optimized Hyperparameters)")
plt.tight_layout()
plt.show()