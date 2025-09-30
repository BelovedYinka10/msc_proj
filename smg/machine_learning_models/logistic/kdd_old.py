# NSL-KDD Logistic Regression - Following Paper's Methodology
# Paper: "Distributed Anomaly Detection in Smart Grids: A Federated Learning-Based Approach"
# Table: Penalty=L2, C=10, Solver=lbfgs, Max Iterations=3000

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression

# -------- Configuration from Paper --------
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

# -------- File Paths --------
TRAIN_PATH = "/Users/mac/Desktop/machine learning/KDD/KDD Train+.csv"
TEST_PATH  = "/Users/mac/Desktop/machine learning/KDD/KDDTest+.csv"

print("=" * 70)
print("NSL-KDD Logistic Regression - Paper Implementation")
print("=" * 70)

# -------- Load Data --------
def load_nsl_kdd(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    return df

print(f"\nLoading training data from: {TRAIN_PATH}")
train_df = load_nsl_kdd(TRAIN_PATH)
print(f"Training samples: {len(train_df)}")

print(f"Loading test data from: {TEST_PATH}")
test_df = load_nsl_kdd(TEST_PATH)
print(f"Test samples: {len(test_df)}")

# -------- Preprocessing --------
print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)

# Find label column
target_col = None
for cand in ["label", "class", "Class", "target"]:
    if cand in train_df.columns:
        target_col = cand
        break
if target_col is None:
    raise ValueError("Could not find label column")
print(f"Label column: {target_col}")

# Binary labels: normal -> 0, others -> 1
def to_binary(labels: pd.Series) -> pd.Series:
    return labels.astype(str).str.strip().str.lower().apply(lambda x: 0 if x == "normal" else 1)

y_train = to_binary(train_df[target_col])
y_test  = to_binary(test_df[target_col])

print(f"\nClass distribution (train):\n{y_train.value_counts()}")
print(f"\nClass distribution (test):\n{y_test.value_counts()}")

# Features
available_features = [f for f in FEATURES if f in train_df.columns]
X_train = train_df[available_features].copy()
X_test  = test_df[available_features].copy()

print(f"\nFeatures used: {len(available_features)}")
numerical_features   = [f for f in available_features if f not in CATEGORICAL_FEATURES]
categorical_features = [f for f in CATEGORICAL_FEATURES if f in available_features]
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Coerce types
for col in numerical_features:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col]  = pd.to_numeric(X_test[col], errors='coerce')
for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col]  = X_test[col].astype(str)

# -------- Preprocessing pipeline --------
print("\nBuilding preprocessing pipeline...")
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])
cat_pipe = Pipeline([
    ('onehot', ohe)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipe, numerical_features),
        ('cat', cat_pipe, categorical_features)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# -------- Logistic Regression (EXACT hyper-params from table) --------
print("\nCreating Logistic Regression with exact table hyper-parameters:")
print("  Penalty=L2, C=10, solver=lbfgs, max_iter=3000")

model = LogisticRegression(
    penalty='l2',
    C=10.0,                 # <-- table Best Value
    solver='lbfgs',
    max_iter=3000,          # <-- table Best Value
    n_jobs=-1,
    random_state=42
    # class_weight not specified in table → leave as default (None)
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# -------- Training --------
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)
t0 = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - t0
print(f"✓ Training completed in {train_time:.2f}s")

# -------- Prediction --------
print("\n" + "=" * 70)
print("PREDICTION & THRESHOLD ANALYSIS")
print("=" * 70)
t1 = time.time()
y_proba = pipeline.predict_proba(X_test)[:, 1]
pred_time = time.time() - t1
print(f"✓ Prediction completed in {pred_time:.4f}s")

# Threshold sweep for F1
ths = np.arange(0.10, 0.90, 0.05)
best_t, best_f1 = 0.50, -1
results = []
for t in ths:
    y_hat = (y_proba >= t).astype(int)
    pr = precision_score(y_test, y_hat, zero_division=0)
    rc = recall_score(y_test, y_hat, zero_division=0)
    f1 = f1_score(y_test, y_hat, zero_division=0)
    results.append({"threshold": t, "precision": pr, "recall": rc, "f1": f1})
    if f1 > best_f1:
        best_t, best_f1 = t, f1
print(f"Best threshold by F1: {best_t:.2f} (F1={best_f1:.4f})")

# You can change which threshold to use; here we keep 0.50 to mirror many papers
final_threshold = 0.50
y_pred = (y_proba >= final_threshold).astype(int)
print(f"\n*** USING THRESHOLD: {final_threshold:.2f} ***")

# -------- Evaluation --------
print("\n" + "=" * 70)
print("PERFORMANCE METRICS")
print("=" * 70)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
try:
    auc = roc_auc_score(y_test, y_proba)
except Exception:
    auc = None

if auc is not None:
    print(f"AUC:       {auc:.4f}")
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

print("\n" + "-" * 70)
print("DETAILED CLASSIFICATION REPORT")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)'], digits=4))

cm = confusion_matrix(y_test, y_pred)
print("\n" + "-" * 70)
print("CONFUSION MATRIX")
print("-" * 70)
print(f"                Predicted")
print(f"                Normal    Attack")
print(f"Actual Normal   {cm[0,0]:<9} {cm[0,1]:<9}")
print(f"Actual Attack   {cm[1,0]:<9} {cm[1,1]:<9}")

tn, fp, fn, tp = cm.ravel()
tpr = tp / (tp + fn) if (tp + fn) else 0.0
fpr = fp / (fp + tn) if (fp + tn) else 0.0
print(f"\nTrue Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")
print(f"\nAttack Detection Rate (Recall): {tpr:.4f} ({tpr*100:.2f}%)")
print(f"False Alarm Rate:               {fpr:.4f} ({fpr*100:.2f}%)")

# -------- Visualization --------
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1) Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Attack']).plot(cmap='Blues', ax=ax1)
ax1.set_title(f'Confusion Matrix (Threshold={final_threshold:.2f})')

# 2) ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
if auc is not None:
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
    ax2.plot(fpr_curve, tpr_curve, label=f'ROC (AUC={auc:.4f})', linewidth=2)
    ax2.plot([0,1], [0,1], 'k--', linewidth=1, label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate (Recall)')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(alpha=0.3)

# 3) Threshold analysis
ax3 = fig.add_subplot(gs[1, :])
res_df = pd.DataFrame(results)
ax3.plot(res_df['threshold'], res_df['precision'], marker='o', label='Precision')
ax3.plot(res_df['threshold'], res_df['recall'], marker='s', label='Recall')
ax3.plot(res_df['threshold'], res_df['f1'], marker='^', label='F1')
ax3.axvline(final_threshold, color='red', linestyle='--', label=f'Selected ({final_threshold:.2f})')
ax3.set_xlabel('Decision Threshold'); ax3.set_ylabel('Score'); ax3.legend(); ax3.grid(alpha=0.3)
ax3.set_title('Threshold vs Metrics')

# 4) Class distribution
ax4 = fig.add_subplot(gs[2, 0])
class_dist = pd.DataFrame({
    'Actual': y_test.value_counts().sort_index(),
    'Predicted': pd.Series(y_pred).value_counts().sort_index()
})
class_dist.plot(kind='bar', ax=ax4, color=['#2E86AB', '#A23B72'])
ax4.set_title('Class Distribution: Actual vs Predicted')
ax4.set_xlabel('Class (0=Normal, 1=Attack)'); ax4.set_ylabel('Count'); ax4.set_xticklabels(['Normal','Attack'], rotation=0)
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('NSL-KDD Logistic Regression — Exact Hyperparameters', fontsize=14, fontweight='bold', y=0.995)
plt.show()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
