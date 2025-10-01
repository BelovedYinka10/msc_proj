import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from datetime import datetime
from collections import Counter

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

# NEW: ClusterCentroids under-sampler
from imblearn.under_sampling import ClusterCentroids

# ================================
# File paths for 5-class KDD dataset
# ================================
TRAIN_PATH = "/Users/mac/Desktop/machine learning/KDD/KDDTrain+_5class.csv"
TEST_PATH  = "/Users/mac/Desktop/machine learning//KDD/KDDTest+_5class.csv"

print("="*70)
print("KDD MODEL WITH BEST HYPERBAND HYPERPARAMETERS + CLUSTER CENTROIDS (CC)")
print("="*70)
print("🎯 USING EXACT OPTIMAL HYPERPARAMETERS")
print("✅ Architecture: 224 units (tanh) → 480 units (relu)")
print("✅ Learning Rate: 0.00030418 (Adam optimizer)")
print("✅ Batch Size: 32")
print("✅ Under-sampling: ClusterCentroids on TRAIN only")
print("="*70)

# ================================
# Load datasets
# ================================
print("Loading KDD 5-class datasets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print(f"Train data shape: {train_df.shape}")
print(f"Test  data shape: {test_df.shape}")

# Target column (assumes last col is class label)
target_col = train_df.columns[-1]
print(f"Target column: {target_col}")

# ================================
# Preprocessing
# ================================
def preprocess_kdd_5class(train_data, test_data, target_col):
    """
    Preprocess KDD 5-class data:
      - Label-encode categorical feature columns (shared fit on train+test categories)
      - MinMax scale features
      - Label-encode targets + one-hot for Keras
    """
    print("\nPreprocessing data...")
    X_train = train_data.drop(columns=[target_col]).copy()
    y_train = train_data[target_col].copy()
    X_test  = test_data.drop(columns=[target_col]).copy()
    y_test  = test_data[target_col].copy()

    # Encode categorical columns with LabelEncoder (simple & fast)
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        print(f"Encoding categorical columns: {categorical_columns}")
        for col in categorical_columns:
            le_col = LabelEncoder()
            combined = pd.concat([X_train[col], X_test[col]], axis=0)
            le_col.fit(combined.astype(str))
            X_train[col] = le_col.transform(X_train[col].astype(str))
            X_test[col]  = le_col.transform(X_test[col].astype(str))

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Encode labels
    label_encoder   = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded  = label_encoder.transform(y_test)

    # One-hot for Keras
    num_classes         = len(label_encoder.classes_)
    y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
    y_test_categorical  = tf.keras.utils.to_categorical(y_test_encoded, num_classes)

    print(f"Classes: {label_encoder.classes_}")

    return (X_train_scaled, X_test_scaled,
            y_train_categorical, y_test_categorical,
            y_train_encoded, y_test_encoded,
            label_encoder.classes_)

# Run preprocessing
X_train, X_test, y_train_cat, y_test_cat, y_train_enc, y_test_enc, class_names = preprocess_kdd_5class(
    train_df, test_df, target_col
)

# ================================
# ClusterCentroids under-sampling (TRAIN ONLY)
# ================================
print("\n" + "="*50)
print("CLUSTER CENTROIDS (under-sampling, training set only)")
print("="*50)

counts_before = Counter(y_train_enc)
print("Class counts BEFORE CC:", {class_names[k]: v for k, v in counts_before.items()})

min_class_count = min(counts_before.values())
print(f"Minority class size = {min_class_count} → all classes will be reduced to this size.")

cc = ClusterCentroids(random_state=42)  # uses KMeans by default
t_us0 = time.time()
X_train_cc, y_train_cc = cc.fit_resample(X_train, y_train_enc)
t_us1 = time.time()

print(f"ClusterCentroids time: {t_us1 - t_us0:.2f}s")
counts_after = Counter(y_train_cc)
print("Class counts AFTER  CC:", {class_names[k]: v for k, v in counts_after.items()})
print(f"Training samples after CC: {X_train_cc.shape[0]} (down from {X_train.shape[0]})")

num_classes = len(class_names)
y_train_cc_cat = tf.keras.utils.to_categorical(y_train_cc, num_classes)

# Guard: if the downsampled set is very small, skip validation_split to avoid errors
val_split = 0.15 if X_train_cc.shape[0] >= 200 else 0.0
if val_split == 0.0:
    print("⚠️ Very small training set after CC → using validation_split=0.0")

# ================================
# Best hyperparameters (from Hyperband)
# ================================
print("\n" + "="*50)
print("BEST HYPERBAND HYPERPARAMETERS")
print("="*50)

best_hyperparams = {
    "num_layers": 2,
    "units_0": 224,
    "activation_0": "tanh",
    "kernel_init": "he_normal",
    "batch_norm_0": True,
    "dropout_0": 0.0,
    "units_1": 480,
    "activation_1": "relu",
    "batch_norm_1": True,
    "dropout_1": 0.1,
    "optimizer": "adam",
    "learning_rate": 0.00030418
}

print("Optimal Architecture Found by Hyperband:")
print(f"├── Number of layers: {best_hyperparams['num_layers']}")
print(f"├── Layer 1: {best_hyperparams['units_0']} units, {best_hyperparams['activation_0']} activation")
print(f"│   ├── Batch Norm: {best_hyperparams['batch_norm_0']}")
print(f"│   └── Dropout: {best_hyperparams['dropout_0']}")
print(f"├── Layer 2: {best_hyperparams['units_1']} units, {best_hyperparams['activation_1']} activation")
print(f"│   ├── Batch Norm: {best_hyperparams['batch_norm_1']}")
print(f"│   └── Dropout: {best_hyperparams['dropout_1']}")
print(f"├── Optimizer: {best_hyperparams['optimizer']}")
print(f"├── Learning Rate: {best_hyperparams['learning_rate']:.6f}")
print(f"└── Kernel Initializer: {best_hyperparams['kernel_init']}")

# ================================
# Model definition
# ================================
def create_optimal_model(input_dim, num_classes):
    """Create model using best Hyperband hyperparameters."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    # Layer 1
    model.add(tf.keras.layers.Dense(224, activation='tanh', kernel_initializer='he_normal'))
    model.add(tf.keras.layers.BatchNormalization())

    # Layer 2
    model.add(tf.keras.layers.Dense(480, activation='relu', kernel_initializer='he_normal'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))

    # Output
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00030418)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ================================
# Train
# ================================
print("\n" + "="*50)
print("BUILDING AND TRAINING MODEL (CC)")
print("="*50)

model = create_optimal_model(X_train_cc.shape[1], len(class_names))
model.summary()

print("\n--- Training Model with Optimal Hyperparameters + ClusterCentroids ---")
start_train = time.time()
history = model.fit(
    X_train_cc, y_train_cc_cat,
    epochs=20,
    batch_size=32,
    validation_split=val_split,
    verbose=1
)
end_train = time.time()

# ================================
# Training curves
# ================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2.5)
if 'val_accuracy' in history.history:
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2.5)
ax1.set_title('Training & Validation Accuracy')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Train Loss', linewidth=2.5)
if 'val_loss' in history.history:
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2.5)
ax2.set_title('Training & Validation Loss')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================
# Final evaluation (predictions)
# ================================
print("\n" + "="*50)
print("FINAL EVALUATION")
print("="*50)

start_test = time.time()
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
end_test = time.time()

# Macro metrics
acc = accuracy_score(y_test_enc, y_pred)
prec = precision_score(y_test_enc, y_pred, average='macro', zero_division=0)
rec  = recall_score(y_test_enc, y_pred, average='macro', zero_division=0)
f1   = f1_score(y_test_enc, y_pred, average='macro', zero_division=0)

print(f"Training Time: {end_train - start_train:.2f} seconds")
print(f"Testing  Time: {end_test - start_test:.2f} seconds")
print(f"Epochs Completed: {len(history.history['accuracy'])}")

print("\n🎯 MODEL PERFORMANCE:")
print(f"├── Accuracy:  {acc * 100:.2f}%")
print(f"├── Precision: {prec * 100:.2f}%")
print(f"├── Recall:    {rec * 100:.2f}%")
print(f"└── F1-Score:  {f1 * 100:.2f}%")

# ================================
# Confusion matrices
# ================================
cm = confusion_matrix(y_test_enc, y_pred, labels=list(range(len(class_names))))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax1, cbar_kws={'label': 'Count'}, annot_kws={'size': 12, 'weight': 'bold'})
ax1.set_title('Confusion Matrix (Counts)')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax2, cbar_kws={'label': 'Proportion'}, annot_kws={'size': 12, 'weight': 'bold'})
ax2.set_title('Confusion Matrix (Normalized)')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

plt.tight_layout()
plt.show()

# ================================
# Per-class report (precision/recall/F1/support) + class-wise accuracy
# ================================
report = classification_report(
    y_test_enc, y_pred, target_names=class_names, output_dict=True, zero_division=0
)

print("\n" + "="*80)
print("DETAILED PERFORMANCE ANALYSIS (WITH CLASS-WISE ACCURACY)")
print("="*80)

class_accuracies = cm.diagonal() / cm.sum(axis=1)
class_performance = []
for i, class_name in enumerate(class_names):
    metrics = report[class_name]
    class_performance.append({
        'Class': class_name,
        'Accuracy': f"{class_accuracies[i]*100:.2f}%",
        'Precision': f"{metrics['precision']*100:.2f}%",
        'Recall': f"{metrics['recall']*100:.2f}%",
        'F1-Score': f"{metrics['f1-score']*100:.2f}%",
        'Support': int(metrics['support'])
    })
df_performance = pd.DataFrame(class_performance)
print(df_performance.to_string(index=False))

print("\n" + "="*50)
print("CLASS-WISE ACCURACY BREAKDOWN")
print("="*50)
for i, class_name in enumerate(class_names):
    print(f"{class_name:<15}: {class_accuracies[i]*100:6.2f}%")

# ================================
# One-vs-rest per-class metrics (incl. specificity, ROC-AUC, PR-AUC)
# ================================
print("\n" + "="*80)
print("PER-CLASS METRICS (ONE-VS-REST)")
print("="*80)

N = cm.sum()
rows = []
for i, name in enumerate(class_names):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = N - TP - FN - FP

    prec_i = TP / (TP + FP) if (TP + FP) else 0.0
    rec_tpr = TP / (TP + FN) if (TP + FN) else 0.0
    spec_tnr = TN / (TN + FP) if (TN + FP) else 0.0
    f1_i = 2 * prec_i * rec_tpr / (prec_i + rec_tpr) if (prec_i + rec_tpr) else 0.0
    acc_ovr = (TP + TN) / N if N else 0.0

    # Probability scores for AUCs
    y_true_bin = (y_test_enc == i).astype(int)
    y_score = y_pred_probs[:, i]

    if y_true_bin.max() > 0 and y_true_bin.min() == 0:
        try:
            roc_auc = roc_auc_score(y_true_bin, y_score)
        except Exception:
            roc_auc = np.nan
        try:
            pr_auc = average_precision_score(y_true_bin, y_score)
        except Exception:
            pr_auc = np.nan
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    rows.append({
        "class": name,
        "support": int(cm[i, :].sum()),
        "precision": prec_i,
        "recall_TPR": rec_tpr,
        "specificity_TNR": spec_tnr,
        "f1": f1_i,
        "accuracy_ovr": acc_ovr,
        "roc_auc_ovr": float(roc_auc) if roc_auc == roc_auc else None,
        "pr_auc_ovr":  float(pr_auc) if pr_auc == pr_auc else None
    })

per_class_df = pd.DataFrame(rows).sort_values("class")
pd.options.display.float_format = lambda v: f"{v:.4f}"
print(per_class_df.to_string(index=False))

# Save per-class tables
per_class_df.to_csv("per_class_metrics_cc.csv", index=False)
with open("per_class_metrics_cc.json", "w") as f:
    json.dump(per_class_df.to_dict(orient="records"), f, indent=2)

print("\nSaved per-class metrics to:")
print("  • per_class_metrics_cc.csv")
print("  • per_class_metrics_cc.json")

# ================================
# Attack detection (Normal vs Attack)
# ================================
print("\n" + "="*50)
print("ATTACK DETECTION PERFORMANCE")
print("="*50)
# Find "normal" label index (case-insensitive). Fallback to 0 if absent.
normal_id_arr = np.where(np.array([c.lower() for c in class_names]) == "normal")[0]
normal_id = int(normal_id_arr[0]) if len(normal_id_arr) else 0

y_test_binary = (y_test_enc != normal_id).astype(int)  # 0: Normal, 1: Attack
y_pred_binary = (y_pred != normal_id).astype(int)

binary_acc  = accuracy_score(y_test_binary, y_pred_binary)
binary_prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
binary_rec  = recall_score(y_test_binary, y_pred_binary, zero_division=0)
binary_f1   = f1_score(y_test_binary, y_pred_binary, zero_division=0)

print("Normal vs Attack Classification:")
print(f"├── Accuracy:  {binary_acc * 100:.2f}%")
print(f"├── Precision: {binary_prec * 100:.2f}%")
print(f"├── Recall:    {binary_rec * 100:.2f}%")
print(f"└── F1-Score:  {binary_f1 * 100:.2f}%")

# ================================
# Save model and results
# ================================
model.save('kdd_optimal_model_cc.keras')

results_summary = {
    "timestamp": datetime.now().isoformat(),
    "model_type": "Optimal Hyperband Model + ClusterCentroids",
    "hyperparameters": {
        **best_hyperparams,
        "cc_estimator": "KMeans(default)",  # informative note
        "cc_strategy": "auto (to minority)"
    },
    "epochs_completed": int(len(history.history['accuracy'])),
    "training_time_seconds": float(end_train - start_train),
    "testing_time_seconds": float(end_test - start_test),
    "macro_performance": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    },
    "class_wise_accuracy": {
        str(class_names[i]): float(class_accuracies[i]) for i in range(len(class_names))
    },
    "per_class_metrics": per_class_df.to_dict(orient="records"),
    "attack_detection": {
        "binary_accuracy": float(binary_acc),
        "binary_precision": float(binary_prec),
        "binary_recall": float(binary_rec),
        "binary_f1": float(binary_f1)
    },
    "train_class_counts_before": {str(class_names[k]): int(v) for k, v in counts_before.items()},
    "train_class_counts_after":  {str(class_names[k]): int(v) for k, v in counts_after.items()}
}

with open("optimal_model_results_cc.json", 'w') as f:
    json.dump(results_summary, f, indent=4)

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE (CC)!")
print("="*60)
print(f"🎯 Final Accuracy: {acc*100:.2f}%")
print("📁 Model saved as: kdd_optimal_model_cc.keras")
print("📁 Results saved as: optimal_model_results_cc.json")
print("📁 Per-class CSV:   per_class_metrics_cc.csv")
print("📁 Per-class JSON:  per_class_metrics_cc.json")
