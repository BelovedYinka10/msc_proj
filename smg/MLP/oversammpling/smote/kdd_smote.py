import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# File paths for 5-class KDD dataset
TRAIN_PATH = "/Users/mac/Desktop/machine learning/KDD/KDDTrain+_5class.csv"
TEST_PATH = "/Users/mac/Desktop/machine learning//KDD/KDDTest+_5class.csv"

print("=" * 70)
print("KDD MODEL WITH BEST HYPERBAND HYPERPARAMETERS")
print("=" * 70)
print("🎯 USING EXACT OPTIMAL HYPERPARAMETERS")
print("✅ Architecture: 224 units (tanh) → 480 units (relu)")
print("✅ Learning Rate: 0.00030418 (Adam optimizer)")
print("✅ Batch Size: 32")
print("✅ Simple training - no complications!")
print("=" * 70)

# Load datasets
print("Loading KDD 5-class datasets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Get target column
target_col = train_df.columns[-1]
print(f"Target column: {target_col}")


# Preprocessing function
def preprocess_kdd_5class(train_data, test_data, target_col):
    """Preprocess KDD 5-class data"""
    print("\nPreprocessing data...")

    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    # Handle categorical features
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        print(f"Encoding categorical columns: {categorical_columns}")
        for col in categorical_columns:
            le = LabelEncoder()
            combined_col = pd.concat([X_train[col], X_test[col]], axis=0)
            le.fit(combined_col.astype(str))
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Convert to categorical
    num_classes = len(label_encoder.classes_)
    y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
    y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, num_classes)

    print(f"Classes: {label_encoder.classes_}")

    return (X_train_scaled, X_test_scaled, y_train_categorical, y_test_categorical,
            y_train_encoded, y_test_encoded, label_encoder.classes_)


# Preprocess data
X_train, X_test, y_train_cat, y_test_cat, y_train_enc, y_test_enc, class_names = preprocess_kdd_5class(
    train_df, test_df, target_col
)

# ---------- SMOTE sampling (inserted here; fractional targets to avoid huge blow-up) ----------
print("\n" + "=" * 60)
print("APPLYING SMOTE TO TRAINING SET (fractional targets)")
print("=" * 60)
try:
    from imblearn.over_sampling import SMOTE
    from collections import Counter

    # Choose fraction of majority to upsample minorities to (0 < SMOTE_FRACTION <= 1.0)
    # Example: 0.30 means each minority class will be upsampled to 30% of the majority class count.
    SMOTE_FRACTION = 0.30

    # Compute original class counts
    orig_counts = Counter(y_train_enc)
    majority_label, majority_count = max(orig_counts.items(), key=lambda kv: kv[1])

    print(f"Original training class distribution (majority={class_names[majority_label]}: {majority_count}):")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {orig_counts.get(idx, 0)}")

    # Build sampling_target dict: for each minority cls, target = max(current_count, int(SMOTE_FRACTION * majority_count))
    sampling_target = {}
    for cls_idx, cnt in orig_counts.items():
        if cls_idx == majority_label:
            # keep majority unchanged
            sampling_target[cls_idx] = cnt
        else:
            target = max(cnt, int(SMOTE_FRACTION * majority_count))
            sampling_target[cls_idx] = target

    # Print targets
    print("\nSMOTE sampling targets (per-class):")
    for idx, name in enumerate(class_names):
        tgt = sampling_target.get(idx, 0)
        print(f"  {name}: target -> {tgt}")

    # Create SMOTE with k_neighbors (defaults 5). SMOTE accepts a dict for sampling_strategy.
    sm = SMOTE(sampling_strategy=sampling_target, random_state=42, k_neighbors=5)

    # Fit-resample
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc)

    res_counts = Counter(y_train_res)
    print("\nAfter SMOTE training class distribution:")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {res_counts.get(idx, 0)}")

    # Convert resampled labels to categorical for Keras
    num_classes = len(class_names)
    y_train_res_cat = tf.keras.utils.to_categorical(y_train_res, num_classes)

    # Replace training inputs used for model.fit below
    X_train_used = X_train_res
    y_train_used_cat = y_train_res_cat
    y_train_used_enc = y_train_res

    print("✅ SMOTE applied successfully.")
except Exception as e:
    print("⚠️ imbalanced-learn (imblearn) not available or SMOTE failed:", e)
    print("Proceeding WITHOUT SMOTE. The original training set will be used.")
    X_train_used = X_train
    y_train_used_cat = y_train_cat
    y_train_used_enc = y_train_enc

# Best hyperparameters from your Hyperband results
print(f"\n" + "=" * 50)
print("BEST HYPERBAND HYPERPARAMETERS")
print("=" * 50)

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


# Create model with best hyperparameters
def create_optimal_model(input_dim, num_classes):
    """Create model using best Hyperband hyperparameters"""

    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    # Layer 1: 224 units, tanh activation
    model.add(tf.keras.layers.Dense(
        224,
        activation='tanh',
        kernel_initializer='he_normal'
    ))
    model.add(tf.keras.layers.BatchNormalization())
    # No dropout for layer 1 (dropout_0 = 0.0)

    # Layer 2: 480 units, relu activation
    model.add(tf.keras.layers.Dense(
        480,
        activation='relu',
        kernel_initializer='he_normal'
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))

    # Output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # Compile with optimal settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00030418)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Create and train the model
print(f"\n" + "=" * 50)
print("BUILDING AND TRAINING MODEL")
print("=" * 50)

model = create_optimal_model(X_train_used.shape[1], len(class_names))
model.summary()

# Train the model - simple and straightforward
print(f"\n--- Training Model with Optimal Hyperparameters ---")

start_train = time.time()

history = model.fit(
    X_train_used, y_train_used_cat,
    epochs=20,  # Reasonable number of epochs
    batch_size=32,  # Standard batch size
    validation_split=0.15,
    verbose=1
)

end_train = time.time()

# Training visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2.5, color='blue')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2.5, color='red')
ax1.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history.history['loss'], label='Train Loss', linewidth=2.5, color='blue')
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2.5, color='red')
ax2.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final evaluation
print(f"\n" + "=" * 50)
print("FINAL EVALUATION")
print("=" * 50)

start_test = time.time()
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
end_test = time.time()

# Calculate metrics
acc = accuracy_score(y_test_enc, y_pred)
prec = precision_score(y_test_enc, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test_enc, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test_enc, y_pred, average='macro', zero_division=0)

print(f"Training Time: {end_train - start_train:.2f} seconds")
print(f"Testing Time: {end_test - start_test:.2f} seconds")
print(f"Epochs Completed: {len(history.history['accuracy'])}")

print(f"\n🎯 MODEL PERFORMANCE:")
print(f"├── Accuracy:  {acc * 100:.2f}%")
print(f"├── Precision: {prec * 100:.2f}%")
print(f"├── Recall:    {rec * 100:.2f}%")
print(f"└── F1-Score:  {f1 * 100:.2f}%")

# Confusion matrix visualization
cm = confusion_matrix(y_test_enc, y_pred)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax1, cbar_kws={'label': 'Count'},
            annot_kws={'size': 12, 'weight': 'bold'})
ax1.set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Predicted Label', fontsize=12)
ax1.set_ylabel('True Label', fontsize=12)

# Normalized
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax2, cbar_kws={'label': 'Proportion'},
            annot_kws={'size': 12, 'weight': 'bold'})
ax2.set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Predicted Label', fontsize=12)
ax2.set_ylabel('True Label', fontsize=12)

plt.tight_layout()
plt.show()

# Per-class analysis WITH class-wise accuracy
report = classification_report(y_test_enc, y_pred, target_names=class_names,
                               output_dict=True, zero_division=0)

print(f"\n" + "=" * 80)
print("DETAILED PERFORMANCE ANALYSIS (WITH CLASS-WISE ACCURACY)")
print("=" * 80)

# Calculate class-wise accuracy from confusion matrix
class_accuracies = cm.diagonal() / cm.sum(axis=1)

class_performance = []
for i, class_name in enumerate(class_names):
    metrics = report[class_name]
    class_performance.append({
        'Class': class_name,
        'Accuracy': f"{class_accuracies[i] * 100:.2f}%",  # Added class-wise accuracy
        'Precision': f"{metrics['precision'] * 100:.2f}%",
        'Recall': f"{metrics['recall'] * 100:.2f}%",
        'F1-Score': f"{metrics['f1-score'] * 100:.2f}%",
        'Support': int(metrics['support'])
    })

df_performance = pd.DataFrame(class_performance)
print(df_performance.to_string(index=False))

# Additional class-wise accuracy section
print(f"\n" + "=" * 50)
print("CLASS-WISE ACCURACY BREAKDOWN")
print("=" * 50)

for i, class_name in enumerate(class_names):
    print(f"{class_name:<15}: {class_accuracies[i] * 100:6.2f}%")

# Attack detection performance
y_test_binary = (y_test_enc != 0).astype(int)  # Normal vs Attack
y_pred_binary = (y_pred != 0).astype(int)

binary_acc = accuracy_score(y_test_binary, y_pred_binary)
binary_prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
binary_rec = recall_score(y_test_binary, y_pred_binary, zero_division=0)
binary_f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)

print(f"\n" + "=" * 50)
print("ATTACK DETECTION PERFORMANCE")
print("=" * 50)
print(f"Normal vs Attack Classification:")
print(f"├── Accuracy:  {binary_acc * 100:.2f}%")
print(f"├── Precision: {binary_prec * 100:.2f}%")
print(f"├── Recall:    {binary_rec * 100:.2f}%")
print(f"└── F1-Score:  {binary_f1 * 100:.2f}%")

# Save model and results
model.save('kdd_optimal_model.keras')

results_summary = {
    "timestamp": datetime.now().isoformat(),
    "model_type": "Optimal Hyperband Model",
    "hyperparameters": best_hyperparams,
    "epochs_completed": len(history.history['accuracy']),
    "training_time_seconds": end_train - start_train,
    "testing_time_seconds": end_test - start_test,
    "performance": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    },
    "class_wise_accuracy": {
        class_names[i]: float(class_accuracies[i]) for i in range(len(class_names))
    },
    "attack_detection": {
        "binary_accuracy": float(binary_acc),
        "binary_precision": float(binary_prec),
        "binary_recall": float(binary_rec),
        "binary_f1": float(binary_f1)
    }
}

with open("optimal_model_results.json", 'w') as f:
    json.dump(results_summary, f, indent=4)

print(f"\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"🎯 Final Accuracy: {acc * 100:.2f}%")
print("📁 Model saved as: kdd_optimal_model.keras")
print("📁 Results saved as: optimal_model_results.json")
print("✅ Class-wise accuracy included in detailed analysis!")
