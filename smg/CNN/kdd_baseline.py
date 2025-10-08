# kdd_cnn_from_tuner_values.py
# Build a CNN for NSL-KDD (5-class) using the provided tuner hyperparameters
# for the Dense head (n_layers, units_i, activation_i, batch_norm_i, dropout_i, learning_rate).

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from tensorflow import keras

# -----------------------
# Paths (NSL-KDD 5-class)
# -----------------------
TRAIN_PATH = "/Users/mac/Desktop/machine learning/KDD/KDDTrain+_5class.csv"
TEST_PATH  = "/Users/mac/Desktop/machine learning//KDD/KDDTest+_5class.csv"
TARGET_COL = "class5"   # grouped 5-class label

np.random.seed(42)
tf.random.set_seed(42)

# -----------------------
# Your tuner values (as provided)
# -----------------------
# NOTE: These define the *Dense head* on top of a small Conv1D feature extractor.
HP = {
    "n_layers": 4,
    "units_0": 832,
    "activation_0": "tanh",
    "batch_norm_0": False,
    "dropout_0": 0.15,

    "units_1": 128,
    "activation_1": "selu",
    "batch_norm_1": False,
    "dropout_1": 0.45,

    "units_2": 704,
    "activation_2": "relu",
    "batch_norm_2": True,
    "dropout_2": 0.25,

    "units_3": 768,
    "activation_3": "tanh",
    "batch_norm_3": False,
    "dropout_3": 0.05,

    "learning_rate": 0.0038393,
}

print("="*70)
print("NSL-KDD (5-class) — CNN with Dense Head from Tuner Values")
print("="*70)
for k, v in HP.items():
    print(f"{k}: {v}")

# -----------------------
# Preprocess
# -----------------------
def load_and_preprocess(train_path, test_path, target_col):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Drop constant columns (except target)
    for df in (train_df, test_df):
        nunique = df.nunique()
        drop_cols = [c for c in df.columns if (nunique[c] <= 1 and c != target_col)]
        df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # Remove extra label-ish columns if present
    extra = ["label", "difficulty", "binary"]
    train_df.drop(columns=[c for c in extra if c in train_df.columns], errors="ignore", inplace=True)
    test_df.drop(columns=[c for c in extra if c in test_df.columns], errors="ignore", inplace=True)

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' missing in train and/or test.")

    y_train_raw = train_df[target_col].astype(str)
    y_test_raw  = test_df[target_col].astype(str)
    X_train_df  = train_df.drop(columns=[target_col]).copy()
    X_test_df   = test_df.drop(columns=[target_col]).copy()

    # Numerics to numeric
    num_cols = X_train_df.select_dtypes(exclude=["object"]).columns.tolist()
    for col in num_cols:
        X_train_df[col] = pd.to_numeric(X_train_df[col], errors="coerce")
        X_test_df[col]  = pd.to_numeric(X_test_df[col], errors="coerce")

    # One-hot encode categoricals
    cat_cols = X_train_df.select_dtypes(include=["object"]).columns.tolist()
    X_train_oh = pd.get_dummies(X_train_df, columns=cat_cols, drop_first=False)
    X_test_oh  = pd.get_dummies(X_test_df,  columns=cat_cols, drop_first=False)

    # Align columns (add missing test cols as 0)
    X_train_oh, X_test_oh = X_train_oh.align(X_test_oh, join="left", axis=1, fill_value=0)

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_oh.fillna(0))
    X_test  = scaler.transform(X_test_oh.fillna(0))

    # Stable label encoding
    le = LabelEncoder()
    le.fit(pd.concat([y_train_raw, y_test_raw], axis=0))
    y_train_enc = le.transform(y_train_raw.values)
    y_test_enc  = le.transform(y_test_raw.values)
    class_names = le.classes_.tolist()

    # One-hot targets for Keras
    num_classes = len(class_names)
    y_train_cat = keras.utils.to_categorical(y_train_enc, num_classes)
    y_test_cat  = keras.utils.to_categorical(y_test_enc, num_classes)

    return (X_train, X_test, y_train_cat, y_test_cat,
            y_train_enc, y_test_enc, class_names)

X_train, X_test, y_train_cat, y_test_cat, y_train_enc, y_test_enc, class_names = load_and_preprocess(
    TRAIN_PATH, TEST_PATH, TARGET_COL
)
input_dim = X_train.shape[1]
num_classes = y_train_cat.shape[1]

print(f"\nClasses: {class_names}")
print(f"Train: X={X_train.shape}, y={y_train_cat.shape} | Test: X={X_test.shape}, y={y_test_cat.shape}")

# -----------------------
# Model
# -----------------------
def add_dense_block(model, units, activation, use_bn, dropout):
    # sensible initializer: selu -> lecun_normal, else he_normal
    kernel_init = "lecun_normal" if activation == "selu" else "he_normal"
    model.add(keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_init))
    if use_bn:
        model.add(keras.layers.BatchNormalization())
    if dropout and dropout > 0:
        model.add(keras.layers.Dropout(dropout))

def build_cnn_with_dense_head(hp: dict, input_dim: int, num_classes: int) -> keras.Model:
    m = keras.Sequential()
    # 1D Conv feature extractor
    m.add(keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)))
    m.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu"))
    m.add(keras.layers.BatchNormalization())
    m.add(keras.layers.MaxPooling1D(pool_size=2))
    m.add(keras.layers.Dropout(0.10))

    m.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
    m.add(keras.layers.BatchNormalization())
    m.add(keras.layers.GlobalMaxPooling1D())
    m.add(keras.layers.Dropout(0.30))

    # Dense head from HP
    n_layers = int(hp.get("n_layers", 1))
    for i in range(n_layers):
        units = int(hp.get(f"units_{i}", 128))
        activation = str(hp.get(f"activation_{i}", "relu"))
        use_bn = bool(hp.get(f"batch_norm_{i}", False))
        dropout = float(hp.get(f"dropout_{i}", 0.0))
        add_dense_block(m, units, activation, use_bn, dropout)

    # Output
    m.add(keras.layers.Dense(num_classes, activation="softmax"))

    lr = float(hp.get("learning_rate", 3e-4))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
    return m

model = build_cnn_with_dense_head(HP, input_dim, num_classes)
model.summary()

# -----------------------
# Train
# -----------------------
epochs = 20
batch_size = 32
es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

print("\n--- Training CNN (tuner-configured dense head) ---")
t0 = time.time()
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.1,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[es],
    verbose=1
)
t1 = time.time()

# -----------------------
# Curves
# -----------------------
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('CNN (Dense head from tuner) — Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Acc')
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('CNN (Dense head from tuner) — Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.tight_layout(); plt.show()

# -----------------------
# Predict & Evaluate
# -----------------------
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

acc  = accuracy_score(y_test_enc, y_pred)
prec = precision_score(y_test_enc, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test_enc, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)

print("\n=== Overall Metrics ===")
print(f"Accuracy:        {acc*100:.2f}%")
print(f"Precision (mac): {prec*100:.2f}%")
print(f"Recall (mac):    {rec*100:.2f}%")
print(f"F1 Score (mac):  {f1*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test_enc, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('CNN (tuner dense head) — Confusion Matrix (NSL-KDD 5-class)')
plt.tight_layout(); plt.show()

# Per-class metrics w/ class-wise accuracy
report = classification_report(y_test_enc, y_pred, target_names=class_names, output_dict=True, zero_division=0)
print("\n=== Per-Class Metrics ===")
for i, cname in enumerate(class_names):
    m = report[cname]
    true_mask = (y_test_enc == i)
    correct = np.sum((y_pred == i) & true_mask)
    total = np.sum(true_mask)
    acc_cls = correct / total if total > 0 else 0.0
    print(f"\nClass: {cname}")
    print(f"  Accuracy:  {acc_cls*100:.2f}%")
    print(f"  Precision: {m['precision']*100:.2f}%")
    print(f"  Recall:    {m['recall']*100:.2f}%")
    print(f"  F1-score:  {m['f1-score']*100:.2f}%")

print("\n=== Timing ===")
print(f"Training Time: {t1 - t0:.2f} seconds")
