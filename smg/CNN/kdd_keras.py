# kdd_baseline_fnn.py
# Load best hyperparameters from JSON, rebuild the exact FNN, train once, and evaluate on NSL-KDD 5-class.

import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from tensorflow import keras

# -----------------------
# Paths / Config
# -----------------------
TRAIN_PATH = "/Users/mac/Desktop/machine learning/KDD/KDDTrain+_5class.csv"
TEST_PATH  = "/Users/mac/Desktop/machine learning//KDD/KDDTest+_5class.csv"
TARGET_COL = "class5"
BEST_JSON  = "kdd_best_fnn_hyperparams.json"

np.random.seed(42)
tf.random.set_seed(42)

# -----------------------
# Preprocessing (same as tuner)
# -----------------------
def load_and_preprocess(train_path, test_path, target_col):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    for df in (train_df, test_df):
        nunique = df.nunique()
        drop_cols = [c for c in df.columns if (nunique[c] <= 1 and c != target_col)]
        df.drop(columns=drop_cols, errors="ignore", inplace=True)

    extra = ["label", "difficulty", "binary"]
    train_df.drop(columns=[c for c in extra if c in train_df.columns], errors="ignore", inplace=True)
    test_df.drop(columns=[c for c in extra if c in test_df.columns], errors="ignore", inplace=True)

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' missing in train and/or test.")

    y_train_raw = train_df[target_col].astype(str)
    y_test_raw  = test_df[target_col].astype(str)
    X_train_df  = train_df.drop(columns=[target_col]).copy()
    X_test_df   = test_df.drop(columns=[target_col]).copy()

    num_cols = X_train_df.select_dtypes(exclude=["object"]).columns.tolist()
    for col in num_cols:
        X_train_df[col] = pd.to_numeric(X_train_df[col], errors="coerce")
        X_test_df[col]  = pd.to_numeric(X_test_df[col], errors="coerce")

    cat_cols = X_train_df.select_dtypes(include=["object"]).columns.tolist()
    X_train_oh = pd.get_dummies(X_train_df, columns=cat_cols, drop_first=False)
    X_test_oh  = pd.get_dummies(X_test_df,  columns=cat_cols, drop_first=False)

    X_train_oh, X_test_oh = X_train_oh.align(X_test_oh, join="left", axis=1, fill_value=0)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_oh.fillna(0))
    X_test  = scaler.transform(X_test_oh.fillna(0))

    le = LabelEncoder()
    le.fit(pd.concat([y_train_raw, y_test_raw], axis=0))
    y_train_enc = le.transform(y_train_raw.values)
    y_test_enc  = le.transform(y_test_raw.values)
    class_names = le.classes_.tolist()

    num_classes = len(class_names)
    y_train_cat = keras.utils.to_categorical(y_train_enc, num_classes)
    y_test_cat  = keras.utils.to_categorical(y_test_enc, num_classes)

    return X_train, X_test, y_train_cat, y_test_cat, y_train_enc, y_test_enc, class_names

X_train, X_test, y_train_cat, y_test_cat, y_test_enc_y, y_test_enc, class_names = load_and_preprocess(
    TRAIN_PATH, TEST_PATH, TARGET_COL
)
# note: y_test_enc_y is y_train_enc in order; variable kept for API parity

print("="*70)
print("NSL-KDD (5-class) — FNN Baseline with Best Hyperparameters")
print("="*70)
print(f"Classes: {class_names}")
print(f"Train: X={X_train.shape}, y={y_train_cat.shape} | Test: X={X_test.shape}, y={y_test_cat.shape}")

# -----------------------
# Load best hyperparameters
# -----------------------
with open(BEST_JSON, "r") as f:
    cfg = json.load(f)
best = cfg["best_hyperparameters"]

def build_fnn_from_best(hp_dict, input_dim, num_classes):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))

    n_layers = int(hp_dict.get("n_layers", 2))
    for i in range(n_layers):
        units = int(hp_dict.get(f"units_{i}", 256))
        act   = hp_dict.get(f"activation_{i}", "relu")
        model.add(keras.layers.Dense(units, activation=act, kernel_initializer="he_normal"))
        if bool(hp_dict.get(f"batch_norm_{i}", False)):
            model.add(keras.layers.BatchNormalization())
        dr = float(hp_dict.get(f"dropout_{i}", 0.0))
        if dr > 0:
            model.add(keras.layers.Dropout(dr))

    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    lr = float(hp_dict.get("learning_rate", 3e-4))
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

input_dim = X_train.shape[1]
num_classes = y_train_cat.shape[1]
model = build_fnn_from_best(best, input_dim, num_classes)
model.summary()

# -----------------------
# Train
# -----------------------
epochs = 20
batch_size = 32
es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

print("\n--- Training FNN (best hparams) ---")
t0 = time.time()
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.15,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[es],
    verbose=1
)
t1 = time.time()

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
plt.title('FNN — Confusion Matrix (NSL-KDD 5-class)')
plt.tight_layout(); plt.show()

# Per-class report incl. class-wise accuracy
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
