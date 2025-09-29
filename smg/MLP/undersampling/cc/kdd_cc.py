# kdd_cc_fast.py
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
from collections import Counter

# File paths (adjust if needed)
TRAIN_PATH = "/Users/mac/Desktop/machine learning/KDD/KDDTrain+_5class.csv"
TEST_PATH  = "/Users/mac/Desktop/machine learning//KDD/KDDTest+_5class.csv"

print("=" * 70)
print("KDD MODEL WITH FAST CLUSTERCENTROIDS (MiniBatchKMeans) PREPROCESSING")
print("=" * 70)

# Load datasets
print("Loading KDD 5-class datasets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"Train data shape: {train_df.shape}")
print(f"Test  data shape: {test_df.shape}")

# target
target_col = train_df.columns[-1]
print(f"Target column: {target_col}")

def preprocess_kdd_5class(train_data, test_data, target_col):
    print("\nPreprocessing data...")
    X_train = train_data.drop(columns=[target_col]).copy()
    y_train = train_data[target_col].copy()
    X_test  = test_data.drop(columns=[target_col]).copy()
    y_test  = test_data[target_col].copy()

    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        print(f"Encoding categorical columns: {categorical_columns}")
        for col in categorical_columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col]  = le.transform(X_test[col].astype(str))

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc  = label_encoder.transform(y_test)

    num_classes = len(label_encoder.classes_)
    y_train_cat = tf.keras.utils.to_categorical(y_train_enc, num_classes)
    y_test_cat  = tf.keras.utils.to_categorical(y_test_enc, num_classes)

    print(f"Classes: {label_encoder.classes_}")
    return X_train_scaled, X_test_scaled, y_train_cat, y_test_cat, y_train_enc, y_test_enc, label_encoder.classes_

# Preprocess
X_train, X_test, y_train_cat, y_test_cat, y_train_enc, y_test_enc, class_names = preprocess_kdd_5class(
    train_df, test_df, target_col
)

# ----------------------------
# FAST ClusterCentroids undersampling
# ----------------------------
print("\n" + "="*60)
print("APPLYING FAST ClusterCentroids (MiniBatchKMeans) TO TRAINING SET")
print("="*60)
try:
    from imblearn.under_sampling import ClusterCentroids
    from sklearn.cluster import MiniBatchKMeans
    import math

    # Controls (tune these)
    CC_FRACTION = 0.25      # target fraction of majority count (before cap)
    CC_MAX_CLUSTERS = 10000 # absolute cap for clusters (to avoid enormous KMeans jobs)
    MBATCH_BATCH_SIZE = 2048
    MBATCH_MAX_ITER = 200

    orig_counts = Counter(y_train_enc)
    majority_label, majority_count = max(orig_counts.items(), key=lambda kv: kv[1])

    print(f"Original training class distribution (majority={class_names[majority_label]}: {majority_count}):")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {orig_counts.get(idx, 0)}")

    # compute desired cluster target
    proposed = int(CC_FRACTION * majority_count)
    max_minority = max(cnt for cls, cnt in orig_counts.items() if cls != majority_label)

    # final target: don't drop below largest minority, cap by CC_MAX_CLUSTERS
    cc_target_for_majority = max(
        max_minority,
        min(proposed, CC_MAX_CLUSTERS)
    )

    # Safety: we must also ensure cc_target_for_majority <= number of unique majority rows
    # compute unique rows count for majority class (round to reduce floating uniqueness noise)
    maj_mask = (y_train_enc == majority_label)
    X_maj = X_train[maj_mask]
    try:
        # round to 6 decimals to avoid trivial floating-point differences
        unique_rows = np.unique(np.round(X_maj, 6), axis=0)
        unique_count = unique_rows.shape[0]
    except Exception:
        unique_count = X_maj.shape[0]
    if unique_count < cc_target_for_majority:
        print(f"Note: unique majority rows ({unique_count}) < target clusters ({cc_target_for_majority}).")
        cc_target_for_majority = unique_count
        print(f"Adjusted cc_target_for_majority -> {cc_target_for_majority}")

    # sampling target: majority reduced, minorities unchanged
    sampling_target = {cls_idx: (cc_target_for_majority if cls_idx == majority_label else cnt)
                       for cls_idx, cnt in orig_counts.items()}

    print("\nClusterCentroids sampling targets (per-class):")
    for idx, name in enumerate(class_names):
        print(f"  {name}: target -> {sampling_target.get(idx, 0)}")

    # build a fast MiniBatchKMeans estimator with the appropriate n_clusters when called per-class
    # ClusterCentroids expects an estimator with .n_clusters attribute set; MiniBatchKMeans will be cloned internally.
    # We create one estimator with n_clusters equal to cc_target_for_majority; imblearn will clone/adjust per-class.
    est = MiniBatchKMeans(
        n_clusters=max(2, int(cc_target_for_majority)),  # k must be >=2
        batch_size=MBATCH_BATCH_SIZE,
        max_iter=MBATCH_MAX_ITER,
        random_state=42,
        init='k-means++'
    )

    # Time it
    t0 = time.time()
    cc = ClusterCentroids(sampling_strategy=sampling_target, random_state=42, estimator=est)
    X_train_res, y_train_res = cc.fit_resample(X_train, y_train_enc)
    t1 = time.time()

    print(f"\nClusterCentroids finished in {(t1-t0):.1f} s")

    res_counts = Counter(y_train_res)
    print("\nAfter ClusterCentroids training class distribution:")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {res_counts.get(idx, 0)}")

    # prepare for Keras
    num_classes = len(class_names)
    y_train_res_cat = tf.keras.utils.to_categorical(y_train_res, num_classes)

    X_train_used = X_train_res
    y_train_used_cat = y_train_res_cat
    y_train_used_enc = y_train_res

    print("✅ Fast ClusterCentroids applied successfully.")
except Exception as e:
    print("⚠️ ClusterCentroids failed or imbalanced-learn not installed:", e)
    print("Falling back to RandomUnderSampler for a quick, lightweight undersample.")
    try:
        from imblearn.under_sampling import RandomUnderSampler
        rus_fraction = 0.5
        # compute rus target similarly (keep at least max_minority)
        rus_target = max_minority if 'max_minority' in locals() else None
        if rus_target is None:
            orig_counts = Counter(y_train_enc)
            majority_label, majority_count = max(orig_counts.items(), key=lambda kv: kv[1])
            rus_target = int(rus_fraction * majority_count)
        sampling_target = {cls_idx: (rus_target if cls_idx == majority_label else cnt)
                           for cls_idx, cnt in orig_counts.items()}
        rus = RandomUnderSampler(sampling_strategy=sampling_target, random_state=42)
        X_train_res, y_train_res = rus.fit_resample(X_train, y_train_enc)
        res_counts = Counter(y_train_res)
        print("\nAfter RandomUnderSampler (fallback) class distribution:")
        for idx, name in enumerate(class_names):
            print(f"  {name}: {res_counts.get(idx, 0)}")
        y_train_res_cat = tf.keras.utils.to_categorical(y_train_res, len(class_names))
        X_train_used = X_train_res
        y_train_used_cat = y_train_res_cat
        y_train_used_enc = y_train_res
        print("✅ RandomUnderSampler fallback applied.")
    except Exception as e2:
        print("⚠️ RandomUnderSampler also failed:", e2)
        print("Using original training data (no resampling).")
        X_train_used = X_train
        y_train_used_cat = y_train_cat
        y_train_used_enc = y_train_enc

# ----------------------------
# Model / training / evaluation (keeps your original NN & flow)
# ----------------------------
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

def create_optimal_model(input_dim, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(224, activation='tanh', kernel_initializer='he_normal'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(480, activation='relu', kernel_initializer='he_normal'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_hyperparams["learning_rate"])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n" + "="*50)
print("BUILDING AND TRAINING MODEL")
print("="*50)
model = create_optimal_model(X_train_used.shape[1], len(class_names))
model.summary()

start_train = time.time()
history = model.fit(
    X_train_used, y_train_used_cat,
    epochs=20,
    batch_size=32,
    validation_split=0.15,
    verbose=1
)
end_train = time.time()

# quick evaluation (same as your pipeline)
start_test = time.time()
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
end_test = time.time()

acc = accuracy_score(y_test_enc, y_pred)
f1 = f1_score(y_test_enc, y_pred, average='macro', zero_division=0)
print(f"\nTraining time: {end_train-start_train:.2f}s  Test time: {end_test-start_test:.2f}s")
print(f"Accuracy: {acc*100:.2f}%  F1-macro: {f1*100:.2f}%")

# save results summary
results_summary = {
    "timestamp": datetime.now().isoformat(),
    "resampler": "ClusterCentroids (fast)" if 'X_train_res' in locals() else "fallback",
    "cc_fraction": CC_FRACTION if 'CC_FRACTION' in locals() else None,
    "cc_max_clusters": CC_MAX_CLUSTERS if 'CC_MAX_CLUSTERS' in locals() else None,
    "training_time_seconds": float(end_train-start_train),
    "testing_time_seconds": float(end_test-start_test),
    "accuracy": float(acc),
    "f1_macro": float(f1)
}
with open("kdd_cc_fast_results.json", "w") as fh:
    json.dump(results_summary, fh, indent=2)

print("Saved results -> kdd_cc_fast_results.json")
