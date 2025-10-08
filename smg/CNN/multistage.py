import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
import time

# -----------------------
# 1. Load & Preprocess Data
# -----------------------
df = pd.read_csv("../../EPIC/dataset_EPICA_raw 1.csv")


def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    features = data.drop(columns=['status'])
    labels = data['status']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels


X, y = preprocess_epica(df)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
y_categorical = pd.get_dummies(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# -----------------------
# 2. Parameters
# -----------------------
F_MIN = 0.70
EPOCHS = 30
BATCH_SIZE = 128
VAL_SPLIT = 0.1
VERBOSE = 0

# Timers
total_train_time = 0.0
total_test_time = 0.0


# -----------------------
# 3. CNN Model Builder
# -----------------------
def build_cnn(input_dim, output_dim, binary=False):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),

        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='sigmoid' if binary else 'softmax')
    ])
    loss = 'binary_crossentropy' if binary else 'categorical_crossentropy'
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
    return model


# -----------------------
# 4. Evaluation Functions
# -----------------------
def evaluate_multiclass(model, X_test, y_test, class_names):
    global total_test_time
    y_true = np.argmax(y_test.values, axis=1)

    start_test = time.time()
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    end_test = time.time()
    total_test_time += (end_test - start_test)

    cm = confusion_matrix(y_true, y_pred)
    total = np.sum(cm)

    print("\n=== Multi-class Evaluation ===")
    metrics = []
    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = total - (TP + FP + FN)
        acc_cls = (TP + TN) / total
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        metrics.append((cls, acc_cls, prec, rec, f1))
        print(
            f"{cls}: Accuracy={acc_cls:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Support={np.sum(cm[i, :])}")

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")

    passed = all(f1 >= F_MIN for (_, _, _, _, f1) in metrics)
    return overall_acc, metrics, passed


def evaluate_binary(model, X_test, y_test):
    global total_test_time
    y_true = y_test.flatten()

    start_test = time.time()
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    end_test = time.time()
    total_test_time += (end_test - start_test)

    cm = confusion_matrix(y_true, y_pred)
    total = np.sum(cm)
    labels = ['Normal', 'Attack']

    print("\n=== Binary Evaluation ===")
    metrics = []
    for i, cls in enumerate(labels):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = total - (TP + FP + FN)
        acc_cls = (TP + TN) / total
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        metrics.append((cls, acc_cls, prec, rec, f1))
        print(
            f"{cls}: Accuracy={acc_cls:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Support={np.sum(cm[i, :])}")

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")

    passed = all(f1 >= F_MIN for (_, _, _, _, f1) in metrics)
    return overall_acc, metrics, passed


def compute_cmr(y_labels):
    unique, counts = np.unique(y_labels, return_counts=True)
    avg = np.mean(counts)
    return {cls: cnt / avg for cls, cnt in zip(unique, counts)}


# -----------------------
# 5. Stage 1: Multi-class
# -----------------------
print("\n=== Stage 1: CNN Multi-class ===")
model_multi = build_cnn(X.shape[1], y_train.shape[1])

start_train = time.time()
model_multi.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VAL_SPLIT, verbose=VERBOSE)
end_train = time.time()
total_train_time += (end_train - start_train)
print(f"Stage 1 Training Time: {end_train - start_train:.2f} seconds")

acc, metrics, passed = evaluate_multiclass(model_multi, X_test, y_test, class_names)

if passed:
    print("✅ Stage 1 passed.")
else:
    print("❌ Stage 1 failed. Proceeding to Stage 2.")

    # -----------------------
    # 6. Stage 2: Binary Normal vs Attack
    # -----------------------
    normal_class = list(class_names).index('Normal')
    y_train_bin = (np.argmax(y_train.values, axis=1) != normal_class).astype(int)
    y_test_bin = (np.argmax(y_test.values, axis=1) != normal_class).astype(int)

    model_bin = build_cnn(X.shape[1], 1, binary=True)

    start_train = time.time()
    model_bin.fit(X_train, y_train_bin, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VAL_SPLIT, verbose=VERBOSE)
    end_train = time.time()
    total_train_time += (end_train - start_train)
    print(f"Stage 2 Training Time: {end_train - start_train:.2f} seconds")

    acc_bin, metrics_bin, passed_bin = evaluate_binary(model_bin, X_test, y_test_bin)

    if passed_bin:
        print("✅ Stage 2 passed. Proceeding to Stage 3.")

        # -----------------------
        # 7. Stage 3: Attack-only Multi-class
        # -----------------------
        attack_mask_train = (np.argmax(y_train.values, axis=1) != normal_class)
        attack_mask_test = (np.argmax(y_test.values, axis=1) != normal_class)
        X_attack_train = X_train[attack_mask_train]
        X_attack_test = X_test[attack_mask_test]
        y_attack_train = y_train[attack_mask_train].drop(columns=['Normal'], errors='ignore')
        y_attack_test = y_test[attack_mask_test].drop(columns=['Normal'], errors='ignore')
        attack_classes = [c for c in class_names if c != 'Normal']

        print("\n=== Stage 3: CNN Multi-class on Attacks ===")
        model_attack = build_cnn(X_attack_train.shape[1], y_attack_train.shape[1])

        start_train = time.time()
        model_attack.fit(X_attack_train, y_attack_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         validation_split=VAL_SPLIT, verbose=VERBOSE)
        end_train = time.time()
        total_train_time += (end_train - start_train)
        print(f"Stage 3 Training Time: {end_train - start_train:.2f} seconds")

        acc_att, metrics_att, passed_att = evaluate_multiclass(model_attack, X_attack_test, y_attack_test,
                                                               attack_classes)

        if passed_att:
            print("✅ Stage 3 passed.")
        else:
            print("❌ Stage 3 failed. Proceeding to Stage 4.")

            # -----------------------
            # 8. Stage 4: Recursive CMR-based Splitting
            # -----------------------
            y_attack_train_idx = np.argmax(y_attack_train.values, axis=1)
            while len(np.unique(y_attack_train_idx)) > 2:
                cmr_values = compute_cmr(y_attack_train_idx)
                worst_class_idx = min(cmr_values, key=cmr_values.get)
                worst_class = attack_classes[worst_class_idx]
                print(f"\nRecursive step: Class with lowest CMR = {worst_class} ({cmr_values[worst_class_idx]:.3f})")

                y_binary = (y_attack_train_idx == worst_class_idx).astype(int)
                model_bin_rec = build_cnn(X_attack_train.shape[1], 1, binary=True)

                start_train = time.time()
                model_bin_rec.fit(X_attack_train, y_binary, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                  validation_split=VAL_SPLIT, verbose=VERBOSE)
                end_train = time.time()
                total_train_time += (end_train - start_train)
                print(f"Recursive Training Time: {end_train - start_train:.2f} seconds")

                acc_b, metrics_b, passed_b = evaluate_binary(model_bin_rec, X_attack_train, y_binary)

                if passed_b:
                    print(f"Class {worst_class} separated successfully. Removing and continuing.")
                    keep_mask = y_attack_train_idx != worst_class_idx
                    X_attack_train = X_attack_train[keep_mask]
                    y_attack_train_idx = y_attack_train_idx[keep_mask]
                    attack_classes = [c for i, c in enumerate(attack_classes) if i != worst_class_idx]
                else:
                    print("Binary recursion failed to meet F_min. Stopping recursion.")
                    break

            if len(np.unique(y_attack_train_idx)) == 2:
                print("\nFinal binary classification for last two attack classes...")
                y_binary_final = (y_attack_train_idx == np.unique(y_attack_train_idx)[0]).astype(int)
                model_final = build_cnn(X_attack_train.shape[1], 1, binary=True)

                start_train = time.time()
                model_final.fit(X_attack_train, y_binary_final, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_split=VAL_SPLIT, verbose=VERBOSE)
                end_train = time.time()
                total_train_time += (end_train - start_train)
                print(f"Final Binary Training Time: {end_train - start_train:.2f} seconds")

                evaluate_binary(model_final, X_attack_train, y_binary_final)

    else:
        print("❌ Stage 2 failed. Cannot proceed further.")

# -----------------------
# Final Timings
# -----------------------
print("\n=== Total Timings ===")
print(f"Total Training Time: {total_train_time:.2f} seconds")
print(f"Total Testing Time:  {total_test_time:.2f} seconds")
