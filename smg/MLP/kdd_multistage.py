import pandas as pd
import numpy as np
import time
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# =======================
# Config
# =======================
TRAIN_CSV = "/Users/mac/Desktop/machine learning/KDD/KDDTrain+_5class.csv"
TEST_CSV  = "/Users/mac/Desktop/machine learning//KDD/KDDTest+_5class.csv"

# ⬇️ NEW: point to your FNN best-params JSON (if you have one).
# Expected keys (optional): learning_rate, epochs, batch_size,
# units_0, activation_0, batch_norm_0, dropout_0,
# units_1, activation_1, batch_norm_1, dropout_1
PARAMS_JSON = "/Users/mac/Desktop/machine learning/smg/machine_learning_models/fnn/kdd_best_fnn_hyperparameters.json"

LABEL_COL = "class5"  # NSL-KDD 5-class label column
F_MIN = 0.60

np.random.seed(42)
tf.random.set_seed(42)


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


# =======================
# Load & Preprocess
# =======================
def load_nsl_kdd(train_path, test_path, label_col):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    # drop constant columns (except label)
    def _clean(df):
        nunique = df.nunique()
        drop_cols = [c for c in df.columns if (nunique[c] <= 1 and c != label_col)]
        return df.drop(columns=drop_cols, errors="ignore")

    train = _clean(train)
    test  = _clean(test)

    if label_col not in train.columns or label_col not in test.columns:
        raise ValueError(f"Label column '{label_col}' not present in both train/test")

    y_train_raw = train[label_col].astype(str).values
    y_test_raw  = test[label_col].astype(str).values
    X_train_df  = train.drop(columns=[label_col])
    X_test_df   = test.drop(columns=[label_col])

    # One-hot categorical features
    cat_cols = X_train_df.select_dtypes(include=["object"]).columns.tolist()
    X_train_df = pd.get_dummies(X_train_df, columns=cat_cols, drop_first=False)
    X_test_df  = pd.get_dummies(X_test_df, columns=cat_cols, drop_first=False)

    # Align columns
    X_train_df, X_test_df = X_train_df.align(X_test_df, join="left", axis=1, fill_value=0)

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test  = scaler.transform(X_test_df)

    # Stable class ordering
    class_names = np.array(sorted(np.unique(np.concatenate([y_train_raw, y_test_raw]))))
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    y_train = np.array([cls_to_idx[c] for c in y_train_raw])
    y_test  = np.array([cls_to_idx[c] for c in y_test_raw])

    return X_train, X_test, y_train, y_test, class_names


X_train, X_test, y_train, y_test, class_names = load_nsl_kdd(TRAIN_CSV, TEST_CSV, LABEL_COL)
print("Classes:", class_names.tolist())


# =======================
# Load tuned FNN params (kept function name to preserve structure)
# =======================
def load_best_dt_params(json_path):
    # Defaults = your Hyperband "best"
    params = {
        "learning_rate": 0.00030418,
        "epochs": 20,
        "batch_size": 32,
        "units_0": 224,
        "activation_0": "tanh",
        "batch_norm_0": True,
        "dropout_0": 0.0,
        "units_1": 480,
        "activation_1": "relu",
        "batch_norm_1": True,
        "dropout_1": 0.10
    }
    try:
        with open(json_path, "r") as f:
            best = json.load(f)
        # accept either flat dict or {"best_parameters": {...}}
        if "best_parameters" in best:
            best = best["best_parameters"]
        for k in params.keys():
            if k in best:
                params[k] = best[k]
        print(f"🔧 Using FNN params from {Path(json_path).name}: {params}")
    except Exception as e:
        print(f"ℹ️ Could not load tuned FNN params ({e}); using defaults {params}")
    return params


fnn_params = load_best_dt_params(PARAMS_JSON)


# =======================
# FNN builder + train/predict utility
# =======================
def build_fnn(input_dim: int, num_classes: int, hp: dict) -> tf.keras.Model:
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    # Layer 1
    model.add(Dense(int(hp["units_0"]), activation=hp["activation_0"], kernel_initializer="he_normal"))
    if hp.get("batch_norm_0", True):
        model.add(BatchNormalization())
    if hp.get("dropout_0", 0.0) and hp["dropout_0"] > 0:
        model.add(Dropout(hp["dropout_0"]))

    # Layer 2
    model.add(Dense(int(hp["units_1"]), activation=hp["activation_1"], kernel_initializer="he_normal"))
    if hp.get("batch_norm_1", True):
        model.add(BatchNormalization())
    if hp.get("dropout_1", 0.0) and hp["dropout_1"] > 0:
        model.add(Dropout(hp["dropout_1"]))

    # Output
    model.add(Dense(num_classes, activation="softmax"))

    opt = Adam(learning_rate=float(hp["learning_rate"]))
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def fit_predict_fnn(X_tr, y_tr_int, X_te, num_classes: int, hp: dict):
    """Fit an FNN on (X_tr, y_tr_int) and predict classes on X_te. Returns (y_pred_int, train_time, test_time)."""
    y_tr_cat = to_categorical(y_tr_int, num_classes=num_classes)

    # Avoid validation issues with tiny sets in later stages
    val_split = 0.15 if X_tr.shape[0] >= 200 else 0.0

    model = build_fnn(X_tr.shape[1], num_classes, hp)

    t0 = time.time()
    model.fit(
        X_tr, y_tr_cat,
        epochs=int(hp["epochs"]),
        batch_size=int(hp["batch_size"]),
        validation_split=val_split,
        verbose=0
    )
    t1 = time.time()
    y_prob = model.predict(X_te, verbose=0)
    t2 = time.time()

    y_pred = np.argmax(y_prob, axis=1)
    return y_pred, (t1 - t0), (t2 - t1)


# =======================
# Metrics helpers (percent)
# =======================
def evaluate_multiclass(y_true, y_pred, class_names, f_min=F_MIN):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    print("\n=== Multi-class Evaluation ===")
    metrics = []
    for i, cls in enumerate(class_names):
        TP = cm[i, i]; FP = cm[:, i].sum() - TP; FN = cm[i, :].sum() - TP
        TN = total - (TP + FP + FN)
        acc_cls = (TP + TN) / total if total else 0
        prec = TP / (TP + FP) if (TP + FP) else 0
        rec  = TP / (TP + FN) if (TP + FN) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        support = int(cm[i, :].sum())
        print(f"{cls:>7}: Acc={pct(acc_cls)}  P={pct(prec)}  R={pct(rec)}  F1={pct(f1)}  Support={support}")
        metrics.append((cls, acc_cls, prec, rec, f1, support))
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {pct(overall_acc)}")
    passed = all(f1 >= f_min for (_, _, _, _, f1, _) in metrics)
    return overall_acc, metrics, passed, cm


def evaluate_binary(y_true, y_pred, f_min=F_MIN, labels=("Pos", "Neg")):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = cm.sum()
    print("\n=== Binary Evaluation ===")
    metrics = []
    for i, lab in enumerate(labels):
        TP = cm[i, i]; FP = cm[:, i].sum() - TP; FN = cm[i, :].sum() - TP
        TN = total - (TP + FP + FN)
        acc_cls = (TP + TN) / total if total else 0
        prec = TP / (TP + FP) if (TP + FP) else 0
        rec  = TP / (TP + FN) if (TP + FN) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        support = int(cm[i, :].sum())
        print(f"{lab:>7}: Acc={pct(acc_cls)}  P={pct(prec)}  R={pct(rec)}  F1={pct(f1)}  Support={support}")
        metrics.append((lab, acc_cls, prec, rec, f1, support))
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {pct(overall_acc)}")
    passed = all(f1 >= f_min for (_, _, _, _, f1, _) in metrics)
    return overall_acc, metrics, passed, cm


def compute_cmr(y_labels):
    unique, counts = np.unique(y_labels, return_counts=True)
    avg = np.mean(counts)
    return {int(cls): float(cnt) / avg for cls, cnt in zip(unique, counts)}


def normal_index(class_names):
    for i, c in enumerate(class_names):
        if c.lower() == "normal":
            return i
    raise ValueError("No 'normal' class found; check labels")


# =======================
# Multi-Stage Pipeline (paper: highest CMR) using FNN
# =======================
total_train_time = 0.0
total_test_time  = 0.0

# Stage 1: multiclass on all classes
print("\n=== Stage 1: FNN Multi-class (all classes) ===")
y_pred_multi, stage_train, stage_test = fit_predict_fnn(
    X_train, y_train, X_test, num_classes=len(class_names), hp=fnn_params
)
total_train_time += stage_train; total_test_time += stage_test
print(f"⏱ Stage 1 Train: {stage_train:.3f}s  Test: {stage_test:.4f}s")
acc1, metrics1, passed1, cm1 = evaluate_multiclass(y_test, y_pred_multi, class_names, f_min=F_MIN)

if passed1:
    print("✅ Stage 1 passed. Stopping.")
else:
    print("❌ Stage 1 failed → Stage 2.")

    # Stage 2: Normal vs Attack binary
    n_idx = normal_index(class_names)
    y_train_bin = (y_train != n_idx).astype(int)
    y_test_bin  = (y_test  != n_idx).astype(int)

    print("\n=== Stage 2: FNN Binary (Normal vs Attack) ===")
    # build 2-class data, same X
    y_pred_bin, stage_train, stage_test = fit_predict_fnn(
        X_train, y_train_bin, X_test, num_classes=2, hp=fnn_params
    )
    total_train_time += stage_train; total_test_time += stage_test
    print(f"⏱ Stage 2 Train: {stage_train:.3f}s  Test: {stage_test:.4f}s")
    acc2, metrics2, passed2, cm2 = evaluate_binary(y_test_bin, y_pred_bin, f_min=F_MIN, labels=("Normal", "Attack"))

    if not passed2:
        print("❌ Stage 2 failed. Cannot proceed.")
    else:
        print("✅ Stage 2 passed → Stage 3.")

        # Stage 3: multiclass only among attacks
        attack_mask_train = (y_train != n_idx)
        attack_mask_test  = (y_test  != n_idx)
        X_attack_train = X_train[attack_mask_train]
        X_attack_test  = X_test[attack_mask_test]
        y_attack_train = y_train[attack_mask_train]
        y_attack_test  = y_test[attack_mask_test]

        attack_class_ids   = [i for i, c in enumerate(class_names) if i != n_idx]
        attack_class_names = class_names[attack_class_ids]

        # remap attack labels old->0..A-1
        remap = {old: new for new, old in enumerate(attack_class_ids)}
        y_attack_train_m = np.array([remap[int(i)] for i in y_attack_train])
        y_attack_test_m  = np.array([remap[int(i)] for i in y_attack_test])

        print("\n=== Stage 3: FNN Multi-class (attacks only) ===")
        y_pred_attack, stage_train, stage_test = fit_predict_fnn(
            X_attack_train, y_attack_train_m, X_attack_test, num_classes=len(attack_class_names), hp=fnn_params
        )
        total_train_time += stage_train; total_test_time += stage_test
        print(f"⏱ Stage 3 Train: {stage_train:.3f}s  Test: {stage_test:.4f}s")
        acc3, metrics3, passed3, cm3 = evaluate_multiclass(y_attack_test_m, y_pred_attack, attack_class_names, f_min=F_MIN)

        if passed3:
            print("✅ Stage 3 passed. Stopping.")
        else:
            print("❌ Stage 3 failed → Stage 4 recursion (paper: highest CMR first).")

            # Stage 4 recursive splitting: TRAIN on attack-train, EVAL on attack-test
            X_curr_train = X_attack_train.copy()
            y_curr_train = y_attack_train_m.copy()
            X_curr_test  = X_attack_test.copy()
            y_curr_test  = y_attack_test_m.copy()
            curr_names   = list(attack_class_names)

            while len(np.unique(y_curr_train)) > 2 and len(y_curr_test) > 0:
                cmr = compute_cmr(y_curr_train)   # local_id -> ratio
                # pick highest CMR class (majority class) to split off first
                pick_local = max(cmr, key=cmr.get)
                pick_name  = curr_names[pick_local]
                print(f"\n[Stage 4] Picked class with HIGHEST CMR: {pick_name} (CMR={cmr[pick_local]:.3f})")

                # Make binary labels such that 0 = picked class, 1 = others
                y_bin_train = (y_curr_train != pick_local).astype(int)
                y_bin_test  = (y_curr_test  != pick_local).astype(int)

                print("\n=== Stage 4: FNN Binary (picked vs others) ===")
                y_pred_b, stage_train, stage_test = fit_predict_fnn(
                    X_curr_train, y_bin_train, X_curr_test, num_classes=2, hp=fnn_params
                )
                total_train_time += stage_train; total_test_time += stage_test
                print(f"⏱ Stage 4 Train: {stage_train:.3f}s  Test: {stage_test:.4f}s")

                # Evaluate: labels tuple = (pick_name, "Others") corresponds to rows [0,1]
                acc_b, metrics_b, passed_b, cm_b = evaluate_binary(
                    y_bin_test, y_pred_b, f_min=F_MIN, labels=(pick_name, "Others")
                )

                if passed_b:
                    # remove picked class from BOTH train & test, and reindex labels
                    keep_train = (y_curr_train != pick_local)
                    keep_test  = (y_curr_test  != pick_local)

                    X_curr_train = X_curr_train[keep_train]
                    y_curr_train = y_curr_train[keep_train]
                    X_curr_test  = X_curr_test[keep_test]
                    y_curr_test  = y_curr_test[keep_test]

                    # remove name
                    curr_names.pop(pick_local)

                    # reindex remaining labels to 0..k-1 consistently
                    uniq = np.unique(y_curr_train)
                    remap2 = {old: i for i, old in enumerate(uniq)}
                    y_curr_train = np.array([remap2[int(v)] for v in y_curr_train])
                    if y_curr_test.size > 0:
                        y_curr_test = np.array([remap2[int(v)] for v in y_curr_test])
                else:
                    print("Binary recursion did not meet F_MIN on held-out test; stopping recursion.")
                    break

            # ---------- Final binary step with real class names ----------
            if len(np.unique(y_curr_train)) == 2 and len(y_curr_test) > 0:
                print("\n[Stage 4] Final binary split on the last two attack classes (evaluated on TEST).")
                uniq_local = np.unique(y_curr_train)
                local_a, local_b = int(uniq_local[0]), int(uniq_local[1])
                name_a = curr_names[local_a]
                name_b = curr_names[local_b]
                print(f"Remaining classes (local idx -> name): {local_a} -> {name_a}, {local_b} -> {name_b}")

                # Map labels so that 0 corresponds to name_a, 1 corresponds to name_b
                y_final_train = (y_curr_train != local_a).astype(int)  # 0 if name_a, 1 if name_b
                y_final_test  = (y_curr_test  != local_a).astype(int)

                print("\n=== Stage 4 Final: FNN Binary (last two attacks) ===")
                y_pred_final, stage_train, stage_test = fit_predict_fnn(
                    X_curr_train, y_final_train, X_curr_test, num_classes=2, hp=fnn_params
                )
                total_train_time += stage_train; total_test_time += stage_test
                print(f"⏱ Stage 4 Final Train: {stage_train:.3f}s  Test: {stage_test:.4f}s")
                _ = evaluate_binary(y_final_test, y_pred_final, f_min=F_MIN, labels=(name_a, name_b))

# Runtime summary
print("\n" + "=" * 60)
print("⏱ TOTAL RUNTIME SUMMARY (final fit/predict only)")
print("=" * 60)
print(f"Total Training Time: {total_train_time:.3f} s")
print(f"Total Testing Time:  {total_test_time:.4f} s")
