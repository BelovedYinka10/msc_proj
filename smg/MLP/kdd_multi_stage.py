# kdd_multi_stage_print_labels_fix.py
"""
Multi-stage DNN pipeline for KDD 5-class (prints only).
This version forces meaningful label names everywhere and prints class counts
before/after stages and at each recursion step.
"""

import time
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# -----------------------
# CONFIG
# -----------------------
TRAIN_PATH = "/Users/mac/Desktop/machine learning/KDD/KDDTrain+_5class.csv"
TEST_PATH  = "/Users/mac/Desktop/machine learning/KDD/KDDTest+_5class.csv"

F_MIN = 0.50
EPOCHS = 20
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.12
VERBOSE = 1

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

LEAK_COLS = ['label', 'attack_cat', 'class', 'binary', 'difficulty']

# -----------------------
# UTILITIES
# -----------------------
def pct(x): return f"{x*100:.2f}%"
def counts_to_str(cnt):
    return ", ".join([f"{k}:{v}" for k,v in cnt.items()])

def print_counts(title, labels, class_names):
    cnt = Counter(labels)
    print(f"\n{title} counts:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {cnt.get(i,0)}")
    return cnt

def build_model(input_dim, output_dim, binary=False, params=best_hyperparams):
    u0 = params["units_0"]; u1 = params["units_1"]
    act0 = params["activation_0"]; act1 = params["activation_1"]
    dr0 = params["dropout_0"]; dr1 = params["dropout_1"]
    bn0 = params["batch_norm_0"]; bn1 = params["batch_norm_1"]
    lr = params["learning_rate"]

    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(u0, activation=act0, kernel_initializer=params["kernel_init"]))
    if bn0: model.add(BatchNormalization())
    if dr0 and dr0 > 0: model.add(Dropout(dr0))

    model.add(Dense(u1, activation=act1, kernel_initializer=params["kernel_init"]))
    if bn1: model.add(BatchNormalization())
    if dr1 and dr1 > 0: model.add(Dropout(dr1))

    if binary:
        model.add(Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(Dense(output_dim, activation="softmax"))
        loss = "categorical_crossentropy"

    model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=["accuracy"])
    return model

def evaluate_multiclass(model, X, y_cat, classes):
    y_true = np.argmax(y_cat, axis=1)
    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    total = cm.sum()
    print("\n=== Multi-class Evaluation ===")
    per = []
    for i, cname in enumerate(classes):
        TP = cm[i,i]
        FP = cm[:,i].sum()-TP
        FN = cm[i,:].sum()-TP
        TN = total - (TP+FP+FN)
        acc_ovr = (TP+TN)/total if total else 0.0
        prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
        rec = TP/(TP+FN) if (TP+FN)>0 else 0.0
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
        support = int(cm[i,:].sum())
        print(f"{cname:10s}: Acc_ovr={pct(acc_ovr)}  P={pct(prec)}  R={pct(rec)}  F1={pct(f1)}  Support={support}")
        per.append((cname, acc_ovr, prec, rec, f1, support))
    overall = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {pct(overall)}")
    passed = all(m[4] >= F_MIN for m in per)
    return overall, per, passed

def evaluate_binary(model, X, y_bin, label_names):
    if not label_names or len(label_names)!=2:
        raise ValueError("Binary evaluation requires label_names list of length 2.")
    y_true = np.asarray(y_bin).astype(int).reshape(-1)
    y_prob = model.predict(X, verbose=0).reshape(-1)
    y_pred = (y_prob > 0.5).astype(int)
    # ensure order: [neg, pos] => labels [0,1]
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    total = cm.sum() if cm.size else 0
    print("\n=== Binary Evaluation ===")
    per = []
    for i, lname in enumerate(label_names):
        TP = cm[i,i] if cm.shape[0]>i and cm.shape[1]>i else 0
        FP = cm[:,i].sum()-TP if cm.shape[1]>i else 0
        FN = cm[i,:].sum()-TP if cm.shape[0]>i else 0
        TN = total - (TP+FP+FN)
        acc_ovr = (TP+TN)/total if total else 0.0
        prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
        rec = TP/(TP+FN) if (TP+FN)>0 else 0.0
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
        support = int(cm[i,:].sum()) if cm.shape[0]>i else 0
        print(f"{lname:30s}: Acc_ovr={pct(acc_ovr)}  P={pct(prec)}  R={pct(rec)}  F1={pct(f1)}  Support={support}")
        per.append((lname, acc_ovr, prec, rec, f1, support))
    overall = accuracy_score(y_true, y_pred) if total else 0.0
    print(f"\nOverall Accuracy: {pct(overall)}")
    passed = all(m[4] >= F_MIN for m in per)
    return overall, per, passed

def compute_cmr(labels_1d):
    unique, counts = np.unique(labels_1d, return_counts=True)
    avg = np.mean(counts) if len(counts)>0 else 1.0
    return {int(u): float(c)/avg for u,c in zip(unique,counts)}

# -----------------------
# LOAD + PREPROCESS
# -----------------------
print("Loading CSVs...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

target_col = train_df.columns[-1]
print("Detected target column:", target_col)

# drop leak columns (if present)
drop_cols = [c for c in LEAK_COLS if c in train_df.columns and c != target_col]
if drop_cols:
    print("Dropping columns:", drop_cols)
    train_df = train_df.drop(columns=drop_cols, errors='ignore')
    test_df  = test_df.drop(columns=drop_cols, errors='ignore')

X_train_df = train_df.drop(columns=[target_col]).copy()
y_train_ser = train_df[target_col].astype(str).str.strip().copy()
X_test_df  = test_df.drop(columns=[target_col]).copy()
y_test_ser  = test_df[target_col].astype(str).str.strip().copy()

# encode categorical features
cat_cols = X_train_df.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    print("Encoding categorical columns:", cat_cols)
    for c in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train_df[c].astype(str), X_test_df[c].astype(str)], axis=0)
        le.fit(combined)
        X_train_df[c] = le.transform(X_train_df[c].astype(str))
        X_test_df[c]  = le.transform(X_test_df[c].astype(str))

# coerce to numeric and drop rows with NaNs
for df_ in (X_train_df, X_test_df):
    for col in X_train_df.columns:
        df_[col] = pd.to_numeric(df_[col], errors='coerce')

train_nan_idx = X_train_df[X_train_df.isna().any(axis=1)].index
if len(train_nan_idx)>0:
    print("Dropping train rows with NaNs:", len(train_nan_idx))
    X_train_df = X_train_df.drop(index=train_nan_idx); y_train_ser = y_train_ser.drop(index=train_nan_idx)

test_nan_idx = X_test_df[X_test_df.isna().any(axis=1)].index
if len(test_nan_idx)>0:
    print("Dropping test rows with NaNs:", len(test_nan_idx))
    X_test_df = X_test_df.drop(index=test_nan_idx); y_test_ser = y_test_ser.drop(index=test_nan_idx)

# scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_df)
X_test  = scaler.transform(X_test_df)

# target encoding
le_y = LabelEncoder()
le_y.fit(pd.concat([y_train_ser, y_test_ser], axis=0))
y_train_enc = le_y.transform(y_train_ser)
y_test_enc  = le_y.transform(y_test_ser)
class_names = list(le_y.classes_)
print("Classes detected:", class_names)

y_train_cat = to_categorical(y_train_enc, num_classes=len(class_names))
y_test_cat  = to_categorical(y_test_enc,  num_classes=len(class_names))

print_counts("TRAIN initial", y_train_enc, class_names)
print_counts("TEST  initial", y_test_enc, class_names)

# -----------------------
# STAGE 1: Multi-class (all classes)
# -----------------------
print("\n=== Stage 1: Multi-class (all classes) ===")
model_multi = build_model(X_train.shape[1], y_train_cat.shape[1], binary=False)
t0 = time.time()
model_multi.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
t1 = time.time()
print(f"⏱ Stage 1 training time: {t1-t0:.2f}s")

acc1, metrics1, passed1 = evaluate_multiclass(model_multi, X_test, y_test_cat, class_names)
if passed1:
    print("✅ Stage 1 PASSED — pipeline stops here.")
else:
    print("❌ Stage 1 FAILED. Proceeding to Stage 2.")

    # -----------------------
    # STAGE 2: Binary Normal vs Attack (explicit)
    # -----------------------
    # find class named 'normal' (case-insensitive). If none, pick the class with largest count as 'normal'
    normal_candidates = [i for i,c in enumerate(class_names) if c.lower()=='normal']
    if normal_candidates:
        normal_idx = normal_candidates[0]
        normal_name = class_names[normal_idx]
    else:
        cnts = Counter(y_train_enc)
        normal_idx = max(cnts, key=cnts.get)
        normal_name = class_names[normal_idx]
        print("No explicit 'normal' label found; using majority class as 'normal':", normal_name)

    y_train_bin = (y_train_enc != normal_idx).astype(int)
    y_test_bin  = (y_test_enc  != normal_idx).astype(int)
    print_counts("TRAIN before Stage2 (original labels)", y_train_enc, class_names)
    print(f"Binary labels for Stage2: 0 -> {normal_name}, 1 -> Attack (all others)")

    print("\n=== Stage 2: Binary Normal vs Attack ===")
    model_bin = build_model(X_train.shape[1], 1, binary=True)
    t0 = time.time()
    model_bin.fit(X_train, y_train_bin, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
    t1 = time.time()
    print(f"⏱ Stage 2 training time: {t1-t0:.2f}s")
    acc2, metrics2, passed2 = evaluate_binary(model_bin, X_test, y_test_bin, label_names=[normal_name, "Attack (others)"])

    if passed2:
        print("✅ Stage 2 PASSED. Proceeding to Stage 3 (attacks only).")

        # -----------------------
        # STAGE 3: Multi-class on attack samples only
        # -----------------------
        attack_mask_train = (y_train_enc != normal_idx)
        attack_mask_test  = (y_test_enc  != normal_idx)
        X_attack_train = X_train[attack_mask_train]
        X_attack_test  = X_test[attack_mask_test]
        attack_orig_idxs = [i for i in range(len(class_names)) if i != normal_idx]
        attack_class_names = [class_names[i] for i in attack_orig_idxs]
        # map original label -> 0..k-1
        orig2new = {orig: new for new, orig in enumerate(attack_orig_idxs)}
        y_attack_train_new = np.array([orig2new[v] for v in y_train_enc[attack_mask_train]])
        y_attack_test_new  = np.array([orig2new[v] for v in y_test_enc[attack_mask_test]])

        print_counts("ATTACK TRAIN (by name)", y_attack_train_new, attack_class_names)
        print_counts("ATTACK TEST  (by name)", y_attack_test_new, attack_class_names)

        print("\n=== Stage 3: Multi-class (attacks only) ===")
        model_attack = build_model(X_attack_train.shape[1], len(attack_class_names), binary=False)
        t0 = time.time()
        model_attack.fit(X_attack_train, to_categorical(y_attack_train_new, num_classes=len(attack_class_names)),
                         epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
        t1 = time.time()
        print(f"⏱ Stage 3 training time: {t1-t0:.2f}s")
        acc3, metrics3, passed3 = evaluate_multiclass(model_attack, X_attack_test,
                                                      to_categorical(y_attack_test_new), attack_class_names)
        if passed3:
            print("✅ Stage 3 PASSED — pipeline ends.")
        else:
            print("❌ Stage 3 FAILED. Proceeding to Stage 4 recursion.")

            # -----------------------
            # STAGE 4: Recursion on attack classes
            # -----------------------
            X_rec_train = X_attack_train.copy()
            X_rec_test  = X_attack_test.copy()
            labels_rec_train = y_attack_train_new.copy()
            labels_rec_test  = y_attack_test_new.copy()
            rec_class_names = attack_class_names.copy()

            step = 0
            while len(np.unique(labels_rec_train)) > 2:
                step += 1
                cmr = compute_cmr(labels_rec_train)
                pick = max(cmr, key=cmr.get)  # highest CMR as per paper
                pick_name = rec_class_names[pick]
                remaining_names = [n for i,n in enumerate(rec_class_names) if i != pick]
                neg_label_name = "Others: " + ", ".join(remaining_names)

                print(f"\n[Recursion {step}] Picked '{pick_name}' (CMR={cmr[pick]:.3f})")
                print_counts("Recursion TRAIN (rec indices)", labels_rec_train, rec_class_names)
                print_counts("Recursion TEST  (rec indices)", labels_rec_test, rec_class_names)

                y_bin_train = (labels_rec_train == pick).astype(int)
                y_bin_test  = (labels_rec_test == pick).astype(int)

                model_bin_rec = build_model(X_rec_train.shape[1], 1, binary=True)
                t0 = time.time()
                model_bin_rec.fit(X_rec_train, y_bin_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                  validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
                t1 = time.time()
                print(f"⏱ Recursion {step} training time: {t1-t0:.2f}s")

                # pass explicit label names: [negative, positive]
                acc_rec, metrics_rec, passed_rec = evaluate_binary(model_bin_rec, X_rec_test, y_bin_test,
                                                                   label_names=[neg_label_name, pick_name])
                if passed_rec:
                    print(f"✅ Recursion {step} accepted for '{pick_name}'. Removing that class and continuing.")
                    keep_train = (labels_rec_train != pick)
                    keep_test  = (labels_rec_test != pick)
                    X_rec_train = X_rec_train[keep_train]
                    X_rec_test  = X_rec_test[keep_test]
                    labels_rec_train = labels_rec_train[keep_train]
                    labels_rec_test  = labels_rec_test[keep_test]
                    rec_class_names.pop(pick)
                    unique_vals = sorted(np.unique(labels_rec_train))
                    old_to_new = {old:new for new,old in enumerate(unique_vals)}
                    labels_rec_train = np.array([old_to_new[v] for v in labels_rec_train])
                    labels_rec_test  = np.array([old_to_new[v] for v in labels_rec_test])
                else:
                    print(f"❌ Recursion {step} rejected for '{pick_name}'. Stop recursion.")
                    break

            remaining = np.unique(labels_rec_train)
            if len(remaining) == 2:
                pos_idx = remaining[0]; neg_idx = remaining[1]  # indices in rec_class_names
                pos_name = rec_class_names[pos_idx]; neg_name = rec_class_names[neg_idx]
                print(f"\nFinal binary between '{neg_name}' and '{pos_name}'")
                y_bin_train_final = (labels_rec_train == pos_idx).astype(int)
                y_bin_test_final  = (labels_rec_test  == pos_idx).astype(int)
                model_final = build_model(X_rec_train.shape[1], 1, binary=True)
                t0 = time.time()
                model_final.fit(X_rec_train, y_bin_train_final, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
                t1 = time.time()
                print(f"⏱ Final binary training time: {t1-t0:.2f}s")
                evaluate_binary(model_final, X_rec_test, y_bin_test_final, label_names=[neg_name, pos_name])
            else:
                print("\nRecursion ended before two classes remained.")
    else:
        print("❌ Stage 2 FAILED. Pipeline stops.")

print("\nPIPELINE COMPLETE — printed results only.")
