import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Config
# -----------------------------
F_MIN = 0.81  # your chosen threshold (paper used 0.99)
EPOCHS = 30
BATCH_SIZE = 128
VAL_SPLIT = 0.1
VERBOSE = 0

# -----------------------------
# Data
# -----------------------------
df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")

def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    X = data.drop(columns=['status'])
    y = data['status']
    X = MinMaxScaler().fit_transform(X)
    return X, y

X, y = preprocess_epica(df)
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_
y_oh = pd.get_dummies(y_enc)

X_train, X_test, y_train_oh, y_test_oh = train_test_split(
    X, y_oh, test_size=0.2, random_state=42, stratify=y_oh
)

y_train = np.argmax(y_train_oh.values, axis=1)
y_test  = np.argmax(y_test_oh.values,  axis=1)

# -----------------------------
# Utils
# -----------------------------
def build_dnn(input_dim, output_dim, binary=False):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='sigmoid' if binary else 'softmax')
    ])
    loss = 'binary_crossentropy' if binary else 'categorical_crossentropy'
    model.compile(optimizer=Adam(1e-3), loss=loss, metrics=['accuracy'])
    return model

def multiClassifier(X_tr, y_tr_idx, X_te, y_te_idx, class_names_subset):
    y_tr_oh = pd.get_dummies(y_tr_idx)
    y_te_oh = pd.get_dummies(y_te_idx)

    # Align columns (when some classes disappear in subsets)
    y_tr_oh = y_tr_oh.reindex(columns=sorted(y_tr_oh.columns), fill_value=0)
    y_te_oh = y_te_oh.reindex(columns=y_tr_oh.columns, fill_value=0)

    model = build_dnn(X_tr.shape[1], y_tr_oh.shape[1], binary=False)
    model.fit(X_tr, y_tr_oh, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_split=VAL_SPLIT, verbose=VERBOSE)

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    f1s = f1_score(y_te_idx, y_pred, average=None, zero_division=1)
    acc = accuracy_score(y_te_idx, y_pred)
    # Map indices back to names given the subset
    report = classification_report(
        y_te_idx, y_pred,
        target_names=[class_names_subset[i] for i in sorted(np.unique(y_tr_idx))],
        zero_division=1
    )
    return model, f1s, acc, report

def binaryClassifier(X_tr, y_tr_bin, X_te, y_te_bin, labels=('Normal','Attack')):
    model = build_dnn(X_tr.shape[1], 1, binary=True)
    model.fit(X_tr, y_tr_bin, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_split=VAL_SPLIT, verbose=VERBOSE)

    y_pred = (model.predict(X_te, verbose=0) > 0.5).astype(int).flatten()
    f1s = f1_score(y_te_bin, y_pred, average=None, zero_division=1)
    acc = accuracy_score(y_te_bin, y_pred)
    report = classification_report(y_te_bin, y_pred, target_names=labels, zero_division=1)
    return model, f1s, acc, report

def compute_cmr(y_idx):
    _, counts = np.unique(y_idx, return_counts=True)
    avg = counts.mean()
    cmr = counts / avg
    # Return a dict class_index -> cmr
    cls = np.unique(y_idx)
    return dict(zip(cls, cmr))

# -----------------------------
# Stage 1: Multi-class on ALL
# -----------------------------
print("\n=== Stage 1: Multi-class on ALL classes ===")
m1, f1s_all, acc_all, rep_all = multiClassifier(
    X_train, y_train, X_test, y_test, class_names
)
print(rep_all)
print("Per-class F1:", f1s_all)
print("Accuracy:", acc_all)

if np.all(f1s_all >= F_MIN):
    print("✅ Stage 1 passed — stop.")
else:
    print("❌ Stage 1 failed — go to Stage 2 (Normal vs Attack)")

    # -----------------------------
    # Stage 2: Binary Normal vs Attack
    # -----------------------------
    print("\n=== Stage 2: Binary Normal vs Attack ===")
    try:
        normal_idx = list(class_names).index('Normal')
    except ValueError:
        raise ValueError("'Normal' class not found in labels.")

    y_train_bin = (y_train != normal_idx).astype(int)
    y_test_bin  = (y_test  != normal_idx).astype(int)

    m2, f1s_bin, acc_bin, rep_bin = binaryClassifier(
        X_train, y_train_bin, X_test, y_test_bin, labels=('Normal','Attack')
    )
    print(rep_bin)
    print("Binary F1 (Normal, Attack):", f1s_bin)
    print("Binary Accuracy:", acc_bin)

    if np.all(f1s_bin >= F_MIN):
        print("✅ Stage 2 passed — now **multi-class ONLY on attack traffic** (exactly as in the paper)")

        # -----------------------------
        # Stage 2b: Multi-class on ATTACK traffic only
        # -----------------------------
        attack_mask_tr = (y_train != normal_idx)
        attack_mask_te = (y_test  != normal_idx)

        X_tr_att = X_train[attack_mask_tr]
        y_tr_att = y_train[attack_mask_tr]
        X_te_att = X_test [attack_mask_te]
        y_te_att = y_test [attack_mask_te]

        # Map attack class indices to contiguous indices for the subset
        att_unique = np.unique(y_tr_att)
        old2new = {old:i for i, old in enumerate(att_unique)}
        new2old = {v:k for k,v in old2new.items()}

        y_tr_att_mapped = np.vectorize(old2new.get)(y_tr_att)
        y_te_att_mapped = np.vectorize(old2new.get)(y_te_att)
        class_names_att = class_names[att_unique]

        print("\n=== Multi-class on ATTACKS only ===")
        m3, f1s_att, acc_att, rep_att = multiClassifier(
            X_tr_att, y_tr_att_mapped, X_te_att, y_te_att_mapped, class_names_att
        )
        print(rep_att)
        print("Per-attack-class F1:", f1s_att)
        print("Attack-only Accuracy:", acc_att)

        if np.all(f1s_att >= F_MIN):
            print("✅ Attack-only multi-class passed — stop.")
        else:
            print("❌ Attack-only multi-class failed — Stage 3 recursion begins")

            # -----------------------------
            # Stage 3: Recursive split using CMR (or worst F1)
            # -----------------------------
            X_cur = X_tr_att
            y_cur = y_tr_att_mapped
            names_cur = class_names_att

            while len(np.unique(y_cur)) > 2:
                # (i) compute CMR on current attack labels
                cmr = compute_cmr(y_cur)
                # take the **lowest** CMR (most minority) class
                worst_local = min(cmr, key=cmr.get)
                worst_name = names_cur[worst_local]
                print(f"\n--> Lowest CMR class: {worst_name}")

                # (ii) binary worst vs others
                y_bin_cur = (y_cur == worst_local).astype(int)
                m_bin_cur, f1s_w, acc_w, rep_w = binaryClassifier(
                    X_cur, y_bin_cur, X_cur, y_bin_cur, labels=(worst_name, 'Others')
                )
                print(rep_w)
                print(f"F1 for {worst_name} vs Others:", f1s_w, "Acc:", acc_w)

                if np.all(f1s_w >= F_MIN):
                    print(f"✅ Accept {worst_name}, remove it and continue …")
                    keep = (y_cur != worst_local)
                    X_cur = X_cur[keep]
                    y_cur = y_cur[keep]

                    # re-map indices to 0..k-1
                    uniq = np.unique(y_cur)
                    remap = {old:i for i,old in enumerate(uniq)}
                    y_cur = np.vectorize(remap.get)(y_cur)
                    names_cur = names_cur[uniq]
                else:
                    print("❌ Could not separate this class above F_MIN — stop recursion.")
                    break

            if len(np.unique(y_cur)) == 2:
                print("\nFinal binary on the last two attack classes …")
                a, b = np.unique(y_cur)
                labels_last = (names_cur[a], names_cur[b])
                y_last_bin = (y_cur == a).astype(int)
                m_last, f1s_last, acc_last, rep_last = binaryClassifier(
                    X_cur, y_last_bin, X_cur, y_last_bin, labels=labels_last
                )
                print(rep_last)
                print("Final 2-class F1:", f1s_last, "Acc:", acc_last)

    else:
        print("❌ Stage 2 failed — cannot proceed to Stage 3.")
