import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------
# 1. Load & Preprocess Data
# -----------------------
df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")


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

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------
# 2. Parameters
# -----------------------
F_MIN = 0.86


# -----------------------
# 3. Evaluation Functions
# -----------------------
def evaluate_multiclass(y_true, y_pred, class_names):
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


def evaluate_binary(y_true, y_pred):
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
# 4. Stage 1: Multi-class
# -----------------------
print("\n=== Stage 1: Decision Tree Multi-class ===")
model_multi = DecisionTreeClassifier(random_state=42)
model_multi.fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)
acc, metrics, passed = evaluate_multiclass(y_test, y_pred_multi, class_names)

if passed:
    print("✅ Stage 1 passed.")
else:
    print("❌ Stage 1 failed. Proceeding to Stage 2.")

    # -----------------------
    # 5. Stage 2: Binary Normal vs Attack
    # -----------------------
    normal_class = list(class_names).index('Normal')
    y_train_bin = (y_train != normal_class).astype(int)
    y_test_bin = (y_test != normal_class).astype(int)

    model_bin = DecisionTreeClassifier(random_state=42)
    model_bin.fit(X_train, y_train_bin)
    y_pred_bin = model_bin.predict(X_test)
    acc_bin, metrics_bin, passed_bin = evaluate_binary(y_test_bin, y_pred_bin)

    if passed_bin:
        print("✅ Stage 2 passed. Proceeding to Stage 3.")

        # -----------------------
        # 6. Stage 3: Attack-only Multi-class
        # -----------------------
        attack_mask_train = (y_train != normal_class)
        attack_mask_test = (y_test != normal_class)
        X_attack_train = X_train[attack_mask_train]
        X_attack_test = X_test[attack_mask_test]
        y_attack_train = y_train[attack_mask_train]
        y_attack_test = y_test[attack_mask_test]
        attack_classes = [c for c in class_names if c != 'Normal']

        print("\n=== Stage 3: Decision Tree Multi-class on Attacks ===")
        model_attack = DecisionTreeClassifier(random_state=42)
        model_attack.fit(X_attack_train, y_attack_train)
        y_pred_attack = model_attack.predict(X_attack_test)
        acc_att, metrics_att, passed_att = evaluate_multiclass(y_attack_test, y_pred_attack, attack_classes)

        if passed_att:
            print("✅ Stage 3 passed.")
        else:
            print("❌ Stage 3 failed. Proceeding to Stage 4.")

            # -----------------------
            # 7. Stage 4: Recursive CMR-based Splitting
            # -----------------------
            y_attack_train_idx = y_attack_train.copy()
            while len(np.unique(y_attack_train_idx)) > 2:
                cmr_values = compute_cmr(y_attack_train_idx)
                worst_class_idx = min(cmr_values, key=cmr_values.get)
                worst_class = attack_classes[worst_class_idx]
                print(f"\nRecursive step: Class with lowest CMR = {worst_class} ({cmr_values[worst_class_idx]:.3f})")

                y_binary = (y_attack_train_idx == worst_class_idx).astype(int)
                model_bin_rec = DecisionTreeClassifier(random_state=42)
                model_bin_rec.fit(X_attack_train, y_binary)
                y_pred_b = model_bin_rec.predict(X_attack_train)
                acc_b, metrics_b, passed_b = evaluate_binary(y_binary, y_pred_b)

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
                model_final = DecisionTreeClassifier(random_state=42)
                model_final.fit(X_attack_train, y_binary_final)
                y_pred_final = model_final.predict(X_attack_train)
                evaluate_binary(y_binary_final, y_pred_final)
    else:
        print("❌ Stage 2 failed. Cannot proceed further.")
