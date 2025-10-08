import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import BorderlineSMOTE  # <-- only BorderlineSMOTE

# Load your dataset
df = pd.read_csv("../../../../../EPIC/dataset_EPICA_raw 1.csv")

# Preprocess data
def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]  # remove constant columns
    data = data.dropna()
    features = data.drop(columns=['status'])
    labels = data['status']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels

# Apply preprocessing
X, y = preprocess_epica(df)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# ===== BorderlineSMOTE oversampling on the training set ONLY =====
borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=5, m_neighbors=10)
X_train_res, y_train_res = borderline_smote.fit_resample(X_train, y_train)

# === Train model with your best params ===
clf = DecisionTreeClassifier(
    random_state=42,
    class_weight=None,
    criterion='entropy',
    max_depth=20,
    min_samples_leaf=8,
    min_samples_split=20
)
clf.fit(X_train_res, y_train_res)

# Predict
y_pred = clf.predict(X_test)

# Evaluate overall metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# Classification report
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)
class_totals = conf_matrix.sum(axis=1)

print("\n=== Overall Metrics (BorderlineSMOTE) ===")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision (macro): {prec * 100:.2f}%")
print(f"Recall (macro): {rec * 100:.2f}%")
print(f"F1 Score (macro): {f1 * 100:.2f}%")

# Per-class metrics
print("\n=== Per-Class Metrics (BorderlineSMOTE) ===")
for idx, class_name in enumerate(class_names):
    tp = conf_matrix[idx][idx]
    total = class_totals[idx]
    approx_acc = tp / total * 100 if total > 0 else 0
    precision = report[class_name]['precision'] * 100
    recall = report[class_name]['recall'] * 100
    f1_score_val = report[class_name]['f1-score'] * 100
    print(f"\nClass: {class_name}")
    print(f"  Accuracy:  {approx_acc:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print(f"  F1-score:  {f1_score_val:.2f}%")
