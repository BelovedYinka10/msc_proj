import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json

# Load dataset
df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")

# Preprocess
def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    features = data.drop(columns=['status'])
    labels = data['status']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)
    return X, labels

X, y = preprocess_epica(df)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Load best hyperparameters
with open('best_knn_hyperparameters.json', 'r') as f:
    saved = json.load(f)
best_params = saved['best_parameters']

print("\nUsing saved hyperparameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# Build final model
clf = KNeighborsClassifier(**best_params)

# Time ONLY the final fit
print("\n⏳ Training final KNN...")
start_train = time.time()
clf.fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train
print(f"✅ Training time (final fit only): {train_time:.5f} s")

# Time ONLY the predict
start_test = time.time()
y_pred = clf.predict(X_test)
end_test = time.time()
test_time = end_test - start_test
print(f"✅ Testing time (predict): {test_time:.5f} s")

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)
class_totals = conf_matrix.sum(axis=1)

print("\n=== Overall Metrics ===")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision (macro): {prec * 100:.2f}%")
print(f"Recall (macro): {rec * 100:.2f}%")
print(f"F1 Score (macro): {f1 * 100:.2f}%")

print("\n=== Per-Class Metrics ===")
for idx, name in enumerate(class_names):
    tp = conf_matrix[idx][idx]
    total = class_totals[idx]
    approx_acc = tp / total * 100 if total > 0 else 0
    print(f"\nClass: {name}")
    print(f"  Accuracy:  {approx_acc:.2f}%")
    print(f"  Precision: {report[name]['precision']*100:.2f}%")
    print(f"  Recall:    {report[name]['recall']*100:.2f}%")
    print(f"  F1-score:  {report[name]['f1-score']*100:.2f}%")

# Confusion matrix (Blue, consistent)
print("\n=== Confusion Matrix ===")
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("KNN — Confusion Matrix")
plt.show()

# Save run results
with open('knn_baseline_results.json', 'w') as f:
    json.dump({
        'used_parameters': best_params,
        'model_type': 'KNeighborsClassifier',
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'training_time': train_time,
        'testing_time': test_time,
        'class_names': list(map(str, class_names))
    }, f, indent=2)

print("\n✅ Saved run metrics to 'knn_baseline_results.json'")
