import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json

print("🚀 Logistic Regression — Baseline from saved hyperparameters")
print("="*70)

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

print(f"📊 Dataset Info:")
print(f"   Features: {X.shape[1]}")
print(f"   Samples: {X.shape[0]}")
print(f"   Classes: {list(class_names)}")

# Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"   Train samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")

# Load best hyperparameters
with open('best_lr_hyperparameters.json', 'r') as f:
    saved = json.load(f)
best_params = saved['best_parameters']

print("\nUsing saved hyperparameters:")
for k, v in best_params.items():
    print(f"   {k}: {v}")

# Build final model with loaded params
clf = LogisticRegression(multi_class="auto", **best_params)

# Time ONLY the final fit
print("\n⏳ Training Logistic Regression (final fit only)...")
start_train = time.time()
clf.fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train
print(f"✅ Training completed! ⏱ {train_time:.3f} seconds")

# Time ONLY the predict
print("\n📈 Evaluating on test data...")
start_test = time.time()
y_pred = clf.predict(X_test)
end_test = time.time()
test_time = end_test - start_test
print(f"✅ Inference completed! ⏱ {test_time:.5f} seconds")

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)
class_totals = conf_matrix.sum(axis=1)

print("\n" + "="*60)
print("🎯 LOGISTIC REGRESSION RESULTS (Final Fit)")
print("="*60)

print(f"\n⏱ Timing:")
print("-" * 35)
print(f"Training Time: {train_time:.3f} s")
print(f"Testing Time:  {test_time:.5f} s")

print(f"\n📊 Overall Performance Metrics:")
print("-" * 35)
print(f"Accuracy:         {acc * 100:.2f}%")
print(f"Precision (macro): {prec * 100:.2f}%")
print(f"Recall (macro):    {rec * 100:.2f}%")
print(f"F1 Score (macro):  {f1 * 100:.2f}%")

print(f"\n📋 Per-Class Performance:")
print("-" * 35)
for idx, class_name in enumerate(class_names):
    tp = conf_matrix[idx][idx]
    total = class_totals[idx]
    class_accuracy = tp / total * 100 if total > 0 else 0
    class_precision = report[class_name]['precision'] * 100
    class_recall = report[class_name]['recall'] * 100
    class_f1 = report[class_name]['f1-score'] * 100
    print(f"\n🏷️  Class: {class_name}")
    print(f"    Accuracy:  {class_accuracy:.2f}%")
    print(f"    Precision: {class_precision:.2f}%")
    print(f"    Recall:    {class_recall:.2f}%")
    print(f"    F1-score:  {class_f1:.2f}%")

# Coefficient importance (optional but useful)
print(f"\n🔍 Feature Importance Analysis (Coefficient magnitudes):")
try:
    feature_names = df.drop(columns=['status', 'Unnamed: 0', 'Unnamed: 0.1'], errors='ignore').columns.tolist()
except:
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

if len(class_names) > 2:
    import numpy as np
    feature_importance = np.mean(np.abs(clf.coef_), axis=0)
else:
    feature_importance = np.abs(clf.coef_[0])

importance_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient_magnitude': feature_importance
}).sort_values('coefficient_magnitude', ascending=False)

print("Top 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

# Confusion Matrix (Blue, consistent)
print("\n=== Confusion Matrix ===")
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Logistic Regression — Confusion Matrix")
plt.show()

# Save run results
with open('lr_baseline_results.json', 'w') as f:
    json.dump({
        'used_parameters': best_params,
        'model_type': 'LogisticRegression',
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'training_time': train_time,
        'testing_time': test_time,
        'class_names': list(map(str, class_names))
    }, f, indent=2)

print("\n✅ Saved run metrics to 'lr_baseline_results.json'")
