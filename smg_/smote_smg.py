import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# === Load dataset ===
df = pd.read_csv("../EPIC/dataset_EPICA_raw 1.csv")

# === Preprocessing ===
def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    X = data.drop(columns=['status'])
    y = data['status']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns.tolist()

X_scaled, y_raw, feature_names = preprocess_epica(df)

# === Apply SMOTE ===
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_raw)

# === One-hot encode labels ===
y_encoded = pd.get_dummies(y_resampled)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_encoded, test_size=0.3, random_state=42)

# === Define MLP Model ===
def create_mlp(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === Model Creation & Training with Timing ===
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = create_mlp(input_dim, output_dim)

start_train = time()
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)
end_train = time()

# === Plot Accuracy ===
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# === Evaluation with Timing ===
start_test = time()
y_pred = np.argmax(model.predict(X_test), axis=1)
end_test = time()

y_true = y_test.to_numpy().argmax(axis=1)
class_names = y_test.columns.tolist()

# === Confusion Matrix and Per-Class Accuracy ===
cm = confusion_matrix(y_true, y_pred)
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# === Classification Report ===
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

print("=== Classification Report (Per-Class) ===")
for i, label in enumerate(class_names):
    precision = report[label]["precision"] * 100
    recall = report[label]["recall"] * 100
    f1 = report[label]["f1-score"] * 100
    acc = class_accuracy[i] * 100
    print(f"{label}:")
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print(f"  F1-score:  {f1:.2f}%\n")

# === Aggregated Metrics ===
acc = accuracy_score(y_true, y_pred)
macro_precision = precision_score(y_true, y_pred, average='macro')
macro_recall = recall_score(y_true, y_pred, average='macro')
macro_f1 = f1_score(y_true, y_pred, average='macro')

print("=== Aggregated Metrics ===")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision (macro): {macro_precision * 100:.2f}%")
print(f"Recall (macro): {macro_recall * 100:.2f}%")
print(f"F1 Score (macro): {macro_f1 * 100:.2f}%")

# === Timing Info ===
print("\n=== Timing ===")
print(f"Training Time: {end_train - start_train:.2f} seconds")
print(f"Testing Time: {end_test - start_test:.2f} seconds")
