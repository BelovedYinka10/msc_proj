import numpy as np
import pandas as pd
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.under_sampling import ClusterCentroids
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../EPIC/dataset_EPICA_raw 1.csv")

# Preprocessing
def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    X = data.drop(columns=['status'])
    y = data['status']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns.tolist()

X, y, feature_names = preprocess_epica(df)

# Apply Cluster Centroids
cc = ClusterCentroids(random_state=42)
X_resampled, y_resampled = cc.fit_resample(X, y)

# One-hot encode labels
y_encoded = pd.get_dummies(y_resampled)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_encoded, test_size=0.3, random_state=42)

# Define model
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

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = create_mlp(input_dim, output_dim)

# Training
start_train = time.time()
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)
end_train = time.time()

# Evaluation
start_test = time.time()
y_pred = np.argmax(model.predict(X_test), axis=1)
end_test = time.time()

y_true = np.argmax(y_test.values, axis=1)
class_labels = y_test.columns.tolist()

# Overall Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro')
rec = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"\n=== Training and Testing Time ===")
print(f"Training Time: {end_train - start_train:.2f} seconds")
print(f"Testing Time:  {end_test - start_test:.2f} seconds")

print("\n=== Overall Metrics ===")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision (macro): {prec * 100:.2f}%")
print(f"Recall (macro): {rec * 100:.2f}%")
print(f"F1 Score (macro): {f1 * 100:.2f}%")

# Per-Class Metrics
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
print("\n=== Per-Class Metrics ===")
for class_name in class_labels:
    metrics = report[class_name]
    class_acc = (metrics['precision'] * metrics['recall']) ** 0.5
    print(f"\nClass: {class_name}")
    print(f"  Accuracy (approx.): {class_acc * 100:.2f}%")
    print(f"  Precision:          {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:             {metrics['recall'] * 100:.2f}%")
    print(f"  F1-score:           {metrics['f1-score'] * 100:.2f}%")
