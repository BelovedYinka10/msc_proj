import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time

# Load dataset
df = pd.read_csv("../EPIC/dataset_EPICA_raw 1.csv")

# Preprocessing
def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    features = data.drop(columns=['status'])
    labels = data['status']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels, features.columns.tolist()

X, y, feature_names = preprocess_epica(df)

# Encode labels (one-hot)
y_encoded = pd.get_dummies(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Reshape data for 1D CNN: (samples, timesteps, features)
X_train_cnn = np.expand_dims(X_train, axis=-1)
X_test_cnn = np.expand_dims(X_test, axis=-1)

# Model input/output dimensions
input_dim = X_train_cnn.shape[1]
output_dim = y_train.shape[1]

# Define 1D CNN model
def create_cnn(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(input_dim, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Instantiate model
model = create_cnn(input_dim, output_dim)

# Train model
start_train = time.time()
history = model.fit(X_train_cnn, y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1)
end_train = time.time()

# Plot accuracy
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Predict
start_test = time.time()
y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
end_test = time.time()

y_true = np.argmax(y_test.values, axis=1)
class_labels = y_test.columns.tolist()

# Evaluate
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro')
rec = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("\n=== Timing ===")
print(f"Training Time: {end_train - start_train:.2f} seconds")
print(f"Testing Time:  {end_test - start_test:.2f} seconds")

print("\n=== Overall Metrics ===")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision (macro): {prec * 100:.2f}%")
print(f"Recall (macro): {rec * 100:.2f}%")
print(f"F1 Score (macro): {f1 * 100:.2f}%")

# Per-class evaluation
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
conf_matrix = confusion_matrix(y_true, y_pred)
class_totals = conf_matrix.sum(axis=1)

print("\n=== Per-Class Metrics ===")
for idx, class_name in enumerate(class_labels):
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
