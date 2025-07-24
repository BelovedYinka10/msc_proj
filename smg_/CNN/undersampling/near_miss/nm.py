import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import NearMiss
import os

nme="NearMiss"
# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load dataset
# NOTE: Ensure the path to the CSV file is correct in your environment.
try:
    df = pd.read_csv("../../../../EPIC/dataset_EPICA_raw 1.csv")
except FileNotFoundError:
    print("Error: dataset_EPICA_raw 1.csv not found. Please ensure the path is correct.")
    exit()

print(df.columns.tolist())

# Preprocessing
def preprocess_epica(data):
    """Handles data cleaning, feature selection, and scaling."""
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    # Remove columns with only one unique value
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    features = data.drop(columns=['status'])
    labels = data['status']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels, features.columns.tolist()

X, y, feature_names = preprocess_epica(df)

# Encode labels
y_encoded = pd.get_dummies(y)
class_labels = y_encoded.columns.tolist()

# Split data
# X_train, X_test will be used for ADASYN, y_train, y_test for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Model configuration and hyperparameters (made configurable as requested)
# Updated CNN hyperparameters based on the user's request
# Updated CNN hyperparameters based on the user's request
cnn_params = {

    "batch_size": 32,                     # unchanged, not in model definition
    "filters1": 64,                          # corresponds to Conv1D(filters=64, ...)
    "dropout1": 0.1,                       # first Dropout(0.1)
    "filters2": 32,                          # corresponds to Conv1D(filters=32, ...)
    "dropout2": 0.3,                       # second Dropout(0.3)
    "optimizer": "adam",                  # optimizer used is Adam
    "lr": 0.0032164230487929987,      # learning_rate passed to Adam
    "dense_units" :128

}

# Model configuration
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

# --- ADASYN Oversampling on Training Data ---
print("\nApplying ADASYN oversampling to the training data...")

# 1. Convert one-hot encoded y_train back to labels for ADASYN
y_train_labels = np.argmax(y_train.values, axis=1)

# 2. Apply ADASYN
adasyn = NearMiss(random_state=42)
X_train_resampled, y_train_resampled_labels = adasyn.fit_resample(X_train, y_train_labels)


# 3. Convert resampled labels back to one-hot encoding for Keras
y_train_resampled_encoded = tf.keras.utils.to_categorical(y_train_resampled_labels, num_classes=output_dim)

print(f"Original training shape: X={X_train.shape}, y={y_train.shape}")
print(f"Resampled training shape: X={X_train_resampled.shape}, y={y_train_resampled_encoded.shape}")
print("-" * 50)


# CNN Model using specified parameters
def create_cnn_model(input_dim, output_dim, params):
    """Creates a CNN model with configurable hyperparameters."""
    model = tf.keras.Sequential([
        # Input layer and Reshape for Conv1D
        tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),

        # First Conv Block (uses filters1 and dropout1)
        tf.keras.layers.Conv1D(filters=params['filters1'], kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(params['dropout1']),

        # Second Conv Block (uses filters2 and dropout2)
        tf.keras.layers.Conv1D(filters=params['filters2'], kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(params['dropout2']),

        # Dense layer (uses dense_units)
        tf.keras.layers.Dense(params['dense_units'], activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])

    # Optimizer and Learning Rate (uses optimizer and lr)
    if params['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=params['lr'])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr']) # Default to Adam

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train model using ADASYN oversampled data
model = create_cnn_model(input_dim, output_dim, cnn_params)
model.summary()

start_train = time.time()
# Train on the ADASYN resampled training data
history = model.fit(X_train_resampled, y_train_resampled_encoded, 
                    epochs=10, 
                    batch_size=cnn_params['batch_size'], 
                    # We can use validation_split on the oversampled data, or evaluate on the test set later.
                    # Maintaining the original structure using validation_split=0.1
                    validation_split=0.1, 
                    verbose=1)
end_train = time.time()

# Accuracy plot
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title(f'CNN Training and Validation Accuracy {nme}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Loss plot
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'CNN Training and Validation Loss {nme}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Prediction on the original (non-oversampled) test set
start_test = time.time()
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
end_test = time.time()

# Convert y_test (one-hot encoded) to labels for evaluation
y_true = np.argmax(y_test.values, axis=1)

# Evaluation
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("\n=== Timing ===")
print(f"Training Time: {end_train - start_train:.2f} seconds")
print(f"Testing Time:  {end_test - start_test:.2f} seconds")

print("\n=== Overall Metrics ===")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision (macro): {prec * 100:.2f}%")
print(f"Recall (macro): {rec * 100:.2f}%")
print(f"F1 Score (macro): {f1 * 100:.2f}%")

# Per-class metrics
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0)
print("\n=== Per-Class Metrics ===")
for class_name in class_labels:
    metrics = report[class_name]
    # Note: Accuracy is typically calculated globally, not per-class. 
    # The approximation used in the original script `(recall * precision)**0.5` is not standard classification accuracy.
    # We will print the standard precision, recall, and f1-score.
    print(f"\nClass: {class_name}")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  F1-score:  {metrics['f1-score'] * 100:.2f}%")