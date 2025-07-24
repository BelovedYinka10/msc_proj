import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# --- Data Loading and Preprocessing ---

# Load EPIC dataset (Ensure this path is correct on your system)
try:
    df = pd.read_csv("../EPIC/dataset_EPICA_raw 1.csv")
except FileNotFoundError:
    print("Error: dataset_EPICA_raw 1.csv not found. Please ensure the file path is correct.")
    exit()

def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    features = data.drop(columns=['status'])
    labels = data['status']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels

# Preprocessing
X, y = preprocess_epica(df)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# --- Define class_names after preprocessing ---
class_names = label_encoder.classes_

# Binary classification: Normal vs Attack
# --- Ensure normal_class_idx is defined *after* class_names ---
normal_class_idx = list(class_names).index('Normal')
y_binary = (y_encoded != normal_class_idx).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# --- Hyperparameter Tuning Setup (Keras Tuner) ---

# Define the model builder for binary classification
# It uses X.shape[1] for input dimension, which is now defined.
def build_binary_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(X.shape[1],)))
    
    # Tunable units and dropout for the first layer
    model.add(tf.keras.layers.Dense(hp.Int('units1', 128, 512, step=64), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))

    # Tunable units and dropout for the second layer
    model.add(tf.keras.layers.Dense(hp.Int('units2', 64, 256, step=64), activation='relu'))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Tunable learning rate
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Set up the Hyperband tuner
tuner = kt.Hyperband(
    build_binary_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='binary_dnn_tuning',
    project_name='normal_vs_attack'
)

# Early stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# --- Search for best hyperparameters ---
# Use X_train and y_train (binary labels for Normal vs Attack)
print("Starting Hyperparameter Search for Stage 2 (Normal vs Attack)...")
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

# --- Evaluation with Best Model ---

# Retrieve best hyperparameters
best_hp = tuner.get_best_hyperparameters(1)[0]

# Build the best model and train it with optimized hyperparameters
model = tuner.hypermodel.build(best_hp)
model.fit(X_train, y_train, epochs=30, batch_size=128, validation_split=0.1, verbose=0)

# Evaluation
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

print("\n📊 Evaluation Report (Stage 2 - Tuned Model):")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:\n")
# Note: y_test and y_pred are binary (0=Normal, 1=Attack)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], zero_division=1))

print("\n✅ Best Hyperparameters found:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")