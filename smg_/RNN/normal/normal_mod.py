import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load dataset
# NOTE: Ensure the path to the CSV file is correct in your environment.
try:
    df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")
except FileNotFoundError:
    print("Error: dataset_EPICA_raw 1.csv not found. Please ensure the path is correct.")
    exit()

print(df.columns.tolist())

# Preprocessing
def preprocess_epica(data):
    """Handles data cleaning, feature selection, and scaling."""
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Model configuration
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
class_labels = y_encoded.columns.tolist()

# --- Configurable RNN parameters ---
# Updated RNN hyperparameters based on the provided Keras Tuner search results ('Value' column)
rnn_params = {
    'batch_size': 64,           # Updated from 32
    'lstm_units1': 128,         # Updated from 64
    'dropout1': 0.3,            # Updated from 0.1
    'lstm_units2': 32,          # Retained from 32
    'dropout2': 0.4,            # Updated from 0.3
    'dense_units': 64,          # Updated from 128
    'optimizer': 'rmsprop',     # Updated from 'adam'
    'lr': 0.00077456            # Updated from 0.0032164230487929987
}

# RNN Model Definition
# Function name remains 'create_cnn_model' as requested, but now accepts and uses 'params'
def create_cnn_model(input_dim, output_dim, params):
    """Creates an RNN (LSTM) model with configurable parameters."""
    model = tf.keras.Sequential([
        # Reshape input for RNN: (batch_size, timesteps, features)
        tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),

        # First RNN Block (LSTM)
        # Using parameters['lstm_units1'] and parameters['dropout1']
        tf.keras.layers.LSTM(units=params['lstm_units1'], activation='relu', return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params['dropout1']),

        # Second RNN Block (LSTM)
        # Using parameters['lstm_units2'] and parameters['dropout2']
        tf.keras.layers.LSTM(units=params['lstm_units2'], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params['dropout2']),

        # Dense layers (using parameters['dense_units'])
        tf.keras.layers.Dense(params['dense_units'], activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])

    # Optimizer and Learning Rate (using parameters['optimizer'] and parameters['lr'])
    if params['optimizer'] == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=params['lr'])
    elif params['optimizer'] == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=params['lr'])
    else:
        # Default to Adam if optimizer choice is invalid or not provided
        opt = tf.keras.optimizers.Adam(learning_rate=params['lr']) 

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train model using configurable parameters
model = create_cnn_model(input_dim, output_dim, rnn_params)
model.summary()

start_train = time.time()
# Use batch_size from rnn_params
history = model.fit(X_train, y_train, epochs=10, batch_size=rnn_params['batch_size'], validation_split=0.1, verbose=1)
end_train = time.time()

# Accuracy plot
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('RNN Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Loss plot
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('RNN Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Prediction
start_test = time.time()
y_pred = np.argmax(model.predict(X_test), axis=1)
end_test = time.time()

y_true = np.argmax(y_test.values, axis=1)

# Evaluation
# Note: Setting zero_division=0 to handle cases where precision/recall is undefined.
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
    # The original script calculated an approximation for 'Accuracy' using (recall*precision)**0.5
    print(f"\nClass: {class_name}")
    print(f"  Accuracy: {(metrics['recall'] * metrics['precision'])**0.5 * 100:.2f}% (Approx)")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  F1-score:  {metrics['f1-score'] * 100:.2f}%")