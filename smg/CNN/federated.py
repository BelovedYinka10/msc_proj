import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load dataset
# NOTE: Ensure the path to the CSV file is correct in your environment.
try:
    df = pd.read_csv("../../EPIC/dataset_EPICA_raw 1.csv")
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

# Federated Learning Parameters
K = 3  # Number of clients
R = 10 # Number of communication rounds
BATCH_SIZE = 32 # Used in the original centralized script
EPOCHS = 10     # Used in the original centralized script

# Split training data among clients (IID assumption for simplicity)
client_data, client_label = [], []
split_size = len(X_train) // K
for i in range(K):
    start, end = i * split_size, (i + 1) * split_size if i < K - 1 else len(X_train)
    client_data.append(X_train[start:end])
    client_label.append(y_train.iloc[start:end])

# Model configuration
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
class_labels = y_encoded.columns.tolist()

# CNN Model Definition
def create_cnn_model(input_dim, output_dim):
    """
    Creates the CNN model with the specified architecture and parameters
    from the centralized training script.
    """
    model = tf.keras.Sequential([
        # Reshape for Conv1D input (batch_size, timesteps, features)
        tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),

        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0032164230487929987)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize Global Model and Weights
global_model = create_cnn_model(input_dim, output_dim)
global_weights = global_model.get_weights()

# Track metrics for plotting
acc_list, prec_list, rec_list, f1_list = [], [], [], []

# Federated Learning Training Loop
start_train = time.time()
print(f"Starting Federated Learning (FedAvg) with K={K} clients and R={R} rounds.")

for r in range(R):
    print(f"\n--- Communication Round {r+1} of {R} ---")
    client_weights = []
    
    # Client training and weight collection
    for k in range(K):
        # Instantiate a new local model for the client
        model = create_cnn_model(input_dim, output_dim)
        
        # Set local model weights to the current global weights
        model.set_weights(global_weights)
        
        # Train the local model on client's data
        # Note: We use the defined BATCH_SIZE and EPOCHS for local training.
        model.fit(client_data[k], client_label[k], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        
        # Collect the updated local weights
        client_weights.append(model.get_weights())
    
    # FedAvg Aggregation
    # Average the weights of all clients layer by layer
    new_weights = []
    for weights_in_layer in zip(*client_weights):
        new_weights.append(np.mean(weights_in_layer, axis=0))
    
    # Update the global model with aggregated weights
    global_model.set_weights(new_weights)
    global_weights = new_weights

    # Evaluation of the global model after aggregation
    y_pred_probs = global_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test.values, axis=1)

    # Store metrics for plotting
    acc_list.append(accuracy_score(y_true, y_pred) * 100)
    prec_list.append(precision_score(y_true, y_pred, average='macro', zero_division=0) * 100)
    rec_list.append(recall_score(y_true, y_pred, average='macro', zero_division=0) * 100)
    f1_list.append(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100)
    
end_train = time.time()

# Final evaluation on the test set
start_test = time.time()
y_pred_probs = global_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
end_test = time.time()
y_true = np.argmax(y_test.values, axis=1)

# Overall Metrics
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

# Per-class metrics (using classification_report for detailed metrics)
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)
class_totals = conf_matrix.sum(axis=1)

print("\n=== Per-Class Metrics ===")
for idx, class_name in enumerate(class_labels):
    metrics = report[class_name]
    # Calculate approximate accuracy (TP/Total for the class)
    tp = conf_matrix[idx][idx]
    total = class_totals[idx]
    approx_acc = tp / total * 100 if total > 0 else 0
    
    print(f"\nClass: {class_name}")
    print(f"  Accuracy:  {approx_acc:.2f}% (Approx)")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  F1-score:  {metrics['f1-score'] * 100:.2f}%")

# Plotting performance over rounds
rounds = list(range(1, R+1))
plt.figure(figsize=(8, 6))
plt.plot(rounds, acc_list, marker='s', label="Accuracy")
plt.plot(rounds, rec_list, marker='o', label="Recall")
plt.plot(rounds, prec_list, marker='^', label="Precision")
plt.plot(rounds, f1_list, marker='o', label="F1 Score")
plt.xlabel("Number of communication rounds")
plt.ylabel("Performance (%)")
plt.title("Federated Learning Performance per Round (CNN Model)")
plt.legend()
plt.grid(True)
plt.show()