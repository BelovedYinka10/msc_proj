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
    df  = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")
except FileNotFoundError:
    print("Error: dataset_EPICA_raw 1.csv not found. Please ensure the path is correct.")
    exit()

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
y_encoded = pd.get_dummies(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Federated parameters
K = 3  # Number of clients
R = 10 # Number of communication rounds
# --- UPDATED BATCH_SIZE and EPOCHS FROM CENTRALIZED CODE ---
BATCH_SIZE = 32
EPOCHS = 10

# Split data among clients
client_data, client_label = [], []
split_size = len(X_train) // K
for i in range(K):
    start, end = i * split_size, (i + 1) * split_size if i < K - 1 else len(X_train)
    client_data.append(X_train[start:end])
    client_label.append(y_train.iloc[start:end])

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
class_labels = y_encoded.columns.tolist()

# --- UPDATED MODEL CREATION FROM CENTRALIZED CODE ---
def create_mlp(input_dim, output_dim):
    """
    Creates a Multi-Layer Perceptron (MLP) model with the specified configuration
    from the provided centralized learning script.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(320, activation='relu', input_shape=(input_dim,)),  # Best units1
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),  # Best dropout
        tf.keras.layers.Dense(224, activation='relu'),  # Best units2
        tf.keras.layers.Dropout(0.1),  # Best dropout again
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    # Best lr and optimizer from the centralized script
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0011140493525027336)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

global_model = create_mlp(input_dim, output_dim)
global_weights = global_model.get_weights()

# Track metrics for plotting
acc_list, prec_list, rec_list, f1_list = [], [], [], []

start_train = time.time()
for r in range(R):
    print(f"\n--- Communication Round {r+1} of {R} ---")
    client_weights = []
    # Train each client's model
    for k in range(K):
        model = create_mlp(input_dim, output_dim)
        model.set_weights(global_weights)
        model.fit(client_data[k], client_label[k], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        client_weights.append(model.get_weights())
    
    # FedAvg aggregation
    new_weights = []
    for weights_in_layer in zip(*client_weights):
        new_weights.append(np.mean(weights_in_layer, axis=0))
    
    global_model.set_weights(new_weights)
    global_weights = new_weights

    # Evaluation after each round on the test set
    y_pred = np.argmax(global_model.predict(X_test), axis=1)
    y_true = np.argmax(y_test.values, axis=1)

    acc_list.append(accuracy_score(y_true, y_pred) * 100)
    prec_list.append(precision_score(y_true, y_pred, average='macro', zero_division=0) * 100)
    rec_list.append(recall_score(y_true, y_pred, average='macro', zero_division=0) * 100)
    f1_list.append(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100)

end_train = time.time()

# Final evaluation
start_test = time.time()
y_pred = np.argmax(global_model.predict(X_test), axis=1)
end_test = time.time()
y_true = np.argmax(y_test.values, axis=1)

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

# Plotting
rounds = list(range(1, R+1))
plt.plot(rounds, acc_list, marker='s', label="Accuracy")
plt.plot(rounds, rec_list, marker='o', label="Recall")
plt.plot(rounds, prec_list, marker='^', label="Precision")
plt.plot(rounds, f1_list, marker='o', label="F1 Score")
plt.xlabel("Number of communication rounds")
plt.ylabel("Performance (%)")
plt.title("Federated Learning Performance per Round")
plt.legend()
plt.grid(True)
plt.show()