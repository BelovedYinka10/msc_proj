import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import time
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
# NOTE: The provided path assumes 'dataset_EPICA_raw 1.csv' is located two directories above the execution path in an 'EPIC' folder.
try:
    df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")
except FileNotFoundError:
    print("Error: dataset_EPICA_raw 1.csv not found. Please ensure the path is correct.")
    exit()

# Preprocess data
def preprocess_epica(data):
    """Handles data cleaning, feature selection, and scaling."""
    # Drop unnecessary columns
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    # Remove columns with only one unique value (constant features)
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna() # Remove any remaining NaN values
    
    features = data.drop(columns=['status'])
    labels = data['status']
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels

# Apply preprocessing
X, y = preprocess_epica(df)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Train-test split (Used for splitting the data for FL clients and the global test set)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# --- Federated Learning Configuration ---
K = 3  # Number of clients
R = 10 # Number of communication rounds

# Split training data among K clients (IID assumption)
client_data_splits = []
split_size = len(X_train_full) // K
for i in range(K):
    start, end = i * split_size, (i + 1) * split_size if i < K - 1 else len(X_train_full)
    client_data_splits.append((X_train_full[start:end], y_train_full[start:end]))

# --- Federated Training (Decision Tree Ensemble Simulation) ---
# Note: For Decision Trees, FedAvg (averaging weights) is not applicable.
# This simulation uses a Federated Ensemble approach: clients train local DTs, 
# and the server aggregates predictions via majority voting over rounds.

# Track metrics for plotting
round_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
start_train = time.time()

# This list will store all models trained across all rounds to form the ensemble
federated_ensemble = []

print(f"Starting Federated Decision Tree Ensemble training with {K} clients and {R} rounds.")

for r in range(R):
    print(f"\n--- Communication Round {r+1} of {R} ---")
    
    # Clients train models locally in this round
    trained_models = []
    for k in range(K):
        X_client, y_client = client_data_splits[k]
        
        # Instantiate and train a Decision Tree Classifier
        client_dt = DecisionTreeClassifier(random_state=42)
        client_dt.fit(X_client, y_client)
        trained_models.append(client_dt)
    
    # Add models from this round to the global ensemble
    federated_ensemble.extend(trained_models)
    
    # --- Server Aggregation (Majority Voting) and Evaluation ---
    
    # 1. Generate predictions from all models in the ensemble on the global test set
    predictions = []
    for model in federated_ensemble:
        predictions.append(model.predict(X_test))
    
    # 2. Aggregate predictions using majority voting
    # We transpose the predictions array (models x samples) to (samples x models) 
    # to find the mode (majority vote) for each sample.
    predictions_array = np.array(predictions).T
    
    # Calculate the mode (most frequent prediction) for each sample
    def majority_vote(predictions_row):
        return np.bincount(predictions_row).argmax()

    aggregated_predictions = np.apply_along_axis(majority_vote, axis=1, arr=predictions_array)
    y_pred = aggregated_predictions
    
    # 3. Evaluate the ensemble performance at this round
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Store metrics for plotting
    round_metrics['accuracy'].append(acc * 100)
    round_metrics['precision'].append(prec * 100)
    round_metrics['recall'].append(rec * 100)
    round_metrics['f1_score'].append(f1 * 100)
    
    print(f"Round {r+1} Metrics: Accuracy: {acc*100:.2f}%, F1 (Macro): {f1*100:.2f}%")

end_train = time.time()

# --- Final Evaluation ---

print("\n=== Final Ensemble Model Metrics ===")
print(f"Training Time: {end_train - start_train:.2f} seconds (Total FL simulation time)")

# Use metrics from the final round (since y_pred and metrics variables hold the final results)
final_acc = round_metrics['accuracy'][-1] / 100
final_prec = round_metrics['precision'][-1] / 100
final_rec = round_metrics['recall'][-1] / 100
final_f1 = round_metrics['f1_score'][-1] / 100

print("\n=== Overall Metrics ===")
print(f"Accuracy: {final_acc * 100:.2f}%")
print(f"Precision (macro): {final_prec * 100:.2f}%")
print(f"Recall (macro): {final_rec * 100:.2f}%")
print(f"F1 Score (macro): {final_f1 * 100:.2f}%")

# Per-class metrics for the final ensemble model
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)
class_totals = conf_matrix.sum(axis=1)

print("\n=== Per-Class Metrics ===")
for idx, class_name in enumerate(class_names):
    # Calculate approx accuracy (TP/Total for the class)
    tp = conf_matrix[idx][idx]
    total = class_totals[idx]
    approx_acc = tp / total * 100 if total > 0 else 0
    precision = report[class_name]['precision'] * 100
    recall = report[class_name]['recall'] * 100
    f1_score_val = report[class_name]['f1-score'] * 100
    print(f"\nClass: {class_name}")
    print(f"  Accuracy:  {approx_acc:.2f}% (Approx)")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print(f"  F1-score:  {f1_score_val:.2f}%")

# --- Plotting Round Performance ---
plt.figure(figsize=(8, 6))
rounds = range(1, R + 1)
plt.plot(rounds, round_metrics['accuracy'], marker='o', label='Accuracy')
plt.plot(rounds, round_metrics['precision'], marker='s', label='Precision (Macro)')
plt.plot(rounds, round_metrics['recall'], marker='^', label='Recall (Macro)')
plt.plot(rounds, round_metrics['f1_score'], marker='d', label='F1 Score (Macro)')
plt.title('Federated Decision Tree Ensemble Performance Over Rounds')
plt.xlabel('Communication Round')
plt.ylabel('Score (%)')
plt.grid(True)
plt.legend()
plt.show()