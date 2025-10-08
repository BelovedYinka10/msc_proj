import pandas as pd
import numpy as np
# CHANGED: use LogisticRegression instead of DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import time
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
try:
    df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")
except FileNotFoundError:
    print("Error: dataset_EPICA_raw 1.csv not found. Please ensure the path is correct.")
    exit()

# Preprocess data
def preprocess_epica(data):
    """Handles data cleaning, feature selection, and scaling."""
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    features = data.drop(columns=['status'])
    labels = data['status']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels

# Apply preprocessing
X, y = preprocess_epica(df)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Train-test split
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

# --- Federated Training (Logistic Regression Ensemble Simulation) ---
# FedAvg isn't applied here; we aggregate predictions via majority voting.

round_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
start_train = time.time()
federated_ensemble = []

print(f"Starting Federated Logistic Regression Ensemble training with {K} clients and {R} rounds.")

for r in range(R):
    print(f"\n--- Communication Round {r+1} of {R} ---")
    
    trained_models = []
    for k in range(K):
        X_client, y_client = client_data_splits[k]
        
        # CHANGED: LogisticRegression model (increase max_iter for convergence)
        client_lr = LogisticRegression(
            random_state=42,
            max_iter=1000,
            n_jobs=None,
            multi_class='auto',
            solver='lbfgs'
        )
        client_lr.fit(X_client, y_client)
        trained_models.append(client_lr)
    
    federated_ensemble.extend(trained_models)
    
    # --- Server Aggregation (Majority Voting) and Evaluation ---
    predictions = []
    for model in trained_models:  # (evaluate current round models only for round metrics)
        predictions.append(model.predict(X_test))
    predictions_array = np.array(predictions).T
    
    def majority_vote(predictions_row):
        return np.bincount(predictions_row).argmax()

    aggregated_predictions = np.apply_along_axis(majority_vote, axis=1, arr=predictions_array)
    y_pred = aggregated_predictions
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    round_metrics['accuracy'].append(acc * 100)
    round_metrics['precision'].append(prec * 100)
    round_metrics['recall'].append(rec * 100)
    round_metrics['f1_score'].append(f1 * 100)
    
    print(f"Round {r+1} Metrics: Accuracy: {acc*100:.2f}%, F1 (Macro): {f1*100:.2f}%")

end_train = time.time()

# --- Final Evaluation ---
print("\n=== Final Ensemble Model Metrics ===")
print(f"Training Time: {end_train - start_train:.2f} seconds (Total FL simulation time)")

# NEW: time ONLY the final ensemble prediction using ALL accumulated models
start_test = time.time()  # NEW
all_preds = [m.predict(X_test) for m in federated_ensemble]  # NEW
all_preds = np.array(all_preds).T  # shape: [n_samples, n_models]  # NEW
y_pred = np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1, arr=all_preds)  # NEW
end_test = time.time()  # NEW
test_time = end_test - start_test  # NEW
print(f"Testing Time: {test_time:.4f} seconds (final ensemble prediction)")  # NEW

final_acc = accuracy_score(y_test, y_pred)
final_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
final_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
final_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\n=== Overall Metrics ===")
print(f"Accuracy: {final_acc * 100:.2f}%")
print(f"Precision (macro): {final_prec * 100:.2f}%")
print(f"Recall (macro): {final_rec * 100:.2f}%")
print(f"F1 Score (macro): {final_f1 * 100:.2f}%")

report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)
class_totals = conf_matrix.sum(axis=1)

print("\n=== Per-Class Metrics ===")
for idx, class_name in enumerate(class_names):
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
plt.title('Federated Logistic Regression Ensemble Performance Over Rounds')
plt.xlabel('Communication Round')
plt.ylabel('Score (%)')
plt.grid(True)
plt.legend()
plt.show()
