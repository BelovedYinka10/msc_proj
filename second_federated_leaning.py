import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix

# Step 1: Load and preprocess the data
data = pd.read_csv("reduced_pre_XIIoTID.csv")

# Preprocessing function (same as yours)
def preprocess_xiiot(data):
    print("Choose the classification scenario:")
    print("1. 2 classes\n2. 10 classes\n3. 19 classes")
    scenario_option = input("Enter the number for the classification scenario: ")
    if scenario_option == '1':
        my_label = data.iloc[:, 61]
    elif scenario_option == '2':
        my_label = data.iloc[:, 60]
    else:
        my_label = data.iloc[:, 59]

    features = data.iloc[:, :59]
    my_label = pd.get_dummies(my_label)
    X_train, X_test, y_train, y_test = train_test_split(features, my_label, test_size=0.3, random_state=142)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_xiiot(data)

# Step 2: Define model architecture
def create_model(feature_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='relu', input_shape=(feature_dim,)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Helper functions for Federated Learning
def get_weights(model):
    return model.get_weights()

def set_weights(model, weights):
    model.set_weights(weights)

def average_weights(weights_list):
    avg_weights = []
    for weights in zip(*weights_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights

# Step 3: Federated Learning Setup
NUM_CLIENTS = 2
NUM_ROUNDS = 5
LOCAL_EPOCHS = 10
feature_dim = X_train.shape[1]
num_classes = y_train.shape[1]

# Split data among clients
client_data = np.array_split(X_train, NUM_CLIENTS)
client_labels = np.array_split(y_train, NUM_CLIENTS)

# Create Global model
global_model = create_model(feature_dim, num_classes)

# Create client models initialized with global model weights
clients = [create_model(feature_dim, num_classes) for _ in range(NUM_CLIENTS)]
for client in clients:
    set_weights(client, get_weights(global_model))

# Training loop
for round_num in range(1, NUM_ROUNDS+1):
    print(f"\n--- Round {round_num} ---")

    # Local training on each client
    for i, client in enumerate(clients):
        print(f"Client {i+1} training...")
        client.fit(client_data[i], client_labels[i], epochs=LOCAL_EPOCHS, batch_size=250, verbose=0)

    # Collect weights
    client_weights = [get_weights(client) for client in clients]

    # Server aggregates the weights
    new_global_weights = average_weights(client_weights)
    set_weights(global_model, new_global_weights)

    # Update all clients with new global weights
    for client in clients:
        set_weights(client, new_global_weights)

# Step 4: Evaluate Global Model
print("\n=== Final Evaluation on Test Data ===")
start_time = time.time()
score = global_model.evaluate(X_test, y_test, verbose=1)
test_time = time.time() - start_time
print(f"Test loss: {score[0]: .4f}")
print(f"Test accuracy: {score[1] * 100:.2f}%")

# Predict and generate confusion matrix
y_pred = np.argmax(global_model.predict(X_test), axis=1)
y_true = np.argmax(np.array(y_test), axis=1)
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Final Global Model')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Extra: Compute TP, FP, FN, TN, Accuracy, Recall, Precision, F1
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (TP + FP + FN)

accuracy = (TP + TN) / (TP + FP + FN + TN)
recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1_score = 2 * (precision * recall) / (precision + recall)

for i in range(len(TP)):
    print(f"Class {i}: Accuracy={accuracy[i]*100:.2f}%, Recall={recall[i]*100:.2f}%, Precision={precision[i]*100:.2f}%, F1 Score={f1_score[i]*100:.2f}%")

print(f"Train time: Not recorded per client in this simple version")

print(f"Test time: {test_time:.4f} seconds")
