import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and preprocess the data
train_data = pd.read_csv("KDD Train+.csv")

test_data = pd.read_csv("KDDTest+.csv")

train_features = train_data.iloc[:, 0:41]
train_labels = train_data.iloc[:, 42]

test_features = test_data.iloc[:, 0:41]
test_labels = test_data.iloc[:, 42]

# Drop categorical features
train_features = train_features.drop(['protocol_type', 'service', 'flag'], axis=1)
test_features = test_features.drop(['protocol_type', 'service', 'flag'], axis=1)

# Scale features
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Step 2: Split training data into 2 clients
client1_data, client2_data, client1_labels, client2_labels = train_test_split(
    train_features, train_labels, test_size=0.5, random_state=42
)

# Step 3: Define model architecture
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(38,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Helper functions
def get_weights(model):
    return model.get_weights()

def set_weights(model, weights):
    model.set_weights(weights)

def average_weights(weights_list):
    avg_weights = list()
    for weights in zip(*weights_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights

# Step 4: Create Global Model and Client Models
global_model = build_model()
client1_model = build_model()
client2_model = build_model()

# Initialize clients with global model weights
set_weights(client1_model, get_weights(global_model))
set_weights(client2_model, get_weights(global_model))

# Step 5: Training loop
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n--- Round {round_num} ---")
    
    # Each client trains locally (continues training)
    client1_model.fit(client1_data, client1_labels, epochs=LOCAL_EPOCHS, batch_size=64, verbose=0)
    client2_model.fit(client2_data, client2_labels, epochs=LOCAL_EPOCHS, batch_size=64, verbose=0)
    
    # Each client sends updated weights
    client1_weights = get_weights(client1_model)
    client2_weights = get_weights(client2_model)
    
    # Server averages the weights
    new_global_weights = average_weights([client1_weights, client2_weights])
    
    # Update global model
    set_weights(global_model, new_global_weights)

# Step 6: Final Evaluation on Test Data

print("\n=== Final Evaluation on Test Data ===")

# Evaluate Client 1
loss_c1, acc_c1 = client1_model.evaluate(test_features, test_labels, verbose=0)
print(f"Client 1 Final Test Accuracy: {acc_c1*100:.2f}%")

# Evaluate Client 2
loss_c2, acc_c2 = client2_model.evaluate(test_features, test_labels, verbose=0)
print(f"Client 2 Final Test Accuracy: {acc_c2*100:.2f}%")

# Evaluate Global Model
loss_g, acc_g = global_model.evaluate(test_features, test_labels, verbose=0)
print(f"Global Model Final Test Accuracy: {acc_g*100:.2f}%")

# Step 7: Confusion Matrix for Global Model
test_predictions = (global_model.predict(test_features) > 0.5).astype(int)

cm = confusion_matrix(test_labels, test_predictions)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Final Global Model (Federated Learning)")
plt.show()
