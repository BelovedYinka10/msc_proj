import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import TomekLinks  # ✅ Updated import

nm = "TomekLinks"  # ✅ Updated method label

# Load dataset
df = pd.read_csv("../../../../EPIC/dataset_EPICA_raw 1.csv")

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

# Encode labels
y_encoded = pd.get_dummies(y)

# Split data
X_train, X_test, y_train_onehot, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
y_train = y_train_onehot.idxmax(axis=1)  # TomekLinks requires labels not one-hot

# Apply TomekLinks
tomek = TomekLinks()
X_train_res, y_train_res = tomek.fit_resample(X_train, y_train)

# Re-encode labels after resampling
y_train_res_onehot = pd.get_dummies(y_train_res)

# Model configuration using best hyperparameters
input_dim = X_train_res.shape[1]
output_dim = y_train_res_onehot.shape[1]

params = {
    "batch_size": 256,
    "units1": 320,
    "dropout1": 0.1,
    "units2": 224,
    "dropout2": 0.1,
    "optimizer": "adam",
    "lr": 0.0014267123289125915
}

def create_mlp(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(params.get("units1"), activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params["dropout1"]),
        tf.keras.layers.Dense(params["units2"], activation='relu'),
        tf.keras.layers.Dropout(params["dropout2"]),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    optimizer = None

    if params["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"])
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=params["lr"])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_mlp(input_dim, output_dim)

# Training
start_train = time.time()
history = model.fit(X_train_res, y_train_res_onehot, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
end_train = time.time()

# Accuracy plot
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title(f'Training and Validation Accuracy {nm}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Loss plot
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'Training and Validation Loss {nm}')
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
class_labels = y_test.columns.tolist()

# Evaluation
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

# Per-class metrics
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
print("\n=== Per-Class Metrics ===")
for class_name in class_labels:
    metrics = report[class_name]
    approx_acc = (metrics['recall'] * metrics['precision'])**0.5 * 100
    print(f"\nClass: {class_name}")
    print(f"  Accuracy: {approx_acc:.2f}% (Approx)")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  F1-score:  {metrics['f1-score'] * 100:.2f}%")
