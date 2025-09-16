import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler  # ✅ Using RUS

method_name = "RandomUnderSampler"

# -------------------------------------
# Load and preprocess dataset
# -------------------------------------
df = pd.read_csv("../../../../EPIC/dataset_EPICA_raw 1.csv")

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

# Encode labels as integers
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)
class_labels = list(label_encoder.classes_)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_int, test_size=0.3, random_state=42, stratify=y_int
)

# -------------------------------------
# Apply RandomUnderSampler
# -------------------------------------
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

# Convert labels to one-hot encoding for NN
num_classes = len(class_labels)
y_train_res_onehot = tf.keras.utils.to_categorical(y_train_res, num_classes=num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# -------------------------------------
# Model parameters
# -------------------------------------
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"]) if params["optimizer"] == "adam" else tf.keras.optimizers.RMSprop(learning_rate=params["lr"])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_mlp(X_train_res.shape[1], num_classes)

start_train = time.time()
history = model.fit(
    X_train_res, y_train_res_onehot,
    epochs=10, batch_size=params["batch_size"],
    validation_split=0.1, verbose=1
)
end_train = time.time()

# -------------------------------------
# Plot accuracy and loss
# -------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title(f'Training and Validation Accuracy ({method_name})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'Training and Validation Loss ({method_name})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------
# Predictions
# -------------------------------------
start_test = time.time()
y_pred = np.argmax(model.predict(X_test), axis=1)
end_test = time.time()

# -------------------------------------
# Evaluation metrics
# -------------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\n=== Timing ===")
print(f"Training Time: {end_train - start_train:.2f} seconds")
print(f"Testing Time:  {end_test - start_test:.2f} seconds")

print("\n=== Overall Metrics ===")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision (macro): {prec * 100:.2f}%")
print(f"Recall (macro): {rec * 100:.2f}%")
print(f"F1 Score (macro): {f1 * 100:.2f}%")

# Per-class metrics
cm = confusion_matrix(y_test, y_pred)
print("\n=== Per-Class Metrics ===")
for i, class_name in enumerate(class_labels):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    acc_c = (TP + TN) / cm.sum()
    prec_c = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec_c = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0

    print(f"\nClass: {class_name}")
    print(f"  Accuracy: {acc_c * 100:.2f}%")
    print(f"  Precision: {prec_c * 100:.2f}%")
    print(f"  Recall:    {rec_c * 100:.2f}%")
    print(f"  F1-score:  {f1_c * 100:.2f}%")
