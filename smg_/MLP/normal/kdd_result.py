import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load NSL-KDD dataset
train_df = pd.read_csv("../../../KDD/KDD Train+.csv")
test_df = pd.read_csv("../../../KDD/KDDTest+.csv")

# Combine train and test for preprocessing
full_df = pd.concat([train_df, test_df], ignore_index=True)

# Drop categorical columns
features = full_df.iloc[:, 0:41].drop(columns=['protocol_type', 'service', 'flag'], errors='ignore')
labels = full_df.iloc[:, 41]

# Encode labels to integers then one-hot
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = pd.get_dummies(labels_encoded)

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, labels_onehot, test_size=0.3, random_state=42
)

# Model configuration
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

def create_mlp(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(320, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(224, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0011140493525027336)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_mlp(input_dim, output_dim)

# Training
start_train = time.time()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
end_train = time.time()

# Accuracy plot
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Loss plot
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
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
class_labels = le.classes_.tolist()

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
    print(f"\nClass: {class_name}")
    print(f"  Accuracy: {(metrics['recall'] * metrics['precision'])**0.5 * 100:.2f}% (Approx)")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  F1-score:  {metrics['f1-score'] * 100:.2f}%")
