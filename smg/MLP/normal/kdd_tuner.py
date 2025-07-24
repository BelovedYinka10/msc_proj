import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import keras_tuner as kt

# Load NSL-KDD dataset
train_df = pd.read_csv("../../../KDD/KDD Train+.csv")
test_df = pd.read_csv("../../../KDD/KDDTest+.csv")

# Combine train and test for preprocessing
full_df = pd.concat([train_df, test_df], ignore_index=True)

# Drop categorical columns
features = full_df.iloc[:, 0:41].drop(columns=['protocol_type', 'service', 'flag'], errors='ignore')
labels = full_df.iloc[:, 41]

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = pd.get_dummies(labels_encoded)

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels_onehot, test_size=0.3, random_state=42)

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

# ✅ Define model builder with batch size as hyperparameter
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        hp.Int('units1', min_value=128, max_value=512, step=64),
        activation='relu',
        input_shape=(input_dim,)
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(
        hp.Int('units2', min_value=64, max_value=256, step=64),
        activation='relu'
    ))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ✅ Tuner setup
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=100,
    factor=3,
    directory='kt_dir',
    project_name='nsl_kdd_tuning_default'
)

# ✅ Early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# ✅ Start tuning
tuner.search(X_train, y_train, validation_split=0.1, callbacks=[stop_early], verbose=1)

# ✅ Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]

print("\n=== Best Hyperparameters Found ===")
print(f"units1:     {best_hps.get('units1')}")
print(f"dropout1:   {best_hps.get('dropout1')}")
print(f"units2:     {best_hps.get('units2')}")
print(f"dropout2:   {best_hps.get('dropout2')}")
print(f"lr:         {best_hps.get('lr')}")

# Note: batch_size is not tunable directly in Hyperband, but we can define it manually here
best_batch_size = 32  # or change manually if desired
print(f"batch_size: {best_batch_size}")

# ✅ Train best model
model = tuner.hypermodel.build(best_hps)
start_train = time.time()
history = model.fit(X_train, y_train, epochs=100, batch_size=best_batch_size, validation_split=0.1, verbose=1)
end_train = time.time()

# ✅ Epochs actually trained (before early stopping)
trained_epochs = len(history.history['loss'])
print(f"Best Epochs Trained: {trained_epochs}")

# ✅ Accuracy plot
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(), plt.title("Accuracy"), plt.show()

# ✅ Loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(), plt.title("Loss"), plt.show()

# ✅ Evaluation
start_test = time.time()
y_pred = np.argmax(model.predict(X_test), axis=1)
end_test = time.time()

y_true = np.argmax(y_test.values, axis=1)
class_labels = le.classes_.tolist()

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

# ✅ Per-class metrics
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
print("\n=== Per-Class Metrics ===")
for class_name in class_labels:
    metrics = report[class_name]
    print(f"\nClass: {class_name}")
    print(f"  Accuracy: {(metrics['recall'] * metrics['precision'])**0.5 * 100:.2f}% (Approx)")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"  F1-score:  {metrics['f1-score'] * 100:.2f}%")
