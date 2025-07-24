import keras_tuner as kt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load dataset
# NOTE: Ensure the path to the CSV file is correct in your environment.
# Assuming the file path provided in the original script is correct relative to the execution environment.
try:
    df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")
except FileNotFoundError:
    print("Error: dataset_EPICA_raw 1.csv not found. Please ensure the path is correct.")
    exit()

# Preprocessing
def preprocess_epica(data):
    """Handles data cleaning, feature selection, and scaling."""
    # Drop unnecessary columns and columns with only one unique value
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    
    # Handle missing values
    data = data.dropna()
    
    # Separate features and labels
    features = data.drop(columns=['status'])
    labels = data['status']
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, labels

# Apply preprocessing and one-hot encode labels
X, y = preprocess_epica(df)
y_encoded = pd.get_dummies(y)
class_labels = y_encoded.columns.tolist()

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --- ADASYN Oversampling on Training Data ---
print("Applying ADASYN oversampling to the training data...")

# 1. Convert one-hot encoded y_train back to labels (imblearn requires integer/string labels)
# We use .values for numpy array conversion, then np.argmax to get the index of the '1' in the one-hot vector.
y_train_labels = np.argmax(y_train.values, axis=1)

# 2. Apply ADASYN
adasyn = SMOTE(random_state=42)
X_train_resampled, y_train_resampled_labels = adasyn.fit_resample(X_train, y_train_labels)

# 3. Convert resampled labels back to one-hot encoding for Keras
# We use tf.keras.utils.to_categorical to convert integer labels back to one-hot encoding.
# The number of classes must match the original output dimension.
output_dim = y_encoded.shape[1]
y_train_resampled_encoded = tf.keras.utils.to_categorical(y_train_resampled_labels, num_classes=output_dim)

print(f"Original training shape: X={X_train.shape}, y={y_train.shape}")
print(f"Resampled training shape: X={X_train_resampled.shape}, y={y_train_resampled_encoded.shape}")
print("-" * 50)

# --- Model Definition and Keras Tuner Setup ---
input_dim = X.shape[1]

# Build CNN model for tuning
def build_model(hp):
    """Defines the CNN model architecture and hyperparameters for tuning."""
    # This choice is handled within the MyTuner class, but defined here for Keras Tuner tracking.
    hp.Choice('batch_size', [32, 64, 128, 256])

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Reshape((input_dim, 1)))  # Reshape for Conv1D

    # First Conv Block
    model.add(tf.keras.layers.Conv1D(filters=hp.Int('filters1', 32, 128, step=32),
                                     kernel_size=3,
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout1', 0.1, 0.5, step=0.1)))

    # Second Conv Block
    model.add(tf.keras.layers.Conv1D(filters=hp.Int('filters2', 32, 128, step=32),
                                     kernel_size=3,
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout2', 0.1, 0.5, step=0.1)))

    # Dense layer
    model.add(tf.keras.layers.Dense(units=hp.Int('dense_units', 64, 256, step=64), activation='relu'))
    model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))

    # Optimizer and Learning Rate
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Custom tuner class to handle the 'batch_size' choice
class MyTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        # We extract 'batch_size' from hyperparameters and pass it to the fit() call within run_trial
        kwargs['batch_size'] = hp.get('batch_size')
        return super().run_trial(trial, *args, **kwargs)

# Tuner setup
tuner = MyTuner(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='kt_results_adasyn',
    project_name='epica_cnn_tuning_adasyn'
)

# Early stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Run search using the RESAMPLED training data and the original validation data
print("Starting Keras Tuner search with ADASYN oversampled training data...")
tuner.search(X_train_resampled, y_train_resampled_encoded, 
             validation_data=(X_val, y_val), 
             epochs=10, 
             callbacks=[stop_early], 
             verbose=1)

# Extract best model and hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]
print("\n✅ Best Hyperparameters (found using oversampled training data):")
for param in best_hps.values.keys():
    print(f"{param}: {best_hps.get(param)}")

# Build and train the final best model
best_model = tuner.hypermodel.build(best_hps)
best_batch_size = best_hps.get('batch_size')
print(f"\n📦 Best Batch Size Used in Final Training: {best_batch_size}")

print("\nStarting final training of the best model using ADASYN oversampled data...")
# Train the best model on the RESAMPLED data
history = best_model.fit(
    X_train_resampled, y_train_resampled_encoded,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=best_batch_size,
    verbose=1
)

# --- Final Evaluation ---

# Predict on the validation set (original data, not resampled)
y_pred_probs = best_model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert y_val (one-hot encoded) back to labels for evaluation
y_true = np.argmax(y_val.values, axis=1)

print("\n=== Final Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_labels))