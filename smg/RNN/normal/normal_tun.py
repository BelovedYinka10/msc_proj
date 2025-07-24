import keras_tuner as kt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load dataset
# NOTE: Ensure the path to the CSV file is correct in your environment.
try:
    df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")
except FileNotFoundError:
    print("Error: dataset_EPICA_raw 1.csv not found. Please ensure the path is correct.")
    exit()

# Preprocessing
def preprocess_epica(data):
    """Handles data cleaning, feature selection, and scaling."""
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1] # Remove columns with only one unique value
    data = data.dropna() # Handle missing values
    
    features = data.drop(columns=['status'])
    labels = data['status']
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels

X, y = preprocess_epica(df)
y_encoded = pd.get_dummies(y)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

input_dim = X.shape[1] # Number of features
output_dim = y_encoded.shape[1] # Number of classes
class_labels = y_encoded.columns.tolist()

# Build RNN (LSTM) model for tuning
def build_model(hp):
    """
    Defines the RNN (LSTM) model architecture and hyperparameters for tuning.
    The input data is treated as sequences where each feature is a timestep
    with a single feature value.
    """
    hp.Choice('batch_size', [32, 64, 128, 256])

    model = tf.keras.Sequential()
    # Reshape for LSTM: (batch_size, timesteps, features)
    # Here, input_dim becomes timesteps, and 1 is the feature dimension per timestep.
    model.add(tf.keras.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Reshape((input_dim, 1))) 

    # First LSTM Block
    # return_sequences=True is crucial for stacking LSTM layers
    model.add(tf.keras.layers.LSTM(units=hp.Int('lstm_units1', 32, 128, step=32),
                                   return_sequences=True,
                                   activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout1', 0.1, 0.5, step=0.1)))

    # Second LSTM Block
    # return_sequences=False (default) as this is the last LSTM layer before Dense
    model.add(tf.keras.layers.LSTM(units=hp.Int('lstm_units2', 32, 128, step=32),
                                   activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
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

# Custom tuner (remains the same as batch_size is a hyperparameter)
class MyTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        kwargs['batch_size'] = hp.get('batch_size')
        return super().run_trial(trial, *args, **kwargs)

# Tuner setup
# The directory and project name are updated to reflect the RNN tuning
tuner = MyTuner(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='kt_results_rnn', # New directory for RNN results
    project_name='epica_rnn_tuning' # New project name
)

# Early stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Run search
print("Starting Keras Tuner search for RNN model...")
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[stop_early], verbose=1)

# Extract best model and hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]
print("\n✅ Best Hyperparameters for RNN Model:")
for param in best_hps.values.keys():
    print(f"{param}: {best_hps.get(param)}")

best_model = tuner.hypermodel.build(best_hps)
best_batch_size = best_hps.get('batch_size')
print(f"\n📦 Best Batch Size Used in Final Training: {best_batch_size}")

# Final training of the best RNN model
print("\nStarting final training of the best RNN model...")
history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=best_batch_size,
    verbose=1
)

# Final evaluation
y_pred_probs = best_model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = np.argmax(y_val.values, axis=1)

print("\n=== Final Classification Report for RNN Model ===")
print(classification_report(y_true, y_pred, target_names=class_labels))