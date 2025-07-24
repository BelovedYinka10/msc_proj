import keras_tuner as kt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("../../EPIC/dataset_EPICA_raw 1.csv")

# Preprocessing
def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    features = data.drop(columns=['status'])
    labels = data['status']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels

X, y = preprocess_epica(df)
y_encoded = pd.get_dummies(y)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
input_dim = X.shape[1]
output_dim = y_encoded.shape[1]
class_labels = y_encoded.columns.tolist()

# Build CNN model for tuning
def build_model(hp):
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

    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Custom tuner
class MyTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        kwargs['batch_size'] = hp.get('batch_size')
        return super().run_trial(trial, *args, **kwargs)

# Tuner setup
tuner = MyTuner(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='kt_results',
    project_name='epica_cnn_tuning'
)

# Early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Run search
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[stop_early], verbose=1)

# Extract best model and params
best_hps = tuner.get_best_hyperparameters(1)[0]
print("\n✅ Best Hyperparameters:")
for param in best_hps.values.keys():
    print(f"{param}: {best_hps.get(param)}")

best_model = tuner.hypermodel.build(best_hps)
best_batch_size = best_hps.get('batch_size')
print(f"\n📦 Best Batch Size Used in Final Training: {best_batch_size}")

# Final training
history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=best_batch_size,
    verbose=1
)

# Final evaluation
y_pred = np.argmax(best_model.predict(X_val), axis=1)
y_true = np.argmax(y_val.values, axis=1)

print("\n=== Final Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_labels))
