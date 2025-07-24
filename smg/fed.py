import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Load dataset
df = pd.read_csv("../EPIC/dataset_EPICA_raw 1.csv")

# Preprocessing
def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1].dropna()
    X = data.drop(columns=['status'])
    y = data['status']
    X_scaled = MinMaxScaler().fit_transform(X)
    return X_scaled, y

X, y = preprocess_epica(df)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
y_categorical = pd.get_dummies(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Define model builder
def model_builder(hp, output_dim, binary=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hp.Int('units1', 128, 512, step=64), activation='relu', input_shape=(X.shape[1],)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))

    model.add(tf.keras.layers.Dense(hp.Int('units2', 64, 256, step=64), activation='relu'))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))

    model.add(tf.keras.layers.Dense(output_dim, activation='sigmoid' if binary else 'softmax'))
    
    loss = 'binary_crossentropy' if binary else 'categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log'))
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# Stage 1: Multi-class Hyperband search
def run_hyperband_multiclass():
    tuner = kt.Hyperband(lambda hp: model_builder(hp, y_train.shape[1]),
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='hyperband_tuning',
                         project_name='multi_class')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    tuner.search(X_train, y_train, epochs=20, validation_split=0.2, callbacks=[stop_early], verbose=0)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("✅ Best multi-class hyperparameters:")
    for k, v in best_hp.values.items():
        print(f"{k}: {v}")
    return best_hp

# Stage 2: Binary Hyperband search
def run_hyperband_binary(normal_class_index):
    y_bin_train = (np.argmax(y_train.values, axis=1) != normal_class_index).astype(int)
    y_bin_test = (np.argmax(y_test.values, axis=1) != normal_class_index).astype(int)

    tuner = kt.Hyperband(lambda hp: model_builder(hp, 1, binary=True),
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='hyperband_tuning',
                         project_name='binary_class')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    tuner.search(X_train, y_bin_train, epochs=20, validation_split=0.2, callbacks=[stop_early], verbose=0)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("✅ Best binary-class hyperparameters:")
    for k, v in best_hp.values.items():
        print(f"{k}: {v}")
    return best_hp

# Execute both stages
normal_class_index = list(class_names).index('Normal')
multi_best_hp = run_hyperband_multiclass()
binary_best_hp = run_hyperband_binary(normal_class_index)
