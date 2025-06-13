import numpy as np
import pandas as pd
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Load and preprocess
df = pd.read_csv("../EPIC/dataset_EPICA_raw 1.csv")
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
df = df.dropna()

# === Stage 1: Normal vs Attack ===
df_stage1 = df.copy()
df_stage1['status'] = df_stage1['status'].replace({'FDIA': 'Attack', 'TDA': 'Attack'})
X1 = df_stage1.drop(columns=['status'])
y1 = df_stage1['status'].map({'Normal': 0, 'Attack': 1}).values
X1 = MinMaxScaler().fit_transform(X1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

class_weights1 = compute_class_weight('balanced', classes=np.unique(y1_train), y=y1_train)
cw_dict1 = dict(enumerate(class_weights1))

model1 = tf.keras.Sequential([
    tf.keras.Input(shape=(X1_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_train1 = time.time()
model1.fit(X1_train, y1_train, epochs=20, batch_size=128, class_weight=cw_dict1, verbose=0)
end_train1 = time.time()

start_test1 = time.time()
y1_pred = (model1.predict(X1_test) > 0.5).astype(int)
end_test1 = time.time()

print("\n=== Stage 1: Normal vs Attack ===")
print(f"Training Time: {end_train1 - start_train1:.2f} seconds")
print(f"Testing Time:  {end_test1 - start_test1:.2f} seconds")
print(f"Accuracy: {accuracy_score(y1_test, y1_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y1_test, y1_pred) * 100:.2f}%")
print(f"Recall:    {recall_score(y1_test, y1_pred) * 100:.2f}%")
print(f"F1 Score:  {f1_score(y1_test, y1_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y1_test, y1_pred, target_names=['Normal', 'Attack']))

# === Stage 2: FDIA vs TDA ===
df_stage2 = df[df['status'] != 'Normal'].copy()
X2 = df_stage2.drop(columns=['status'])
y2 = df_stage2['status'].map({'FDIA': 0, 'TDA': 1}).values
X2 = MinMaxScaler().fit_transform(X2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

class_weights2 = compute_class_weight('balanced', classes=np.unique(y2_train), y=y2_train)
cw_dict2 = dict(enumerate(class_weights2))

model2 = tf.keras.Sequential([
    tf.keras.Input(shape=(X2_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_train2 = time.time()
model2.fit(X2_train, y2_train, epochs=20, batch_size=128, class_weight=cw_dict2, verbose=0)
end_train2 = time.time()

start_test2 = time.time()
y2_pred = (model2.predict(X2_test) > 0.5).astype(int)
end_test2 = time.time()

print("\n=== Stage 2: FDIA vs TDA ===")
print(f"Training Time: {end_train2 - start_train2:.2f} seconds")
print(f"Testing Time:  {end_test2 - start_test2:.2f} seconds")
print(f"Accuracy: {accuracy_score(y2_test, y2_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y2_test, y2_pred) * 100:.2f}%")
print(f"Recall:    {recall_score(y2_test, y2_pred) * 100:.2f}%")
print(f"F1 Score:  {f1_score(y2_test, y2_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y2_test, y2_pred, target_names=['FDIA', 'TDA']))
