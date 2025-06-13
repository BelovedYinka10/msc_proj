import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

# === Load Dataset ===
df = pd.read_csv("dataset_EPICA_raw 1.csv")

# === Preprocess ===
def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    X = data.drop(columns=['status'])
    y = data['status']
    return X, y

X, y = preprocess_epica(df)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === Stage 1: Normal vs Attack ===
y_stage1 = y.apply(lambda x: 'Normal' if x == 'Normal' else 'Attack')
y_stage1_encoded = pd.get_dummies(y_stage1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y_stage1_encoded, test_size=0.3, random_state=42)

# === Build MLP Model ===
def build_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model1 = build_model(X_train1.shape[1], y_train1.shape[1])
model1.fit(X_train1, y_train1, epochs=20, batch_size=128, validation_split=0.1, verbose=0)

# === Evaluate Stage 1 ===
pred1 = np.argmax(model1.predict(X_test1), axis=1)
true1 = np.argmax(y_test1.values, axis=1)

print("\n=== Stage 1: Normal vs Attack ===")
print("Accuracy: {:.2f}%".format(accuracy_score(true1, pred1)*100))
print("Precision: {:.2f}%".format(precision_score(true1, pred1, average='macro')*100))
print("Recall: {:.2f}%".format(recall_score(true1, pred1, average='macro')*100))
print("F1 Score: {:.2f}%".format(f1_score(true1, pred1, average='macro')*100))
print("\nClassification Report:\n", classification_report(true1, pred1, target_names=y_stage1_encoded.columns))

# === Stage 2: Attack Type Classification ===
attack_mask = y != 'Normal'
X_attack = X_scaled[attack_mask]
y_attack = y[attack_mask]
y_attack_encoded = pd.get_dummies(y_attack)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_attack, y_attack_encoded, test_size=0.3, random_state=42)

model2 = build_model(X_train2.shape[1], y_train2.shape[1])
model2.fit(X_train2, y_train2, epochs=20, batch_size=128, validation_split=0.1, verbose=0)

# === Evaluate Stage 2 ===
pred2 = np.argmax(model2.predict(X_test2), axis=1)
true2 = np.argmax(y_test2.values, axis=1)

print("\n=== Stage 2: Attack Type Classification ===")
print("Accuracy: {:.2f}%".format(accuracy_score(true2, pred2)*100))
print("Precision: {:.2f}%".format(precision_score(true2, pred2, average='macro')*100))
print("Recall: {:.2f}%".format(recall_score(true2, pred2, average='macro')*100))
print("F1 Score: {:.2f}%".format(f1_score(true2, pred2, average='macro')*100))
print("\nClassification Report:\n", classification_report(true2, pred2, target_names=y_attack_encoded.columns))
