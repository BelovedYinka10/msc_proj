import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load EPIC dataset
# df = pd.read_csv("dataset_EPICA_raw 1.csv")
df = pd.read_csv("../EPIC/dataset_EPICA_raw 1.csv")


# Preprocess EPICA dataset
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
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
y_categorical = pd.get_dummies(y_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Global F1 score threshold
F_MIN = 0.81

# Helper: build improved DNN model
def build_dnn(input_dim, output_dim, binary=False):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim, activation='sigmoid' if binary else 'softmax'))
    loss = 'binary_crossentropy' if binary else 'categorical_crossentropy'
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
    return model

# Helper: evaluate model
def evaluate_model(model, X_test, y_test, class_names, is_binary=False):
    y_pred_probs = model.predict(X_test)
    if is_binary:
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = y_test.values.flatten()
        f1s = f1_score(y_true, y_pred, average=None, zero_division=1)
        report = classification_report(y_true, y_pred, target_names=['Normal', 'Attack'], zero_division=1)
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test.values, axis=1)
        f1s = f1_score(y_true, y_pred, average=None, zero_division=1)
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=1)
    return f1s, report

# === Stage 1 ===
print("\nStage 1: Multi-class DNN training...")
model_multi = build_dnn(X.shape[1], y_train.shape[1])
model_multi.fit(X_train, y_train, epochs=30, batch_size=128, validation_split=0.1, verbose=0)
f1_scores, report = evaluate_model(model_multi, X_test, y_test, class_names)
print("F1 scores:", f1_scores)
print(report)

if all(f1 >= F_MIN for f1 in f1_scores):
    print("✅ Multi-class model passed.")
else:
    print("\n❌ Multi-class model did not meet F_min. Proceeding to binary classification...")

    # === Stage 2 ===
    print("\nStage 2: Binary DNN - Normal vs Attack")
    normal_class = list(class_names).index('Normal')
    y_binary = (np.argmax(y_train.values, axis=1) != normal_class).astype(int)
    y_test_bin = (np.argmax(y_test.values, axis=1) != normal_class).astype(int)

    model_bin = build_dnn(X.shape[1], 1, binary=True)
    model_bin.fit(X_train, y_binary, epochs=30, batch_size=128, validation_split=0.1, verbose=0)
    f1_scores_bin, report_bin = evaluate_model(model_bin, X_test, pd.DataFrame(y_test_bin), class_names, is_binary=True)
    print("Binary F1 scores:", f1_scores_bin)
    print(report_bin)

    if all(f1_scores_bin >= F_MIN):
        print("\n✅ Binary model passed. Proceeding to attack-type classification...")

        # === Stage 3 ===
        print("\nStage 3: Recursive classification on attack types...")
        attack_mask = (np.argmax(y_train.values, axis=1) != normal_class)
        X_attack = X_train[attack_mask]
        y_attack = np.argmax(y_train.values[attack_mask], axis=1)

        while len(np.unique(y_attack)) > 2:
            model = build_dnn(X_attack.shape[1], len(np.unique(y_attack)))
            y_attack_oh = pd.get_dummies(y_attack)
            model.fit(X_attack, y_attack_oh, epochs=30, batch_size=128, validation_split=0.1, verbose=0)
            y_pred = np.argmax(model.predict(X_attack), axis=1)
            f1s = f1_score(y_attack, y_pred, average=None, zero_division=1)

            worst_class = np.unique(y_attack)[np.argmin(f1s)]
            print(f"Recursive: Class {class_names[worst_class]} has lowest F1: {f1s.min():.4f}")

            is_cmr = (y_attack == worst_class).astype(int)
            model_bin = build_dnn(X_attack.shape[1], 1, binary=True)
            model_bin.fit(X_attack, is_cmr, epochs=30, batch_size=128, validation_split=0.1, verbose=0)
            y_pred_bin = (model_bin.predict(X_attack) > 0.5).astype(int).flatten()
            f1_bin = f1_score(is_cmr, y_pred_bin, zero_division=1)
            print(f"Binary F1 Score (Class {class_names[worst_class]} vs others): {f1_bin:.4f}")

            if f1_bin >= F_MIN:
                print("✅ Class accepted, removing and continuing.")
                keep_mask = y_attack != worst_class
                X_attack = X_attack[keep_mask]
                y_attack = y_attack[keep_mask]
            else:
                print("❌ Binary classifier did not meet F_min. Stopping recursion.")
                break

        if len(np.unique(y_attack)) == 2:
            print("\nFinal binary classification for last two attack classes...")
            binary_labels = (y_attack == np.unique(y_attack)[0]).astype(int)
            model_final = build_dnn(X_attack.shape[1], 1, binary=True)
            model_final.fit(X_attack, binary_labels, epochs=30, batch_size=128, validation_split=0.1, verbose=0)
            y_pred_final = (model_final.predict(X_attack) > 0.5).astype(int).flatten()
            final_f1 = f1_score(binary_labels, y_pred_final, zero_division=1)
            print(f"Final binary F1 score: {final_f1:.4f}")

    else:
        print("❌ Binary model failed. Cannot proceed to Stage 3.")
