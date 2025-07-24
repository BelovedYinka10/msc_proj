import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv("../EPIC/dataset_EPICA_raw 1.csv")

# Preprocess dataset
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
F_MIN = 0.81

# Model builder
def build_dnnm(input_dim, output_dim, binary=False):
    model = Sequential()
    model.add(Dense( 384, activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, activation='sigmoid' if binary else 'softmax'))
    loss = 'binary_crossentropy' if binary else 'categorical_crossentropy'
    model.compile(optimizer=Adam(learning_rate= 0.0008363054140143658), loss=loss, metrics=['accuracy'])
    return model


def build_dnn_binary(input_dim, output_dim, binary=False):
    model = Sequential()
    model.add(Dense(320, activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='sigmoid' if binary else 'softmax'))
    loss = 'binary_crossentropy' if binary else 'categorical_crossentropy'
    model.compile(optimizer=Adam(learning_rate=0.0005167397035568562), loss=loss, metrics=['accuracy'])
    return model

# Evaluator
def evaluate_model(model, X_test, y_test, class_names, is_binary=False):
    y_pred_probs = model.predict(X_test)
    if is_binary:
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = y_test.values.flatten()
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test.values, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=1)
    recall = recall_score(y_true, y_pred, average=None, zero_division=1)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=1)

    results = pd.DataFrame({
        'Class': class_names,
        'Accuracy': [accuracy]*len(class_names),
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })
    return results

# === Stage 1 ===
print("\nStage 1: Multi-class DNN training...")
model_multi = build_dnnm(X.shape[1], y_train.shape[1])
model_multi.fit(X_train, y_train, epochs=30, batch_size=128, validation_split=0.1, verbose=0)
stage1_results = evaluate_model(model_multi, X_test, y_test, class_names)
print(stage1_results)

if all(stage1_results['F1 Score'] >= F_MIN):
    print("\n✅ Multi-class model passed.")
else:
    print("\n❌ Multi-class model did not meet F_min. Proceeding to binary classification...")

    # === Stage 2 ===
    print("\nStage 2: Binary DNN - Normal vs Attack")
    normal_class_idx = list(class_names).index('Normal')
    y_binary = (np.argmax(y_train.values, axis=1) != normal_class_idx).astype(int)
    y_test_bin = (np.argmax(y_test.values, axis=1) != normal_class_idx).astype(int)

    model_bin = build_dnn_binary(X.shape[1], 1, binary=True)
    model_bin.fit(X_train, y_binary, epochs=30, batch_size=128, validation_split=0.1, verbose=0)
    stage2_results = evaluate_model(model_bin, X_test, pd.DataFrame(y_test_bin), ['Normal', 'Attack'], is_binary=True)
    print(stage2_results)

    if all(stage2_results['F1 Score'] >= F_MIN):
        print("\n✅ Binary model passed. Proceeding to attack-type classification...")

        # === Stage 3 ===
        print("\nStage 3: Attack-type classification only")
        attack_mask = (np.argmax(y_train.values, axis=1) != normal_class_idx)
        X_attack_train = X_train[attack_mask]
        y_attack_train = np.argmax(y_train.values[attack_mask], axis=1)
        y_attack_train_onehot = pd.get_dummies(y_attack_train)
        X_attack_test = X_test[(np.argmax(y_test.values, axis=1) != normal_class_idx)]
        y_attack_test = np.argmax(y_test.values, axis=1)[(np.argmax(y_test.values, axis=1) != normal_class_idx)]
        y_attack_test_onehot = pd.get_dummies(y_attack_test)

        attack_class_names = [c for c in class_names if c != 'Normal']

        model_attack = build_dnnm(X.shape[1], len(attack_class_names))
        model_attack.fit(X_attack_train, y_attack_train_onehot, epochs=30, batch_size=128, validation_split=0.1, verbose=0)

        stage3_results = evaluate_model(model_attack, X_attack_test, y_attack_test_onehot, attack_class_names)
        print(stage3_results)

    else:
        print("\n❌ Binary model failed. Cannot proceed to Stage 3.")
