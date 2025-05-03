import pandas as pd
import numpy as np
import time
import io
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
test_data = pd.read_csv("reduced_pre_XIIoTID.csv")
print(test_data.head())
print(test_data.shape)

# Preprocessing function
def preprocess_xiiot(data):
    while True:
        print("Choose the classification scenario:")
        print("1. 2 classes\n2. 10 classes\n3. 19 classes")
        scenario_option = input("Enter the number for the classification scenario: ")
        if scenario_option in ['1', '2', '3']:
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

    if scenario_option == '1':
        my_label = data.iloc[:, 61]  # label_2
    elif scenario_option == '2':
        my_label = data.iloc[:, 60]  # label_10
    else:
        my_label = data.iloc[:, 59]  # label_19

    features = data.iloc[:, :59]
    my_label = pd.get_dummies(my_label)

    X_train, X_test, y_train, y_test = train_test_split(features, my_label, test_size=0.3, random_state=142)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_xiiot(test_data)

# Create model function
def create_model(feature_dim, num_classes):
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(feature_dim,)))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train model
def hold_out_training(model, X_train, y_train):
    start_time = time.time()
    history = model.fit(X_train, y_train, batch_size=250, epochs=10, validation_split=0.3, verbose=1)
    train_time = time.time() - start_time
    return train_time

# Evaluate model
def evaluate_model(model, X_test, y_test, train_time):
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {score[0]: .4f}")
    print(f"Test accuracy: {score[1] * 100:.2f}%")

    start_time = time.time()
    y_pred = np.argmax(model.predict(X_test), axis=1)
    test_time = time.time() - start_time
    y_true = np.argmax(y_test.values, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    return train_time, test_time

# Create and train model
feature_dim = X_train.shape[1]
num_classes = y_test.shape[1]
model = create_model(feature_dim, num_classes)
train_time = hold_out_training(model, X_train, y_train)

# Evaluate model
train_time, test_time = evaluate_model(model, X_test, y_test, train_time)

# =======================
# Add EXPLAINABLE AI (SHAP) using only sample
# =======================
print("\nGenerating SHAP explanations...")

# Sample only 500 rows from X_test for faster SHAP calculation
sampled_X_test = X_test.sample(500, random_state=42)

# Create SHAP explainer and calculate shap values
explainer = shap.Explainer(model, sampled_X_test)
shap_values = explainer(sampled_X_test)

# Summary plot to show feature importance
shap.summary_plot(shap_values, sampled_X_test)

print("\nExplainable AI (SHAP) visualization completed!")
