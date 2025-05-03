import pandas as pd
import io

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix

import pandas as pd
import io

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import io

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

test_data = pd.read_csv("reduced_pre_XIIoTID.csv")

print(test_data.head())

print(test_data.shape)

from sklearn.model_selection import train_test_split

def preprocess_xiiot(data):
    while True:
        # Prompt user to choose the classification scenario
        print("Choose the classification scenario:")
        print("1. 2 classes\n2. 10 classes\n3. 19 classes")

        # Take user input for the classification scenario
        scenario_option = input("Enter the number for the classification scenario: ")

        # Check if the input is valid (1, 2, or 3)
        if scenario_option in ['1', '2', '3']:
            break  # Break the loop if the input is valid
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

    # Logic to select the label based on the chosen scenario
    if scenario_option == '1':
        my_label = data.iloc[:, 61]  # Choose label_2 for 2 classes
    elif scenario_option == '2':
        my_label = data.iloc[:, 60]  # Choose label_10 for 10 classes
    else:
        my_label = data.iloc[:, 59]  # Choose label_19 for 19 classes

    features = data.iloc[:, :59]
    my_label = pd.get_dummies(my_label)

    # split data into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(features, my_label, test_size=0.3, random_state=142)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test =preprocess_xiiot(test_data)


print(
   { 
    "X_train" : X_train,
    "X_test" : X_test,
    "y_train" : y_train,
    "y_test" : y_test
   }
)


import matplotlib.pyplot as plt
import numpy as np

def descriptive_stats(X_train, X_test, y_train, y_test):
    # Summary statistics for features in X_train
    summary_stats = X_train.describe()
    print(summary_stats)

    # Plot histograms for each feature in X_train
    for column in X_train.columns:
        pass
        # plt.figure(figsize=(6, 4))
        # plt.hist(X_train[column], bins=20, color='skyblue', edgecolor='black')
        # plt.title(f'Distribution of {column}')
        # plt.xlabel('Values')
        # plt.ylabel('Frequency')
        # plt.grid(True)
        # plt.show()

    # Unique classes and their counts in y_train
    unique_classes_train, counts_train = np.unique(np.argmax(y_train.values, axis=1), return_counts=True)
    class_counts_train = dict(zip(unique_classes_train, counts_train))

    print("Classes and their counts in y_train:")
    for class_index, count in class_counts_train.items():
        print(f"Class {class_index}: Count - {count}")

    # Unique classes and their counts in y_test
    unique_classes_test, counts_test = np.unique(np.argmax(y_test.values, axis=1), return_counts=True)
    class_counts_test = dict(zip(unique_classes_test, counts_test))

    print("\nClasses and their counts in y_test:")
    for class_index, count in class_counts_test.items():
        print(f"Class {class_index}: Count - {count}")

    return summary_stats


# Descriptive Statistics
summary_stats = descriptive_stats(X_train, X_test, y_train, y_test)




# Define the deep neural network model
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


num_classes = y_test.shape[1]

feature_dim = X_train.shape[1]

# Create the deep neural network model
model = create_model(feature_dim, num_classes)


import time

def hold_out_training(model, X_train, y_train):
    start_time = time.time()
    history = model.fit(X_train, y_train, batch_size=250, epochs=10, validation_split=0.3, verbose=1)
    train_time = time.time() - start_time

    # Plot training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_loss = range(1, len(loss) + 1)
    # plt.figure(1)
    # plt.plot(epochs_loss, loss, 'bo-', label='Training Loss')
    # plt.plot(epochs_loss, val_loss, 'r*-', label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(color='black', linestyle='--', linewidth=1)
    # plt.show()

    return train_time



# Train the model
train_time = hold_out_training(model, X_train, y_train)



from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_model(model, X_test, y_test, train_time):
    # Evaluate the server model on the test data
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {score[0]: .4f}")
    print(f"Test accuracy: {score[1] * 100:.2f}%")

    # Evaluate the server model on the test data
    start_time = time.time()
    y_pred = np.argmax(model.predict(X_test), axis=1)
    test_time = time.time() - start_time
    y_test = np.array(y_test)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Compute TP, FP, FN, TN for each class
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    # Compute accuracy, recall, precision, and F1 score for each class
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Print the results for each class
    for i in range(len(TP)):
        print(f"Class {i+1}:")
        print(f"TP: {TP[i]}")
        print(f"FP: {FP[i]}")
        print(f"FN: {FN[i]}")
        print(f"TN: {TN[i]}")
        print(f"Accuracy: {accuracy[i]* 100:.2f}%")
        print(f"Recall: {recall[i]* 100:.2f}%")
        print(f"Precision: {precision[i]* 100:.2f}%")
        print(f"F1 score: {f1_score[i]* 100:.2f}%")

    print(f"Train time: {train_time : .4f}")
    print(f"Test time: {test_time : .4f}")

    return accuracy, recall, precision, f1_score, train_time, test_time


# Evaluate the performance of the model
accuracy, recall, precision, f1_score, train_time, test_time = evaluate_model(model, X_test, y_test, train_time)


print("recall",recall)
print("precision",precision)
print("accuracy",accuracy)



print(
    {
        "summary_stats" : summary_stats
    }
)