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


from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

train_data = pd.read_csv("KDD Train+.csv")

test_data = pd.read_csv("KDDTest+.csv")

train_features = train_data.iloc[:, 0:41]

test_features = test_data.iloc[:, 0:41]

train_label = train_data.iloc[:, 42]

test_label = test_data.iloc[:, 42]

train_features = train_features.drop(['protocol_type', 'service', 'flag'], axis=1)

test_features = test_features.drop(['protocol_type', 'service', 'flag'], axis=1)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_train_features = scaler.fit_transform(train_features)

scaled_test_features = scaler.fit_transform(test_features)

model = Sequential()

model.add(Dense(256, input_dim=38, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(scaled_train_features, train_label, batch_size=64, epochs=10)

_, accuracy = model.evaluate(scaled_train_features, train_label)

print('Accuracy: %.2f' % (accuracy*100))

test_predictions = (model.predict(scaled_test_features)> 0.5).astype(int)

test_predictions.shape

from sklearn.metrics import confusion_matrix

confusion_matrix(test_label, test_predictions)

true_negative, false_positive, false_negative, true_positive = confusion_matrix(test_label, test_predictions).ravel()


for i in range(15):
  print('%d (expected %d)' % (test_predictions[i], test_label[i]))



# Calculate the confusion matrix
cm = confusion_matrix(test_label, test_predictions)
# Plot the confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted 0", "Predicted1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()