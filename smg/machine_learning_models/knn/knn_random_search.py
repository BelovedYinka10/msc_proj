import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json

# Load dataset
df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")

# Preprocess
def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    features = data.drop(columns=['status'])
    labels = data['status']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)
    return X, labels

X, y = preprocess_epica(df)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Random Search for KNN
param_distributions = {
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    "weights": ["uniform", "distance"],
    "metric": ["minkowski"],   # use p to switch L1/L2
    "p": [1, 2],
    "leaf_size": [15, 20, 30, 40, 50]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rand = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions=param_distributions,
    n_iter=40,                 # random combos
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True
)

print("Starting hyperparameter search (KNN, RandomizedSearchCV)...")
rand.fit(X_train, y_train)

print("\n=== Randomized Search Results (KNN) ===")
print("Best Params:", rand.best_params_)
print(f"Best CV f1_macro: {rand.best_score_:.4f}")

# Save only best params & CV score
best_params = rand.best_params_
json_params = {}
for k, v in best_params.items():
    if isinstance(v, np.floating):
        json_params[k] = float(v)
    elif isinstance(v, np.integer):
        json_params[k] = int(v)
    else:
        json_params[k] = v

with open('best_knn_hyperparameters.json', 'w') as f:
    json.dump({
        'best_parameters': json_params,
        'best_cv_score': float(rand.best_score_),
        'model_type': 'KNeighborsClassifier',
        'class_names': list(map(str, class_names))
    }, f, indent=2)

print("✅ Saved best params to 'best_knn_hyperparameters.json'")
