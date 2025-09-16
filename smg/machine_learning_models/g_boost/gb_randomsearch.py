import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json

# Load dataset
df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")

# Preprocess (same style as your DT code)
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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Search space (GB)
param_dist = {
    'n_estimators': [100, 150, 200, 250, 300, 400],
    'learning_rate': np.linspace(0.03, 0.2, 10),
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 5, 10, 15, 20, 25],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10],
    'subsample': [0.6, 0.8, 1.0],
    'max_features': [None, 'sqrt', 'log2']
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rand = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True
)

print("Starting hyperparameter search (Gradient Boosting)...")
rand.fit(X_train, y_train)

print("\n=== Randomized Search Results (Gradient Boosting) ===")
print("Best Params:", rand.best_params_)
print(f"Best CV f1_macro: {rand.best_score_:.4f}")

# Save only the best parameters & CV score (NO training/testing times here)
best_params = rand.best_params_
json_params = {}
for k, v in best_params.items():
    if isinstance(v, np.floating):
        json_params[k] = float(v)
    elif isinstance(v, np.integer):
        json_params[k] = int(v)
    else:
        json_params[k] = v

with open('best_gb_hyperparameters.json', 'w') as f:
    json.dump({
        'best_parameters': json_params,
        'best_cv_score': float(rand.best_score_),
        'model_type': 'GradientBoostingClassifier',
        'class_names': list(map(str, class_names))
    }, f, indent=2)

print("✅ Saved best params to 'best_gb_hyperparameters.json'")
