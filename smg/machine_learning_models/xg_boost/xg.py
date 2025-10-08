import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load your dataset
df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")

# Preprocess data (unchanged)
def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]  # remove constant columns
    data = data.dropna()
    features = data.drop(columns=['status'])
    labels = data['status']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels

# Apply preprocessing
X, y = preprocess_epica(df)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# ----- Randomized Search on XGBoost -----
param_dist = {
    'n_estimators': [100, 150, 200, 250, 300, 400],
    'learning_rate': np.linspace(0.03, 0.2, 10),  # finer range
    'max_depth': [2, 3, 4, 5, 6],
    'min_child_weight': [1, 3, 5, 7],  # XGBoost equivalent to min_samples_leaf
    'subsample': [0.6, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],  # XGBoost equivalent to max_features
    'colsample_bylevel': [0.6, 0.8, 1.0],  # Additional XGBoost parameter
    'reg_alpha': [0, 0.1, 0.5, 1],  # L1 regularization
    'reg_lambda': [1, 1.5, 2, 5],  # L2 regularization
    'gamma': [0, 0.1, 0.2, 0.5]  # Minimum loss reduction for further split
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rand = RandomizedSearchCV(
    estimator=XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',  # Suppress warning for multiclass
        objective='multi:softprob' if len(class_names) > 2 else 'binary:logistic'
    ),
    param_distributions=param_dist,
    n_iter=50,               # try 50 random combos (tweak if needed)
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True
)

rand.fit(X_train, y_train)

print("\n=== Randomized Search Results (XGBoost) ===")
print("Best Params:", rand.best_params_)
print(f"Best CV f1_macro: {rand.best_score_:.4f}")

# Use best model
clf = rand.best_estimator_

# Predict
y_pred = clf.predict(X_test)

# Evaluate overall metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# Classification report
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)
class_totals = conf_matrix.sum(axis=1)

print("\n=== Overall Metrics ===")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision (macro): {prec * 100:.2f}%")
print(f"Recall (macro): {rec * 100:.2f}%")
print(f"F1 Score (macro): {f1 * 100:.2f}%")

print("\n=== Per-Class Metrics ===")
for idx, class_name in enumerate(class_names):
    tp = conf_matrix[idx][idx]
    total = class_totals[idx]
    approx_acc = tp / total * 100 if total > 0 else 0  # equals recall per class
    precision = report[class_name]['precision'] * 100
    recall = report[class_name]['recall'] * 100
    f1_score_val = report[class_name]['f1-score'] * 100
    print(f"\nClass: {class_name}")
    print(f"  Accuracy:  {approx_acc:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print(f"  F1-score:  {f1_score_val:.2f}%")

# Optional: Feature importance analysis
print("\n=== Feature Importance (Top 10) ===")
if hasattr(clf, 'feature_importances_'):
    # Get feature names if available, otherwise use indices
    try:
        feature_names = df.drop(columns=['status', 'Unnamed: 0', 'Unnamed: 0.1'], errors='ignore').columns
    except:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(10).to_string(index=False))

# Optional: Learning curve visualization (if you want to plot)
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

# Plot validation curve for n_estimators
param_range = [50, 100, 150, 200, 250, 300]
train_scores, test_scores = validation_curve(
    XGBClassifier(random_state=42), X_train, y_train,
    param_name='n_estimators', param_range=param_range,
    cv=3, scoring='f1_macro', n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(param_range, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.xlabel('Number of Estimators')
plt.ylabel('F1 Score (Macro)')
plt.title('XGBoost Validation Curve')
plt.legend()
plt.grid(True)
plt.show()
"""