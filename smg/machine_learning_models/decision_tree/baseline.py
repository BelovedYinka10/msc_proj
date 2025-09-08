import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load your dataset
df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")

# Preprocess data
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

# ----- Randomized Search on Decision Tree -----
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, 15, 20, 25, 30, None],  # None means unlimited depth
    'min_samples_split': [2, 5, 10, 15, 20, 25],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10],
    'max_features': [None, 'sqrt', 'log2'],  # Feature selection at each split
    'class_weight': [None, 'balanced'],  # Handle class imbalance
    'splitter': ['best', 'random'],  # Strategy for choosing splits
    'max_leaf_nodes': [None, 10, 20, 50, 100, 200]  # Control tree size
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rand = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,  # Try 100 random combinations (Decision Trees are fast)
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True
)

print("Starting hyperparameter optimization...")
rand.fit(X_train, y_train)

print("\n=== Randomized Search Results (Decision Tree) ===")
print("Best Params:", rand.best_params_)
print(f"Best CV f1_macro: {rand.best_score_:.4f}")

# ============================================
# EXTRACT BEST HYPERPARAMETERS FOR EASY USE
# ============================================

best_params = rand.best_params_

print("\n" + "="*60)
print("🎯 BEST HYPERPARAMETERS FOR SAMPLING TECHNIQUES")
print("="*60)

print("\n📋 Copy-Paste Ready Configuration:")
print("-" * 40)
print("# Best hyperparameters found:")
for param, value in best_params.items():
    if isinstance(value, float):
        print(f"{param} = {value:.6f}")
    else:
        print(f"{param} = {value}")

print(f"\n🔧 Ready-to-Use DecisionTreeClassifier:")
print("-" * 50)
print("optimal_dt_classifier = DecisionTreeClassifier(")
for i, (param, value) in enumerate(best_params.items()):
    comma = "," if i < len(best_params) - 1 else ""
    if isinstance(value, float):
        print(f"    {param}={value:.6f}{comma}")
    elif value is None:
        print(f"    {param}=None{comma}")
    elif isinstance(value, str):
        print(f"    {param}='{value}'{comma}")
    else:
        print(f"    {param}={value}{comma}")
print("    random_state=42")
print(")")

print(f"\n📊 Alternative Dictionary Format:")
print("-" * 35)
print("best_hyperparameters = {")
for i, (param, value) in enumerate(best_params.items()):
    comma = "," if i < len(best_params) - 1 else ""
    if isinstance(value, float):
        print(f"    '{param}': {value:.6f}{comma}")
    elif value is None:
        print(f"    '{param}': None{comma}")
    elif isinstance(value, str):
        print(f"    '{param}': '{value}'{comma}")
    else:
        print(f"    '{param}': {value}{comma}")
print("}")

print(f"\n🎲 For Use with Sampling Techniques:")
print("-" * 40)
print("# Example with SMOTE:")
print("from imblearn.over_sampling import SMOTE")
print("from sklearn.tree import DecisionTreeClassifier")
print("")
print("# Apply sampling")
print("smote = SMOTE(random_state=42)")
print("X_resampled, y_resampled = smote.fit_resample(X_train, y_train)")
print("")
print("# Use optimal hyperparameters")
print("optimal_classifier = DecisionTreeClassifier(")
for i, (param, value) in enumerate(best_params.items()):
    comma = "," if i < len(best_params) - 1 else ""
    if isinstance(value, float):
        print(f"    {param}={value:.6f}{comma}")
    elif value is None:
        print(f"    {param}=None{comma}")
    elif isinstance(value, str):
        print(f"    {param}='{value}'{comma}")
    else:
        print(f"    {param}={value}{comma}")
print("    random_state=42")
print(")")
print("")
print("# Train on resampled data")
print("optimal_classifier.fit(X_resampled, y_resampled)")

print("\n" + "="*60)

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

# Save best parameters to a file for later use
print("\n💾 Saving best parameters to file...")
import json
with open('best_dt_hyperparameters.json', 'w') as f:
    # Convert numpy types to native Python types for JSON serialization
    json_params = {}
    for key, value in best_params.items():
        if isinstance(value, np.floating):
            json_params[key] = float(value)
        elif isinstance(value, np.integer):
            json_params[key] = int(value)
        else:
            json_params[key] = value
    
    json.dump({
        'best_parameters': json_params,
        'best_cv_score': float(rand.best_score_),
        'model_type': 'DecisionTreeClassifier'
    }, f, indent=2)

print("✅ Best parameters saved to 'best_dt_hyperparameters.json'")

# Function to load and use saved parameters
print("\n🔄 Function to load saved parameters:")
print("-" * 40)
print("""
def load_optimal_dt_classifier():
    import json
    from sklearn.tree import DecisionTreeClassifier
    
    with open('best_dt_hyperparameters.json', 'r') as f:
        saved_data = json.load(f)
    
    params = saved_data['best_parameters']
    classifier = DecisionTreeClassifier(**params, random_state=42)
    
    return classifier, saved_data['best_cv_score']

# Usage:
# optimal_model, cv_score = load_optimal_dt_classifier()
# print(f"Loaded model with CV score: {cv_score:.4f}")
""")