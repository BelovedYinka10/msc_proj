import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.under_sampling import ClusterCentroids
import json
from collections import Counter
import plotly.graph_objects as go

print("🚀 Logistic Regression with ClusterCentroids - Optimized Workflow")
print("="*70)

# Load your dataset
df = pd.read_csv("../../../../../EPIC/dataset_EPICA_raw 1.csv")

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

print(f"📊 Dataset Info:")
print(f"   Features: {X.shape[1]}")
print(f"   Samples: {X.shape[0]}")
print(f"   Classes: {list(class_names)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"   Train samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")

# ============================================
# USE YOUR STORED OPTIMAL LOGISTIC REGRESSION HYPERPARAMETERS
# ============================================

print(f"\n📂 Using your optimal Logistic Regression hyperparameters...")

# Your stored optimal parameters from JSON
best_params = {
    "penalty": "l1",
    "C": 1438.44988828766,
    "random_state": 42
}
best_cv_score = 0.61342749838798

print(f"✅ Using optimal hyperparameters (CV Score: {best_cv_score:.4f})")
print("📋 Parameters:")
for param, value in best_params.items():
    print(f"   {param}: {value}")

# Add solver for L1 penalty
print("📋 Additional parameters:")
print("   solver: liblinear (required for L1 penalty)")
print("   max_iter: 1000 (sufficient iterations)")
print("   class_weight: balanced (handle class imbalance)")

# ============================================
# CHECK ORIGINAL CLASS DISTRIBUTION
# ============================================

print(f"\n📊 Original Class Distribution (Training Data):")
original_counts = Counter(y_train)
total_train = len(y_train)
for class_idx, count in sorted(original_counts.items()):
    percentage = (count / total_train) * 100
    print(f"   Class {class_names[class_idx]}: {count} samples ({percentage:.1f}%)")

# ============================================
# APPLY CLUSTERCENTROIDS SAMPLING
# ============================================

print(f"\n🎲 Applying ClusterCentroids Sampling...")

# Initialize ClusterCentroids
cluster_centroids = ClusterCentroids(
    sampling_strategy='auto',  # Balance all classes by reducing majority class
    random_state=42,
    estimator=None,  # Use default KMeans
    voting='auto'    # Automatic voting strategy
)

# Apply ClusterCentroids to training data ONLY
try:
    X_train_resampled, y_train_resampled = cluster_centroids.fit_resample(X_train, y_train)
    
    print(f"✅ ClusterCentroids sampling completed successfully!")
    print(f"   Original training samples: {X_train.shape[0]}")
    print(f"   Resampled training samples: {X_train_resampled.shape[0]}")
    
    # Show new class distribution
    print(f"\n📈 New Class Distribution (After ClusterCentroids):")
    new_counts = Counter(y_train_resampled)
    total_resampled = len(y_train_resampled)
    for class_idx, count in sorted(new_counts.items()):
        percentage = (count / total_resampled) * 100
        original_count = original_counts[class_idx]
        change = count - original_count
        if change < 0:
            print(f"   Class {class_names[class_idx]}: {count} samples ({percentage:.1f}%) [{change} reduced to centroids]")
        else:
            print(f"   Class {class_names[class_idx]}: {count} samples ({percentage:.1f}%) [unchanged]")
    
except ValueError as e:
    print(f"❌ ClusterCentroids sampling failed: {e}")
    print("⚠️  Using original training data without sampling")
    X_train_resampled, y_train_resampled = X_train, y_train

# ============================================
# CREATE OPTIMAL LOGISTIC REGRESSION MODEL
# ============================================

print(f"\n🔧 Creating optimal Logistic Regression classifier...")

# Create classifier with optimal hyperparameters
optimal_classifier = LogisticRegression(
    penalty=best_params['penalty'],  # L1 regularization
    C=best_params['C'],  # Optimal regularization strength
    solver='liblinear',  # Required for L1 penalty
    max_iter=1000,  # Sufficient iterations
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)

print(f"✅ Logistic Regression classifier created with optimal hyperparameters")

# ============================================
# TRAIN MODEL ON RESAMPLED DATA
# ============================================

print(f"\n⏳ Training Logistic Regression model on ClusterCentroids-resampled data...")

# Train on resampled training data
optimal_classifier.fit(X_train_resampled, y_train_resampled)

print(f"✅ Training completed!")

# ============================================
# EVALUATE ON ORIGINAL TEST DATA
# ============================================

print(f"\n📈 Evaluating on original test data...")

# Predict on ORIGINAL test set (never resampled)
y_pred = optimal_classifier.predict(X_test)

# Calculate overall metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# Classification report
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)
class_totals = conf_matrix.sum(axis=1)

# ============================================
# DISPLAY RESULTS
# ============================================

print("\n" + "="*60)
print("🎯 LOGISTIC REGRESSION + CLUSTERCENTROIDS RESULTS")
print("="*60)

print(f"\n📊 Overall Performance Metrics:")
print("-" * 35)
print(f"Accuracy:         {acc * 100:.2f}%")
print(f"Precision (macro): {prec * 100:.2f}%")
print(f"Recall (macro):    {rec * 100:.2f}%")
print(f"F1 Score (macro):  {f1 * 100:.2f}%")

print(f"\n📋 Per-Class Performance:")
print("-" * 35)
for idx, class_name in enumerate(class_names):
    tp = conf_matrix[idx][idx]
    total = class_totals[idx]
    class_accuracy = tp / total * 100 if total > 0 else 0
    class_precision = report[class_name]['precision'] * 100
    class_recall = report[class_name]['recall'] * 100
    class_f1 = report[class_name]['f1-score'] * 100
    
    print(f"\n🏷️  Class: {class_name}")
    print(f"    Accuracy:  {class_accuracy:.2f}%")
    print(f"    Precision: {class_precision:.2f}%")
    print(f"    Recall:    {class_recall:.2f}%")
    print(f"    F1-score:  {class_f1:.2f}%")

# ============================================
# FEATURE IMPORTANCE ANALYSIS (COEFFICIENTS)
# ============================================

print(f"\n🔍 Feature Importance Analysis (Logistic Regression Coefficients):")
print("-" * 60)

# Get feature names
try:
    feature_names = df.drop(columns=['status', 'Unnamed: 0', 'Unnamed: 0.1'], errors='ignore').columns.tolist()
except:
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

# For multiclass, take the absolute mean of coefficients across all classes
if len(class_names) > 2:
    # Multiclass: coefficients shape is (n_classes, n_features)
    feature_importance = np.mean(np.abs(optimal_classifier.coef_), axis=0)
else:
    # Binary: coefficients shape is (1, n_features)  
    feature_importance = np.abs(optimal_classifier.coef_[0])

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient_magnitude': feature_importance
}).sort_values('coefficient_magnitude', ascending=False)

print(f"Top 10 Most Important Features (by coefficient magnitude):")
print(importance_df.head(10).to_string(index=False))
print(f"\nNote: Higher coefficient magnitude = stronger influence on prediction")

# ============================================
# INTERACTIVE CONFUSION MATRIX WITH PLOTLY
# ============================================

print(f"\n📊 Generating Interactive Confusion Matrix...")

# Create interactive heatmap with Plotly
fig = go.Figure(data=go.Heatmap(
    z=conf_matrix,
    x=[f'Predicted: {class_name}' for class_name in class_names],
    y=[f'Actual: {class_name}' for class_name in class_names],
    colorscale='Purples',  # Purple color for Logistic Regression
    showscale=True,
    colorbar=dict(title="Count"),
    text=conf_matrix,
    texttemplate="%{text}",
    textfont={"size": 14},
    hovertemplate='<b>Actual</b>: %{y}<br>' +
                  '<b>Predicted</b>: %{x}<br>' +
                  '<b>Count</b>: %{z}<br>' +
                  '<extra></extra>'
))

fig.update_layout(
    title={
        'text': 'Confusion Matrix - Logistic Regression + ClusterCentroids',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'color': 'purple'}
    },
    xaxis_title="Predicted Class",
    yaxis_title="Actual Class",
    width=600,
    height=500,
    font=dict(size=12)
)

# Display the interactive confusion matrix
fig.show()

print(f"✅ Interactive confusion matrix created!")
print(f"📊 Confusion Matrix Summary:")
print("-" * 30)
print(f"Classes: {list(class_names)}")
print(f"Matrix Shape: {conf_matrix.shape}")
print(f"Total Test Samples: {np.sum(conf_matrix)}")

# ============================================
# SUMMARY
# ============================================

print(f"\n" + "="*60)
print("📝 EXPERIMENT SUMMARY")
print("="*60)
print(f"🔧 Algorithm: Logistic Regression")
print(f"🎲 Sampling: ClusterCentroids")
print(f"⚙️  Hyperparameters: Optimal (C={best_params['C']:.2f}, L1 penalty)")
print(f"📈 CV Score: {best_cv_score:.4f}")
print(f"🎯 Test F1-Score: {f1:.4f}")
print(f"⏱️  Training Time: Very Fast (optimized linear model)")
print(f"🔍 Samples Removed: {X_train.shape[0] - X_train_resampled.shape[0]}")

print(f"\n✅ Ready for comparison with other algorithms and sampling techniques!")
print(f"💡 Comparison ready with:")
print(f"   - Gradient Boosting + ADASYN (oversampling)")
print(f"   - Random Forest + ROS (oversampling)")
print(f"   - Logistic Regression + ClusterCentroids (undersampling)")
print(f"   - XGBoost + ClusterCentroids (next?)")

print(f"\n🎉 Logistic Regression + ClusterCentroids experiment completed successfully!")