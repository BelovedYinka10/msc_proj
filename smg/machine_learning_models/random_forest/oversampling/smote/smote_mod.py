import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
import json
from collections import Counter
import plotly.graph_objects as go

print("🚀 Random Forest with SMOTE Sampling - Optimized Workflow")
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
# USE YOUR STORED OPTIMAL HYPERPARAMETERS
# ============================================

print(f"\n📂 Using your optimal hyperparameters...")

# Your stored parameters from JSON
best_params = {
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "max_depth": 22,
    "n_estimators": 300,
    "bootstrap": True,
    "n_jobs": -1
}
best_cv_score = 0.9087766002063082

print(f"✅ Using optimal hyperparameters (CV Score: {best_cv_score:.4f})")
print("📋 Parameters:")
for param, value in best_params.items():
    print(f"   {param}: {value}")

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
# APPLY SMOTE SAMPLING
# ============================================

print(f"\n🎲 Applying SMOTE Sampling...")

# Initialize SMOTE
smote = SMOTE(
    sampling_strategy='auto',  # Balance all classes to majority class size
    random_state=42,
    k_neighbors=5
)

# Apply SMOTE to training data ONLY
try:
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"✅ SMOTE sampling completed successfully!")
    print(f"   Original training samples: {X_train.shape[0]}")
    print(f"   Resampled training samples: {X_train_resampled.shape[0]}")
    
    # Show new class distribution
    print(f"\n📈 New Class Distribution (After SMOTE):")
    new_counts = Counter(y_train_resampled)
    total_resampled = len(y_train_resampled)
    for class_idx, count in sorted(new_counts.items()):
        percentage = (count / total_resampled) * 100
        original_count = original_counts[class_idx]
        increase = count - original_count
        print(f"   Class {class_names[class_idx]}: {count} samples ({percentage:.1f}%) [+{increase} synthetic]")
    
except ValueError as e:
    print(f"❌ SMOTE sampling failed: {e}")
    print("⚠️  Using original training data without sampling")
    X_train_resampled, y_train_resampled = X_train, y_train

# ============================================
# CREATE OPTIMAL MODEL WITH SAVED PARAMETERS
# ============================================

print(f"\n🔧 Creating optimal Random Forest classifier...")

# Create classifier with optimal hyperparameters
optimal_classifier = RandomForestClassifier(
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    n_estimators=best_params['n_estimators'],
    bootstrap=best_params['bootstrap'],
    n_jobs=best_params['n_jobs'],
    random_state=42
)

print(f"✅ Random Forest classifier created with optimal hyperparameters")

# ============================================
# TRAIN MODEL ON RESAMPLED DATA
# ============================================

print(f"\n⏳ Training Random Forest model on BorderlineSMOTE-resampled data...")

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
print("🎯 RANDOM FOREST + BORDERLINESMOTE RESULTS")
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
# FEATURE IMPORTANCE ANALYSIS
# ============================================

print(f"\n🔍 Feature Importance Analysis:")
print("-" * 35)

# Get feature names
try:
    feature_names = df.drop(columns=['status', 'Unnamed: 0', 'Unnamed: 0.1'], errors='ignore').columns.tolist()
except:
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': optimal_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print(f"Top 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

# ============================================
# INTERACTIVE CONFUSION MATRIX WITH PLOTLY
# ============================================

print(f"\n📊 Generating Interactive Confusion Matrix...")

# Create interactive heatmap with Plotly
fig = go.Figure(data=go.Heatmap(
    z=conf_matrix,
    x=[f'Predicted: {class_name}' for class_name in class_names],
    y=[f'Actual: {class_name}' for class_name in class_names],
    colorscale='Greens',  # Different color for Random Forest
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
        'text': 'Confusion Matrix - Random Forest + BorderlineSMOTE',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'color': 'darkgreen'}
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
print(f"🔧 Algorithm: Random Forest")
print(f"🎲 Sampling: BorderlineSMOTE")
print(f"⚙️  Hyperparameters: Optimal (loaded from previous search)")
print(f"📈 CV Score: {best_cv_score:.4f}")
print(f"🎯 Test F1-Score: {f1:.4f}")
print(f"⏱️  Training Time: Fast (no hyperparameter search)")
print(f"🔍 Synthetic Samples Added: {X_train_resampled.shape[0] - X_train.shape[0]}")

print(f"\n✅ Ready for comparison with other algorithms and sampling techniques!")
print(f"💡 Comparison ready with:")
print(f"   - Gradient Boosting + ADASYN (previous experiment)")
print(f"   - Random Forest + BorderlineSMOTE (this experiment)")
print(f"   - XGBoost + BorderlineSMOTE (next?)")
print(f"   - Decision Tree + BorderlineSMOTE (next?)")

print(f"\n🎉 Random Forest + BorderlineSMOTE experiment completed successfully!")