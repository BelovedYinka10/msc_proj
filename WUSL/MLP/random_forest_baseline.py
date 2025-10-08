# baseline_ids_random_forest.py
# Random Forest Baseline for IIoT IDS (NO Multi-Stage)
# Direct comparison to MSDL approach

import os
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")

random.seed(42)
np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


@dataclass
class BaselineConfig:
    # Dataset
    csv_path: str = "../wustl_iiot_2021.csv"
    label_col: str = "Traffic"
    drop_cols: Tuple[str, ...] = ("StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId")

    # Split
    test_size: float = 0.30
    random_state: int = 42

    # Random Forest parameters
    n_estimators: int = 100
    max_depth: int = None  # None means unlimited
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"  # 'sqrt', 'log2', or None for all features
    n_jobs: int = -1  # Use all available cores

    # Class weighting options
    use_class_weight: bool = True  # Set to False for no class weights

    # Output
    save_dir: str = "../baseline_outputs"


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric"""
    out = df.copy()
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def per_class_report(y_true, y_pred, labels: List[str]) -> pd.DataFrame:
    """Generate precision, recall, F1, accuracy, support for each class"""
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Calculate per-class accuracy
    per_class_acc = []
    for label in labels:
        mask = y_true == label
        if mask.sum() > 0:
            acc = (y_pred[mask] == label).sum() / mask.sum() * 100
        else:
            acc = 0.0
        per_class_acc.append(acc)

    # Calculate overall accuracy
    overall_acc = accuracy_score(y_true, y_pred) * 100

    df = pd.DataFrame({
        "Class": labels,
        "Precision": p * 100,
        "Recall": r * 100,
        "F1": f1 * 100,
        "Accuracy": per_class_acc,
        "Support": s
    })

    # Add overall accuracy as a summary row
    summary_row = pd.DataFrame({
        "Class": ["OVERALL"],
        "Precision": [df["Precision"].mean()],
        "Recall": [df["Recall"].mean()],
        "F1": [df["F1"].mean()],
        "Accuracy": [overall_acc],
        "Support": [df["Support"].sum()]
    })

    df = pd.concat([df, summary_row], ignore_index=True)

    return df


class BaselineIDS:
    def __init__(self, cfg: BaselineConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load dataset"""
        print("Loading dataset...")
        df = pd.read_csv(self.cfg.csv_path)

        # Drop columns
        for c in self.cfg.drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

        y = df[self.cfg.label_col].astype(str)
        X = df.drop(columns=[self.cfg.label_col])
        X = ensure_numeric(X)

        return X, y

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series):
        """Train baseline Random Forest model and evaluate"""

        print(f"\nDataset: {len(X):,} samples, {len(X.columns)} features, {len(y.unique())} classes")
        print("\nClass distribution:")
        for cls, count in y.value_counts().sort_index().items():
            pct = count / len(y) * 100
            print(f"  {cls:20s}: {count:7,} ({pct:5.2f}%)")

        # Split data
        print(f"\nSplitting data: {int((1 - self.cfg.test_size) * 100)}% train, {int(self.cfg.test_size * 100)}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y
        )

        print(f"Train: {len(y_train):,} samples, Test: {len(y_test):,} samples")

        # Prepare labels
        labels = sorted(y.unique())

        # Preprocessing
        print("\nPreprocessing...")
        imp = SimpleImputer(strategy="median")
        X_train_imp = imp.fit_transform(X_train)
        X_test_imp = imp.transform(X_test)

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # Build Random Forest model
        print("\nBuilding Random Forest model...")
        print(f"  n_estimators: {self.cfg.n_estimators}")
        print(f"  max_depth: {self.cfg.max_depth if self.cfg.max_depth else 'unlimited'}")
        print(f"  max_features: {self.cfg.max_features}")
        print(f"  min_samples_split: {self.cfg.min_samples_split}")
        print(f"  min_samples_leaf: {self.cfg.min_samples_leaf}")
        print(f"  n_jobs: {self.cfg.n_jobs}")

        # Configure class weights
        class_weight_param = "balanced" if self.cfg.use_class_weight else None

        if self.cfg.use_class_weight:
            print(f"\nClass weights: ENABLED (balanced)")
        else:
            print("\nClass weights: DISABLED")

        model = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_split=self.cfg.min_samples_split,
            min_samples_leaf=self.cfg.min_samples_leaf,
            max_features=self.cfg.max_features,
            class_weight=class_weight_param,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
            verbose=0
        )

        # Training
        print("\nTraining...")
        model.fit(X_train_scaled, y_train.values)
        print("Training completed")

        # Feature importance (top 10)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")

        # Save feature importance
        feat_imp_path = os.path.join(self.cfg.save_dir, "feature_importance.csv")
        feature_importance.to_csv(feat_imp_path, index=False)

        # Predict
        print("\nEvaluating on test set...")
        y_pred = model.predict(X_test_scaled)

        # Generate report
        print("\n" + "=" * 70)
        print("RANDOM FOREST BASELINE RESULTS")
        print("=" * 70)

        report = per_class_report(y_test.values, y_pred, labels)
        print("\n" + report.to_string(index=False))

        # Save report
        report_path = os.path.join(self.cfg.save_dir, "baseline_results.csv")
        report.to_csv(report_path, index=False)

        # Confusion matrix
        cm = confusion_matrix(y_test.values, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_path = os.path.join(self.cfg.save_dir, "confusion_matrix.csv")
        cm_df.to_csv(cm_path)

        print(f"\n✓ Results saved to: {self.cfg.save_dir}/")
        print(f"  - baseline_results.csv")
        print(f"  - confusion_matrix.csv")
        print(f"  - feature_importance.csv")

        # Summary statistics
        overall_acc = accuracy_score(y_test.values, y_pred) * 100
        avg_f1 = report[report["Class"] != "OVERALL"]["F1"].mean()
        min_f1 = report[report["Class"] != "OVERALL"]["F1"].min()
        max_f1 = report[report["Class"] != "OVERALL"]["F1"].max()

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Overall Accuracy:  {overall_acc:.2f}%")
        print(f"Average F1 Score:  {avg_f1:.2f}%")
        print(f"Min F1 Score:      {min_f1:.2f}%")
        print(f"Max F1 Score:      {max_f1:.2f}%")
        print("=" * 70)

        return report


def main():
    # Configuration
    cfg = BaselineConfig(
        csv_path="../wustl_iiot_2021.csv",
        label_col="Traffic",
        drop_cols=("StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId"),
        test_size=0.30,
        random_state=42,
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",
        use_class_weight=True,  # Change to False for no class weights
        save_dir="../baseline_outputs"
    )

    print("=" * 70)
    print("RANDOM FOREST BASELINE FOR IIoT IDS")
    print("Single-Stage Multi-Class Classification")
    print("=" * 70)

    baseline = BaselineIDS(cfg)
    X, y = baseline.load_dataset()
    baseline.train_and_evaluate(X, y)


if __name__ == "__main__":
    main()