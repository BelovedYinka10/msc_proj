# baseline_ids_simple.py
# Simple Baseline DNN for IIoT IDS (NO Multi-Stage)
# Direct comparison to MSDL approach
# Enhanced with ADASYN for handling class imbalance

import os
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "0")

import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from imblearn.over_sampling import ADASYN

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


@dataclass
class BaselineConfig:
    # Dataset
    csv_path: str = "../wustl_iiot_2021.csv"
    label_col: str = "Traffic"
    drop_cols: Tuple[str, ...] = ("StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId")

    # Split
    test_size: float = 0.30
    random_state: int = 42

    # Training parameters
    learning_rate: float = 5e-4
    batch_size: int = 128
    epochs: int = 100
    val_ratio: float = 0.30

    # ADASYN parameters
    use_adasyn: bool = True  # Set to False to disable ADASYN
    adasyn_sampling_strategy: str = "auto"  # 'auto', 'minority', 'not majority', or dict
    adasyn_n_neighbors: int = 5  # Number of nearest neighbors for ADASYN

    # Class weighting options
    use_class_weight: bool = False  # Set to False when using ADASYN (usually not needed together)

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


def build_baseline_model(input_dim: int, n_classes: int) -> keras.Model:
    """
    Build simple DNN baseline (same architecture as MSDL)
    - 3 hidden layers
    - 50 neurons each
    - ReLU activation
    - Dropout for regularization
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(50, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(50, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(50, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax")
    ])
    return model


def make_callbacks():
    """Standard callbacks"""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        ),
    ]


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
        """Train baseline model and evaluate"""

        print(f"\nDataset: {len(X):,} samples, {len(X.columns)} features, {len(y.unique())} classes")
        print("\nClass distribution (BEFORE ADASYN):")
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
        label2id = {lab: i for i, lab in enumerate(labels)}

        y_train_idx = np.array([label2id[lab] for lab in y_train], dtype="int32")
        y_test_idx = np.array([label2id[lab] for lab in y_test], dtype="int32")

        # Preprocessing
        print("\nPreprocessing...")
        imp = SimpleImputer(strategy="median")
        X_train_imp = imp.fit_transform(X_train)
        X_test_imp = imp.transform(X_test)

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # Apply ADASYN if enabled
        if self.cfg.use_adasyn:
            print(f"\nApplying ADASYN...")
            print(f"  Sampling strategy: {self.cfg.adasyn_sampling_strategy}")
            print(f"  N neighbors: {self.cfg.adasyn_n_neighbors}")

            print(f"\nClass distribution before ADASYN:")
            unique, counts = np.unique(y_train_idx, return_counts=True)
            for idx, count in zip(unique, counts):
                print(f"  {labels[idx]:20s}: {count:7,}")

            adasyn = ADASYN(
                sampling_strategy=self.cfg.adasyn_sampling_strategy,
                n_neighbors=self.cfg.adasyn_n_neighbors,
                random_state=self.cfg.random_state
            )

            try:
                X_train_scaled, y_train_idx = adasyn.fit_resample(X_train_scaled, y_train_idx)

                print(f"\nClass distribution after ADASYN:")
                unique, counts = np.unique(y_train_idx, return_counts=True)
                for idx, count in zip(unique, counts):
                    print(f"  {labels[idx]:20s}: {count:7,}")

                print(f"\nTotal training samples after ADASYN: {len(y_train_idx):,}")
            except Exception as e:
                print(f"\nWarning: ADASYN failed with error: {e}")
                print("Continuing without ADASYN...")
        else:
            print("\nADASYN: DISABLED")

        # Validation split
        val_size = max(1, int(len(X_train_scaled) * self.cfg.val_ratio))
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=self.cfg.random_state
        )
        tr_idx, val_idx = next(sss.split(X_train_scaled, y_train_idx))

        # Build model
        print("\nBuilding model...")
        print(f"  Architecture: 3 hidden layers (50 neurons each)")
        print(f"  Dropout: 0.2")
        print(f"  Learning rate: {self.cfg.learning_rate}")
        print(f"  Batch size: {self.cfg.batch_size}")
        print(f"  Epochs: {self.cfg.epochs}")

        model = build_baseline_model(X_train_scaled.shape[1], len(labels))

        opt = keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Training
        fit_kwargs = dict(
            x=X_train_scaled[tr_idx],
            y=y_train_idx[tr_idx],
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.epochs,
            verbose=0,
            validation_data=(X_train_scaled[val_idx], y_train_idx[val_idx]),
            shuffle=True,
            callbacks=make_callbacks(),
        )

        if self.cfg.use_class_weight:
            from sklearn.utils.class_weight import compute_class_weight
            cw = compute_class_weight(
                class_weight="balanced",
                classes=np.arange(len(labels)),
                y=y_train_idx[tr_idx]
            )
            class_weight_dict = {i: float(w) for i, w in enumerate(cw)}
            fit_kwargs["class_weight"] = class_weight_dict

            print(f"\nClass weights enabled:")
            for i, lab in enumerate(labels):
                print(f"  {lab:20s}: {class_weight_dict[i]:8.2f}")
        else:
            print("\nClass weights: DISABLED")

        print("\nTraining...")
        history = model.fit(**fit_kwargs)

        print(f"Training completed in {len(history.history['loss'])} epochs")

        # Predict
        print("\nEvaluating on test set...")
        y_prob = model.predict(X_test_scaled, verbose=0)
        y_pred_idx = np.argmax(y_prob, axis=1)
        y_pred = np.array([labels[i] for i in y_pred_idx])

        # Generate report
        print("\n" + "=" * 70)
        print("BASELINE RESULTS (WITH ADASYN)" if self.cfg.use_adasyn else "BASELINE RESULTS")
        print("=" * 70)

        report = per_class_report(y_test.values, y_pred, labels)
        print("\n" + report.to_string(index=False))

        # Save report
        suffix = "_adasyn" if self.cfg.use_adasyn else ""
        report_path = os.path.join(self.cfg.save_dir, f"baseline_results{suffix}.csv")
        report.to_csv(report_path, index=False)

        # Confusion matrix
        cm = confusion_matrix(y_test.values, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_path = os.path.join(self.cfg.save_dir, f"confusion_matrix{suffix}.csv")
        cm_df.to_csv(cm_path)

        print(f"\n✓ Results saved to: {self.cfg.save_dir}/")
        print(f"  - baseline_results{suffix}.csv")
        print(f"  - confusion_matrix{suffix}.csv")

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
        learning_rate=5e-4,
        batch_size=128,
        epochs=100,
        use_adasyn=True,  # Enable ADASYN
        adasyn_sampling_strategy="auto",  # Balance all minority classes
        adasyn_n_neighbors=5,
        use_class_weight=False,  # Usually not needed with ADASYN
        save_dir="../baseline_outputs"
    )

    print("=" * 70)
    print("BASELINE DNN FOR IIoT IDS")
    print("Single-Stage Multi-Class Classification")
    print("Enhanced with ADASYN Oversampling")
    print("=" * 70)

    baseline = BaselineIDS(cfg)
    X, y = baseline.load_dataset()
    baseline.train_and_evaluate(X, y)


if __name__ == "__main__":
    main()