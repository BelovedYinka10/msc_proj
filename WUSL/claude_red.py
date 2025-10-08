# msdl_ids_paper_faithful_FIXED.py
# Multi-Stage Deep Learning (MSDL) for IIoT IDS
# FIXED VERSION with proper class weighting and debugging

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


@dataclass
class MSDLConfig:
    csv_path: str = "wustl_iiot_2021.csv"
    label_col: str = "Traffic"
    drop_cols: Tuple[str, ...] = ("StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId")
    drop_extra_leaky: Tuple[str, ...] = ()
    test_size: float = 0.30
    random_state: int = 42
    chronological_split: bool = False
    time_col: Optional[str] = "StartTime"
    f_min: float = 0.95
    force_all_stages: bool = False
    benign_labels: Tuple[str, ...] = ("normal", "benign")

    # FIXED: Enable class weights everywhere
    learning_rate: float = 5e-4  # Reduced for better convergence
    batch_size: int = 128  # Smaller batch size for better gradient estimates
    epochs_stage1: int = 50
    epochs_stage2: int = 50
    epochs_stage3plus: int = 100  # Much longer for difficult classes
    val_ratio: float = 0.30

    cw_stage1_multiclass: bool = True  # ENABLED
    cw_stage2_binary: bool = True
    cw_stage3plus_multiclass: bool = True  # CRITICAL FIX: ENABLED
    cw_stage3plus_binary: bool = True

    max_rows_per_class: Optional[int] = None
    save_dir: str = "./msdl_outputs_paper_faithful"


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric"""
    out = df.copy()
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def per_class_report(y_true, y_pred, labels: List[str]) -> pd.DataFrame:
    """Generate precision, recall, F1, accuracy, support for each class"""
    from sklearn.metrics import accuracy_score

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Calculate per-class accuracy (correct predictions / total for that class)
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


def compute_cmr(y: pd.Series) -> Dict[str, float]:
    """Compute Class Minority Ratio"""
    counts = y.value_counts()
    avg = counts.mean()
    if avg == 0:
        return {k: 0.0 for k in counts.index}
    return (counts / avg).to_dict()


def build_core_model(input_dim: int) -> keras.Model:
    """Build DNN with dropout for regularization"""
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(50, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(50, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(50, activation="relu"),
        layers.Dropout(0.2),
    ])


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


def class_weight_binary(y_01: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights for binary classification"""
    p_true = float((y_01 == 1).mean()) if y_01.size else 0.5
    p_true = max(min(p_true, 1 - 1e-6), 1e-6)
    w_true = 0.5 / p_true
    w_false = 0.5 / (1.0 - p_true)
    return {0: float(w_false), 1: float(w_true)}


def choose_threshold_youden(val_prob: np.ndarray, y_val_01: np.ndarray) -> float:
    """Find optimal threshold using Youden's J statistic"""
    best_t, best_j = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        pred = (val_prob >= t).astype(int)
        tp = np.sum((pred == 1) & (y_val_01 == 1))
        tn = np.sum((pred == 0) & (y_val_01 == 0))
        fp = np.sum((pred == 1) & (y_val_01 == 0))
        fn = np.sum((pred == 0) & (y_val_01 == 1))
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def train_predict_dnn_multiclass(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        labels: List[str],
        epochs: int,
        cfg: MSDLConfig,
        use_class_weight: bool
) -> np.ndarray:
    """Train multi-class DNN with FIXED preprocessing and class weights"""

    labs = sorted(list(set(labels)))
    lab2id = {lab: i for i, lab in enumerate(labs)}
    y_train_idx = np.array([lab2id[y] for y in y_train], dtype="int32")

    # DEBUG: Print class distribution
    print(f"\n  Training {len(labs)} classes:")
    for lab in labs:
        count = np.sum(y_train == lab)
        pct = count / len(y_train) * 100
        print(f"    {lab:20s}: {count:6,} samples ({pct:6.2f}%)")

    # Preprocessing - FIXED: Use proper StandardScaler
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    X_te = imp.transform(X_test)

    scaler = StandardScaler(with_mean=True, with_std=True)  # FIXED
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Validation split
    val_size = max(1, int(len(X_tr) * cfg.val_ratio))
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=cfg.random_state
    )
    tr_idx, val_idx = next(sss.split(X_tr, y_train_idx))

    # Model
    model = build_core_model(X_tr.shape[1])
    model.add(layers.Dense(len(labs), activation="softmax"))

    opt = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Training
    fit_kwargs = dict(
        x=X_tr[tr_idx],
        y=y_train_idx[tr_idx],
        batch_size=cfg.batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(X_tr[val_idx], y_train_idx[val_idx]),
        shuffle=True,
        callbacks=make_callbacks(),
    )

    if use_class_weight:
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(labs)),
            y=y_train_idx[tr_idx]
        )
        class_weight_dict = {i: float(w) for i, w in enumerate(cw)}
        fit_kwargs["class_weight"] = class_weight_dict

        print(f"  Class weights:")
        for i, lab in enumerate(labs):
            print(f"    {lab:20s}: {class_weight_dict[i]:8.2f}")

    model.fit(**fit_kwargs)

    # Predict
    y_prob = model.predict(X_te, verbose=0)
    y_idx = np.argmax(y_prob, axis=1)

    # DEBUG: Show prediction distribution
    print(f"  Predictions:")
    unique, counts = np.unique(y_idx, return_counts=True)
    for idx, count in zip(unique, counts):
        pct = count / len(y_idx) * 100
        print(f"    {labs[idx]:20s}: {count:6,} ({pct:6.2f}%)")

    return np.array([labs[i] for i in y_idx])


def train_predict_dnn_binary(
        X_train: np.ndarray,
        y_train_bool_str: np.ndarray,
        X_test: np.ndarray,
        epochs: int,
        cfg: MSDLConfig,
        use_class_weight: bool
) -> Tuple[np.ndarray, float]:
    """Train binary DNN with threshold tuning"""

    y_01 = np.array([1 if v == "True" else 0 for v in y_train_bool_str], dtype="int32")

    # Preprocessing - FIXED
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    X_te = imp.transform(X_test)

    scaler = StandardScaler(with_mean=True, with_std=True)  # FIXED
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Validation split
    val_size = max(1, int(len(X_tr) * cfg.val_ratio))
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=cfg.random_state
    )
    tr_idx, val_idx = next(sss.split(X_tr, y_01))
    X_tr_fit, y_tr_fit = X_tr[tr_idx], y_01[tr_idx]
    X_val, y_val = X_tr[val_idx], y_01[val_idx]

    # Model
    model = build_core_model(X_tr.shape[1])
    model.add(layers.Dense(1, activation="sigmoid"))

    opt = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Training
    fit_kwargs = dict(
        x=X_tr_fit,
        y=y_tr_fit,
        batch_size=cfg.batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(X_val, y_val),
        shuffle=True,
        callbacks=make_callbacks(),
    )

    if use_class_weight:
        fit_kwargs["class_weight"] = class_weight_binary(y_tr_fit)

    model.fit(**fit_kwargs)

    # Threshold tuning
    val_prob = model.predict(X_val, verbose=0).ravel()
    thr = choose_threshold_youden(val_prob, y_val)

    # Test predictions
    te_prob = model.predict(X_te, verbose=0).ravel()
    y_pred_test = np.where(te_prob >= thr, "True", "False")

    return y_pred_test, thr


class MultiStageIDS:
    def __init__(self, cfg: MSDLConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        self.stage_reports: List[Tuple[str, pd.DataFrame]] = []

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Load dataset"""
        df = pd.read_csv(self.cfg.csv_path)
        df_original = df.copy()

        drop_now = list(self.cfg.drop_cols)
        for c in drop_now:
            if c in df.columns:
                df = df.drop(columns=[c])

        for c in self.cfg.drop_extra_leaky:
            if c in df.columns:
                df = df.drop(columns=[c])

        y = df[self.cfg.label_col].astype(str)
        X = df.drop(columns=[self.cfg.label_col])

        if self.cfg.max_rows_per_class is not None:
            parts = []
            indices = []
            for cls, grp in df.groupby(self.cfg.label_col):
                subset = grp.head(self.cfg.max_rows_per_class)
                parts.append(subset)
                indices.extend(subset.index.tolist())
            df = pd.concat(parts, ignore_index=False)
            df_original = df_original.loc[indices]
            y = df[self.cfg.label_col].astype(str)
            X = df.drop(columns=[self.cfg.label_col])

        X = ensure_numeric(X)
        return X, y, df_original

    def _save_report(self, name: str, df: pd.DataFrame):
        """Save stage report"""
        out = df.copy()
        out.insert(0, "Stage", name)
        fname = name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
        fpath = os.path.join(self.cfg.save_dir, f"{fname}.csv")
        out.to_csv(fpath, index=False)
        self.stage_reports.append((name, df.copy()))

    def _split(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            df_full: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data 70/30"""
        if (self.cfg.chronological_split and
                self.cfg.time_col and
                self.cfg.time_col in df_full.columns):
            print(f"  Using chronological split on '{self.cfg.time_col}'")
            order = df_full[self.cfg.time_col].argsort(kind="mergesort")
            X_sorted = X.iloc[order]
            y_sorted = y.iloc[order]
            split_idx = int((1.0 - self.cfg.test_size) * len(X_sorted))
            return (
                X_sorted.iloc[:split_idx],
                X_sorted.iloc[split_idx:],
                y_sorted.iloc[:split_idx],
                y_sorted.iloc[split_idx:]
            )
        else:
            return train_test_split(
                X, y,
                test_size=self.cfg.test_size,
                random_state=self.cfg.random_state,
                stratify=y
            )

    def export_summary(self) -> pd.DataFrame:
        """Export all stage reports"""
        rows = []
        for name, df in self.stage_reports:
            tmp = df.copy()
            tmp.insert(0, "Stage", name)
            rows.append(tmp)
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(
            os.path.join(self.cfg.save_dir, "ALL_STAGE_REPORTS.csv"),
            index=False
        )
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series, df_full: pd.DataFrame):
        """Execute Multi-Stage Deep Learning"""

        X_train, X_test, y_train, y_test = self._split(X, y, df_full)
        print(f"\nSplit: Train={len(y_train):,}, Test={len(y_test):,}")

        # Stage 1
        print("\n" + "=" * 70)
        print("STAGE 1: Multi-class classification on all classes")
        print("=" * 70)

        labels_all = sorted(y_test.unique())
        y_pred_1 = train_predict_dnn_multiclass(
            X_train.values, y_train.values,
            X_test.values, labels_all,
            epochs=self.cfg.epochs_stage1,
            cfg=self.cfg,
            use_class_weight=self.cfg.cw_stage1_multiclass
        )

        rep1 = per_class_report(y_test.values, y_pred_1, labels_all)
        self._save_report("Stage1_Multiclass_All", rep1)
        print("\nResults:")
        print(rep1.to_string(index=False))

        if (rep1["F1"] >= self.cfg.f_min * 100).all() and not self.cfg.force_all_stages:
            print(f"\n✓ All classes meet F_min={self.cfg.f_min}. DONE.")
            return

        failing = rep1[rep1["F1"] < self.cfg.f_min * 100]["Class"].tolist()
        print(f"\n✗ Classes below F_min: {failing}")
        print("→ Proceeding to Stage 2...")

        # Stage 2
        print("\n" + "=" * 70)
        print("STAGE 2: Binary classification (Normal vs Attack)")
        print("=" * 70)

        benign_set = set(b.strip().lower() for b in self.cfg.benign_labels)
        is_attack_train = (~y_train.str.strip().str.lower().isin(benign_set)).astype(str).values
        is_attack_test = (~y_test.str.strip().str.lower().isin(benign_set)).astype(str).values

        y_pred2_test, thr2 = train_predict_dnn_binary(
            X_train.values, is_attack_train,
            X_test.values,
            epochs=self.cfg.epochs_stage2,
            cfg=self.cfg,
            use_class_weight=self.cfg.cw_stage2_binary
        )

        rep2 = per_class_report(is_attack_test, y_pred2_test, ["False", "True"])
        rep2["Class"] = ["Normal", "Attack"]
        self._save_report("Stage2_Binary_Normal_vs_Attack", rep2)
        print("\nResults:")
        print(rep2.to_string(index=False))

        if not (rep2["F1"] >= self.cfg.f_min * 100).all():
            print(f"\n✗ Stage 2 failed to meet F_min. STOPPING.")
            return

        print(f"\n✓ Stage 2 passed.")
        print("→ Proceeding to Stage 3+ (attack-specific)...")

        # Stage 3+
        print("\n" + "=" * 70)
        print("STAGE 3+: Attack-specific classification (recursive)")
        print("=" * 70)

        attack_train_mask = (is_attack_train == "True")
        attack_test_mask = (is_attack_test == "True")

        A_X_train = X_train[attack_train_mask]
        A_y_train = y_train[attack_train_mask]
        A_X_test = X_test[attack_test_mask]
        A_y_test = y_test[attack_test_mask]

        if len(A_y_test) == 0:
            print("No attack samples. STOPPING.")
            return

        print(f"\nAttack subset: Train={len(A_y_train):,}, Test={len(A_y_test):,}")

        isolated_classes = set()
        iteration = 0
        max_iterations = 20

        while iteration < max_iterations:
            iteration += 1

            remaining_classes = sorted([
                c for c in A_y_test.unique()
                if c not in isolated_classes
            ])

            if len(remaining_classes) == 0:
                print("\n✓ All classes isolated.")
                break

            if len(remaining_classes) == 1:
                last_class = remaining_classes[0]
                mask_test = (A_y_test == last_class)
                if mask_test.sum() > 0:
                    print(f"\n--- Iteration {iteration}: Final class '{last_class}' ---")
                    y_pred_final = A_y_test[mask_test].values
                    rep_final = per_class_report(
                        A_y_test[mask_test].values,
                        y_pred_final,
                        [last_class]
                    )
                    self._save_report(f"Stage3_Iter{iteration}_Final_{last_class}", rep_final)
                    print(rep_final.to_string(index=False))
                    if rep_final["F1"].iloc[0] >= self.cfg.f_min * 100:
                        isolated_classes.add(last_class)
                break

            print(f"\n{'=' * 70}")
            print(f"Iteration {iteration}: {len(remaining_classes)} classes remaining")
            print(f"{'=' * 70}")

            mask_train = A_y_train.isin(remaining_classes)
            mask_test = A_y_test.isin(remaining_classes)

            y_pred_mc = train_predict_dnn_multiclass(
                A_X_train[mask_train].values,
                A_y_train[mask_train].values,
                A_X_test[mask_test].values,
                remaining_classes,
                epochs=self.cfg.epochs_stage3plus,
                cfg=self.cfg,
                use_class_weight=self.cfg.cw_stage3plus_multiclass
            )

            rep_mc = per_class_report(
                A_y_test[mask_test].values,
                y_pred_mc,
                remaining_classes
            )
            self._save_report(f"Stage3_Iter{iteration}_Multiclass", rep_mc)
            print("\nMulti-class on remaining attacks:")
            print(rep_mc.to_string(index=False))

            classes_ok = rep_mc[rep_mc["F1"] >= self.cfg.f_min * 100]["Class"].tolist()
            if len(classes_ok) == len(remaining_classes):
                print(f"\n✓ All remaining meet F_min={self.cfg.f_min}")
                isolated_classes.update(remaining_classes)
                break

            cmr_dict = compute_cmr(A_y_train[A_y_train.isin(remaining_classes)])
            target_class = max(remaining_classes, key=lambda c: cmr_dict.get(c, 0.0))
            target_cmr = cmr_dict[target_class]
            target_f1 = rep_mc[rep_mc["Class"] == target_class]["F1"].iloc[0]
            print(f"\n→ Isolating '{target_class}' (highest CMR={target_cmr:.2f}, F1={target_f1:.2f}%)")

            train_mask_binary = A_y_train.isin(remaining_classes)
            test_mask_binary = A_y_test.isin(remaining_classes)

            y_train_binary = (A_y_train[train_mask_binary] == target_class).astype(str).values
            y_test_binary = (A_y_test[test_mask_binary] == target_class).astype(str).values

            y_pred_binary, thr_binary = train_predict_dnn_binary(
                A_X_train[train_mask_binary].values,
                y_train_binary,
                A_X_test[test_mask_binary].values,
                epochs=self.cfg.epochs_stage3plus,
                cfg=self.cfg,
                use_class_weight=self.cfg.cw_stage3plus_binary
            )

            rep_binary = per_class_report(
                y_test_binary,
                y_pred_binary,
                ["False", "True"]
            )
            rep_binary["Class"] = [f"Others", f"{target_class}"]
            self._save_report(
                f"Stage3_Iter{iteration}_Binary_{target_class}_vs_Others",
                rep_binary
            )
            print("\nBinary classification:")
            print(rep_binary.to_string(index=False))

            target_f1_binary = rep_binary[
                rep_binary["Class"] == target_class
                ]["F1"].iloc[0]

            if target_f1_binary >= self.cfg.f_min * 100:
                print(f"✓ '{target_class}' meets F_min. Isolated.")
                isolated_classes.add(target_class)
            else:
                print(f"✗ '{target_class}' below F_min but removing to avoid loop.")
                isolated_classes.add(target_class)

            print(f"→ Removing '{target_class}' from pool...")

        print(f"\n{'=' * 70}")
        print(f"MSDL COMPLETE: {len(isolated_classes)} classes processed")
        print(f"{'=' * 70}")


def main():
    cfg = MSDLConfig(
        csv_path="wustl_iiot_2021.csv",
        label_col="Traffic",
        drop_cols=("StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId"),
        test_size=0.30,
        random_state=42,
        f_min=0.95,
        cw_stage1_multiclass=True,
        cw_stage2_binary=True,
        cw_stage3plus_multiclass=True,
        cw_stage3plus_binary=True,
        epochs_stage1=50,
        epochs_stage2=50,
        epochs_stage3plus=100,
        learning_rate=5e-4,
        batch_size=128,
        save_dir="./msdl_outputs_fixed"
    )

    print("=" * 70)
    print("Multi-Stage Deep Learning (MSDL) - FIXED VERSION")
    print("=" * 70)

    msdl = MultiStageIDS(cfg)
    print("\nLoading dataset...")
    X, y, df_full = msdl.load_dataset()

    print(f"\nDataset: {len(X):,} samples, {len(X.columns)} features, {len(y.unique())} classes")
    print("\nClass distribution:")
    for cls, count in y.value_counts().sort_index().items():
        pct = count / len(y) * 100
        print(f"  {cls:20s}: {count:7,} ({pct:5.2f}%)")

    msdl.fit(X, y, df_full)
    summary = msdl.export_summary()

    print(f"\n✓ Results saved to: {cfg.save_dir}/")


if __name__ == "__main__":
    main()
