# plot_kdd_5class_distribution.py
# Shows class distribution bar charts for KDDTrain+_5class.csv and KDDTest+_5class.csv

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== CONFIG ====
TARGET_COL = "class5"  # change if your grouped column has a different name
ROOT = Path.cwd()
TRAIN_PATH = ROOT /   "KDDTrain+_5class.csv"
TEST_PATH  = ROOT /  "KDDTest+_5class.csv"
# ===============

def load_df(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    if TARGET_COL not in df.columns:
        raise ValueError(f"{p.name} missing '{TARGET_COL}'. Columns: {df.columns.tolist()}")
    return df

def order_labels(idx):
    """Order classes in the familiar sequence if present, else keep others at the end."""
    pref = {"normal":0, "dos":1, "probe":2, "r2l":3, "u2r":4}
    return sorted(idx, key=lambda x: pref.get(str(x).lower(), 99))

print(f"Using:\n  Train: {TRAIN_PATH}\n  Test : {TEST_PATH}")
train_df = load_df(TRAIN_PATH)
test_df  = load_df(TEST_PATH)

# Counts
train_counts = train_df[TARGET_COL].value_counts()
test_counts  = test_df[TARGET_COL].value_counts()

# Reorder nicely
ordered_labels = order_labels(train_counts.index.union(test_counts.index))
train_counts = train_counts.reindex(ordered_labels, fill_value=0)
test_counts  = test_counts.reindex(ordered_labels, fill_value=0)

# Percentages
train_pct = (train_counts / train_counts.sum() * 100.0)
test_pct  = (test_counts  / test_counts.sum()  * 100.0)

# ---- Print tables ----
pd.options.display.float_format = lambda v: f"{v:.2f}"
print("\n=== Train distribution (counts) ===")
print(train_counts.to_frame("count"))
print("\n=== Train distribution (%) ===")
print(train_pct.to_frame("percent"))

print("\n=== Test distribution (counts) ===")
print(test_counts.to_frame("count"))
print("\n=== Test distribution (%) ===")
print(test_pct.to_frame("percent"))

# ---- Plot 1: Train & Test counts (two subplots) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Train counts
axes[0].bar(np.arange(len(train_counts)), train_counts.values)
axes[0].set_title("KDDTrain+ (5-class) — Counts")
axes[0].set_xticks(np.arange(len(train_counts)))
axes[0].set_xticklabels(train_counts.index, rotation=45, ha="right")
axes[0].set_ylabel("Count")
for i, v in enumerate(train_counts.values):
    axes[0].text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)

# Test counts
axes[1].bar(np.arange(len(test_counts)), test_counts.values)
axes[1].set_title("KDDTest+ (5-class) — Counts")
axes[1].set_xticks(np.arange(len(test_counts)))
axes[1].set_xticklabels(test_counts.index, rotation=45, ha="right")
axes[1].set_ylabel("Count")
for i, v in enumerate(test_counts.values):
    axes[1].text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()

# ---- Plot 2: Side-by-side % comparison (Train vs Test) ----
x = np.arange(len(ordered_labels))
width = 0.4

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(x - width/2, train_pct.values, width, label="Train")
ax2.bar(x + width/2, test_pct.values,  width, label="Test")

ax2.set_title("Class Distribution — Percentage (Train vs Test)")
ax2.set_xticks(x)
ax2.set_xticklabels(ordered_labels, rotation=45, ha="right")
ax2.set_ylabel("Percentage (%)")
ax2.legend()

# Annotate bars with percentages
for i, v in enumerate(train_pct.values):
    ax2.text(i - width/2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
for i, v in enumerate(test_pct.values):
    ax2.text(i + width/2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()
