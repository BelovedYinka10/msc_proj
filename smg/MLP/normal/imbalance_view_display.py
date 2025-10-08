import pandas as pd
import matplotlib.pyplot as plt

# === Load dataset ===
df = pd.read_csv("../../../EPIC/dataset_EPICA_raw 1.csv")

# === Preprocessing ===
def preprocess_epica(data):
    data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
    data = data.loc[:, data.nunique() > 1]
    data = data.dropna()
    return data

# Raw counts (before dropping NaNs)
df_raw = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
raw_counts = df_raw['status'].value_counts()
raw_total = raw_counts.sum()

# Cleaned counts (after dropping NaNs & constants)
df_clean = preprocess_epica(df.copy())
clean_counts = df_clean['status'].value_counts()
clean_total = clean_counts.sum()

# Stats about what was removed
removed_counts = raw_counts - clean_counts
removed_counts = removed_counts.fillna(0).astype(int)
removed_total = raw_total - clean_total

# Print stats
print("=== BEFORE CLEANING ===")
print(raw_counts)
print(f"Total: {raw_total}\n")

print("=== AFTER CLEANING ===")
print(clean_counts)
print(f"Total: {clean_total}\n")

print("=== REMOVED DURING CLEANING ===")
print(removed_counts)
print(f"Total Removed: {removed_total}")

# === Plot before vs after ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before
raw_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[0])
axes[0].set_title("Before Cleaning (Raw Data)")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Number of Samples")
for idx, value in enumerate(raw_counts):
    axes[0].text(idx, value + (0.01 * max(raw_counts)), str(value), 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# After
clean_counts.plot(kind='bar', color='lightgreen', edgecolor='black', ax=axes[1])
axes[1].set_title("After Cleaning (No NaNs)")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Number of Samples")
for idx, value in enumerate(clean_counts):
    axes[1].text(idx, value + (0.01 * max(clean_counts)), str(value), 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle("EPICA Dataset: Class Distribution Before vs After Cleaning", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
