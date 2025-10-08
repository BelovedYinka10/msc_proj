# plot_log_scale_only.py
# Clean log-scale visualization of class distribution

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
CSV_PATH = "../wustl_iiot_2021.csv"
LABEL_COL = "Traffic"
OUTPUT_DIR = "../plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

# Get class distribution
class_counts = df[LABEL_COL].value_counts().sort_index()
total_samples = len(df)

print(f"\nDataset: {total_samples:,} samples")
print("\nClass distribution:")
for cls, count in class_counts.items():
    pct = count / total_samples * 100
    print(f"  {cls:20s}: {count:7,} ({pct:5.2f}%)")

# Create single large figure with log scale
fig, ax = plt.subplots(figsize=(14, 8))

# Create bars with gradient colors
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_counts)))
bars = ax.bar(range(len(class_counts)), class_counts.values,
              color=colors, edgecolor='black', linewidth=2, alpha=0.85)

# Set log scale
ax.set_yscale('log')

# Customize x-axis
ax.set_xticks(range(len(class_counts)))
ax.set_xticklabels(class_counts.index, rotation=0, ha='center', fontsize=14, fontweight='bold')

# Customize y-axis
ax.set_ylabel('Number of Samples (Log Scale)', fontsize=16, fontweight='bold')
ax.tick_params(axis='y', labelsize=12)

# Title
ax.set_title('WUSTL-IIoT-2021: Class Distribution (Log Scale)',
             fontsize=18, fontweight='bold', pad=20)

# Grid
ax.grid(axis='y', alpha=0.4, which='both', linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Add value labels above each bar
for i, (bar, count) in enumerate(zip(bars, class_counts.values)):
    height = bar.get_height()
    pct = count / total_samples * 100

    # Position label above bar
    label_y = height * 1.5

    # Format: count and percentage
    label_text = f'{int(count):,}\n({pct:.2f}%)'

    ax.text(bar.get_x() + bar.get_width() / 2., label_y,
            label_text,
            ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.8))

# Add horizontal reference lines for key magnitudes
ax.axhline(y=100, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='100 samples')
ax.axhline(y=1000, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='1K samples')
ax.axhline(y=10000, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='10K samples')
ax.axhline(y=100000, color='blue', linestyle=':', linewidth=1.5, alpha=0.5, label='100K samples')

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Set y-axis limits for better visualization
ax.set_ylim(bottom=50, top=class_counts.max() * 2)

# Tight layout
plt.tight_layout()

# Save
output_path = os.path.join(OUTPUT_DIR, 'class_distribution_log_scale.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Plot saved to: {output_path}")

plt.show()

# Print summary
print("\n" + "=" * 70)
print("CLASS DISTRIBUTION SUMMARY (Log Scale Visualization)")
print("=" * 70)
print(f"{'Class':<20} {'Count':>12} {'Percentage':>12} {'Log10(Count)':>15}")
print("-" * 70)
for cls, count in class_counts.items():
    pct = count / total_samples * 100
    log_count = np.log10(count)
    print(f"{cls:<20} {count:>12,} {pct:>11.2f}% {log_count:>14.2f}")
print("-" * 70)
print(f"Total: {total_samples:,} samples")
print("=" * 70)