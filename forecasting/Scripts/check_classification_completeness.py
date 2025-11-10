#!/usr/bin/env python3
"""Check if all periods are classified as stable or unstable"""

import pandas as pd
from pathlib import Path

# Load stability labels
stability_file = Path(__file__).parent.parent.parent / "Data" / "processed" / "stability_labels.csv"

if not stability_file.exists():
    print(f"ERROR: Stability labels file not found: {stability_file}")
    exit(1)

df = pd.read_csv(stability_file)
df['datetime'] = pd.to_datetime(df['datetime'])

print("="*60)
print("CLASSIFICATION COMPLETENESS CHECK")
print("="*60)

# Check temporal coverage
print(f"\nTemporal Coverage:")
print(f"  Total hours: {len(df)}")
print(f"  Expected for 2024 (leap year): 8784")
print(f"  Coverage: {len(df)/8784*100:.2f}%")
print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Check for gaps
df_sorted = df.sort_values('datetime')
gaps = df_sorted['datetime'].diff().dt.total_seconds() / 3600
non_hourly = (gaps != 1.0).sum() - 1  # -1 because first row has NaN
print(f"  Gaps in timeline: {non_hourly}")

# Check classification completeness
print(f"\nClassification Status:")
print(f"  unstable_gmm missing values: {df['unstable_gmm'].isna().sum()}")
print(f"  unstable_gmm unique values: {sorted(df['unstable_gmm'].unique())}")

# Value distribution
stable_count = (df['unstable_gmm'] == 0).sum()
unstable_count = (df['unstable_gmm'] == 1).sum()
total = len(df)

print(f"\nClassification Distribution:")
print(f"  Stable (0): {stable_count} hours ({stable_count/total*100:.2f}%)")
print(f"  Unstable (1): {unstable_count} hours ({unstable_count/total*100:.2f}%)")
print(f"  Total classified: {stable_count + unstable_count} hours")

# Check GMM cluster labels
print(f"\nGMM Cluster Details:")
print(f"  stability_label_gmm unique values: {sorted(df['stability_label_gmm'].unique())}")
print(f"  Cluster distribution:")
cluster_counts = df['stability_label_gmm'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"    Cluster {cluster}: {count} hours ({count/total*100:.2f}%)")

# Verify binary mapping
print(f"\nBinary Classification Mapping:")
for cluster in sorted(df['stability_label_gmm'].unique()):
    cluster_df = df[df['stability_label_gmm'] == cluster]
    unstable_ratio = cluster_df['unstable_gmm'].mean()
    print(f"  Cluster {cluster}: {unstable_ratio*100:.1f}% marked as unstable")

# Final verification
print(f"\n" + "="*60)
if df['unstable_gmm'].isna().sum() == 0 and len(df) == 8784:
    print("OK ALL PERIODS CLASSIFIED")
    print(f"  - Complete temporal coverage (8784/8784 hours)")
    print(f"  - No missing classifications")
    print(f"  - Binary classification: {stable_count} stable, {unstable_count} unstable")
else:
    print("WARNING: INCOMPLETE CLASSIFICATION")
    if df['unstable_gmm'].isna().sum() > 0:
        print(f"  - Missing classifications: {df['unstable_gmm'].isna().sum()}")
    if len(df) != 8784:
        print(f"  - Missing hours: {8784 - len(df)}")
print("="*60)

