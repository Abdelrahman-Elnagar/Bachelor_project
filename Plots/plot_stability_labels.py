import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Read the stability labels data
print("Reading stability_labels.csv...")
df = pd.read_csv('../Data/processed/stability_labels.csv')

# Convert datetime column to datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Filter data from 2024-10-19 18:00:00 onwards (test set period)
start_date = pd.Timestamp('2024-10-19 18:00:00')
df_filtered = df[df['datetime'] >= start_date].copy()

print(f"Filtered data: {len(df_filtered)} rows")
print(f"Date range: {df_filtered['datetime'].iloc[0]} to {df_filtered['datetime'].iloc[-1]}")

# Create the plot
fig, ax = plt.subplots(figsize=(16, 8))

# Plot the three variables with different colors
ax.plot(df_filtered['datetime'], df_filtered['stability_probability_gmm'], 
        label='Stability Probability (GMM)', color='blue', linewidth=1.5, alpha=0.8)

ax.plot(df_filtered['datetime'], df_filtered['unstable_gmm'], 
        label='Unstable (GMM)', color='orange', linewidth=1.5, alpha=0.8)

ax.plot(df_filtered['datetime'], df_filtered['stability_label_kmeans'], 
        label='Stability Label (K-Means)', color='green', linewidth=1.5, alpha=0.8)

# Formatting
ax.set_xlabel('Datetime', fontsize=12, fontweight='bold')
ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_title('Stability Labels Analysis (Test Period: Oct 19 - Dec 31, 2024)', 
             fontsize=14, fontweight='bold', pad=20)

# Format x-axis to show dates nicely
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45, ha='right')

# Add legend
ax.legend(loc='best', fontsize=10, framealpha=0.9)

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Tight layout
plt.tight_layout()

# Save the plot
output_file = 'stability_labels_test_period.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_file}")

# Show the plot
plt.show()

print("Done!")

