import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pathlib import Path

# Read the stability labels data
print("Reading stability_labels.csv...")
stability_df = pd.read_csv('../../../../Data/processed/stability_labels.csv')
stability_df['datetime'] = pd.to_datetime(stability_df['datetime'])

# Filter stability data from 2024-10-19 18:00:00 onwards (test set period)
start_date = pd.Timestamp('2024-10-19 18:00:00')
stability_filtered = stability_df[stability_df['datetime'] >= start_date].copy()
stability_filtered = stability_filtered.reset_index(drop=True)

print(f"Stability data: {len(stability_filtered)} rows")
print(f"Date range: {stability_filtered['datetime'].iloc[0]} to {stability_filtered['datetime'].iloc[-1]}\n")

# Get all CSV files in the parent directory (detailed_results)
csv_files = [f for f in os.listdir('..') if f.endswith('.csv') and f != 'all_predictions.csv']

print(f"Found {len(csv_files)} CSV files to process\n")

success_count = 0
error_count = 0

for csv_file in sorted(csv_files):
    try:
        print(f"Processing: {csv_file}")
        
        # Read the model predictions file
        model_df = pd.read_csv(f'../{csv_file}')
        model_df['datetime'] = pd.to_datetime(model_df['datetime'])
        
        # Get the number of rows
        num_rows = len(model_df)
        
        # Match stability data to model data based on datetime
        # Use only the rows that match the model's datetime range
        matched_stability = stability_filtered.iloc[:num_rows].copy()
        
        # Normalize absolute_error to 0-1 range
        abs_error = model_df['absolute_error'].values
        if abs_error.max() > abs_error.min():
            abs_error_normalized = (abs_error - abs_error.min()) / (abs_error.max() - abs_error.min())
        else:
            abs_error_normalized = abs_error * 0  # All zeros if constant
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot the three stability variables
        ax.plot(matched_stability['datetime'], matched_stability['stability_probability_gmm'], 
                label='Stability Probability (GMM)', color='blue', linewidth=1.5, alpha=0.8)
        
        ax.plot(matched_stability['datetime'], matched_stability['unstable_gmm'], 
                label='Unstable (GMM)', color='orange', linewidth=1.5, alpha=0.8)
        
        ax.plot(matched_stability['datetime'], matched_stability['stability_label_kmeans'], 
                label='Stability Label (K-Means)', color='green', linewidth=1.5, alpha=0.8)
        
        # Plot the normalized absolute error from the model
        ax.plot(model_df['datetime'], abs_error_normalized, 
                label='Absolute Error (Normalized)', color='red', linewidth=1.5, alpha=0.8)
        
        # Extract model name and experiment name for title
        model_name = model_df['model'].iloc[0] if 'model' in model_df.columns else csv_file.split('.')[0]
        experiment_name = model_df['experiment'].iloc[0] if 'experiment' in model_df.columns else ''
        
        # Formatting
        ax.set_xlabel('Datetime', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        title = f'{model_name.upper()} - {experiment_name.replace("_", " ").title()}\nStability Analysis with Prediction Error'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
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
        
        # Create output filename based on CSV filename
        plot_name = csv_file.replace('.csv', '_plot.png')
        output_file = plot_name
        
        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_file}")
        success_count += 1
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        error_count += 1

print(f"\n{'='*70}")
print(f"Summary:")
print(f"  Successfully created: {success_count} plots")
print(f"  Errors: {error_count} plots")
print(f"{'='*70}")

