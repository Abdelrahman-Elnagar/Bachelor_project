import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

def analyze_stability_impact():
    print("Starting stability impact analysis...")
    
    # Load the combined data
    input_file = 'combined_stability_and_errors.csv'
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded data: {len(df)} rows")

    # Identify model error columns (those starting with 'absoluteerror_')
    error_cols = [col for col in df.columns if col.startswith('absoluteerror_')]
    print(f"Found {len(error_cols)} model error columns to analyze.")

    results = []

    for col in error_cols:
        # Extract model info from column name
        # Format: absoluteerror_modelname-variant-target
        model_name_full = col.replace('absoluteerror_', '')
        
        # Get clean model name for grouping (e.g., 'catboost', 'xgboost')
        model_family = model_name_full.split('-')[0]
        
        # Separate data into Stable vs Unstable groups using 'unstable_gmm'
        stable_errors = df[df['unstable_gmm'] == 0][col].dropna()
        unstable_errors = df[df['unstable_gmm'] == 1][col].dropna()

        # 1. Mean Error Ratio
        mean_stable = stable_errors.mean()
        mean_unstable = unstable_errors.mean()
        
        # Avoid division by zero
        if mean_stable == 0:
            impact_ratio = np.nan
        else:
            impact_ratio = mean_unstable / mean_stable

        # 2. Mann-Whitney U Test
        # Compare distributions of errors in Stable vs Unstable
        try:
            u_stat, p_value_u = stats.mannwhitneyu(stable_errors, unstable_errors, alternative='two-sided')
        except ValueError:
            u_stat, p_value_u = np.nan, np.nan

        # 3. Spearman Correlation
        # Correlation between error and continuous stability probability
        # Drop NaNs to ensure valid correlation
        valid_data = df[[col, 'stability_probability_gmm']].dropna()
        if len(valid_data) > 1:
            corr, p_value_corr = stats.spearmanr(valid_data[col], valid_data['stability_probability_gmm'])
        else:
            corr, p_value_corr = np.nan, np.nan

        results.append({
            'Model_Full': model_name_full,
            'Model_Family': model_family,
            'Mean_Error_Stable': mean_stable,
            'Mean_Error_Unstable': mean_unstable,
            'Impact_Ratio': impact_ratio,
            'MannWhitney_P_Value': p_value_u,
            'Spearman_Correlation': corr,
            'Spearman_P_Value': p_value_corr,
            'Is_Significant': p_value_u < 0.05  # Using Mann-Whitney as primary significance test
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Impact Ratio (descending) to see most affected models first
    results_df = results_df.sort_values('Impact_Ratio', ascending=False)

    # Save summary CSV
    output_csv = 'stability_impact_summary.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nSummary report saved to: {output_csv}")
    
    # Print top 5 most affected models
    print("\nTop 5 Models Most Affected by Instability (Highest Impact Ratio):")
    print(results_df[['Model_Full', 'Impact_Ratio', 'Spearman_Correlation']].head(5).to_string(index=False))

    # Generate Visualization
    create_visualization(results_df)

def create_visualization(results_df):
    print("\nGenerating visualization...")
    
    # Aggregate by Model Family to see broader trends
    family_stats = results_df.groupby('Model_Family')['Impact_Ratio'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    # We can plot individual models or families. Plotting top 20 most affected models might be clearer.
    # Let's plot the top 15 most affected models to keep it readable
    top_plot_df = results_df.head(20)
    
    sns.barplot(x='Impact_Ratio', y='Model_Full', data=top_plot_df, palette='viridis')
    
    plt.title('Top 20 Models: Impact of Weather Instability on Prediction Error\n(Ratio of Mean Error in Unstable vs. Stable Conditions)', fontsize=14, fontweight='bold')
    plt.xlabel('Impact Ratio (Unstable Error / Stable Error)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    
    # Add a vertical line at Ratio = 1 (No impact)
    plt.axvline(x=1, color='red', linestyle='--', label='No Impact (Ratio=1)')
    plt.legend()
    
    plt.tight_layout()
    
    output_plot = 'stability_impact_ranking.png'
    plt.savefig(output_plot, dpi=300)
    print(f"Visualization saved to: {output_plot}")

if __name__ == "__main__":
    analyze_stability_impact()




