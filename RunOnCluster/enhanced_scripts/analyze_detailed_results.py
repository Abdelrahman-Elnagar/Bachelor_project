"""
analyze_detailed_results.py
----------------------------
Comprehensive analysis script for detailed predictions and residuals.
Run this after model training to generate insights from per-sample errors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils_detailed_metrics import (
    get_summary_statistics, 
    analyze_residuals_by_range,
    export_summary_report,
    MASTER_FILE,
    DETAILED_DIR
)
import os

def load_all_predictions():
    """Load the master predictions file."""
    if not os.path.exists(MASTER_FILE):
        print(f"[ERROR] Master file not found: {MASTER_FILE}")
        print("       Run model scripts with detailed tracking first.")
        return None
    
    df = pd.read_csv(MASTER_FILE)
    print(f"[LOADED] {len(df):,} predictions from {df['model'].nunique()} models")
    return df

def compare_models_by_experiment():
    """Compare all models for each experiment."""
    df = load_all_predictions()
    if df is None:
        return
    
    summary = get_summary_statistics()
    
    print("\n" + "=" * 100)
    print("MODEL COMPARISON BY EXPERIMENT")
    print("=" * 100)
    
    for experiment in df['experiment'].unique():
        print(f"\n{experiment.upper()}:")
        print("-" * 100)
        
        exp_data = summary[summary['experiment'] == experiment].sort_values('MAE')
        
        print(exp_data[['model', 'MAE', 'RMSE', 'MAE_std', 'n_samples']].to_string(index=False))
        
        # Best model
        best_model = exp_data.iloc[0]
        print(f"\n→ BEST: {best_model['model']} (MAE: {best_model['MAE']:.2f}, RMSE: {best_model['RMSE']:.2f})")

def analyze_error_distribution(model_name=None, experiment_name=None):
    """Analyze the distribution of errors."""
    df = load_all_predictions()
    if df is None:
        return
    
    if model_name:
        df = df[df['model'] == model_name]
    if experiment_name:
        df = df[df['experiment'] == experiment_name]
    
    print("\n" + "=" * 100)
    print(f"ERROR DISTRIBUTION ANALYSIS")
    if model_name:
        print(f"Model: {model_name}")
    if experiment_name:
        print(f"Experiment: {experiment_name}")
    print("=" * 100)
    
    print(f"\nAbsolute Error Statistics:")
    print(df['absolute_error'].describe())
    
    print(f"\nResidual Statistics:")
    print(df['residual'].describe())
    
    print(f"\nPercentage Error Statistics:")
    print(df['percentage_error'].describe())
    
    # Quartile analysis
    quartiles = df['absolute_error'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(f"\nError Quartiles:")
    for q, val in quartiles.items():
        print(f"  {int(q*100)}th percentile: {val:.4f}")

def find_worst_predictions(n=20):
    """Find the worst predictions across all models."""
    df = load_all_predictions()
    if df is None:
        return
    
    print("\n" + "=" * 100)
    print(f"TOP {n} WORST PREDICTIONS (Highest Absolute Errors)")
    print("=" * 100)
    
    worst = df.nlargest(n, 'absolute_error')[['model', 'experiment', 'sample_index', 
                                                 'actual', 'predicted', 'absolute_error']]
    print(worst.to_string(index=False))

def compare_by_value_range():
    """Compare model performance across different value ranges."""
    df = load_all_predictions()
    if df is None:
        return
    
    print("\n" + "=" * 100)
    print("PERFORMANCE BY VALUE RANGE (Binned Analysis)")
    print("=" * 100)
    
    for experiment in df['experiment'].unique():
        print(f"\n{experiment.upper()}:")
        exp_df = df[df['experiment'] == experiment]
        
        # Create bins
        exp_df['value_bin'] = pd.qcut(exp_df['actual'], q=5, duplicates='drop')
        
        # Analyze by model and bin
        for model in exp_df['model'].unique():
            model_df = exp_df[exp_df['model'] == model]
            analysis = model_df.groupby('value_bin')['absolute_error'].mean()
            
            print(f"\n  {model}:")
            for bin_range, mae in analysis.items():
                print(f"    {bin_range}: MAE = {mae:.4f}")

def generate_ranking_table():
    """Generate a ranking table of all models across all experiments."""
    df = load_all_predictions()
    if df is None:
        return
    
    summary = get_summary_statistics()
    
    print("\n" + "=" * 100)
    print("OVERALL MODEL RANKING")
    print("=" * 100)
    
    # Rank by average MAE across all experiments
    ranking = summary.groupby('model').agg({
        'MAE': 'mean',
        'RMSE': 'mean',
        'MAE_std': 'mean',
        'n_samples': 'sum'
    }).sort_values('MAE')
    
    ranking.columns = ['Avg_MAE', 'Avg_RMSE', 'Avg_MAE_std', 'Total_samples']
    ranking['Rank'] = range(1, len(ranking) + 1)
    
    print(ranking.to_string())
    
    # Save to file
    ranking_file = os.path.join(DETAILED_DIR, "model_rankings.csv")
    ranking.to_csv(ranking_file)
    print(f"\n[SAVED] Rankings saved to {ranking_file}")

def export_all_analyses():
    """Export all analysis to files."""
    print("\n" + "=" * 100)
    print("EXPORTING ALL ANALYSES")
    print("=" * 100)
    
    # Summary report
    export_summary_report()
    
    # Summary statistics
    summary = get_summary_statistics()
    if summary is not None:
        summary_file = os.path.join(DETAILED_DIR, "summary_statistics.csv")
        summary.to_csv(summary_file, index=False)
        print(f"[SAVED] Summary statistics: {summary_file}")
    
    # Model rankings
    generate_ranking_table()
    
    # Per-experiment winners
    df = load_all_predictions()
    if df is not None:
        winners = []
        summary = get_summary_statistics()
        for experiment in df['experiment'].unique():
            exp_summary = summary[summary['experiment'] == experiment]
            best_mae = exp_summary.loc[exp_summary['MAE'].idxmin()]
            best_rmse = exp_summary.loc[exp_summary['RMSE'].idxmin()]
            
            winners.append({
                'experiment': experiment,
                'best_MAE_model': best_mae['model'],
                'best_MAE_value': best_mae['MAE'],
                'best_RMSE_model': best_rmse['model'],
                'best_RMSE_value': best_rmse['RMSE']
            })
        
        winners_df = pd.DataFrame(winners)
        winners_file = os.path.join(DETAILED_DIR, "experiment_winners.csv")
        winners_df.to_csv(winners_file, index=False)
        print(f"[SAVED] Experiment winners: {winners_file}")

def main():
    """Run all analyses."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE DETAILED RESULTS ANALYSIS")
    print("=" * 100)
    
    # Load and validate data
    df = load_all_predictions()
    if df is None:
        return
    
    # Run analyses
    compare_models_by_experiment()
    analyze_error_distribution()
    find_worst_predictions(n=20)
    compare_by_value_range()
    generate_ranking_table()
    
    # Export everything
    export_all_analyses()
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print(f"All results saved to: {DETAILED_DIR}/")
    print("=" * 100)

if __name__ == "__main__":
    main()

