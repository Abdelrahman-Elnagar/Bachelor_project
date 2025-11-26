"""
utils_detailed_metrics.py
--------------------------
Utility module for saving detailed per-sample predictions and residuals.
Creates a unified CSV with all model predictions for comprehensive error analysis.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

DETAILED_DIR = "detailed_results"
MASTER_FILE = os.path.join(DETAILED_DIR, "all_predictions.csv")

os.makedirs(DETAILED_DIR, exist_ok=True)

def save_detailed_predictions(model_name, experiment_name, y_true, y_pred, 
                               best_params=None, additional_info=None):
    """
    Saves detailed per-sample predictions and residuals to CSV.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., 'xgboost', 'lstm', 'nhits')
    experiment_name : str
        Name of the experiment (e.g., 'with_wsi_solar')
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    best_params : dict, optional
        Best hyperparameters found
    additional_info : dict, optional
        Any additional metadata
    
    Returns:
    --------
    str : Path to saved file
    """
    
    # Ensure arrays
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    # Calculate residuals and errors
    residuals = y_pred - y_true
    absolute_errors = np.abs(residuals)
    squared_errors = residuals ** 2
    percentage_errors = (absolute_errors / (y_true + 1e-8)) * 100  # Avoid division by zero
    
    # Create detailed dataframe
    df = pd.DataFrame({
        'model': model_name,
        'experiment': experiment_name,
        'sample_index': range(len(y_true)),
        'actual': y_true,
        'predicted': y_pred,
        'residual': residuals,
        'absolute_error': absolute_errors,
        'squared_error': squared_errors,
        'percentage_error': percentage_errors,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Add parameters as separate columns if provided
    if best_params:
        for key, value in best_params.items():
            df[f'param_{key}'] = str(value)
    
    # Add additional info if provided
    if additional_info:
        for key, value in additional_info.items():
            df[f'info_{key}'] = str(value)
    
    # Save individual model+experiment file
    individual_file = os.path.join(DETAILED_DIR, f"{model_name}_{experiment_name}.csv")
    df.to_csv(individual_file, index=False)
    print(f"   [DETAILED] Saved {len(df)} predictions to {individual_file}")
    
    # Append to master file
    if os.path.exists(MASTER_FILE):
        df.to_csv(MASTER_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(MASTER_FILE, mode='w', header=True, index=False)
    print(f"   [DETAILED] Appended to master file: {MASTER_FILE}")
    
    return individual_file


def get_summary_statistics(model_name=None, experiment_name=None):
    """
    Generate summary statistics from the master predictions file.
    
    Parameters:
    -----------
    model_name : str, optional
        Filter by specific model
    experiment_name : str, optional
        Filter by specific experiment
    
    Returns:
    --------
    pd.DataFrame : Summary statistics
    """
    
    if not os.path.exists(MASTER_FILE):
        print(f"[WARNING] Master file not found: {MASTER_FILE}")
        return None
    
    df = pd.read_csv(MASTER_FILE)
    
    # Apply filters
    if model_name:
        df = df[df['model'] == model_name]
    if experiment_name:
        df = df[df['experiment'] == experiment_name]
    
    # Group by model and experiment
    summary = df.groupby(['model', 'experiment']).agg({
        'absolute_error': ['mean', 'std', 'min', 'max', 'median'],
        'squared_error': 'mean',  # MSE
        'percentage_error': ['mean', 'median'],
        'sample_index': 'count'  # Number of samples
    }).round(4)
    
    summary.columns = ['MAE', 'MAE_std', 'MAE_min', 'MAE_max', 'MAE_median', 
                       'MSE', 'MAPE', 'MdAPE', 'n_samples']
    
    # Calculate RMSE
    summary['RMSE'] = np.sqrt(summary['MSE'])
    
    return summary.reset_index()


def analyze_residuals_by_range(model_name, experiment_name, bins=10):
    """
    Analyze residuals by actual value ranges (binned analysis).
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    experiment_name : str
        Name of the experiment
    bins : int
        Number of bins for actual values
    
    Returns:
    --------
    pd.DataFrame : Residual statistics by value range
    """
    
    if not os.path.exists(MASTER_FILE):
        print(f"[WARNING] Master file not found: {MASTER_FILE}")
        return None
    
    df = pd.read_csv(MASTER_FILE)
    df = df[(df['model'] == model_name) & (df['experiment'] == experiment_name)]
    
    if len(df) == 0:
        print(f"[WARNING] No data found for {model_name} - {experiment_name}")
        return None
    
    # Create bins based on actual values
    df['value_bin'] = pd.qcut(df['actual'], q=bins, duplicates='drop')
    
    # Analyze by bin
    analysis = df.groupby('value_bin').agg({
        'actual': ['mean', 'min', 'max', 'count'],
        'absolute_error': ['mean', 'std'],
        'residual': ['mean', 'std']
    }).round(4)
    
    analysis.columns = ['actual_mean', 'actual_min', 'actual_max', 'count',
                        'MAE', 'MAE_std', 'bias', 'bias_std']
    
    return analysis.reset_index()


def clear_master_file():
    """Remove the master file to start fresh."""
    if os.path.exists(MASTER_FILE):
        os.remove(MASTER_FILE)
        print(f"[CLEARED] Removed {MASTER_FILE}")
    else:
        print(f"[INFO] No master file to clear")


def export_summary_report(output_file="detailed_results/summary_report.txt"):
    """
    Generate a comprehensive text report from all predictions.
    
    Parameters:
    -----------
    output_file : str
        Path to output report file
    """
    
    if not os.path.exists(MASTER_FILE):
        print(f"[WARNING] Master file not found: {MASTER_FILE}")
        return
    
    df = pd.read_csv(MASTER_FILE)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE MODEL PREDICTIONS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Predictions: {len(df):,}\n")
        f.write(f"Number of Models: {df['model'].nunique()}\n")
        f.write(f"Number of Experiments: {df['experiment'].nunique()}\n")
        f.write(f"Models: {', '.join(df['model'].unique())}\n")
        f.write(f"Experiments: {', '.join(df['experiment'].unique())}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("SUMMARY STATISTICS BY MODEL AND EXPERIMENT\n")
        f.write("-" * 80 + "\n\n")
        
        summary = get_summary_statistics()
        f.write(summary.to_string(index=False))
        f.write("\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("BEST MODELS BY METRIC\n")
        f.write("-" * 80 + "\n\n")
        
        for experiment in df['experiment'].unique():
            exp_summary = get_summary_statistics(experiment_name=experiment)
            
            f.write(f"\n{experiment.upper()}:\n")
            f.write(f"  Best MAE:  {exp_summary.loc[exp_summary['MAE'].idxmin(), 'model']}")
            f.write(f" ({exp_summary['MAE'].min():.2f})\n")
            f.write(f"  Best RMSE: {exp_summary.loc[exp_summary['RMSE'].idxmin(), 'model']}")
            f.write(f" ({exp_summary['RMSE'].min():.2f})\n")
    
    print(f"[REPORT] Summary report saved to {output_file}")

