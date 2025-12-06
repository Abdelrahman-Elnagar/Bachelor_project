"""
Main training script for energy prediction models.
Trains 7 models (Prophet, TBATS, SARIMAX, XGBoost, LightGBM, CatBoost, Random Forest)
on Solar and Wind datasets with hyperparameter tuning using Hyperopt.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from hyperopt_tuning import tune_hyperparameters

# Model names
MODELS = ['Prophet', 'TBATS', 'SARIMAX', 'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']

# Hyperparameter tuning max evaluations
MAX_EVALS_STATISTICAL = 50  # For Prophet, TBATS, SARIMAX
MAX_EVALS_ML = 100  # For ML models


def load_and_split_data(data_path, target_col, train_start='2019-01-01', train_end='2023-12-31 23:00:00', 
                       test_start='2024-01-01 00:00:00', test_end='2024-12-31 23:00:00'):
    """
    Load data and split into training (2019-2023) and test (2024) sets.
    
    Args:
        data_path: Path to CSV file
        target_col: Name of target column
        train_start: Start date for training
        train_end: End date for training
        test_start: Start date for testing
        test_end: End date for testing
        
    Returns:
        X_train, y_train, X_test, y_test, test_datetimes
    """
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Convert datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Split data
    train_mask = (df['datetime'] >= train_start) & (df['datetime'] <= train_end)
    test_mask = (df['datetime'] >= test_start) & (df['datetime'] <= test_end)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Training data: {len(train_df)} samples ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"Test data: {len(test_df)} samples ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    test_datetimes = test_df['datetime']
    
    return X_train, y_train, X_test, y_test, test_datetimes


def calculate_rmse_per_timestamp(y_true, y_pred):
    """Calculate RMSE for each timestamp."""
    return np.sqrt(np.square(y_true - y_pred))


def train_all_models(data_path, target_col, dataset_name, max_evals_stat=MAX_EVALS_STATISTICAL, max_evals_ml=MAX_EVALS_ML):
    """
    Train all models for a given dataset.
    
    Args:
        data_path: Path to CSV file
        target_col: Name of target column
        dataset_name: Name of dataset (for output files)
        max_evals_stat: Max evaluations for statistical models
        max_evals_ml: Max evaluations for ML models
        
    Returns:
        errors_df: DataFrame with RMSE per timestamp for each model
    """
    print("\n" + "="*80)
    print(f"TRAINING MODELS FOR {dataset_name.upper()}")
    print("="*80)
    
    # Load and split data
    X_train, y_train, X_test, y_test, test_datetimes = load_and_split_data(data_path, target_col)
    
    # Create validation set from last 20% of training data
    val_size = int(len(X_train) * 0.2)
    X_val = X_train.iloc[-val_size:].copy()
    y_val = y_train.iloc[-val_size:].copy()
    X_train_subset = X_train.iloc[:-val_size].copy()
    y_train_subset = y_train.iloc[:-val_size].copy()
    
    print(f"\nUsing {len(X_train_subset)} samples for training, {len(X_val)} for validation")
    
    # Dictionary to store results
    results = {
        'datetime': test_datetimes.values,
        'y_true': y_test.values
    }
    
    # Train each model
    for model_name in MODELS:
        print(f"\n{'='*80}")
        print(f"Training {model_name}...")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Determine max_evals based on model type
            if model_name in ['Prophet', 'TBATS', 'SARIMAX']:
                max_evals = max_evals_stat
            else:
                max_evals = max_evals_ml
            
            # Tune hyperparameters and get best model
            print(f"Tuning hyperparameters (max_evals={max_evals})...")
            best_params, best_model, best_val_rmse = tune_hyperparameters(
                model_name, X_train_subset, y_train_subset, X_val, y_val, max_evals=max_evals
            )
            
            print(f"\nBest validation RMSE: {best_val_rmse:.4f}")
            print(f"Best parameters: {best_params}")
            
            # Retrain on full training set with best parameters
            print("Retraining on full training set...")
            if model_name == 'SARIMAX':
                order = best_params['order']
                seasonal_order = best_params['seasonal_order']
                best_model.fit(X_train, y_train, order=order, seasonal_order=seasonal_order)
            else:
                best_model.fit(X_train, y_train, **best_params)
            
            # Make predictions on test set
            print("Making predictions on test set...")
            y_pred = best_model.predict(X_test)
            
            # Calculate overall RMSE
            overall_rmse = np.sqrt(np.mean(np.square(y_test.values - y_pred)))
            print(f"\nOverall Test RMSE: {overall_rmse:.4f}")
            
            # Calculate RMSE per timestamp
            rmse_per_timestamp = calculate_rmse_per_timestamp(y_test.values, y_pred)
            results[f'{model_name}_RMSE'] = rmse_per_timestamp
            
            elapsed_time = time.time() - start_time
            print(f"Completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"ERROR training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fill with NaN if model fails
            results[f'{model_name}_RMSE'] = np.nan
    
    # Create results DataFrame
    errors_df = pd.DataFrame(results)
    
    # Reorder columns: datetime, then model RMSE columns
    model_rmse_cols = [f'{model}_RMSE' for model in MODELS]
    errors_df = errors_df[['datetime', 'y_true'] + model_rmse_cols]
    
    return errors_df


def main():
    """Main function to train all models on all datasets."""
    print("="*80)
    print("ENERGY PREDICTION MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define datasets
    datasets = [
        {
            'name': 'Solar',
            'path': '../Trainingdata_5years/Solar_data.csv',
            'target': 'solar',
            'output': 'Solar_errors.csv'
        },
        {
            'name': 'Wind_Offshore',
            'path': '../Trainingdata_5years/Wind_data.csv',
            'target': 'Wind Offshore',
            'output': 'Wind_Offshore_errors.csv'
        },
        {
            'name': 'Wind_Onshore',
            'path': '../Trainingdata_5years/Wind_data.csv',
            'target': 'Wind Onshore',
            'output': 'Wind_Onshore_errors.csv'
        }
    ]
    
    # Train models for each dataset
    for dataset in datasets:
        try:
            errors_df = train_all_models(
                dataset['path'],
                dataset['target'],
                dataset['name']
            )
            
            # Save errors to CSV
            output_path = dataset['output']
            errors_df.to_csv(output_path, index=False)
            print(f"\n{'='*80}")
            print(f"Saved errors to {output_path}")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"\nERROR processing {dataset['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
