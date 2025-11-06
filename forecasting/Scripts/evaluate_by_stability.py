#!/usr/bin/env python3
"""
Evaluate Forecasting Models by Weather Stability Regime

This script evaluates model performance stratified by stability regime,
performs statistical tests, computes effect sizes, and generates SHAP analyses.

Author: Energy Forecasting Pipeline
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
import logging
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
np.random.seed(42)


class StabilityEvaluator:
    """Evaluate models stratified by stability regime"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.base_dir = Path(__file__).parent.parent
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.base_dir / "figures"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.logs_dir = self.base_dir / "logs"
        
        for dir_path in [self.results_dir, self.figures_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        log_file = self.logs_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_training_results(self, source_type: str) -> Dict:
        """Load training results"""
        results_file = self.results_dir / f"{source_type}_training_results.pkl"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Training results not found: {results_file}")
        
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        return results
    
    def load_stability_labels(self) -> pd.DataFrame:
        """Load stability labels"""
        stability_file = Path(__file__).parent.parent.parent / "Data" / "processed" / "stability_labels.csv"
        
        if not stability_file.exists():
            raise FileNotFoundError(f"Stability labels not found: {stability_file}")
        
        df = pd.read_csv(stability_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    def compute_metrics(self, predictions: List[float], actuals: List[float]) -> Dict:
        """Compute forecasting metrics"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        predictions = predictions[mask]
        actuals = actuals[mask]
        
        if len(predictions) == 0:
            return {}
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        # MAPE
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
        
        # sMAPE
        smape = 100 * np.mean(2 * np.abs(actuals - predictions) / (np.abs(actuals) + np.abs(predictions) + 1e-8))
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'sMAPE': smape,
            'n_samples': len(predictions)
        }
    
    def evaluate_by_stability(self, results: Dict, source_type: str) -> Dict:
        """Evaluate models stratified by stability regime"""
        self.logger.info("="*60)
        self.logger.info(f"EVALUATING {source_type.upper()} MODELS BY STABILITY")
        self.logger.info("="*60)
        
        # Load stability labels
        stability_df = self.load_stability_labels()
        
        evaluation_results = {}
        
        for model_name, model_results in results.items():
            if model_results is None:
                continue
            
            self.logger.info(f"\nEvaluating {model_name}...")
            
            # Create predictions dataframe
            pred_df = pd.DataFrame({
                'datetime': model_results['timestamps'],
                'prediction': model_results['predictions'],
                'actual': model_results['actuals']
            })
            pred_df['datetime'] = pd.to_datetime(pred_df['datetime'])
            
            # Merge with stability labels
            pred_df = pred_df.merge(stability_df, on='datetime', how='left')
            
            # Compute errors
            pred_df['error'] = pred_df['prediction'] - pred_df['actual']
            pred_df['abs_error'] = np.abs(pred_df['error'])
            
            # Split by stability regime
            stable_mask = pred_df['unstable_gmm'] == 0
            unstable_mask = pred_df['unstable_gmm'] == 1
            
            stable_errors = pred_df[stable_mask]['abs_error'].values
            unstable_errors = pred_df[unstable_mask]['abs_error'].values
            
            # Remove NaN
            stable_errors = stable_errors[~np.isnan(stable_errors)]
            unstable_errors = unstable_errors[~np.isnan(unstable_errors)]
            
            if len(stable_errors) == 0 or len(unstable_errors) == 0:
                self.logger.warning(f"Insufficient data for {model_name}. Skipping.")
                continue
            
            # Compute metrics for each regime
            stable_metrics = self.compute_metrics(
                pred_df[stable_mask]['prediction'].values,
                pred_df[stable_mask]['actual'].values
            )
            unstable_metrics = self.compute_metrics(
                pred_df[unstable_mask]['prediction'].values,
                pred_df[unstable_mask]['actual'].values
            )
            
            # Overall metrics
            overall_metrics = self.compute_metrics(
                pred_df['prediction'].values,
                pred_df['actual'].values
            )
            
            # Statistical tests
            # Mann-Whitney U test (non-parametric)
            statistic_mwu, pvalue_mwu = stats.mannwhitneyu(
                stable_errors, unstable_errors, alternative='two-sided'
            )
            
            # Welch's t-test (doesn't assume equal variances)
            statistic_welch, pvalue_welch = stats.ttest_ind(
                stable_errors, unstable_errors, equal_var=False
            )
            
            # Effect sizes
            # Cohen's d
            pooled_std = np.sqrt(
                (np.var(stable_errors, ddof=1) + np.var(unstable_errors, ddof=1)) / 2
            )
            cohens_d = (np.mean(unstable_errors) - np.mean(stable_errors)) / (pooled_std + 1e-8)
            
            # Percentage increase
            pct_increase = ((np.mean(unstable_errors) - np.mean(stable_errors)) / 
                           (np.mean(stable_errors) + 1e-8)) * 100
            
            # Rank-biserial correlation
            all_errors = np.concatenate([stable_errors, unstable_errors])
            ranks = stats.rankdata(all_errors)
            stable_ranks = ranks[:len(stable_errors)]
            unstable_ranks = ranks[len(stable_errors):]
            rank_biserial = (np.mean(unstable_ranks) - np.mean(stable_ranks)) / len(all_errors)
            
            evaluation_results[model_name] = {
                'overall': overall_metrics,
                'stable': stable_metrics,
                'unstable': unstable_metrics,
                'statistical_tests': {
                    'mann_whitney_u': {
                        'statistic': float(statistic_mwu),
                        'pvalue': float(pvalue_mwu)
                    },
                    'welch_t_test': {
                        'statistic': float(statistic_welch),
                        'pvalue': float(pvalue_welch)
                    }
                },
                'effect_sizes': {
                    'cohens_d': float(cohens_d),
                    'percentage_increase': float(pct_increase),
                    'rank_biserial_correlation': float(rank_biserial)
                },
                'n_stable': len(stable_errors),
                'n_unstable': len(unstable_errors)
            }
            
            self.logger.info(f"  Stable MAE: {stable_metrics['MAE']:.2f}")
            self.logger.info(f"  Unstable MAE: {unstable_metrics['MAE']:.2f}")
            self.logger.info(f"  Cohen's d: {cohens_d:.3f}")
            self.logger.info(f"  p-value (MWU): {pvalue_mwu:.4f}")
        
        return evaluation_results
    
    def compute_robustness_metrics(self, evaluation_results: Dict) -> Dict:
        """Compute robustness metrics (RPD, CV, ranking stability)"""
        self.logger.info("\nComputing robustness metrics...")
        
        robustness = {}
        
        for model_name, results in evaluation_results.items():
            stable_mae = results['stable']['MAE']
            unstable_mae = results['unstable']['MAE']
            
            # Relative Performance Degradation (RPD)
            rpd = ((unstable_mae - stable_mae) / (stable_mae + 1e-8)) * 100
            
            # Coefficient of Variation
            cv = (np.std([stable_mae, unstable_mae]) / 
                  (np.mean([stable_mae, unstable_mae]) + 1e-8)) * 100
            
            robustness[model_name] = {
                'RPD': float(rpd),
                'coefficient_of_variation': float(cv)
            }
        
        return robustness
    
    def save_results(self, evaluation_results: Dict, robustness: Dict, source_type: str):
        """Save evaluation results"""
        output = {
            'source_type': source_type,
            'evaluation_results': evaluation_results,
            'robustness_metrics': robustness,
            'timestamp': datetime.now().isoformat()
        }
        
        output_file = self.results_dir / f"{source_type}_evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        self.logger.info(f"Saved evaluation results to: {output_file}")
        
        # Also save as text summary
        summary_file = self.results_dir / f"{source_type}_evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Evaluation Summary for {source_type.upper()}\n")
            f.write("="*60 + "\n\n")
            
            for model_name, results in evaluation_results.items():
                f.write(f"{model_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Overall MAE: {results['overall']['MAE']:.2f}\n")
                f.write(f"Stable MAE: {results['stable']['MAE']:.2f}\n")
                f.write(f"Unstable MAE: {results['unstable']['MAE']:.2f}\n")
                f.write(f"Cohen's d: {results['effect_sizes']['cohens_d']:.3f}\n")
                f.write(f"RPD: {robustness.get(model_name, {}).get('RPD', 0):.2f}%\n")
                f.write(f"p-value (MWU): {results['statistical_tests']['mann_whitney_u']['pvalue']:.4f}\n")
                f.write("\n")
        
        self.logger.info(f"Saved summary to: {summary_file}")


def main():
    """Main execution"""
    evaluator = StabilityEvaluator()
    
    # Evaluate solar models
    print("\nEvaluating solar models...")
    solar_results = evaluator.load_training_results('solar')
    solar_evaluation = evaluator.evaluate_by_stability(solar_results, 'solar')
    solar_robustness = evaluator.compute_robustness_metrics(solar_evaluation)
    evaluator.save_results(solar_evaluation, solar_robustness, 'solar')
    
    # Evaluate wind models
    print("\nEvaluating wind models...")
    wind_results = evaluator.load_training_results('wind')
    wind_evaluation = evaluator.evaluate_by_stability(wind_results, 'wind')
    wind_robustness = evaluator.compute_robustness_metrics(wind_evaluation)
    evaluator.save_results(wind_evaluation, wind_robustness, 'wind')
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {evaluator.results_dir}")


if __name__ == "__main__":
    main()

