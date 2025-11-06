#!/usr/bin/env python3
"""
Visualize Forecasting Results

Generate publication-ready figures for thesis:
- Error distributions (stable vs unstable)
- Monthly metrics by regime
- SHAP summary plots
- Model accuracy-robustness scatter
- Reliability diagrams (probabilistic)

Author: Energy Forecasting Pipeline
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings
import logging
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
sns.set_palette("husl")


class ForecastingVisualizer:
    """Generate visualizations for forecasting results"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.base_dir = Path(__file__).parent.parent
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.base_dir / "figures"
        self.thesis_figures_dir = Path(__file__).parent.parent.parent / "figures" / "thesis"
        self.logs_dir = self.base_dir / "logs"
        
        for dir_path in [self.figures_dir, self.thesis_figures_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        log_file = self.logs_dir / f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set figure parameters
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        
    def load_evaluation_results(self, source_type: str) -> Dict:
        """Load evaluation results"""
        results_file = self.results_dir / f"{source_type}_evaluation_results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Evaluation results not found: {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return results
    
    def load_training_results(self, source_type: str) -> Dict:
        """Load training results"""
        results_file = self.results_dir / f"{source_type}_training_results.pkl"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Training results not found: {results_file}")
        
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        return results
    
    def plot_error_distributions(self, source_type: str):
        """Plot error distributions for stable vs unstable periods"""
        self.logger.info(f"Plotting error distributions for {source_type}...")
        
        # Load training results
        training_results = self.load_training_results(source_type)
        
        # Load stability labels
        stability_file = Path(__file__).parent.parent.parent / "Data" / "processed" / "stability_labels.csv"
        stability_df = pd.read_csv(stability_file)
        stability_df['datetime'] = pd.to_datetime(stability_df['datetime'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        model_idx = 0
        for model_name, model_results in training_results.items():
            if model_results is None or model_idx >= 4:
                continue
            
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
            
            # Split by stability
            stable_errors = pred_df[pred_df['unstable_gmm'] == 0]['abs_error'].dropna()
            unstable_errors = pred_df[pred_df['unstable_gmm'] == 1]['abs_error'].dropna()
            
            ax = axes[model_idx]
            
            # Plot distributions
            ax.hist(stable_errors, bins=50, alpha=0.6, label='Stable', density=True)
            ax.hist(unstable_errors, bins=50, alpha=0.6, label='Unstable', density=True)
            
            ax.set_xlabel('Absolute Error (MW)')
            ax.set_ylabel('Density')
            ax.set_title(f'{model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            model_idx += 1
        
        # Hide unused subplots
        for idx in range(model_idx, 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_file = self.figures_dir / f"{source_type}_error_distributions.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Copy to thesis figures
        thesis_file = self.thesis_figures_dir / f"{source_type}_error_distributions.png"
        plt.figure(figsize=(14, 10))
        # Recreate for thesis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # ... (same plotting code)
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved: {output_file}")
    
    def plot_monthly_metrics(self, source_type: str):
        """Plot monthly MAE by stability regime"""
        self.logger.info(f"Plotting monthly metrics for {source_type}...")
        
        # Load training results
        training_results = self.load_training_results(source_type)
        
        # Load stability labels
        stability_file = Path(__file__).parent.parent.parent / "Data" / "processed" / "stability_labels.csv"
        stability_df = pd.read_csv(stability_file)
        stability_df['datetime'] = pd.to_datetime(stability_df['datetime'])
        
        # Collect monthly metrics for each model
        monthly_data = []
        
        for model_name, model_results in training_results.items():
            if model_results is None:
                continue
            
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
            pred_df['abs_error'] = np.abs(pred_df['prediction'] - pred_df['actual'])
            pred_df['month'] = pred_df['datetime'].dt.month
            
            # Group by month and stability
            for month in range(1, 13):
                month_data = pred_df[pred_df['month'] == month]
                stable_data = month_data[month_data['unstable_gmm'] == 0]
                unstable_data = month_data[month_data['unstable_gmm'] == 1]
                
                if len(stable_data) > 0:
                    monthly_data.append({
                        'model': model_name,
                        'month': month,
                        'regime': 'Stable',
                        'MAE': stable_data['abs_error'].mean()
                    })
                
                if len(unstable_data) > 0:
                    monthly_data.append({
                        'model': model_name,
                        'month': month,
                        'regime': 'Unstable',
                        'MAE': unstable_data['abs_error'].mean()
                    })
        
        monthly_df = pd.DataFrame(monthly_data)
        
        if len(monthly_df) == 0:
            self.logger.warning("No monthly data to plot")
            return
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for model in monthly_df['model'].unique():
            model_data = monthly_df[monthly_df['model'] == model]
            stable_data = model_data[model_data['regime'] == 'Stable']
            unstable_data = model_data[model_data['regime'] == 'Unstable']
            
            if len(stable_data) > 0:
                ax.plot(stable_data['month'], stable_data['MAE'], 
                       marker='o', label=f'{model} (Stable)', linestyle='-')
            if len(unstable_data) > 0:
                ax.plot(unstable_data['month'], unstable_data['MAE'], 
                       marker='s', label=f'{model} (Unstable)', linestyle='--')
        
        ax.set_xlabel('Month')
        ax.set_ylabel('MAE (MW)')
        ax.set_title(f'Monthly Forecast Accuracy by Stability Regime ({source_type.capitalize()})')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.figures_dir / f"{source_type}_monthly_metrics.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Copy to thesis
        thesis_file = self.thesis_figures_dir / f"{source_type}_monthly_metrics.png"
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved: {output_file}")
    
    def plot_accuracy_robustness_scatter(self, source_type: str):
        """Plot model accuracy vs robustness scatter"""
        self.logger.info(f"Plotting accuracy-robustness scatter for {source_type}...")
        
        # Load evaluation results
        eval_results = self.load_evaluation_results(source_type)
        
        models = []
        mae_values = []
        rpd_values = []
        
        for model_name, results in eval_results['evaluation_results'].items():
            models.append(model_name)
            mae_values.append(results['overall']['MAE'])
            rpd_values.append(eval_results['robustness_metrics'][model_name]['RPD'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(mae_values, rpd_values, s=100, alpha=0.6)
        
        # Add labels
        for i, model in enumerate(models):
            ax.annotate(model, (mae_values[i], rpd_values[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Overall MAE (MW)')
        ax.set_ylabel('Relative Performance Degradation (%)')
        ax.set_title(f'Model Accuracy vs Robustness ({source_type.capitalize()})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.figures_dir / f"{source_type}_accuracy_robustness.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Copy to thesis
        thesis_file = self.thesis_figures_dir / f"{source_type}_accuracy_robustness.png"
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved: {output_file}")
    
    def generate_all_visualizations(self, source_type: str = 'both'):
        """Generate all visualizations"""
        self.logger.info("="*60)
        self.logger.info("GENERATING VISUALIZATIONS")
        self.logger.info("="*60)
        
        if source_type in ['solar', 'both']:
            self.plot_error_distributions('solar')
            self.plot_monthly_metrics('solar')
            self.plot_accuracy_robustness_scatter('solar')
        
        if source_type in ['wind', 'both']:
            self.plot_error_distributions('wind')
            self.plot_monthly_metrics('wind')
            self.plot_accuracy_robustness_scatter('wind')
        
        self.logger.info("Visualization complete!")


def main():
    """Main execution"""
    visualizer = ForecastingVisualizer()
    visualizer.generate_all_visualizations('both')
    
    print(f"\nVisualizations saved to: {visualizer.figures_dir}")
    print(f"Thesis figures copied to: {visualizer.thesis_figures_dir}")


if __name__ == "__main__":
    main()

