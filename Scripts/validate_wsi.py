#!/usr/bin/env python3
"""
WSI Validation and Visualization

This script implements Phase 4 of the WSI pipeline:
- Validates WSI and classification results statistically
- Compares different classification methods
- Creates visualizations (timeline, distributions, correlations, cluster quality)
- Generates validation report

Scientific justification: Validation ensures that the WSI and classification methods
produce meteorologically meaningful results. Visualizations aid in understanding
patterns and detecting issues. Statistical validation metrics provide quantitative
evidence of classification quality (Rousseeuw, 1987; Caliński & Harabasz, 1974).

Author: Weather Stability Index Implementation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class WSIValidator:
    """Validate and visualize WSI and classification results"""
    
    def __init__(self):
        """Initialize validator"""
        # Data paths
        self.stability_labels_file = Path("data/processed/stability_labels.csv")
        self.wsi_timeline_file = Path("data/processed/wsi_timeline.csv")
        self.features_file = Path("data/features/weather_features.csv")
        self.classification_metrics_file = Path("models/classification_metrics.json")
        
        # Output paths
        self.figures_dir = Path("figures")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load all required data"""
        print("="*60)
        print("LOADING DATA FOR VALIDATION")
        print("="*60)
        
        data = {}
        
        # Load stability labels
        if self.stability_labels_file.exists():
            data['stability'] = pd.read_csv(self.stability_labels_file)
            data['stability']['datetime'] = pd.to_datetime(data['stability']['datetime'])
            print(f"OK Loaded stability labels: {len(data['stability'])} rows")
        else:
            raise FileNotFoundError(f"Stability labels not found: {self.stability_labels_file}")
        
        # Load WSI timeline
        if self.wsi_timeline_file.exists():
            data['wsi'] = pd.read_csv(self.wsi_timeline_file)
            data['wsi']['datetime'] = pd.to_datetime(data['wsi']['datetime'])
            print(f"OK Loaded WSI timeline: {len(data['wsi'])} rows")
        else:
            print(f"  Warning: WSI timeline not found: {self.wsi_timeline_file}")
        
        # Load features (for correlation analysis)
        if self.features_file.exists():
            data['features'] = pd.read_csv(self.features_file)
            print(f"OK Loaded features: {len(data['features'])} rows")
        else:
            print(f"  Warning: Features file not found: {self.features_file}")
        
        # Load classification metrics
        if self.classification_metrics_file.exists():
            with open(self.classification_metrics_file, 'r') as f:
                data['metrics'] = json.load(f)
            print(f"OK Loaded classification metrics")
        else:
            print(f"  Warning: Classification metrics not found: {self.classification_metrics_file}")
        
        return data
    
    def validate_cluster_sizes(self, df):
        """Validate cluster sizes and balance"""
        print("\n" + "="*60)
        print("CLUSTER SIZE VALIDATION")
        print("="*60)
        
        if 'unstable_gmm' not in df.columns:
            print("  No GMM classification found")
            return {}
        
        stable_count = (df['unstable_gmm'] == 0).sum()
        unstable_count = (df['unstable_gmm'] == 1).sum()
        total = len(df)
        
        stable_prop = stable_count / total
        unstable_prop = unstable_count / total
        
        print(f"  Stable periods: {stable_count} ({stable_prop*100:.1f}%)")
        print(f"  Unstable periods: {unstable_count} ({unstable_prop*100:.1f}%)")
        
        # Check balance
        if stable_prop > 0.9 or unstable_prop > 0.9:
            print(f"  WARNING: Extreme imbalance detected!")
            status = "extreme_imbalance"
        elif stable_prop < 0.4 or unstable_prop < 0.4:
            print(f"  WARNING: Moderate imbalance")
            status = "moderate_imbalance"
        else:
            print(f"  OK Cluster balance acceptable")
            status = "balanced"
        
        return {
            'stable_count': int(stable_count),
            'unstable_count': int(unstable_count),
            'stable_proportion': float(stable_prop),
            'unstable_proportion': float(unstable_prop),
            'status': status
        }
    
    def compare_classification_methods(self, df):
        """Compare different classification methods"""
        print("\n" + "="*60)
        print("CLASSIFICATION METHOD COMPARISON")
        print("="*60)
        
        comparison = {}
        
        # Check which methods are available
        methods = {}
        if 'unstable_gmm' in df.columns:
            methods['GMM'] = df['unstable_gmm']
        if 'unstable_kmeans' in df.columns:
            methods['K-Means'] = df['unstable_kmeans']
        if 'unstable_percentile' in df.columns:
            methods['Percentile'] = df['unstable_percentile']
        
        if len(methods) < 2:
            print("  Not enough methods for comparison")
            return comparison
        
        # Compute agreement metrics
        method_names = list(methods.keys())
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names[i+1:], start=i+1):
                labels1 = methods[method1]
                labels2 = methods[method2]
                
                # Cohen's kappa
                kappa = cohen_kappa_score(labels1, labels2)
                
                # Percentage agreement
                agreement = (labels1 == labels2).mean()
                
                print(f"  {method1} vs {method2}:")
                print(f"    Cohen's kappa: {kappa:.4f}")
                print(f"    Agreement: {agreement*100:.1f}%")
                
                comparison[f"{method1}_vs_{method2}"] = {
                    'cohens_kappa': float(kappa),
                    'agreement': float(agreement)
                }
        
        return comparison
    
    def plot_wsi_timeline(self, df):
        """Plot WSI timeline with stability labels"""
        print("\nCreating WSI timeline plot...")
        
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # Plot WSI
        if 'WSI_smoothed' in df.columns:
            ax.plot(df['datetime'], df['WSI_smoothed'], 
                   linewidth=1.5, alpha=0.7, label='WSI (smoothed)', color='blue')
        
        # Color-code by stability regime
        if 'unstable_gmm' in df.columns:
            # Plot stable periods
            stable_mask = df['unstable_gmm'] == 0
            ax.scatter(df.loc[stable_mask, 'datetime'], 
                      df.loc[stable_mask, 'WSI_smoothed'],
                      c='green', alpha=0.3, s=10, label='Stable', marker='o')
            
            # Plot unstable periods
            unstable_mask = df['unstable_gmm'] == 1
            ax.scatter(df.loc[unstable_mask, 'datetime'], 
                      df.loc[unstable_mask, 'WSI_smoothed'],
                      c='red', alpha=0.3, s=10, label='Unstable', marker='o')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weather Stability Index (WSI)', fontsize=12, fontweight='bold')
        ax.set_title('Weather Stability Index Timeline (2024)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        output_file = self.figures_dir / "wsi_timeline_2024.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  OK Saved to: {output_file}")
        return output_file
    
    def plot_wsi_distribution(self, df):
        """Plot WSI distribution by regime"""
        print("\nCreating WSI distribution plot...")
        
        if 'WSI_smoothed' not in df.columns or 'unstable_gmm' not in df.columns:
            print("  Required columns not found")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        stable_data = df[df['unstable_gmm'] == 0]['WSI_smoothed']
        unstable_data = df[df['unstable_gmm'] == 1]['WSI_smoothed']
        
        axes[0].hist(stable_data, bins=50, alpha=0.7, label='Stable', color='green', density=True)
        axes[0].hist(unstable_data, bins=50, alpha=0.7, label='Unstable', color='red', density=True)
        axes[0].set_xlabel('WSI Value', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
        axes[0].set_title('WSI Distribution by Regime', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        data_for_box = [stable_data, unstable_data]
        axes[1].boxplot(data_for_box, labels=['Stable', 'Unstable'])
        axes[1].set_ylabel('WSI Value', fontsize=12, fontweight='bold')
        axes[1].set_title('WSI Distribution by Regime (Box Plot)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.figures_dir / "wsi_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  OK Saved to: {output_file}")
        return output_file

    def plot_monthly_stability(self, df):
        """Create monthly visualizations of stability (hours and percentage)."""
        print("\nCreating monthly stability plots...")
        work = df.copy()
        work['month'] = work['datetime'].dt.to_period('M').dt.to_timestamp()
        if 'unstable_gmm' not in work.columns and 'stability_label_gmm' in work.columns:
            work['unstable_gmm'] = (work['stability_label_gmm'].str.lower() == 'unstable').astype(int)
        monthly = work.groupby('month')['unstable_gmm'].agg([
            ('unstable_hours', 'sum'),
            ('total_hours', 'count')
        ]).reset_index()
        monthly['stable_hours'] = monthly['total_hours'] - monthly['unstable_hours']
        monthly['pct_unstable'] = 100.0 * monthly['unstable_hours'] / monthly['total_hours']

        # Stacked hours per month
        plt.figure(figsize=(12, 5))
        months = monthly['month'].dt.strftime('%b')
        plt.bar(months, monthly['stable_hours'], label='Stable hours', color='#4CAF50')
        plt.bar(months, monthly['unstable_hours'], bottom=monthly['stable_hours'], label='Unstable hours', color='#F44336')
        plt.title('Monthly Stable vs Unstable Hours (GMM)')
        plt.xlabel('Month')
        plt.ylabel('Hours')
        plt.legend()
        plt.tight_layout()
        out1 = self.figures_dir / 'monthly_stability_hours.png'
        plt.savefig(out1, dpi=200)
        plt.close()
        print(f"  OK Saved to: {out1}")

        # Percentage unstable per month
        plt.figure(figsize=(12, 4))
        plt.bar(months, monthly['pct_unstable'], color='#FF9800')
        for i, v in enumerate(monthly['pct_unstable']):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)
        plt.ylim(0, max(100, monthly['pct_unstable'].max() + 5))
        plt.title('Monthly Percentage Unstable (GMM)')
        plt.xlabel('Month')
        plt.ylabel('Unstable (%)')
        plt.tight_layout()
        out2 = self.figures_dir / 'monthly_unstable_percentage.png'
        plt.savefig(out2, dpi=200)
        plt.close()
        print(f"  OK Saved to: {out2}")
        return out1, out2
    
    def plot_cluster_validation(self, df, metrics):
        """Plot cluster validation metrics"""
        print("\nCreating cluster validation plot...")
        
        if 'WSI_smoothed' not in df.columns or 'stability_label_gmm' not in df.columns:
            print("  Required columns not found")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Silhouette plot (simplified - show distribution)
        wsi_values = df['WSI_smoothed'].values.reshape(-1, 1)
        labels = df['stability_label_gmm'].values
        
        # Plot WSI by cluster
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            axes[0].scatter(range(cluster_mask.sum()), 
                           df.loc[cluster_mask, 'WSI_smoothed'],
                           alpha=0.5, s=20, label=f'Cluster {cluster_id}')
        
        axes[0].set_xlabel('Sample Index (within cluster)', fontsize=11)
        axes[0].set_ylabel('WSI Value', fontsize=12, fontweight='bold')
        axes[0].set_title('WSI Distribution by Cluster', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Metrics summary
        metrics_text = []
        if 'silhouette_score' in metrics:
            metrics_text.append(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        if 'davies_bouldin_index' in metrics and metrics['davies_bouldin_index'] is not None:
            metrics_text.append(f"Davies-Bouldin: {metrics['davies_bouldin_index']:.4f}")
        if 'calinski_harabasz_score' in metrics and metrics['calinski_harabasz_score'] is not None:
            metrics_text.append(f"Calinski-Harabasz: {metrics['calinski_harabasz_score']:.4f}")
        
        axes[1].axis('off')
        axes[1].text(0.1, 0.5, '\n'.join(metrics_text), 
                    fontsize=12, verticalalignment='center',
                    family='monospace')
        axes[1].set_title('Cluster Quality Metrics', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = self.figures_dir / "cluster_validation.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  OK Saved to: {output_file}")
        return output_file
    
    def generate_validation_report(self, data, cluster_validation, method_comparison):
        """Generate validation report"""
        print("\n" + "="*60)
        print("GENERATING VALIDATION REPORT")
        print("="*60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'cluster_validation': cluster_validation,
            'method_comparison': method_comparison,
            'classification_metrics': data.get('metrics', {})
        }
        
        # Write text report
        report_file = self.results_dir / "wsi_validation_report.txt"
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("WSI VALIDATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CLUSTER VALIDATION\n")
            f.write("-"*60 + "\n")
            if cluster_validation:
                f.write(f"Stable periods: {cluster_validation.get('stable_count', 'N/A')} "
                       f"({cluster_validation.get('stable_proportion', 0)*100:.1f}%)\n")
                f.write(f"Unstable periods: {cluster_validation.get('unstable_count', 'N/A')} "
                       f"({cluster_validation.get('unstable_proportion', 0)*100:.1f}%)\n")
                f.write(f"Status: {cluster_validation.get('status', 'N/A')}\n")
            f.write("\n")
            
            f.write("CLASSIFICATION METRICS\n")
            f.write("-"*60 + "\n")
            metrics = data.get('metrics', {})
            if 'silhouette_score' in metrics:
                f.write(f"Silhouette Score: {metrics['silhouette_score']:.4f} (target: > 0.4)\n")
            if 'davies_bouldin_index' in metrics and metrics['davies_bouldin_index'] is not None:
                f.write(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f} (lower is better)\n")
            if 'calinski_harabasz_score' in metrics and metrics['calinski_harabasz_score'] is not None:
                f.write(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f} (higher is better)\n")
            f.write("\n")
            
            f.write("METHOD COMPARISON\n")
            f.write("-"*60 + "\n")
            if method_comparison:
                for comparison, values in method_comparison.items():
                    f.write(f"{comparison}:\n")
                    f.write(f"  Cohen's kappa: {values.get('cohens_kappa', 'N/A'):.4f}\n")
                    f.write(f"  Agreement: {values.get('agreement', 'N/A')*100:.1f}%\n")
            f.write("\n")
        
        print(f"OK Saved validation report to: {report_file}")
        
        # Also save as JSON
        json_file = self.results_dir / "wsi_validation_report.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"OK Saved validation report (JSON) to: {json_file}")
        
        return report_file
    
    def run(self):
        """Run complete validation and visualization pipeline"""
        print("="*60)
        print("WSI VALIDATION AND VISUALIZATION")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load data
        data = self.load_data()
        
        df = data['stability']
        
        # Step 2: Validate cluster sizes
        cluster_validation = self.validate_cluster_sizes(df)
        
        # Step 3: Compare classification methods
        method_comparison = self.compare_classification_methods(df)
        
        # Step 4: Create visualizations
        self.plot_wsi_timeline(df)
        self.plot_wsi_distribution(df)
        self.plot_monthly_stability(df)
        if 'metrics' in data:
            self.plot_cluster_validation(df, data['metrics'])
        
        # Step 5: Generate validation report
        report_file = self.generate_validation_report(data, cluster_validation, method_comparison)
        
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)
        print(f"OK Validation report: {report_file}")
        print(f"OK Visualizations saved to: {self.figures_dir}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return cluster_validation, method_comparison


def main():
    """Main function"""
    validator = WSIValidator()
    cluster_validation, method_comparison = validator.run()
    
    return cluster_validation, method_comparison


if __name__ == "__main__":
    main()

