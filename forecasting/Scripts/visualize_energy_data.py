#!/usr/bin/env python3
"""
Visualize Energy Production Data

Generate exploratory figures for solar and wind energy data:
- Time series plots
- Daily/seasonal patterns
- Distribution analysis
- Correlation analysis

Author: Energy Forecasting Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
sns.set_palette("husl")

# Set figure parameters
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


class EnergyDataVisualizer:
    """Visualize energy production data"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.base_dir = Path(__file__).parent.parent
        self.processed_dir = self.base_dir / "data" / "processed"
        self.figures_dir = self.base_dir / "figures"
        self.thesis_figures_dir = Path(__file__).parent.parent.parent / "figures" / "thesis"
        
        for dir_path in [self.figures_dir, self.thesis_figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.energy_file = self.processed_dir / "energy_production_2024.csv"
        if not self.energy_file.exists():
            raise FileNotFoundError(f"Energy data not found: {self.energy_file}")
        
        print(f"Loading energy data from: {self.energy_file}")
        self.df = pd.read_csv(self.energy_file)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        
        print(f"Loaded {len(self.df)} rows")
        print(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
    
    def plot_time_series(self):
        """Plot time series of solar and wind generation"""
        print("\nPlotting time series...")
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Solar
        axes[0].plot(self.df['datetime'], self.df['solar_generation_mw'], 
                    linewidth=0.5, alpha=0.7, color='orange')
        axes[0].set_title('Solar PV Generation - 2024 (Germany)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Generation (MW)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlabel('Date')
        
        # Wind
        axes[1].plot(self.df['datetime'], self.df['wind_generation_mw'], 
                    linewidth=0.5, alpha=0.7, color='blue')
        axes[1].set_title('Wind Power Generation - 2024 (Germany)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Generation (MW)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel('Date')
        
        plt.tight_layout()
        
        output_file = self.figures_dir / "energy_time_series_2024.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Copy to thesis
        thesis_file = self.thesis_figures_dir / "energy_time_series_2024.png"
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file}")
    
    def plot_daily_patterns(self):
        """Plot average daily patterns"""
        print("\nPlotting daily patterns...")
        
        # Add hour of day
        self.df['hour'] = self.df['datetime'].dt.hour
        
        # Calculate average by hour
        solar_hourly = self.df.groupby('hour')['solar_generation_mw'].mean()
        wind_hourly = self.df.groupby('hour')['wind_generation_mw'].mean()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Solar
        axes[0].plot(solar_hourly.index, solar_hourly.values, 
                    marker='o', linewidth=2, markersize=8, color='orange')
        axes[0].fill_between(solar_hourly.index, solar_hourly.values, alpha=0.3, color='orange')
        axes[0].set_title('Average Daily Solar Generation Pattern', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Average Generation (MW)')
        axes[0].set_xticks(range(0, 24, 2))
        axes[0].grid(True, alpha=0.3)
        
        # Wind
        axes[1].plot(wind_hourly.index, wind_hourly.values, 
                    marker='s', linewidth=2, markersize=8, color='blue')
        axes[1].fill_between(wind_hourly.index, wind_hourly.values, alpha=0.3, color='blue')
        axes[1].set_title('Average Daily Wind Generation Pattern', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Average Generation (MW)')
        axes[1].set_xticks(range(0, 24, 2))
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.figures_dir / "daily_patterns.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Copy to thesis
        thesis_file = self.thesis_figures_dir / "daily_patterns.png"
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file}")
    
    def plot_seasonal_patterns(self):
        """Plot seasonal patterns by month"""
        print("\nPlotting seasonal patterns...")
        
        # Add month
        self.df['month'] = self.df['datetime'].dt.month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Calculate average by month
        solar_monthly = self.df.groupby('month')['solar_generation_mw'].agg(['mean', 'std'])
        wind_monthly = self.df.groupby('month')['wind_generation_mw'].agg(['mean', 'std'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Solar
        x_pos = range(len(solar_monthly))
        axes[0].bar(x_pos, solar_monthly['mean'], yerr=solar_monthly['std'],
                   color='orange', alpha=0.7, capsize=5)
        axes[0].set_title('Average Monthly Solar Generation', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Average Generation (MW)')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(month_names)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Wind
        x_pos = range(len(wind_monthly))
        axes[1].bar(x_pos, wind_monthly['mean'], yerr=wind_monthly['std'],
                   color='blue', alpha=0.7, capsize=5)
        axes[1].set_title('Average Monthly Wind Generation', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Average Generation (MW)')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(month_names)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = self.figures_dir / "seasonal_patterns.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Copy to thesis
        thesis_file = self.thesis_figures_dir / "seasonal_patterns.png"
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file}")
    
    def plot_distributions(self):
        """Plot distribution of generation values"""
        print("\nPlotting distributions...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Solar distribution
        axes[0].hist(self.df['solar_generation_mw'], bins=50, alpha=0.7, 
                   color='orange', edgecolor='black')
        axes[0].axvline(self.df['solar_generation_mw'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {self.df["solar_generation_mw"].mean():.0f} MW')
        axes[0].axvline(self.df['solar_generation_mw'].median(), color='green', 
                       linestyle='--', linewidth=2, label=f'Median: {self.df["solar_generation_mw"].median():.0f} MW')
        axes[0].set_title('Solar Generation Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Generation (MW)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Wind distribution
        axes[1].hist(self.df['wind_generation_mw'], bins=50, alpha=0.7, 
                   color='blue', edgecolor='black')
        axes[1].axvline(self.df['wind_generation_mw'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {self.df["wind_generation_mw"].mean():.0f} MW')
        axes[1].axvline(self.df['wind_generation_mw'].median(), color='green', 
                       linestyle='--', linewidth=2, label=f'Median: {self.df["wind_generation_mw"].median():.0f} MW')
        axes[1].set_title('Wind Generation Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Generation (MW)')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = self.figures_dir / "generation_distributions.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Copy to thesis
        thesis_file = self.thesis_figures_dir / "generation_distributions.png"
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file}")
    
    def plot_correlation_with_wsi(self):
        """Plot correlation between generation and WSI"""
        print("\nPlotting correlation with WSI...")
        
        if 'WSI_smoothed' not in self.df.columns:
            print("WSI data not available, skipping correlation plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Solar vs WSI
        axes[0].scatter(self.df['WSI_smoothed'], self.df['solar_generation_mw'], 
                       alpha=0.3, s=10, color='orange')
        axes[0].set_title('Solar Generation vs Weather Stability Index', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('WSI (smoothed)')
        axes[0].set_ylabel('Solar Generation (MW)')
        axes[0].grid(True, alpha=0.3)
        
        # Calculate correlation
        corr_solar = self.df[['WSI_smoothed', 'solar_generation_mw']].corr().iloc[0, 1]
        axes[0].text(0.05, 0.95, f'Correlation: {corr_solar:.3f}', 
                    transform=axes[0].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Wind vs WSI
        axes[1].scatter(self.df['WSI_smoothed'], self.df['wind_generation_mw'], 
                       alpha=0.3, s=10, color='blue')
        axes[1].set_title('Wind Generation vs Weather Stability Index', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('WSI (smoothed)')
        axes[1].set_ylabel('Wind Generation (MW)')
        axes[1].grid(True, alpha=0.3)
        
        # Calculate correlation
        corr_wind = self.df[['WSI_smoothed', 'wind_generation_mw']].corr().iloc[0, 1]
        axes[1].text(0.05, 0.95, f'Correlation: {corr_wind:.3f}', 
                    transform=axes[1].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        output_file = self.figures_dir / "wsi_correlation.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Copy to thesis
        thesis_file = self.thesis_figures_dir / "wsi_correlation.png"
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file}")
    
    def plot_stability_comparison(self):
        """Plot generation by stability regime"""
        print("\nPlotting stability comparison...")
        
        if 'unstable_gmm' not in self.df.columns:
            print("Stability labels not available, skipping stability comparison")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Solar by stability
        stable_solar = self.df[self.df['unstable_gmm'] == 0]['solar_generation_mw']
        unstable_solar = self.df[self.df['unstable_gmm'] == 1]['solar_generation_mw']
        
        axes[0].hist(stable_solar, bins=50, alpha=0.6, label='Stable', color='green', density=True)
        axes[0].hist(unstable_solar, bins=50, alpha=0.6, label='Unstable', color='red', density=True)
        axes[0].set_title('Solar Generation Distribution by Stability Regime', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Generation (MW)')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Wind by stability
        stable_wind = self.df[self.df['unstable_gmm'] == 0]['wind_generation_mw']
        unstable_wind = self.df[self.df['unstable_gmm'] == 1]['wind_generation_mw']
        
        axes[1].hist(stable_wind, bins=50, alpha=0.6, label='Stable', color='green', density=True)
        axes[1].hist(unstable_wind, bins=50, alpha=0.6, label='Unstable', color='red', density=True)
        axes[1].set_title('Wind Generation Distribution by Stability Regime', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Generation (MW)')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = self.figures_dir / "stability_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Copy to thesis
        thesis_file = self.thesis_figures_dir / "stability_comparison.png"
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file}")
    
    def generate_summary_statistics(self):
        """Generate and print summary statistics"""
        print("\n" + "="*60)
        print("ENERGY DATA SUMMARY STATISTICS")
        print("="*60)
        
        print("\nSolar Generation:")
        print(self.df['solar_generation_mw'].describe())
        
        print("\nWind Generation:")
        print(self.df['wind_generation_mw'].describe())
        
        if 'WSI_smoothed' in self.df.columns:
            print("\nCorrelation with WSI:")
            print(f"  Solar-WSI: {self.df[['solar_generation_mw', 'WSI_smoothed']].corr().iloc[0,1]:.3f}")
            print(f"  Wind-WSI: {self.df[['wind_generation_mw', 'WSI_smoothed']].corr().iloc[0,1]:.3f}")
        
        if 'unstable_gmm' in self.df.columns:
            print("\nStability Regime Statistics:")
            stable_count = (self.df['unstable_gmm'] == 0).sum()
            unstable_count = (self.df['unstable_gmm'] == 1).sum()
            print(f"  Stable hours: {stable_count} ({stable_count/len(self.df)*100:.1f}%)")
            print(f"  Unstable hours: {unstable_count} ({unstable_count/len(self.df)*100:.1f}%)")
            print(f"\n  Solar mean - Stable: {self.df[self.df['unstable_gmm']==0]['solar_generation_mw'].mean():.1f} MW")
            print(f"  Solar mean - Unstable: {self.df[self.df['unstable_gmm']==1]['solar_generation_mw'].mean():.1f} MW")
            print(f"  Wind mean - Stable: {self.df[self.df['unstable_gmm']==0]['wind_generation_mw'].mean():.1f} MW")
            print(f"  Wind mean - Unstable: {self.df[self.df['unstable_gmm']==1]['wind_generation_mw'].mean():.1f} MW")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("="*60)
        print("GENERATING ENERGY DATA VISUALIZATIONS")
        print("="*60)
        
        self.plot_time_series()
        self.plot_daily_patterns()
        self.plot_seasonal_patterns()
        self.plot_distributions()
        self.plot_correlation_with_wsi()
        self.plot_stability_comparison()
        self.generate_summary_statistics()
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)
        print(f"Figures saved to: {self.figures_dir}")
        print(f"Thesis figures copied to: {self.thesis_figures_dir}")


def main():
    """Main execution"""
    visualizer = EnergyDataVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()

