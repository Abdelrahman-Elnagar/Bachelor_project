#!/usr/bin/env python3
"""
Weather Distribution Visualization Script

This script creates comprehensive visualizations showing weather attribute distributions
over the course of the year with hourly data on the x-axis and attributes on the y-axis.
Each Bundesland is represented as a separate line on the charts.

Author: Weather Data Analysis Project
Date: 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WeatherDistributionVisualizer:
    def __init__(self, data_path="Data/Bundesland_aggregation", metric_subdir: str = "Temp_Bundesland_Aggregated", value_column: str = "TT_TU_mean", y_label: str = "Temperature (°C)", title_metric: str = "Temperature"):
        self.data_path = data_path
        self.metric_subdir = metric_subdir
        self.value_column = value_column
        self.y_label = y_label
        self.title_metric = title_metric
        self.bundesland_mapping = {
            'Baden-Württemberg': 'BW',
            'Bayern': 'BY', 
            'Berlin': 'BE',
            'Brandenburg': 'BB',
            'Bremen': 'HB',
            'Hamburg': 'HH',
            'Hessen': 'HE',
            'Mecklenburg-Vorpommern': 'MV',
            'Niedersachsen': 'NI',
            'Nordrhein-Westfalen': 'NW',
            'Rheinland-Pfalz': 'RP',
            'Saarland': 'SL',
            'Sachsen': 'SN',
            'Sachsen-Anhalt': 'ST',
            'Schleswig-Holstein': 'SH',
            'Thüringen': 'TH'
        }
        # Build lookup from file-friendly names to canonical names
        self.file_to_canonical = self._build_file_name_lookup()

    def _normalize_for_file(self, name: str) -> str:
        """Create a filename-friendly version of a Bundesland name."""
        replacements = {
            'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue', 'ß': 'ss'
        }
        out = name
        for src, dst in replacements.items():
            out = out.replace(src, dst)
        out = out.replace(' ', '-')  # keep hyphens as-is
        return out

    def _build_file_name_lookup(self) -> dict:
        """Map filename-style names to canonical Bundesland names."""
        lookup = {}
        for canonical in self.bundesland_mapping.keys():
            file_style = self._normalize_for_file(canonical)
            lookup[file_style] = canonical
        return lookup
        
    def load_temperature_data(self):
        """Load temperature data from all Bundesland files"""
        temp_data = {}
        temp_path = os.path.join(self.data_path, self.metric_subdir)
        
        print(f"Looking for temperature files in: {temp_path}")
        
        files = []
        files.extend(glob.glob(os.path.join(temp_path, "*_aggregated.csv")))
        files.extend(glob.glob(os.path.join(temp_path, "*.csv")))
        files = sorted(set(files))

        if not files:
            print("No CSV files found. Ensure the directory and filenames are correct.")
        
        for file in files:
            base = os.path.basename(file)
            name_no_ext = os.path.splitext(base)[0]
            # Accept both patterns: NAME_aggregated and NAME
            raw_name = name_no_ext.replace('_aggregated', '')
            # Try canonical direct match
            canonical = None
            if raw_name in self.bundesland_mapping:
                canonical = raw_name
            else:
                # Try file-style normalized match
                canonical = self.file_to_canonical.get(raw_name)

            print(f"Found file: {base}; interpreted as: {raw_name}; canonical: {canonical}")

            if canonical is None:
                # As a last resort, try reverse normalization for cases where files use canonical and mapping expects it
                # but we keep behavior simple: skip if unknown
                print(f"Skipping {raw_name} - not recognized as a Bundesland")
                continue

            try:
                df = pd.read_csv(file)
                df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'])
                df['hour_of_year'] = df['MESS_DATUM'].dt.dayofyear * 24 + df['MESS_DATUM'].dt.hour
                temp_data[canonical] = df
                print(f"Successfully loaded {len(df)} records for {canonical}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        print(f"Loaded data for {len(temp_data)} Bundesländer")
        return temp_data
    
    def create_temperature_distribution_plot(self, save_path="temperature_distribution_2024.png", sample_every: int = 6):
        """Create comprehensive temperature distribution visualization"""
        temp_data = self.load_temperature_data()
        
        if not temp_data:
            print("No temperature data found!")
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{self.title_metric} Distribution Analysis Across German Bundesländer (2024)', 
                     fontsize=20, fontweight='bold')
        
        # 1. Hourly Trends
        ax1 = axes[0, 0]
        for bundesland, df in temp_data.items():
            # Downsample to reduce overplotting while preserving distribution shape
            df_plot = df.iloc[::sample_every] if sample_every and sample_every > 1 else df
            ax1.plot(df_plot['hour_of_year'], df_plot[self.value_column], 
                    label=bundesland, alpha=0.7, linewidth=1.5)
        
        suffix = f" (sampled every {sample_every} h)" if sample_every and sample_every > 1 else ""
        ax1.set_title(f'Hourly {self.title_metric} Trends Throughout 2024{suffix}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hour of Year (0-8784)', fontsize=12)
        ax1.set_ylabel(self.y_label, fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 2. Daily Average Distribution
        ax2 = axes[0, 1]
        daily_temps = {}
        for bundesland, df in temp_data.items():
            daily_avg = df.groupby(df['MESS_DATUM'].dt.date)[self.value_column].mean()
            daily_temps[bundesland] = daily_avg.values
        
        # Create box plot for daily temperature distributions
        box_data = [daily_temps[bundesland] for bundesland in daily_temps.keys()]
        box_labels = [self.bundesland_mapping.get(bundesland, bundesland) for bundesland in daily_temps.keys()]
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title(f'Daily {self.title_metric} Distribution by Bundesland', fontsize=14, fontweight='bold')
        ax2.set_ylabel(self.y_label, fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Monthly Patterns
        ax3 = axes[1, 0]
        monthly_temps = {}
        for bundesland, df in temp_data.items():
            monthly_avg = df.groupby(df['MESS_DATUM'].dt.month)[self.value_column].mean()
            monthly_temps[bundesland] = monthly_avg
        
        for bundesland, monthly_data in monthly_temps.items():
            ax3.plot(monthly_data.index, monthly_data.values, 
                    marker='o', label=bundesland, linewidth=2, markersize=6)
        
        ax3.set_title(f'Monthly Average {self.title_metric} Patterns', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Month', fontsize=12)
        ax3.set_ylabel(self.y_label, fontsize=12)
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax3.grid(True, alpha=0.3)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 4. Daily Range Analysis
        ax4 = axes[1, 1]
        temp_ranges = {}
        for bundesland, df in temp_data.items():
            daily_min = df.groupby(df['MESS_DATUM'].dt.date)[self.value_column].min()
            daily_max = df.groupby(df['MESS_DATUM'].dt.date)[self.value_column].max()
            daily_range = daily_max - daily_min
            temp_ranges[bundesland] = daily_range.mean()
        
        bundesland_names = list(temp_ranges.keys())
        range_values = list(temp_ranges.values())
        colors = plt.cm.viridis(np.linspace(0, 1, len(bundesland_names)))
        
        bars = ax4.bar(range(len(bundesland_names)), range_values, color=colors, alpha=0.8)
        ax4.set_title(f'Average Daily {self.title_metric} Range by Bundesland', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Bundesland', fontsize=12)
        ax4.set_ylabel(f'{self.title_metric} Range', fontsize=12)
        ax4.set_xticks(range(len(bundesland_names)))
        ax4.set_xticklabels([self.bundesland_mapping.get(name, name) for name in bundesland_names], 
                           rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, range_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}°C', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Temperature distribution plot saved as: {save_path}")
        
    def create_detailed_hourly_plot(self, save_path="detailed_hourly_temperature.png", sample_every: int = 6):
        """Create a detailed plot focusing on hourly patterns"""
        temp_data = self.load_temperature_data()
        
        if not temp_data:
            print("No temperature data found!")
            return
            
        fig, ax = plt.subplots(figsize=(24, 12))
        
        # Plot each Bundesland with different colors and styles
        colors = plt.cm.tab20(np.linspace(0, 1, len(temp_data)))
        
        for i, (bundesland, df) in enumerate(temp_data.items()):
            # Downsample to reduce overplotting while preserving distribution shape
            df_plot = df.iloc[::sample_every] if sample_every and sample_every > 1 else df
            ax.plot(df_plot['hour_of_year'], df_plot[self.value_column], 
                   label=f"{self.bundesland_mapping.get(bundesland, bundesland)} ({bundesland})",
                   color=colors[i], linewidth=2, alpha=0.8)
        
        # Add vertical lines for month boundaries
        month_boundaries = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan+1']
        
        for boundary in month_boundaries[1:-1]:  # Skip first and last
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        suffix = f" (sampled every {sample_every} h)" if sample_every and sample_every > 1 else ""
        ax.set_title(f'Detailed Hourly {self.title_metric} Distribution Across German Bundesländer (2024){suffix}', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Hour of Year (0-8784)', fontsize=14)
        ax.set_ylabel(self.y_label, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Create custom legend
        legend_elements = []
        for i, (bundesland, df) in enumerate(temp_data.items()):
            legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=3, 
                                            label=f"{self.bundesland_mapping.get(bundesland, bundesland)}"))
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=10, ncol=2)
        
        # Add month labels on x-axis
        ax.set_xticks(month_boundaries)
        ax.set_xticklabels(month_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Detailed hourly temperature plot saved as: {save_path}")
        
    def create_statistical_summary(self):
        """Create statistical summary of temperature data"""
        temp_data = self.load_temperature_data()
        
        if not temp_data:
            print("No temperature data found!")
            return
            
        summary_stats = []
        
        for bundesland, df in temp_data.items():
            stats = {
                'Bundesland': bundesland,
                'Code': self.bundesland_mapping.get(bundesland, bundesland),
                f'Mean_{self.title_metric}': df[self.value_column].mean(),
                f'Min_{self.title_metric}': df[self.value_column].min(),
                f'Max_{self.title_metric}': df[self.value_column].max(),
                f'Std_{self.title_metric}': df[self.value_column].std(),
                f'{self.title_metric}_Range': df[self.value_column].max() - df[self.value_column].min(),
                'Data_Points': len(df)
            }
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df.sort_values('Mean_Temp', ascending=False)
        
        print("\n" + "="*80)
        print("TEMPERATURE STATISTICAL SUMMARY BY BUNDESLAND (2024)")
        print("="*80)
        print(summary_df.to_string(index=False, float_format='%.2f'))
        
        # Save summary to CSV
        summary_df.to_csv('temperature_summary_2024.csv', index=False)
        print(f"\nSummary saved to: temperature_summary_2024.csv")
        
        return summary_df

def main():
    """Main function to run the visualization"""
    print("Starting Weather Distribution Visualization...")
    
    # Initialize visualizer
    visualizer = WeatherDistributionVisualizer(
        data_path="Data/Bundesland_aggregation",
        metric_subdir="Cloudness_Bundesland_Aggregated",
        value_column="V_N_mean",
        y_label="Cloudiness (oktas)",
        title_metric="Cloudiness"
    )
    
    # Create temperature distribution plots
    print("\n1. Creating comprehensive temperature distribution plot...")
    visualizer.create_temperature_distribution_plot(sample_every=6)
    
    print("\n2. Creating detailed hourly temperature plot...")
    visualizer.create_detailed_hourly_plot(sample_every=6)
    
    print("\n3. Generating statistical summary...")
    summary_df = visualizer.create_statistical_summary()
    
    print("\nVisualization complete! Check the generated PNG files and CSV summary.")

if __name__ == "__main__":
    main()
