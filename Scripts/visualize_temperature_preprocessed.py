#!/usr/bin/env python3
"""
Temperature Preprocessed Data Visualization Script

This script generates visualizations from preprocessed temperature data:
- 12 monthly plots (each Bundesland + Germany aggregation)
- 1 full-year plot
- Monthly distribution comparison plots
- Statistical summary tables

Uses preprocessed data (outliers and NaN removed) from:
- Data/Bundesland_aggregation/Temp_Bundesland_Aggregated_preprocessed
- Data/Germany_aggregation/Temp_Germany_Aggregated_preprocessed

Outputs to:
- visualizations/temperature_preprocessed/

Author: Weather Data Analysis Project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TemperaturePreprocessedVisualizer:
    def __init__(self, config_path="Scripts/preprocessing_config.yaml"):
        """Initialize visualizer with configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Use preprocessed data folders
        self.base_path = Path("Data")
        self.bundesland_folder = self.base_path / "Bundesland_aggregation" / "Temp_Bundesland_Aggregated_preprocessed"
        self.germany_folder = self.base_path / "Germany_aggregation" / "Temp_Germany_Aggregated_preprocessed"
        
        self.viz_config = self.config['visualization']
        self.bundesland_mapping = self.config['bundesland_mapping']
        self.temperature_config = self.config['attribute_column_mappings']['temperature']
        
        # Value column for temperature
        self.value_columns = self.temperature_config['value_columns']
        self.y_label = self.temperature_config['y_label']
        
        # Build filename to canonical name mapping
        self.file_to_canonical = self._build_file_name_lookup()
        
        # Create visualizations directory
        self.viz_dir = Path("visualizations") / "temperature_preprocessed"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def _normalize_for_file(self, name: str) -> str:
        """Create a filename-friendly version of a Bundesland name"""
        replacements = {
            'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue', 'ß': 'ss'
        }
        out = name
        for src, dst in replacements.items():
            out = out.replace(src, dst)
        out = out.replace(' ', '-')
        return out
    
    def _build_file_name_lookup(self) -> dict:
        """Map filename-style names to canonical Bundesland names"""
        lookup = {}
        for canonical in self.bundesland_mapping.keys():
            file_style = self._normalize_for_file(canonical)
            lookup[file_style] = canonical
        return lookup
    
    def _get_canonical_name(self, filename):
        """Convert filename to canonical Bundesland name"""
        base = os.path.splitext(filename)[0]
        base = base.replace('_aggregated', '')
        
        if base in self.bundesland_mapping:
            return base
        elif base in self.file_to_canonical:
            return self.file_to_canonical[base]
        else:
            # Try reverse lookup
            for canonical, file_style in self.file_to_canonical.items():
                if base == file_style or base.startswith(file_style):
                    return canonical
            return base
    
    def load_bundesland_data(self):
        """Load all Bundesland CSV files from preprocessed folder"""
        data_dict = {}
        csv_files = list(self.bundesland_folder.glob("*.csv"))
        
        for file_path in csv_files:
            # Skip summary files
            if '_SUMMARY' in file_path.name.upper() or \
               '_AGGREGATION' in file_path.name.upper():
                continue
            
            try:
                df = pd.read_csv(file_path, low_memory=False)
                
                # Ensure datetime column is datetime
                datetime_col = 'datetime' if 'datetime' in df.columns else 'MESS_DATUM'
                if datetime_col in df.columns:
                    df[datetime_col] = pd.to_datetime(df[datetime_col])
                    if datetime_col != 'datetime':
                        df.rename(columns={datetime_col: 'datetime'}, inplace=True)
                    
                    # Find value column
                    value_col = None
                    for col in self.value_columns:
                        if col in df.columns:
                            value_col = col
                            break
                    
                    if value_col:
                        bundesland_name = self._get_canonical_name(file_path.stem)
                        data_dict[bundesland_name] = {
                            'df': df,
                            'value_column': value_col
                        }
                    else:
                        print(f"  Warning: No value column found in {file_path.name}. Available columns: {list(df.columns)}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return data_dict
    
    def load_germany_data(self):
        """Load Germany aggregation data from preprocessed folder"""
        csv_files = list(self.germany_folder.glob("*.csv"))
        
        for file_path in csv_files:
            # Look for Germany_total or similar
            if 'Germany' in file_path.name or 'germany' in file_path.name.lower():
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    
                    # Handle datetime column
                    datetime_col = 'datetime' if 'datetime' in df.columns else 'MESS_DATUM'
                    if datetime_col in df.columns:
                        df[datetime_col] = pd.to_datetime(df[datetime_col])
                        if datetime_col != 'datetime':
                            df.rename(columns={datetime_col: 'datetime'}, inplace=True)
                        
                        # Find value column
                        value_col = None
                        for col in self.value_columns:
                            if col in df.columns:
                                value_col = col
                                break
                        
                        if value_col:
                            return {'df': df, 'value_column': value_col}
                        else:
                            print(f"  Warning: No value column found. Available columns: {list(df.columns)}")
                except Exception as e:
                    print(f"Error loading Germany data from {file_path}: {e}")
        
        return None
    
    def create_monthly_plots(self, bundesland_data, germany_data, value_column, output_dir):
        """Create 12 monthly plots"""
        monthly_dir = output_dir / "monthly"
        monthly_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        all_data = {}
        for bundesland, data in bundesland_data.items():
            df = data['df'].copy()
            if 'datetime' in df.columns:
                df = df[df['datetime'].notna()].sort_values('datetime')
            all_data[bundesland] = df
        
        # Germany data
        if germany_data:
            df_germany = germany_data['df'].copy()
            if 'datetime' in df_germany.columns:
                df_germany = df_germany[df_germany['datetime'].notna()].sort_values('datetime')
        
        months = range(1, 13)
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        try:
            colors = plt.colormaps[self.viz_config['color_palette']](np.linspace(0, 1, len(all_data)))
        except (KeyError, AttributeError):
            # Fallback for older matplotlib versions
            colors = plt.cm.get_cmap(self.viz_config['color_palette'])(np.linspace(0, 1, len(all_data)))
        
        for month in months:
            fig, ax = plt.subplots(figsize=self.viz_config['figure_size_monthly'])
            
            # Plot each Bundesland
            for i, (bundesland, df) in enumerate(all_data.items()):
                month_data = df[df['datetime'].dt.month == month].copy()
                if len(month_data) > 0:
                    month_data = month_data.sort_values('datetime')
                    # Calculate hour of month from datetime
                    month_data['hour'] = (month_data['datetime'].dt.day - 1) * 24 + month_data['datetime'].dt.hour
                    # Data is already cleaned, but filter just in case
                    plot_data = month_data[[value_column, 'hour']].dropna()
                    if len(plot_data) > 0:
                        ax.plot(plot_data['hour'], plot_data[value_column],
                               label=self.bundesland_mapping.get(bundesland, bundesland),
                               color=colors[i], linewidth=self.viz_config['line_width'],
                               alpha=self.viz_config['alpha'])
            
            # Plot Germany aggregation
            if germany_data:
                df_germany_month = df_germany[df_germany['datetime'].dt.month == month].copy()
                if len(df_germany_month) > 0:
                    df_germany_month = df_germany_month.sort_values('datetime')
                    df_germany_month['hour'] = (df_germany_month['datetime'].dt.day - 1) * 24 + df_germany_month['datetime'].dt.hour
                    plot_data_germany = df_germany_month[[value_column, 'hour']].dropna()
                    if len(plot_data_germany) > 0:
                        ax.plot(plot_data_germany['hour'], plot_data_germany[value_column],
                               label='Germany (Aggregated)',
                               linestyle=self.viz_config['germany_line_style'],
                               linewidth=self.viz_config['germany_line_width'],
                               color='black', alpha=0.9)
            
            ax.set_title(f'Temperature (Preprocessed) - {month_names[month-1]} 2024',
                        fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('Hour of Month', fontsize=12)
            ax.set_ylabel(self.y_label, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
            
            plt.tight_layout()
            output_file = monthly_dir / f"temperature_preprocessed_month_{month:02d}_2024.png"
            plt.savefig(output_file, dpi=self.viz_config['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"  Created monthly plot: {output_file}")
    
    def create_full_year_plot(self, bundesland_data, germany_data, value_column, output_dir):
        """Create full year plot"""
        fig, ax = plt.subplots(figsize=self.viz_config['figure_size_full_year'])
        
        # Prepare data
        all_data = {}
        for bundesland, data in bundesland_data.items():
            df = data['df'].copy()
            if 'datetime' in df.columns:
                df = df[df['datetime'].notna()].sort_values('datetime')
            df['hour_of_year'] = range(len(df))
            all_data[bundesland] = df
        
        try:
            colors = plt.colormaps[self.viz_config['color_palette']](np.linspace(0, 1, len(all_data)))
        except (KeyError, AttributeError):
            colors = plt.cm.get_cmap(self.viz_config['color_palette'])(np.linspace(0, 1, len(all_data)))
        
        # Plot each Bundesland
        for i, (bundesland, df) in enumerate(all_data.items()):
            if len(df) > 0:
                plot_data = df[[value_column, 'hour_of_year']].dropna()
                if len(plot_data) > 0:
                    ax.plot(plot_data['hour_of_year'], plot_data[value_column],
                           label=self.bundesland_mapping.get(bundesland, bundesland),
                           color=colors[i], linewidth=self.viz_config['line_width'],
                           alpha=self.viz_config['alpha'])
        
        # Plot Germany aggregation
        if germany_data:
            df_germany = germany_data['df'].copy()
            if 'datetime' in df_germany.columns:
                df_germany = df_germany[df_germany['datetime'].notna()].sort_values('datetime')
            df_germany['hour_of_year'] = range(len(df_germany))
            
            if len(df_germany) > 0:
                plot_data_germany = df_germany[[value_column, 'hour_of_year']].dropna()
                if len(plot_data_germany) > 0:
                    ax.plot(plot_data_germany['hour_of_year'], plot_data_germany[value_column],
                           label='Germany (Aggregated)',
                           linestyle=self.viz_config['germany_line_style'],
                           linewidth=self.viz_config['germany_line_width'],
                           color='black', alpha=0.9)
        
        # Add month boundaries
        month_boundaries = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for boundary in month_boundaries[1:-1]:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_title('Temperature (Preprocessed - Outliers & NaN Removed) - Full Year 2024',
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Hour of Year', fontsize=14)
        ax.set_ylabel(self.y_label, fontsize=14)
        ax.set_xticks(month_boundaries[:-1])
        ax.set_xticklabels(month_labels, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=2)
        
        plt.tight_layout()
        output_file = output_dir / "temperature_preprocessed_full_year_2024.png"
        plt.savefig(output_file, dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"  Created full-year plot: {output_file}")
    
    def create_distribution_plots(self, bundesland_data, germany_data, value_column, output_dir):
        """Create monthly distribution comparison plots"""
        dist_dir = output_dir / "distributions"
        dist_dir.mkdir(parents=True, exist_ok=True)
        
        months = range(1, 13)
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        for month in months:
            fig, ax = plt.subplots(figsize=self.viz_config['figure_size_distribution'])
            
            # Collect data for this month
            plot_data = []
            labels = []
            
            # Bundesland data
            for bundesland, data in bundesland_data.items():
                df = data['df'].copy()
                if 'datetime' in df.columns:
                    month_data = df[(df['datetime'].dt.month == month) & 
                                   (df[value_column].notna())]
                    if len(month_data) > 0:
                        plot_data.append(month_data[value_column].values)
                        labels.append(self.bundesland_mapping.get(bundesland, bundesland))
            
            # Germany data
            if germany_data:
                df_germany = germany_data['df'].copy()
                if 'datetime' in df_germany.columns:
                    month_data_germany = df_germany[
                        (df_germany['datetime'].dt.month == month) & 
                        (df_germany[value_column].notna())
                    ]
                    if len(month_data_germany) > 0:
                        plot_data.append(month_data_germany[value_column].values)
                        labels.append('Germany (Aggregated)')
            
            if plot_data:
                # Create violin plot
                parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                                     showmeans=True, showmedians=True)
                
                # Color violins
                try:
                    colors_violin = plt.colormaps[self.viz_config['color_palette']](
                        np.linspace(0, 1, len(plot_data))
                    )
                except (KeyError, AttributeError):
                    colors_violin = plt.cm.get_cmap(self.viz_config['color_palette'])(
                        np.linspace(0, 1, len(plot_data))
                    )
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors_violin[i])
                    pc.set_alpha(self.viz_config['alpha'])
                
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel(self.y_label, fontsize=12)
                ax.set_title(f'Temperature Distribution (Preprocessed) - {month_names[month-1]} 2024',
                           fontsize=14, fontweight='bold', pad=15)
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                output_file = dist_dir / f"temperature_preprocessed_distribution_month_{month:02d}_2024.png"
                plt.savefig(output_file, dpi=self.viz_config['dpi'], bbox_inches='tight')
                plt.close()
                
                print(f"  Created distribution plot: {output_file}")
    
    def create_statistical_summaries(self, bundesland_data, germany_data, value_column, output_dir):
        """Create statistical summary tables"""
        summary_dir = output_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        summary_data = []
        
        # Process each month
        for month in range(1, 13):
            month_name = datetime(2024, month, 1).strftime('%B')
            
            # Bundesland statistics
            for bundesland, data in bundesland_data.items():
                df = data['df'].copy()
                if 'datetime' in df.columns:
                    month_data = df[(df['datetime'].dt.month == month) & 
                                   (df[value_column].notna())]
                    if len(month_data) > 0:
                        summary_data.append({
                            'Month': month_name,
                            'Region': self.bundesland_mapping.get(bundesland, bundesland),
                            'Region_Type': 'Bundesland',
                            'Mean': month_data[value_column].mean(),
                            'Std': month_data[value_column].std(),
                            'Min': month_data[value_column].min(),
                            'Max': month_data[value_column].max(),
                            'Count': len(month_data)
                        })
            
            # Germany statistics
            if germany_data:
                df_germany = germany_data['df'].copy()
                if 'datetime' in df_germany.columns:
                    month_data_germany = df_germany[
                        (df_germany['datetime'].dt.month == month) & 
                        (df_germany[value_column].notna())
                    ]
                    if len(month_data_germany) > 0:
                        summary_data.append({
                            'Month': month_name,
                            'Region': 'Germany',
                            'Region_Type': 'Germany',
                            'Mean': month_data_germany[value_column].mean(),
                            'Std': month_data_germany[value_column].std(),
                            'Min': month_data_germany[value_column].min(),
                            'Max': month_data_germany[value_column].max(),
                            'Count': len(month_data_germany)
                        })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values(['Month', 'Region'])
            output_file = summary_dir / "temperature_preprocessed_monthly_stats_2024.csv"
            summary_df.to_csv(output_file, index=False)
            print(f"  Created summary table: {output_file}")
    
    def visualize(self):
        """Create all visualizations for preprocessed temperature data"""
        print("="*60)
        print("TEMPERATURE PREPROCESSED DATA VISUALIZATION")
        print("="*60)
        
        # Check if folders exist
        if not self.bundesland_folder.exists():
            print(f"Bundesland folder not found: {self.bundesland_folder}")
            print("Please run preprocessing first!")
            return
        if not self.germany_folder.exists():
            print(f"Germany folder not found: {self.germany_folder}")
            print("Warning: Continuing without Germany aggregation data")
        
        # Load data
        print("Loading Bundesland data...")
        bundesland_data = self.load_bundesland_data()
        print(f"  Loaded {len(bundesland_data)} Bundesländer")
        
        print("Loading Germany data...")
        germany_data = self.load_germany_data()
        if germany_data:
            print("  Loaded Germany aggregation data")
        else:
            print("  Warning: No Germany aggregation data found")
        
        if not bundesland_data:
            print("No data to visualize!")
            return
        
        # Get value column from first Bundesland
        value_column = bundesland_data[list(bundesland_data.keys())[0]]['value_column']
        
        # Create visualizations
        print("\nCreating monthly plots...")
        self.create_monthly_plots(bundesland_data, germany_data, value_column, self.viz_dir)
        
        print("\nCreating full-year plot...")
        self.create_full_year_plot(bundesland_data, germany_data, value_column, self.viz_dir)
        
        print("\nCreating distribution plots...")
        self.create_distribution_plots(bundesland_data, germany_data, value_column, self.viz_dir)
        
        print("\nCreating statistical summaries...")
        self.create_statistical_summaries(bundesland_data, germany_data, value_column, self.viz_dir)
        
        print(f"\n{'='*60}")
        print("VISUALIZATION COMPLETE!")
        print(f"{'='*60}")
        print(f"All visualizations saved to: {self.viz_dir}")


def main():
    """Main visualization function"""
    visualizer = TemperaturePreprocessedVisualizer()
    visualizer.visualize()


if __name__ == "__main__":
    main()

