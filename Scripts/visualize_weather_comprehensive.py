#!/usr/bin/env python3
"""
Comprehensive Weather Data Visualization Script

This script generates:
- 12 monthly plots per attribute (each Bundesland + Germany aggregation)
- 1 full-year plot per attribute
- Monthly distribution comparison plots
- Statistical summary tables

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

class WeatherDataVisualizer:
    def __init__(self, config_path="Scripts/preprocessing_config.yaml"):
        """Initialize visualizer with configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.base_path = Path("Data")
        self.viz_config = self.config['visualization']
        self.bundesland_mapping = self.config['bundesland_mapping']
        self.attribute_mappings = self.config['attribute_column_mappings']
        
        # Build filename to canonical name mapping
        self.file_to_canonical = self._build_file_name_lookup()
        
        # Create visualizations directory
        self.viz_dir = Path("visualizations")
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
    
    def load_bundesland_data(self, folder_path, value_columns):
        """Load all Bundesland CSV files"""
        data_dict = {}
        csv_files = list(folder_path.glob("*.csv"))
        
        for file_path in csv_files:
            # Skip summary files
            if '_SUMMARY' in file_path.name.upper() or \
               '_AGGREGATION' in file_path.name.upper():
                continue
            
            try:
                df = pd.read_csv(file_path, low_memory=False)
                
                # Ensure datetime column is datetime (handle both old and new names)
                datetime_col = 'datetime' if 'datetime' in df.columns else 'MESS_DATUM'
                if datetime_col in df.columns:
                    df[datetime_col] = pd.to_datetime(df[datetime_col])
                    # Rename to standard name for consistency
                    if datetime_col != 'datetime':
                        df.rename(columns={datetime_col: 'datetime'}, inplace=True)
                    
                    # Ensure datetime column exists
                    if 'datetime' not in df.columns and 'MESS_DATUM' in df.columns:
                        df.rename(columns={'MESS_DATUM': 'datetime'}, inplace=True)
                    
                    # Find value column
                    value_col = None
                    for col in value_columns:
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
    
    def load_germany_data(self, folder_path, value_columns):
        """Load Germany aggregation data"""
        csv_files = list(folder_path.glob("*.csv"))
        
        for file_path in csv_files:
            # Look for Germany_total or similar
            if 'Germany' in file_path.name or 'germany' in file_path.name.lower():
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    
                    # Handle datetime column (both old and new names)
                    datetime_col = 'datetime' if 'datetime' in df.columns else 'MESS_DATUM'
                    if datetime_col in df.columns:
                        df[datetime_col] = pd.to_datetime(df[datetime_col])
                        if datetime_col != 'datetime':
                            df.rename(columns={datetime_col: 'datetime'}, inplace=True)
                        
                        # Find value column
                        value_col = None
                        for col in value_columns:
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
    
    def create_monthly_plots(self, attribute_name, bundesland_data, germany_data, 
                            value_column, y_label, output_dir):
        """Create 12 monthly plots"""
        monthly_dir = output_dir / "monthly"
        monthly_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        all_data = {}
        datetime_col = 'datetime' if 'datetime' in list(bundesland_data.values())[0]['df'].columns else 'MESS_DATUM'
        for bundesland, data in bundesland_data.items():
            df = data['df'].copy()
            if datetime_col in df.columns:
                df = df[df[datetime_col].notna()]
                # Standardize column name
                if datetime_col != 'datetime':
                    df.rename(columns={datetime_col: 'datetime'}, inplace=True)
            all_data[bundesland] = df
        
        # Germany data
        if germany_data:
            df_germany = germany_data['df'].copy()
            datetime_col_germany = 'datetime' if 'datetime' in df_germany.columns else 'MESS_DATUM'
            if datetime_col_germany in df_germany.columns:
                df_germany = df_germany[df_germany[datetime_col_germany].notna()]
                if datetime_col_germany != 'datetime':
                    df_germany.rename(columns={datetime_col_germany: 'datetime'}, inplace=True)
        
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
                datetime_col = 'datetime' if 'datetime' in df.columns else 'MESS_DATUM'
                month_data = df[df[datetime_col].dt.month == month].copy()
                if len(month_data) > 0:
                    month_data = month_data.sort_values(datetime_col)
                    # Calculate hour of month from datetime
                    month_data['hour'] = (month_data[datetime_col].dt.day - 1) * 24 + month_data[datetime_col].dt.hour
                    # Filter out NaN values for plotting
                    plot_data = month_data[[value_column, 'hour']].dropna()
                    if len(plot_data) > 0:
                        ax.plot(plot_data['hour'], plot_data[value_column],
                               label=self.bundesland_mapping.get(bundesland, bundesland),
                               color=colors[i], linewidth=self.viz_config['line_width'],
                               alpha=self.viz_config['alpha'])
            
            # Plot Germany aggregation
            if germany_data:
                datetime_col_germany = 'datetime' if 'datetime' in df_germany.columns else 'MESS_DATUM'
                df_germany_month = df_germany[df_germany[datetime_col_germany].dt.month == month].copy()
                if len(df_germany_month) > 0:
                    df_germany_month = df_germany_month.sort_values(datetime_col_germany)
                    # Calculate hour of month from datetime
                    df_germany_month['hour'] = (df_germany_month[datetime_col_germany].dt.day - 1) * 24 + df_germany_month[datetime_col_germany].dt.hour
                    # Filter out NaN values for plotting
                    plot_data_germany = df_germany_month[[value_column, 'hour']].dropna()
                    if len(plot_data_germany) > 0:
                        ax.plot(plot_data_germany['hour'], plot_data_germany[value_column],
                               label='Germany (Aggregated)',
                               linestyle=self.viz_config['germany_line_style'],
                               linewidth=self.viz_config['germany_line_width'],
                               color='black', alpha=0.9)
            
            ax.set_title(f'{attribute_name.title()} - {month_names[month-1]} 2024',
                        fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('Hour of Month', fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
            
            plt.tight_layout()
            output_file = monthly_dir / f"{attribute_name}_month_{month:02d}_2024.png"
            plt.savefig(output_file, dpi=self.viz_config['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"  Created monthly plot: {output_file}")
    
    def create_full_year_plot(self, attribute_name, bundesland_data, germany_data,
                             value_column, y_label, output_dir):
        """Create full year plot"""
        fig, ax = plt.subplots(figsize=self.viz_config['figure_size_full_year'])
        
        # Prepare data
        all_data = {}
        datetime_col = 'datetime' if 'datetime' in list(bundesland_data.values())[0]['df'].columns else 'MESS_DATUM'
        for bundesland, data in bundesland_data.items():
            df = data['df'].copy()
            if datetime_col in df.columns:
                df = df[df[datetime_col].notna()].sort_values(datetime_col)
                if datetime_col != 'datetime':
                    df.rename(columns={datetime_col: 'datetime'}, inplace=True)
            df['hour_of_year'] = range(len(df))
            all_data[bundesland] = df
        
        try:
            colors = plt.colormaps[self.viz_config['color_palette']](np.linspace(0, 1, len(all_data)))
        except (KeyError, AttributeError):
            # Fallback for older matplotlib versions
            colors = plt.cm.get_cmap(self.viz_config['color_palette'])(np.linspace(0, 1, len(all_data)))
        
        # Plot each Bundesland
        for i, (bundesland, df) in enumerate(all_data.items()):
            if len(df) > 0:
                # Filter out NaN values for plotting
                plot_data = df[[value_column, 'hour_of_year']].dropna()
                if len(plot_data) > 0:
                    ax.plot(plot_data['hour_of_year'], plot_data[value_column],
                           label=self.bundesland_mapping.get(bundesland, bundesland),
                           color=colors[i], linewidth=self.viz_config['line_width'],
                           alpha=self.viz_config['alpha'])
        
        # Plot Germany aggregation
        if germany_data:
            df_germany = germany_data['df'].copy()
            datetime_col_germany = 'datetime' if 'datetime' in df_germany.columns else 'MESS_DATUM'
            if datetime_col_germany in df_germany.columns:
                df_germany = df_germany[df_germany[datetime_col_germany].notna()].sort_values(datetime_col_germany)
                if datetime_col_germany != 'datetime':
                    df_germany.rename(columns={datetime_col_germany: 'datetime'}, inplace=True)
            df_germany['hour_of_year'] = range(len(df_germany))
            
            if len(df_germany) > 0:
                # Filter out NaN values for plotting
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
        
        ax.set_title(f'{attribute_name.title()} - Full Year 2024',
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Hour of Year', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_xticks(month_boundaries[:-1])
        ax.set_xticklabels(month_labels, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=2)
        
        plt.tight_layout()
        output_file = output_dir / f"{attribute_name}_full_year_2024.png"
        plt.savefig(output_file, dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"  Created full-year plot: {output_file}")
    
    def create_distribution_plots(self, attribute_name, bundesland_data, germany_data,
                                 value_column, y_label, output_dir):
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
            datetime_col = 'datetime' if 'datetime' in list(bundesland_data.values())[0]['df'].columns else 'MESS_DATUM'
            for bundesland, data in bundesland_data.items():
                df = data['df'].copy()
                if datetime_col in df.columns:
                    month_data = df[(df[datetime_col].dt.month == month) & 
                                   (df[value_column].notna())]
                    if len(month_data) > 0:
                        plot_data.append(month_data[value_column].values)
                        labels.append(self.bundesland_mapping.get(bundesland, bundesland))
            
            # Germany data
            if germany_data:
                df_germany = germany_data['df'].copy()
                datetime_col_germany = 'datetime' if 'datetime' in df_germany.columns else 'MESS_DATUM'
                if datetime_col_germany in df_germany.columns:
                    month_data_germany = df_germany[
                        (df_germany[datetime_col_germany].dt.month == month) & 
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
                    # Fallback for older matplotlib versions
                    colors_violin = plt.cm.get_cmap(self.viz_config['color_palette'])(
                        np.linspace(0, 1, len(plot_data))
                    )
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors_violin[i])
                    pc.set_alpha(self.viz_config['alpha'])
                
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel(y_label, fontsize=12)
                ax.set_title(f'{attribute_name.title()} Distribution - {month_names[month-1]} 2024',
                           fontsize=14, fontweight='bold', pad=15)
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                output_file = dist_dir / f"{attribute_name}_distribution_month_{month:02d}_2024.png"
                plt.savefig(output_file, dpi=self.viz_config['dpi'], bbox_inches='tight')
                plt.close()
                
                print(f"  Created distribution plot: {output_file}")
    
    def create_statistical_summaries(self, attribute_name, bundesland_data, germany_data,
                                    value_column, output_dir):
        """Create statistical summary tables"""
        summary_dir = output_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        summary_data = []
        
        # Process each month
        for month in range(1, 13):
            month_name = datetime(2024, month, 1).strftime('%B')
            
            # Bundesland statistics
            datetime_col = 'datetime' if 'datetime' in list(bundesland_data.values())[0]['df'].columns else 'MESS_DATUM'
            for bundesland, data in bundesland_data.items():
                df = data['df'].copy()
                if datetime_col in df.columns:
                    month_data = df[(df[datetime_col].dt.month == month) & 
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
                datetime_col_germany = 'datetime' if 'datetime' in df_germany.columns else 'MESS_DATUM'
                if datetime_col_germany in df_germany.columns:
                    month_data_germany = df_germany[
                        (df_germany[datetime_col_germany].dt.month == month) & 
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
            output_file = summary_dir / f"{attribute_name}_monthly_stats_2024.csv"
            summary_df.to_csv(output_file, index=False)
            print(f"  Created summary table: {output_file}")
    
    def visualize_attribute(self, attribute_name):
        """Create all visualizations for an attribute"""
        if attribute_name not in self.attribute_mappings:
            print(f"Unknown attribute: {attribute_name}")
            return
        
        attr_config = self.attribute_mappings[attribute_name]
        bundesland_folder = self.base_path / "Bundesland_aggregation" / attr_config['bundesland_folder']
        germany_folder = self.base_path / "Germany_aggregation" / attr_config['germany_folder']
        value_columns = attr_config['value_columns']
        y_label = attr_config['y_label']
        
        print(f"\n{'='*60}")
        print(f"Visualizing {attribute_name}")
        print(f"{'='*60}")
        
        # Check if folders exist
        if not bundesland_folder.exists():
            print(f"Bundesland folder not found: {bundesland_folder}")
            return
        if not germany_folder.exists():
            print(f"Germany folder not found: {germany_folder}")
            return
        
        # Load data
        print("Loading Bundesland data...")
        bundesland_data = self.load_bundesland_data(bundesland_folder, value_columns)
        print(f"  Loaded {len(bundesland_data)} Bundesländer")
        
        print("Loading Germany data...")
        germany_data = self.load_germany_data(germany_folder, value_columns)
        if germany_data:
            print("  Loaded Germany aggregation data")
        else:
            print("  Warning: No Germany aggregation data found")
        
        if not bundesland_data:
            print("No data to visualize!")
            return
        
        # Get value column from first Bundesland
        value_column = bundesland_data[list(bundesland_data.keys())[0]]['value_column']
        
        # Create output directory
        output_dir = self.viz_dir / attribute_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        print("\nCreating monthly plots...")
        self.create_monthly_plots(attribute_name, bundesland_data, germany_data,
                                 value_column, y_label, output_dir)
        
        print("\nCreating full-year plot...")
        self.create_full_year_plot(attribute_name, bundesland_data, germany_data,
                                 value_column, y_label, output_dir)
        
        print("\nCreating distribution plots...")
        self.create_distribution_plots(attribute_name, bundesland_data, germany_data,
                                     value_column, y_label, output_dir)
        
        print("\nCreating statistical summaries...")
        self.create_statistical_summaries(attribute_name, bundesland_data, germany_data,
                                         value_column, output_dir)
        
        print(f"\nCompleted visualization for {attribute_name}")


def main():
    """Main visualization function"""
    visualizer = WeatherDataVisualizer()
    
    # List of attributes to visualize
    attributes = [
        'temperature',
        'cloudiness',
        'wind',
        'wind_synop',
        'precipitation',
        'pressure',
        'dew_point',
        'moisture',
        'extreme_wind',
        'soil_temperature',
        'sun',
        'visibility'
    ]
    
    print("="*60)
    print("WEATHER DATA VISUALIZATION")
    print("="*60)
    print(f"Generating visualizations for {len(attributes)} attributes...")
    
    for attribute in attributes:
        try:
            visualizer.visualize_attribute(attribute)
        except Exception as e:
            print(f"Error visualizing {attribute}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"All visualizations saved to: visualizations/")


if __name__ == "__main__":
    main()

