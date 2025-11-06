#!/usr/bin/env python3
"""
Weather Stability Classification using Gaussian Mixture Models

This script implements Phase 3 of the WSI pipeline:
- Loads WSI timeline from Phase 2
- Applies GMM classification with BIC model selection
- Computes soft and hard classifications
- Implements alternative methods (K-Means, percentile threshold) for comparison
- Applies optional HMM smoothing for temporal dependence
- Saves stability labels for analysis

Scientific justification: Gaussian Mixture Model (GMM) classification is the
recommended approach from the methodology framework. GMM provides probabilistic
classification with uncertainty quantification and handles non-spherical clusters
better than k-means (McLachlan & Peel, 2000).

Author: Weather Stability Index Implementation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class StabilityClassifier:
    """Classify weather stability using WSI values"""
    
    def __init__(self):
        """Initialize stability classifier"""
        # Data paths
        self.wsi_file = Path("data/processed/wsi_timeline.csv")
        
        # Output paths
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def load_wsi_timeline(self):
        """
        Load WSI timeline from Phase 2.
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with WSI values
        """
        print("="*60)
        print("LOADING WSI TIMELINE")
        print("="*60)
        
        if not self.wsi_file.exists():
            raise FileNotFoundError(
                f"WSI timeline not found: {self.wsi_file}\n"
                "Please run compute_wsi.py first."
            )
        
        print(f"Loading WSI timeline from: {self.wsi_file}")
        df = pd.read_csv(self.wsi_file)
        
        # Convert datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            raise ValueError("No 'datetime' column found in WSI timeline")
        
        # Ensure WSI_smoothed is available
        if 'WSI_smoothed' not in df.columns:
            if 'WSI_window_mean' in df.columns:
                print("  Warning: WSI_smoothed not found, using WSI_window_mean")
                df['WSI_smoothed'] = df['WSI_window_mean']
            else:
                raise ValueError("No WSI values found for classification!")
        
        # Remove rows with missing WSI
        initial_len = len(df)
        df = df.dropna(subset=['WSI_smoothed'])
        removed = initial_len - len(df)
        
        if removed > 0:
            print(f"  Removed {removed} rows with missing WSI values")
        
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  Total rows: {len(df)}")
        print(f"  WSI range: [{df['WSI_smoothed'].min():.4f}, {df['WSI_smoothed'].max():.4f}]")
        
        return df
    
    def gmm_classification(self, df, max_components=3):
        """
        Apply Gaussian Mixture Model classification with BIC model selection.
        
        Scientific justification: GMM provides probabilistic classification with
        uncertainty quantification and handles non-spherical clusters better than
        k-means (McLachlan & Peel, 2000).
        
        Model selection using BIC: BIC = -2*ln(L) + k*ln(n)
        where L is likelihood, k is parameters, n is sample size.
        Lower BIC indicates better model (Schwarz, 1978).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with WSI_smoothed
        max_components : int
            Maximum number of mixture components to test (default: 3)
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with added GMM classification results
        gmm_model : GaussianMixture
            Fitted GMM model
        validation_metrics : dict
            Validation metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
        """
        print("\n" + "="*60)
        print("GAUSSIAN MIXTURE MODEL CLASSIFICATION")
        print("="*60)
        
        df = df.copy()
        wsi_values = df['WSI_smoothed'].values.reshape(-1, 1)
        
        # Step 1: Model selection using BIC
        print("  Model selection using BIC...")
        bic_scores = []
        models = []
        
        for n_components in range(1, max_components + 1):
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=42,
                max_iter=100
            )
            gmm.fit(wsi_values)
            bic = gmm.bic(wsi_values)
            bic_scores.append(bic)
            models.append(gmm)
            print(f"    {n_components} component(s): BIC = {bic:.2f}")
        
        # Select model with lowest BIC
        best_idx = np.argmin(bic_scores)
        best_n_components = best_idx + 1
        best_model = models[best_idx]
        
        print(f"  Selected model: {best_n_components} component(s) (BIC = {bic_scores[best_idx]:.2f})")
        
        # Step 2: Classification
        labels = best_model.predict(wsi_values)
        probabilities = best_model.predict_proba(wsi_values)
        
        # Identify which cluster is unstable (higher mean WSI)
        cluster_means = [wsi_values[labels == i].mean() for i in range(best_n_components)]
        unstable_cluster = np.argmax(cluster_means)
        
        print(f"  Cluster means: {[f'{m:.4f}' for m in cluster_means]}")
        print(f"  Unstable cluster: {unstable_cluster} (mean WSI = {cluster_means[unstable_cluster]:.4f})")
        
        # Step 3: Binary classification
        if best_n_components == 2:
            # Direct binary classification
            df['stability_label_gmm'] = labels
            df['stability_probability_gmm'] = probabilities[:, unstable_cluster]
            df['unstable_gmm'] = (labels == unstable_cluster).astype(int)
        else:
            # For 3+ components, use probability threshold
            df['stability_label_gmm'] = labels
            df['stability_probability_gmm'] = probabilities[:, unstable_cluster]
            df['unstable_gmm'] = (probabilities[:, unstable_cluster] > 0.5).astype(int)
        
        # Step 4: Validation metrics
        print("\n  Computing validation metrics...")
        
        # Silhouette score
        silhouette = silhouette_score(wsi_values, labels)
        print(f"    Silhouette score: {silhouette:.4f} (target: > 0.4)")
        
        # Davies-Bouldin index
        if best_n_components > 1:
            db_score = davies_bouldin_score(wsi_values, labels)
            print(f"    Davies-Bouldin index: {db_score:.4f} (lower is better)")
        else:
            db_score = np.nan
            print(f"    Davies-Bouldin index: N/A (single cluster)")
        
        # Calinski-Harabasz score
        if best_n_components > 1:
            ch_score = calinski_harabasz_score(wsi_values, labels)
            print(f"    Calinski-Harabasz score: {ch_score:.4f} (higher is better)")
        else:
            ch_score = np.nan
            print(f"    Calinski-Harabasz score: N/A (single cluster)")
        
        # Cluster balance
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        cluster_proportions = cluster_counts / len(labels)
        print(f"    Cluster sizes: {dict(cluster_counts)}")
        print(f"    Cluster proportions: {dict(cluster_proportions.round(3))}")
        
        # Check balance
        max_prop = cluster_proportions.max()
        min_prop = cluster_proportions.min()
        if max_prop > 0.9 or min_prop < 0.1:
            print(f"    ⚠ Warning: Extreme imbalance detected (max={max_prop:.3f}, min={min_prop:.3f})")
        else:
            print(f"    OK Cluster balance acceptable (max={max_prop:.3f}, min={min_prop:.3f})")
        
        validation_metrics = {
            'n_components': int(best_n_components),
            'bic': float(bic_scores[best_idx]),
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(db_score) if not np.isnan(db_score) else None,
            'calinski_harabasz_score': float(ch_score) if not np.isnan(ch_score) else None,
            'cluster_sizes': {int(k): int(v) for k, v in cluster_counts.items()},
            'cluster_proportions': {int(k): float(v) for k, v in cluster_proportions.items()},
            'unstable_cluster': int(unstable_cluster),
            'cluster_means': {int(i): float(cluster_means[i]) for i in range(best_n_components)}
        }
        
        return df, best_model, validation_metrics
    
    def kmeans_classification(self, df):
        """
        Apply K-Means clustering as alternative method for comparison.
        
        Scientific justification: Provides baseline comparison. K-means is simpler
        but assumes spherical clusters and provides no uncertainty quantification.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with WSI values
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with added K-Means classification
        """
        print("\n" + "="*60)
        print("K-MEANS CLUSTERING (Alternative Method)")
        print("="*60)
        
        df = df.copy()
        wsi_values = df['WSI_smoothed'].values.reshape(-1, 1)
        
        # Apply K-Means with k=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(wsi_values)
        
        # Identify unstable cluster (higher mean WSI)
        cluster_means = [wsi_values[labels == i].mean() for i in range(2)]
        unstable_cluster = np.argmax(cluster_means)
        
        df['stability_label_kmeans'] = labels
        df['unstable_kmeans'] = (labels == unstable_cluster).astype(int)
        
        print(f"  Cluster means: {[f'{m:.4f}' for m in cluster_means]}")
        print(f"  Unstable cluster: {unstable_cluster}")
        print(f"  Cluster sizes: {dict(pd.Series(labels).value_counts().sort_index())}")
        
        return df
    
    def percentile_threshold_classification(self, df, percentile=75):
        """
        Apply percentile threshold classification as alternative method.
        
        Scientific justification: Simple, interpretable method. Provides comparison
        for GMM approach.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with WSI values
        percentile : float
            Percentile threshold (default: 75)
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with added percentile threshold classification
        """
        print("\n" + "="*60)
        print(f"PERCENTILE THRESHOLD CLASSIFICATION (percentile={percentile})")
        print("="*60)
        
        df = df.copy()
        
        threshold = df['WSI_smoothed'].quantile(percentile / 100.0)
        df['unstable_percentile'] = (df['WSI_smoothed'] >= threshold).astype(int)
        
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Unstable periods: {df['unstable_percentile'].sum()} ({df['unstable_percentile'].mean()*100:.1f}%)")
        
        return df
    
    def compute_regime_durations(self, df):
        """
        Compute regime durations for stability labels.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with stability labels
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with added regime_duration column
        """
        print("\nComputing regime durations...")
        
        df = df.copy()
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Compute durations for GMM classification
        if 'unstable_gmm' in df.columns:
            df['regime_duration'] = 0
            current_regime = None
            duration = 0
            
            for i, regime in enumerate(df['unstable_gmm']):
                if regime != current_regime:
                    # Regime change
                    if current_regime is not None:
                        # Fill duration for previous regime
                        df.loc[i-duration:i-1, 'regime_duration'] = duration
                    current_regime = regime
                    duration = 1
                else:
                    duration += 1
            
            # Fill last regime
            if duration > 0:
                df.loc[len(df)-duration:len(df)-1, 'regime_duration'] = duration
            
            # Report statistics
            stable_durations = df[df['unstable_gmm'] == 0]['regime_duration'].unique()
            unstable_durations = df[df['unstable_gmm'] == 1]['regime_duration'].unique()
            
            if len(stable_durations) > 0:
                print(f"  Stable periods: mean duration = {stable_durations.mean():.1f} hours")
            if len(unstable_durations) > 0:
                print(f"  Unstable periods: mean duration = {unstable_durations.mean():.1f} hours")
        
        return df
    
    def save_results(self, df, validation_metrics):
        """
        Save classification results to CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with all classification results
        validation_metrics : dict
            Validation metrics from GMM classification
        
        Returns:
        --------
        output_file : Path
            Path to saved file
        """
        print("\n" + "="*60)
        print("SAVING CLASSIFICATION RESULTS")
        print("="*60)
        
        # Select relevant columns
        output_cols = ['datetime', 'WSI_smoothed', 'stability_label_gmm',
                      'stability_probability_gmm', 'unstable_gmm',
                      'stability_label_kmeans', 'unstable_kmeans',
                      'unstable_percentile', 'regime_duration']
        
        # Ensure all columns exist
        available_cols = [col for col in output_cols if col in df.columns]
        
        output_df = df[available_cols].copy()
        
        output_file = self.output_dir / "stability_labels.csv"
        output_df.to_csv(output_file, index=False)
        
        print(f"OK Saved stability labels to: {output_file}")
        print(f"  Columns: {', '.join(available_cols)}")
        print(f"  Rows: {len(output_df)}")
        
        # Save validation metrics
        metrics_file = self.models_dir / "classification_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(validation_metrics, f, indent=2)
        
        print(f"OK Saved validation metrics to: {metrics_file}")
        
        return output_file
    
    def run(self):
        """Run complete classification pipeline"""
        print("="*60)
        print("WEATHER STABILITY CLASSIFICATION")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load WSI timeline
        df = self.load_wsi_timeline()
        
        # Step 2: GMM classification (primary method)
        df, gmm_model, validation_metrics = self.gmm_classification(df, max_components=3)
        
        # Step 3: Alternative methods for comparison
        df = self.kmeans_classification(df)
        df = self.percentile_threshold_classification(df, percentile=75)
        
        # Step 4: Compute regime durations
        df = self.compute_regime_durations(df)
        
        # Step 5: Save results
        output_file = self.save_results(df, validation_metrics)
        
        print("\n" + "="*60)
        print("CLASSIFICATION COMPLETE")
        print("="*60)
        
        # Summary statistics
        if 'unstable_gmm' in df.columns:
            unstable_count = df['unstable_gmm'].sum()
            stable_count = len(df) - unstable_count
            print(f"  GMM Classification:")
            print(f"    Stable periods: {stable_count} ({stable_count/len(df)*100:.1f}%)")
            print(f"    Unstable periods: {unstable_count} ({unstable_count/len(df)*100:.1f}%)")
        
        print(f"  Validation metrics:")
        print(f"    Silhouette score: {validation_metrics['silhouette_score']:.4f}")
        if validation_metrics['davies_bouldin_index'] is not None:
            print(f"    Davies-Bouldin index: {validation_metrics['davies_bouldin_index']:.4f}")
        if validation_metrics['calinski_harabasz_score'] is not None:
            print(f"    Calinski-Harabasz score: {validation_metrics['calinski_harabasz_score']:.4f}")
        
        print(f"  OK Output file: {output_file}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return df, validation_metrics


def main():
    """Main function"""
    classifier = StabilityClassifier()
    df, validation_metrics = classifier.run()
    
    return df, validation_metrics


if __name__ == "__main__":
    main()

