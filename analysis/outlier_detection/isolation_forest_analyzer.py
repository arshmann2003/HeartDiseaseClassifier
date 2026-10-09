"""
Isolation Forest Outlier Detection Module
Provides functions for detecting and analyzing outliers in heart disease data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class IsolationForestAnalyzer:
    """
    A class for performing Isolation Forest outlier detection analysis.
    """
    
    def __init__(self, X, y=None, random_state=42):
        """
        Initialize the Isolation Forest Analyzer.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature data (should be standardized)
        y : array-like, shape (n_samples,), optional
            Target variable (for analysis purposes)
        random_state : int, default=42
            Random state for reproducibility
        """
        self.X = X
        self.y = y
        self.random_state = random_state
        self.isolation_forest = None
        self.outlier_labels = None
        self.outlier_scores = None
        self.outlier_indices = None
        self.n_outliers = None
        self.n_inliers = None
        
    def fit(self, contamination=0.1, n_estimators=100, max_samples='auto', verbose=True):
        """
        Fit Isolation Forest model to detect outliers.
        
        Parameters:
        -----------
        contamination : float, default=0.1
            Expected proportion of outliers (0.0 to 0.5)
        n_estimators : int, default=100
            Number of trees in the forest
        max_samples : int or 'auto', default='auto'
            Number of samples to draw for each tree
        verbose : bool, default=True
            Whether to print progress
            
        Returns:
        --------
        outlier_labels : array
            Outlier labels (-1 for outliers, 1 for inliers)
        """
        if verbose:
            print(f"Fitting Isolation Forest with contamination={contamination}...")
        
        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.outlier_labels = self.isolation_forest.fit_predict(self.X)
        self.outlier_scores = self.isolation_forest.score_samples(self.X)
        
        # Get outlier indices
        self.outlier_indices = np.where(self.outlier_labels == -1)[0]
        self.n_outliers = len(self.outlier_indices)
        self.n_inliers = len(self.X) - self.n_outliers
        
        if verbose:
            print(f"  Detected {self.n_outliers} outliers ({self.n_outliers/len(self.X)*100:.2f}%)")
            print(f"  {self.n_inliers} inliers ({self.n_inliers/len(self.X)*100:.2f}%)")
        
        return self.outlier_labels
    
    def get_outlier_summary(self):
        """Get summary statistics of outliers."""
        if self.outlier_labels is None:
            print("No outlier detection performed yet. Call fit() first.")
            return None
        
        summary = {
            'n_outliers': self.n_outliers,
            'n_inliers': self.n_inliers,
            'outlier_percentage': self.n_outliers / len(self.X) * 100,
            'inlier_percentage': self.n_inliers / len(self.X) * 100,
            'mean_outlier_score': self.outlier_scores[self.outlier_indices].mean() if len(self.outlier_indices) > 0 else None,
            'mean_inlier_score': self.outlier_scores[self.outlier_labels == 1].mean() if len(self.outlier_labels[self.outlier_labels == 1]) > 0 else None
        }
        
        return summary
    
    def print_summary(self):
        """Print summary of outlier detection results."""
        summary = self.get_outlier_summary()
        if summary is None:
            return
        
        print("\n" + "="*60)
        print("OUTLIER DETECTION SUMMARY")
        print("="*60)
        print(f"Total samples: {len(self.X)}")
        print(f"Outliers detected: {summary['n_outliers']} ({summary['outlier_percentage']:.2f}%)")
        print(f"Inliers: {summary['n_inliers']} ({summary['inlier_percentage']:.2f}%)")
        if summary['mean_outlier_score'] is not None:
            print(f"Mean outlier score: {summary['mean_outlier_score']:.4f}")
        if summary['mean_inlier_score'] is not None:
            print(f"Mean inlier score: {summary['mean_inlier_score']:.4f}")
        print("="*60)
    
    def visualize_outliers_pca(self, y_sample=None, figsize=(14, 6), sample_size=None):
        """
        Visualize outliers using PCA (2D scatter plot).
        
        Parameters:
        -----------
        y_sample : array-like, optional
            Target variable for additional analysis
        figsize : tuple, default=(14, 6)
            Figure size
        sample_size : int, optional
            Sample size for visualization (for large datasets)
        """
        if self.outlier_labels is None:
            print("No outlier detection performed yet. Call fit() first.")
            return
        
        # Sample data if specified
        if sample_size and sample_size < len(self.X):
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(len(self.X), size=sample_size, replace=False)
            X_viz = self.X[sample_indices]
            labels_viz = self.outlier_labels[sample_indices]
            scores_viz = self.outlier_scores[sample_indices]
            if y_sample is not None:
                y_viz = y_sample[sample_indices]
            else:
                y_viz = None
        else:
            X_viz = self.X
            labels_viz = self.outlier_labels
            scores_viz = self.outlier_scores
            y_viz = y_sample
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_viz)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Outliers vs Inliers
        ax1 = axes[0]
        inlier_mask = labels_viz == 1
        outlier_mask = labels_viz == -1
        
        ax1.scatter(
            X_pca[inlier_mask, 0], X_pca[inlier_mask, 1],
            c='blue', alpha=0.5, s=20, label='Inliers', edgecolors='black', linewidth=0.3
        )
        ax1.scatter(
            X_pca[outlier_mask, 0], X_pca[outlier_mask, 1],
            c='red', alpha=0.8, s=50, label='Outliers', edgecolors='black', linewidth=0.5, marker='X'
        )
        ax1.set_xlabel(
            f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})',
            fontsize=11
        )
        ax1.set_ylabel(
            f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})',
            fontsize=11
        )
        ax1.set_title('Outlier Detection Results (PCA Visualization)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Outlier scores
        ax2 = axes[1]
        scatter = ax2.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=scores_viz, cmap='RdYlGn', alpha=0.6, s=20, edgecolors='black', linewidth=0.3
        )
        plt.colorbar(scatter, ax=ax2, label='Outlier Score (Lower = More Outlier)')
        ax2.set_xlabel(
            f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})',
            fontsize=11
        )
        ax2.set_ylabel(
            f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})',
            fontsize=11
        )
        ax2.set_title('Outlier Scores (PCA Visualization)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Total explained variance by first 2 components: {pca.explained_variance_ratio_.sum():.2%}")
    
    def visualize_outlier_distribution(self, feature_names=None, top_n=10, figsize=(14, 8)):
        """
        Visualize distribution of outliers across features.
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        top_n : int, default=10
            Number of top features to visualize
        figsize : tuple, default=(14, 8)
            Figure size
        """
        if self.outlier_labels is None:
            print("No outlier detection performed yet. Call fit() first.")
            return
        
        # Create DataFrame
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(self.X.shape[1])]
        
        df_viz = pd.DataFrame(self.X, columns=feature_names)
        df_viz['Outlier'] = self.outlier_labels == -1
        
        # Calculate mean values for outliers vs inliers
        outlier_means = df_viz[df_viz['Outlier']].drop(columns=['Outlier']).mean()
        inlier_means = df_viz[~df_viz['Outlier']].drop(columns=['Outlier']).mean()
        
        # Find features with largest differences
        differences = np.abs(outlier_means - inlier_means).sort_values(ascending=False)
        top_features = differences.head(top_n).index
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Outliers': outlier_means[top_features],
            'Inliers': inlier_means[top_features]
        })
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Bar chart comparison
        ax1 = axes[0]
        x = np.arange(len(top_features))
        width = 0.35
        ax1.bar(x - width/2, comparison_df['Inliers'], width, label='Inliers', alpha=0.8)
        ax1.bar(x + width/2, comparison_df['Outliers'], width, label='Outliers', alpha=0.8)
        ax1.set_xlabel('Features', fontsize=12)
        ax1.set_ylabel('Mean Value', fontsize=12)
        ax1.set_title(f'Top {top_n} Features: Outliers vs Inliers', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(top_features, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Difference plot
        ax2 = axes[1]
        differences_sorted = differences.head(top_n).sort_values()
        ax2.barh(range(len(differences_sorted)), differences_sorted.values, alpha=0.8)
        ax2.set_yticks(range(len(differences_sorted)))
        ax2.set_yticklabels(differences_sorted.index)
        ax2.set_xlabel('Absolute Difference (Outlier Mean - Inlier Mean)', fontsize=12)
        ax2.set_title(f'Top {top_n} Features: Largest Differences', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_outlier_target_relationship(self, y_sample):
        """
        Analyze relationship between outliers and target variable.
        
        Parameters:
        -----------
        y_sample : array-like
            Target variable values
            
        Returns:
        --------
        analysis_df : DataFrame
            Analysis results
        """
        if self.outlier_labels is None:
            print("No outlier detection performed yet. Call fit() first.")
            return None
        
        outlier_analysis = pd.DataFrame({
            'Outlier': self.outlier_labels == -1,
            'Target': y_sample
        })
        
        # Cross-tabulation
        crosstab = pd.crosstab(
            outlier_analysis['Outlier'],
            outlier_analysis['Target'],
            margins=True,
            normalize='index'
        )
        print("Outlier vs Target Distribution (Row Percentages):")
        print(crosstab)
        
        # Count by outlier status
        print("\nOutlier vs Target Counts:")
        crosstab_counts = pd.crosstab(
            outlier_analysis['Outlier'],
            outlier_analysis['Target']
        )
        print(crosstab_counts)
        
        # Calculate target rate
        print("\nTarget Rate by Outlier Status:")
        for is_outlier in [False, True]:
            status_name = "Outliers" if is_outlier else "Inliers"
            status_data = outlier_analysis[outlier_analysis['Outlier'] == is_outlier]
            if len(status_data) > 0:
                target_rate = status_data['Target'].mean()
                print(f"  {status_name}: {target_rate:.2%} "
                      f"({status_data['Target'].sum()}/{len(status_data)} samples)")
        
        return outlier_analysis
    
    def analyze_outlier_characteristics(self, feature_names=None):
        """
        Analyze characteristics of outliers vs inliers.
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features
            
        Returns:
        --------
        comparison_df : DataFrame
            Comparison of outlier vs inlier characteristics
        """
        if self.outlier_labels is None:
            print("No outlier detection performed yet. Call fit() first.")
            return None
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(self.X.shape[1])]
        
        # Create DataFrame
        df_analysis = pd.DataFrame(self.X, columns=feature_names)
        df_analysis['IsOutlier'] = self.outlier_labels == -1
        
        # Calculate statistics
        outlier_stats = df_analysis[df_analysis['IsOutlier']].drop(columns=['IsOutlier']).describe()
        inlier_stats = df_analysis[~df_analysis['IsOutlier']].drop(columns=['IsOutlier']).describe()
        
        # Create comparison
        comparison_df = pd.DataFrame({
            'Outlier_Mean': outlier_stats.loc['mean'],
            'Inlier_Mean': inlier_stats.loc['mean'],
            'Outlier_Std': outlier_stats.loc['std'],
            'Inlier_Std': inlier_stats.loc['std'],
            'Difference': outlier_stats.loc['mean'] - inlier_stats.loc['mean']
        })
        comparison_df['Abs_Difference'] = np.abs(comparison_df['Difference'])
        comparison_df = comparison_df.sort_values('Abs_Difference', ascending=False)
        
        print("Outlier vs Inlier Feature Comparison:")
        print(comparison_df.round(4))
        
        return comparison_df
    
    def get_outlier_data(self):
        """
        Get outlier data points.
        
        Returns:
        --------
        outlier_data : array
            Outlier data points
        """
        if self.outlier_labels is None:
            print("No outlier detection performed yet. Call fit() first.")
            return None
        
        return self.X[self.outlier_indices]
    
    def get_inlier_data(self):
        """
        Get inlier data points.
        
        Returns:
        --------
        inlier_data : array
            Inlier data points
        """
        if self.outlier_labels is None:
            print("No outlier detection performed yet. Call fit() first.")
            return None
        
        inlier_indices = np.where(self.outlier_labels == 1)[0]
        return self.X[inlier_indices]
    
    def remove_outliers(self):
        """
        Remove outliers from the dataset.
        
        Returns:
        --------
        X_cleaned : array
            Data without outliers
        y_cleaned : array or None
            Target without outliers (if y was provided)
        """
        if self.outlier_labels is None:
            print("No outlier detection performed yet. Call fit() first.")
            return None, None
        
        inlier_mask = self.outlier_labels == 1
        X_cleaned = self.X[inlier_mask]
        
        if self.y is not None:
            y_cleaned = self.y[inlier_mask]
        else:
            y_cleaned = None
        
        print(f"Removed {self.n_outliers} outliers. Remaining: {len(X_cleaned)} samples")
        
        return X_cleaned, y_cleaned
    
    def decide_outlier_action(self, y_sample=None, threshold_heart_disease_rate=0.15):
        """
        Decide whether to keep or remove outliers based on analysis.
        
        Parameters:
        -----------
        y_sample : array-like, optional
            Target variable for analysis
        threshold_heart_disease_rate : float, default=0.15
            Threshold for heart disease rate in outliers to consider them informative
            
        Returns:
        --------
        decision : str
            'keep' or 'remove'
        reasoning : str
            Reasoning for the decision
        """
        if self.outlier_labels is None:
            print("No outlier detection performed yet. Call fit() first.")
            return None, None
        
        reasoning_parts = []
        
        # Check outlier percentage
        outlier_pct = self.n_outliers / len(self.X) * 100
        reasoning_parts.append(f"Outlier percentage: {outlier_pct:.2f}%")
        
        # Check relationship with target if available
        if y_sample is not None:
            outlier_analysis = pd.DataFrame({
                'Outlier': self.outlier_labels == -1,
                'Target': y_sample
            })
            
            outlier_target_rate = outlier_analysis[outlier_analysis['Outlier']]['Target'].mean()
            inlier_target_rate = outlier_analysis[~outlier_analysis['Outlier']]['Target'].mean()
            
            reasoning_parts.append(f"Heart disease rate in outliers: {outlier_target_rate:.2%}")
            reasoning_parts.append(f"Heart disease rate in inliers: {inlier_target_rate:.2%}")
            
            # Decision logic
            if outlier_target_rate > threshold_heart_disease_rate:
                decision = 'keep'
                reasoning_parts.append(
                    f"Outliers have higher heart disease rate ({outlier_target_rate:.2%} > {threshold_heart_disease_rate:.2%}). "
                    "They may contain important information about high-risk cases."
                )
            elif abs(outlier_target_rate - inlier_target_rate) < 0.05:
                decision = 'remove'
                reasoning_parts.append(
                    "Outliers have similar heart disease rate to inliers. "
                    "They are likely noise and can be removed."
                )
            else:
                decision = 'keep'
                reasoning_parts.append(
                    "Outliers show different patterns. They may represent edge cases worth investigating."
                )
        else:
            # If no target, use general heuristics
            if outlier_pct > 20:
                decision = 'remove'
                reasoning_parts.append("Too many outliers (>20%). Likely noise. Recommend removal.")
            else:
                decision = 'keep'
                reasoning_parts.append("Moderate number of outliers. May contain valuable information.")
        
        reasoning = "\n".join(reasoning_parts)
        
        print("\n" + "="*60)
        print("OUTLIER DECISION ANALYSIS")
        print("="*60)
        print(reasoning)
        print(f"\nDecision: {decision.upper()} outliers")
        print("="*60)
        
        return decision, reasoning



