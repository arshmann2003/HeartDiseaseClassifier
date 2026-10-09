"""
K-Means Clustering Analysis Module
Provides functions for performing K-Means clustering analysis on heart disease data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class KMeansAnalyzer:
    """
    A class for performing K-Means clustering analysis.
    """
    
    def __init__(self, X, y=None, random_state=42):
        """
        Initialize the K-Means Analyzer.
        
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
        self.kmeans_model = None
        self.cluster_labels = None
        self.optimal_k = None
        self.evaluation_metrics = {}
        
    def find_optimal_k(self, k_range=range(2, 11), sample_size=None, verbose=True):
        """
        Find optimal number of clusters using multiple metrics.
        
        Parameters:
        -----------
        k_range : range, default=range(2, 11)
            Range of k values to test
        sample_size : int, optional
            Sample size for faster computation. If None, uses full dataset.
        verbose : bool, default=True
            Whether to print progress
            
        Returns:
        --------
        results_df : DataFrame
            DataFrame with evaluation metrics for each k
        """
        # Sample data if specified
        if sample_size and sample_size < len(self.X):
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(
                len(self.X), 
                size=sample_size, 
                replace=False
            )
            X_sample = self.X[sample_indices]
        else:
            X_sample = self.X
            
        if verbose:
            print(f"Using sample size: {len(X_sample)} for clustering analysis")
        
        # Test different values of k
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_sample)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_sample, labels))
            calinski_scores.append(calinski_harabasz_score(X_sample, labels))
            davies_scores.append(davies_bouldin_score(X_sample, labels))
            
            if verbose:
                print(f"k={k}: Silhouette={silhouette_scores[-1]:.4f}, "
                      f"Calinski-Harabasz={calinski_scores[-1]:.4f}, "
                      f"Davies-Bouldin={davies_scores[-1]:.4f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'k': list(k_range),
            'Inertia': inertias,
            'Silhouette Score': silhouette_scores,
            'Calinski-Harabasz Index': calinski_scores,
            'Davies-Bouldin Index': davies_scores
        })
        
        # Find optimal k (highest silhouette score)
        self.optimal_k = k_range[np.argmax(silhouette_scores)]
        
        return results_df
    
    def fit(self, n_clusters=None, sample_size=None):
        """
        Fit K-Means clustering model.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters. If None, uses optimal_k from find_optimal_k()
        sample_size : int, optional
            Sample size for faster computation. If None, uses full dataset.
            
        Returns:
        --------
        cluster_labels : array
            Cluster labels for each sample
        """
        if n_clusters is None:
            if self.optimal_k is None:
                raise ValueError("Must specify n_clusters or call find_optimal_k() first")
            n_clusters = self.optimal_k
        
        # Sample data if specified
        if sample_size and sample_size < len(self.X):
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(
                len(self.X), 
                size=sample_size, 
                replace=False
            )
            X_sample = self.X[sample_indices]
            self.sample_indices = sample_indices
        else:
            X_sample = self.X
            self.sample_indices = None
        
        # Fit K-Means
        self.kmeans_model = KMeans(
            n_clusters=n_clusters, 
            random_state=self.random_state, 
            n_init=10
        )
        self.cluster_labels = self.kmeans_model.fit_predict(X_sample)
        
        # Evaluate clustering
        self.evaluation_metrics = {
            'Silhouette Score': silhouette_score(X_sample, self.cluster_labels),
            'Calinski-Harabasz Index': calinski_harabasz_score(X_sample, self.cluster_labels),
            'Davies-Bouldin Index': davies_bouldin_score(X_sample, self.cluster_labels)
        }
        
        return self.cluster_labels
    
    def get_evaluation_metrics(self):
        """Return evaluation metrics."""
        return self.evaluation_metrics
    
    def print_cluster_summary(self):
        """Print summary of clustering results."""
        if self.cluster_labels is None:
            print("No clustering performed yet. Call fit() first.")
            return
        
        n_clusters = len(np.unique(self.cluster_labels))
        print(f"K-Means Clustering Results (k={n_clusters}):")
        for metric, value in self.evaluation_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nCluster sizes:")
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} samples ({count/len(self.cluster_labels)*100:.2f}%)")
    
    def visualize_optimal_k(self, results_df, figsize=(15, 10)):
        """
        Visualize metrics to find optimal k.
        
        Parameters:
        -----------
        results_df : DataFrame
            Results from find_optimal_k()
        figsize : tuple, default=(15, 10)
            Figure size
        """
        k_range = results_df['k'].values
        inertias = results_df['Inertia'].values
        silhouette_scores = results_df['Silhouette Score'].values
        calinski_scores = results_df['Calinski-Harabasz Index'].values
        davies_scores = results_df['Davies-Bouldin Index'].values
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Elbow Method
        axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0, 0].set_ylabel('Inertia', fontsize=12)
        axes[0, 0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette Score
        axes[0, 1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
        axes[0, 1].set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz Index
        axes[1, 0].plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1, 0].set_ylabel('Calinski-Harabasz Index', fontsize=12)
        axes[1, 0].set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Davies-Bouldin Index
        axes[1, 1].plot(k_range, davies_scores, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1, 1].set_ylabel('Davies-Bouldin Index', fontsize=12)
        axes[1, 1].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        if self.optimal_k is not None:
            print(f"\nOptimal number of clusters (based on Silhouette Score): k = {self.optimal_k}")
    
    def visualize_pca(self, y_sample=None, figsize=(14, 6)):
        """
        Visualize clusters using PCA (2D).
        
        Parameters:
        -----------
        y_sample : array-like, optional
            Target variable for comparison plot
        figsize : tuple, default=(14, 6)
            Figure size
        """
        if self.cluster_labels is None:
            print("No clustering performed yet. Call fit() first.")
            return
        
        # Get the data that was used for clustering
        if hasattr(self, 'sample_indices') and self.sample_indices is not None:
            X_sample = self.X[self.sample_indices]
        else:
            X_sample = self.X
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_sample)
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        # Plot 1: Clusters colored by cluster assignment
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1], 
            c=self.cluster_labels, 
            cmap='viridis', 
            alpha=0.6, 
            s=20, 
            edgecolors='black', 
            linewidth=0.5
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(
            f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})', 
            fontsize=11
        )
        plt.ylabel(
            f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})', 
            fontsize=11
        )
        plt.title('K-Means Clustering Results (PCA Visualization)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Clusters colored by target variable (if provided)
        if y_sample is not None:
            plt.subplot(1, 2, 2)
            scatter2 = plt.scatter(
                X_pca[:, 0], X_pca[:, 1], 
                c=y_sample, 
                cmap='RdYlGn_r', 
                alpha=0.6, 
                s=20, 
                edgecolors='black', 
                linewidth=0.5
            )
            plt.colorbar(scatter2, label='Heart Disease (0=No, 1=Yes)')
            plt.xlabel(
                f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})', 
                fontsize=11
            )
            plt.ylabel(
                f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})', 
                fontsize=11
            )
            plt.title('Actual Heart Disease Distribution (PCA Visualization)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Total explained variance by first 2 components: {pca.explained_variance_ratio_.sum():.2%}")
    
    def visualize_tsne(self, y_sample=None, sample_size=5000, perplexity=30, figsize=(14, 6)):
        """
        Visualize clusters using t-SNE (2D).
        
        Parameters:
        -----------
        y_sample : array-like, optional
            Target variable for comparison plot
        sample_size : int, default=5000
            Sample size for t-SNE (computationally expensive)
        perplexity : int, default=30
            t-SNE perplexity parameter
        figsize : tuple, default=(14, 6)
            Figure size
        """
        if self.cluster_labels is None:
            print("No clustering performed yet. Call fit() first.")
            return
        
        # Get the data that was used for clustering
        if hasattr(self, 'sample_indices') and self.sample_indices is not None:
            X_sample = self.X[self.sample_indices]
            labels_sample = self.cluster_labels
        else:
            X_sample = self.X
            labels_sample = self.cluster_labels
        
        # Further sample for t-SNE
        tsne_sample_size = min(sample_size, len(X_sample))
        np.random.seed(self.random_state)
        tsne_indices = np.random.choice(len(X_sample), size=tsne_sample_size, replace=False)
        X_tsne_sample = X_sample[tsne_indices]
        labels_tsne_sample = labels_sample[tsne_indices]
        
        if y_sample is not None:
            if hasattr(self, 'sample_indices') and self.sample_indices is not None:
                y_tsne_sample = y_sample[self.sample_indices][tsne_indices]
            else:
                y_tsne_sample = y_sample[tsne_indices]
        else:
            y_tsne_sample = None
        
        print("Computing t-SNE visualization (this may take a few minutes)...")
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=perplexity, n_iter=1000)
        X_tsne = tsne.fit_transform(X_tsne_sample)
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        # Plot 1: Clusters
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(
            X_tsne[:, 0], X_tsne[:, 1], 
            c=labels_tsne_sample, 
            cmap='viridis', 
            alpha=0.6, 
            s=20, 
            edgecolors='black', 
            linewidth=0.5
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('t-SNE Component 1', fontsize=11)
        plt.ylabel('t-SNE Component 2', fontsize=11)
        plt.title('K-Means Clustering Results (t-SNE Visualization)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Heart Disease (if provided)
        if y_tsne_sample is not None:
            plt.subplot(1, 2, 2)
            scatter2 = plt.scatter(
                X_tsne[:, 0], X_tsne[:, 1], 
                c=y_tsne_sample, 
                cmap='RdYlGn_r', 
                alpha=0.6, 
                s=20, 
                edgecolors='black', 
                linewidth=0.5
            )
            plt.colorbar(scatter2, label='Heart Disease (0=No, 1=Yes)')
            plt.xlabel('t-SNE Component 1', fontsize=11)
            plt.ylabel('t-SNE Component 2', fontsize=11)
            plt.title('Actual Heart Disease Distribution (t-SNE Visualization)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_cluster_target_relationship(self, y_sample):
        """
        Analyze relationship between clusters and target variable.
        
        Parameters:
        -----------
        y_sample : array-like
            Target variable values
            
        Returns:
        --------
        analysis_df : DataFrame
            Analysis results
        """
        if self.cluster_labels is None:
            print("No clustering performed yet. Call fit() first.")
            return None
        
        cluster_analysis = pd.DataFrame({
            'Cluster': self.cluster_labels,
            'HeartDisease': y_sample
        })
        
        # Cross-tabulation
        crosstab = pd.crosstab(
            cluster_analysis['Cluster'], 
            cluster_analysis['HeartDisease'], 
            margins=True, 
            normalize='index'
        )
        print("Cluster vs Heart Disease Distribution (Row Percentages):")
        print(crosstab)
        
        # Count by cluster
        print("\nCluster vs Heart Disease Counts:")
        crosstab_counts = pd.crosstab(
            cluster_analysis['Cluster'], 
            cluster_analysis['HeartDisease']
        )
        print(crosstab_counts)
        
        # Calculate heart disease rate per cluster
        print("\nHeart Disease Rate by Cluster:")
        for cluster_id in sorted(cluster_analysis['Cluster'].unique()):
            cluster_data = cluster_analysis[cluster_analysis['Cluster'] == cluster_id]
            heart_disease_rate = cluster_data['HeartDisease'].mean()
            print(f"  Cluster {cluster_id}: {heart_disease_rate:.2%} "
                  f"({cluster_data['HeartDisease'].sum()}/{len(cluster_data)} samples)")
        
        return cluster_analysis
    
    def get_cluster_centers(self, feature_names, scaler=None):
        """
        Get cluster centers in original feature space.
        
        Parameters:
        -----------
        feature_names : list
            Names of features
        scaler : StandardScaler, optional
            Scaler used to transform data. If provided, centers are inverse transformed.
            
        Returns:
        --------
        centers_df : DataFrame
            Cluster centers DataFrame
        """
        if self.kmeans_model is None:
            print("No clustering performed yet. Call fit() first.")
            return None
        
        cluster_centers = self.kmeans_model.cluster_centers_
        
        # Transform back to original scale if scaler provided
        if scaler is not None:
            cluster_centers_original = scaler.inverse_transform(cluster_centers)
        else:
            cluster_centers_original = cluster_centers
        
        # Create DataFrame
        n_clusters = len(cluster_centers)
        centers_df = pd.DataFrame(
            cluster_centers_original, 
            columns=feature_names
        )
        centers_df.index = [f'Cluster {i}' for i in range(n_clusters)]
        
        return centers_df
    
    def visualize_cluster_characteristics(self, feature_names, scaler=None, top_n=10, figsize=(14, 8)):
        """
        Visualize cluster characteristics - average feature values per cluster.
        
        Parameters:
        -----------
        feature_names : list
            Names of features
        scaler : StandardScaler, optional
            Scaler used to transform data
        top_n : int, default=10
            Number of top features to visualize
        figsize : tuple, default=(14, 8)
            Figure size
        """
        centers_df = self.get_cluster_centers(feature_names, scaler)
        
        if centers_df is None:
            return
        
        print("Average Feature Values per Cluster (Original Scale):")
        print(centers_df.round(2))
        
        # Visualize top features that differ most between clusters
        feature_variance = centers_df.var(axis=0).sort_values(ascending=False)
        top_features = feature_variance.head(top_n).index
        
        plt.figure(figsize=figsize)
        centers_df[top_features].T.plot(kind='bar', width=0.8)
        plt.title(f'Top {top_n} Features: Average Values by Cluster', fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Average Value', fontsize=12)
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

