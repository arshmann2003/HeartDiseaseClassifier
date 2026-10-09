"""
Mutual Information Feature Selection Module
Provides functions for selecting important features using mutual information.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    mutual_info_classif,
    SelectKBest,
    f_classif
)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class MutualInfoSelector:
    """
    A class for performing feature selection using Mutual Information.
    """
    
    def __init__(self, X, y, feature_names=None, random_state=42):
        """
        Initialize the Mutual Information Selector.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature data
        y : array-like, shape (n_samples,)
            Target variable
        feature_names : list, optional
            Names of features
        random_state : int, default=42
            Random state for reproducibility
        """
        self.X = X
        self.y = y
        self.random_state = random_state
        
        if feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names
        
        self.mi_scores = None
        self.selected_features = None
        self.selected_indices = None
        self.selector = None
        
    def calculate_mutual_information(self, random_state=None):
        """
        Calculate mutual information scores for all features.
        
        Parameters:
        -----------
        random_state : int, optional
            Random state for MI calculation
            
        Returns:
        --------
        mi_scores : array
            Mutual information scores for each feature
        """
        if random_state is None:
            random_state = self.random_state
            
        print("Calculating Mutual Information scores...")
        self.mi_scores = mutual_info_classif(
            self.X, 
            self.y, 
            random_state=random_state
        )
        
        # Create DataFrame for easier analysis
        mi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'MI_Score': self.mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        print(f"Calculated MI scores for {len(self.feature_names)} features")
        
        return self.mi_scores, mi_df
    
    def select_features(self, k=None, threshold=None, top_percent=None):
        """
        Select top k features based on mutual information.
        
        Parameters:
        -----------
        k : int, optional
            Number of top features to select
        threshold : float, optional
            Minimum MI score threshold
        top_percent : float, optional
            Top percentage of features to select (0-1)
            
        Returns:
        --------
        selected_features : list
            Names of selected features
        selected_indices : array
            Indices of selected features
        """
        if self.mi_scores is None:
            self.calculate_mutual_information()
        
        # Determine k based on parameters
        if k is None:
            if threshold is not None:
                k = np.sum(self.mi_scores >= threshold)
            elif top_percent is not None:
                k = max(1, int(len(self.feature_names) * top_percent))
            else:
                k = len(self.feature_names)  # Select all
        
        # Ensure k is valid
        k = min(k, len(self.feature_names))
        k = max(1, k)
        
        # Get top k features
        top_indices = np.argsort(self.mi_scores)[-k:][::-1]
        self.selected_indices = top_indices
        self.selected_features = [self.feature_names[i] for i in top_indices]
        
        # Create selector
        self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
        self.selector.fit(self.X, self.y)
        
        print(f"\nSelected {k} features out of {len(self.feature_names)}")
        print(f"Selected features: {', '.join(self.selected_features)}")
        
        return self.selected_features, self.selected_indices
    
    def get_selected_data(self):
        """
        Get data with only selected features.
        
        Returns:
        --------
        X_selected : array
            Data with selected features only
        """
        if self.selected_indices is None:
            print("No features selected yet. Call select_features() first.")
            return None
        
        return self.X[:, self.selected_indices]
    
    def visualize_feature_importance(self, top_n=None, figsize=(12, 8)):
        """
        Visualize mutual information scores for features.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to display
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if self.mi_scores is None:
            self.calculate_mutual_information()
        
        # Create DataFrame
        mi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'MI_Score': self.mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        if top_n is None:
            top_n = len(mi_df)
        else:
            top_n = min(top_n, len(mi_df))
        
        mi_df_top = mi_df.head(top_n)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Bar chart
        ax1 = axes[0]
        ax1.barh(range(len(mi_df_top)), mi_df_top['MI_Score'].values, alpha=0.8)
        ax1.set_yticks(range(len(mi_df_top)))
        ax1.set_yticklabels(mi_df_top['Feature'].values)
        ax1.set_xlabel('Mutual Information Score', fontsize=12)
        ax1.set_title(f'Top {top_n} Features by Mutual Information', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Sorted scores
        ax2 = axes[1]
        ax2.plot(range(len(mi_df)), mi_df['MI_Score'].values, 'o-', linewidth=2, markersize=4)
        ax2.set_xlabel('Feature Rank', fontsize=12)
        ax2.set_ylabel('Mutual Information Score', fontsize=12)
        ax2.set_title('All Features: MI Scores (Sorted)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Highlight selected features if available
        if self.selected_indices is not None:
            selected_ranks = [np.where(mi_df.index == idx)[0][0] for idx in self.selected_indices 
                            if idx in mi_df.index]
            if selected_ranks:
                ax2.scatter(selected_ranks, mi_df.iloc[selected_ranks]['MI_Score'].values, 
                           c='red', s=100, marker='X', label='Selected', zorder=5)
                ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        return mi_df
    
    def compare_with_without_selection(self, classifier=None, cv=5, test_size=0.2):
        """
        Compare model performance with and without feature selection.
        
        Parameters:
        -----------
        classifier : estimator, optional
            Classifier to use. If None, uses RandomForestClassifier
        cv : int, default=5
            Number of cross-validation folds
        test_size : float, default=0.2
            Test set size for train-test split
            
        Returns:
        --------
        comparison_df : DataFrame
            Comparison of performance metrics
        """
        from sklearn.model_selection import train_test_split
        
        if self.selected_indices is None:
            print("No features selected yet. Call select_features() first.")
            return None
        
        if classifier is None:
            classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y
        )
        
        # Get selected features data
        X_train_selected = X_train[:, self.selected_indices]
        X_test_selected = X_test[:, self.selected_indices]
        
        print(f"\nComparing models with {len(self.feature_names)} vs {len(self.selected_features)} features...")
        print("="*60)
        
        results = {}
        
        # Model with all features
        print("\n1. Training with ALL features...")
        classifier_all = type(classifier)(**classifier.get_params())
        classifier_all.set_params(random_state=self.random_state)
        classifier_all.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores_all = cross_val_score(classifier_all, X_train, y_train, cv=cv, scoring='accuracy')
        
        # Test predictions
        y_pred_all = classifier_all.predict(X_test)
        y_pred_proba_all = classifier_all.predict_proba(X_test)[:, 1] if hasattr(classifier_all, 'predict_proba') else None
        
        results['All_Features'] = {
            'n_features': len(self.feature_names),
            'cv_mean': cv_scores_all.mean(),
            'cv_std': cv_scores_all.std(),
            'test_accuracy': accuracy_score(y_test, y_pred_all),
            'test_precision': precision_score(y_test, y_pred_all, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_all, zero_division=0),
            'test_f1': f1_score(y_test, y_pred_all, zero_division=0),
            'test_auc': roc_auc_score(y_test, y_pred_proba_all) if y_pred_proba_all is not None else None
        }
        
        print(f"  CV Accuracy: {cv_scores_all.mean():.4f} (+/- {cv_scores_all.std()*2:.4f})")
        print(f"  Test Accuracy: {results['All_Features']['test_accuracy']:.4f}")
        
        # Model with selected features
        print(f"\n2. Training with SELECTED {len(self.selected_features)} features...")
        classifier_selected = type(classifier)(**classifier.get_params())
        classifier_selected.set_params(random_state=self.random_state)
        classifier_selected.fit(X_train_selected, y_train)
        
        # Cross-validation
        cv_scores_selected = cross_val_score(classifier_selected, X_train_selected, y_train, cv=cv, scoring='accuracy')
        
        # Test predictions
        y_pred_selected = classifier_selected.predict(X_test_selected)
        y_pred_proba_selected = classifier_selected.predict_proba(X_test_selected)[:, 1] if hasattr(classifier_selected, 'predict_proba') else None
        
        results['Selected_Features'] = {
            'n_features': len(self.selected_features),
            'cv_mean': cv_scores_selected.mean(),
            'cv_std': cv_scores_selected.std(),
            'test_accuracy': accuracy_score(y_test, y_pred_selected),
            'test_precision': precision_score(y_test, y_pred_selected, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_selected, zero_division=0),
            'test_f1': f1_score(y_test, y_pred_selected, zero_division=0),
            'test_auc': roc_auc_score(y_test, y_pred_proba_selected) if y_pred_proba_selected is not None else None
        }
        
        print(f"  CV Accuracy: {cv_scores_selected.mean():.4f} (+/- {cv_scores_selected.std()*2:.4f})")
        print(f"  Test Accuracy: {results['Selected_Features']['test_accuracy']:.4f}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        comparison_df['improvement'] = comparison_df['test_accuracy'] - comparison_df.loc['All_Features', 'test_accuracy']
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(comparison_df[['n_features', 'cv_mean', 'test_accuracy', 'test_f1', 'test_auc']].round(4))
        
        # Calculate improvement
        accuracy_improvement = results['Selected_Features']['test_accuracy'] - results['All_Features']['test_accuracy']
        feature_reduction = (1 - len(self.selected_features) / len(self.feature_names)) * 100
        
        print(f"\nFeature Reduction: {feature_reduction:.1f}%")
        print(f"Accuracy Change: {accuracy_improvement:+.4f}")
        
        if accuracy_improvement > 0:
            print("✓ Feature selection improved model performance!")
        elif abs(accuracy_improvement) < 0.01:
            print("≈ Feature selection maintained similar performance with fewer features.")
        else:
            print("⚠ Feature selection slightly decreased performance, but may improve interpretability.")
        
        return comparison_df, results
    
    def plot_confusion_matrices(self, classifier=None, test_size=0.2, figsize=(12, 5)):
        """
        Plot confusion matrices for models with and without feature selection.
        
        Parameters:
        -----------
        classifier : estimator, optional
            Classifier to use
        test_size : float, default=0.2
            Test set size
        figsize : tuple, default=(12, 5)
            Figure size
        """
        from sklearn.model_selection import train_test_split
        
        if self.selected_indices is None:
            print("No features selected yet. Call select_features() first.")
            return
        
        if classifier is None:
            classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y
        )
        
        X_train_selected = X_train[:, self.selected_indices]
        X_test_selected = X_test[:, self.selected_indices]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Model with all features
        classifier_all = type(classifier)(**classifier.get_params())
        classifier_all.set_params(random_state=self.random_state)
        classifier_all.fit(X_train, y_train)
        y_pred_all = classifier_all.predict(X_test)
        cm_all = confusion_matrix(y_test, y_pred_all)
        
        sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_title(f'All Features ({len(self.feature_names)} features)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Predicted', fontsize=11)
        axes[0].set_ylabel('Actual', fontsize=11)
        
        # Model with selected features
        classifier_selected = type(classifier)(**classifier.get_params())
        classifier_selected.set_params(random_state=self.random_state)
        classifier_selected.fit(X_train_selected, y_train)
        y_pred_selected = classifier_selected.predict(X_test_selected)
        cm_selected = confusion_matrix(y_test, y_pred_selected)
        
        sns.heatmap(cm_selected, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
        axes[1].set_title(f'Selected Features ({len(self.selected_features)} features)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted', fontsize=11)
        axes[1].set_ylabel('Actual', fontsize=11)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance_summary(self):
        """
        Get summary of feature importance.
        
        Returns:
        --------
        summary_df : DataFrame
            Summary of feature importance
        """
        if self.mi_scores is None:
            self.calculate_mutual_information()
        
        summary_df = pd.DataFrame({
            'Feature': self.feature_names,
            'MI_Score': self.mi_scores,
            'Selected': [i in self.selected_indices if self.selected_indices is not None else False 
                         for i in range(len(self.feature_names))]
        }).sort_values('MI_Score', ascending=False)
        
        summary_df['Rank'] = range(1, len(summary_df) + 1)
        
        return summary_df



