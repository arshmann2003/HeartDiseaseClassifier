"""
Hyperparameter Tuning Module for Random Forest Classifier
Provides functions for tuning Random Forest hyperparameters using Grid Search or Random Search.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    make_scorer
)
import warnings
warnings.filterwarnings('ignore')


class RandomForestTuner:
    """
    A class for performing hyperparameter tuning on Random Forest classifier.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, feature_names=None, random_state=42):
        """
        Initialize the Random Forest Tuner.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        feature_names : list, optional
            Names of features
        random_state : int, default=42
            Random state for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        
        if feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        else:
            self.feature_names = feature_names
        
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.search_results = None
        self.before_tuning_metrics = {}
        self.after_tuning_metrics = {}
    
    def evaluate_model(self, model, X, y, threshold=0.5):
        """
        Evaluate a model and return comprehensive metrics.
        
        Parameters:
        -----------
        model : RandomForestClassifier
            Trained model
        X : array-like
            Features
        y : array-like
            True labels
        threshold : float, default=0.5
            Classification threshold
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        return metrics
    
    def train_baseline_model(self, **kwargs):
        """
        Train a baseline model with default/default-specified parameters.
        
        Parameters:
        -----------
        **kwargs : dict
            Parameters to pass to RandomForestClassifier
            
        Returns:
        --------
        model : RandomForestClassifier
            Trained baseline model
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        model = RandomForestClassifier(**default_params)
        model.fit(self.X_train, self.y_train)
        
        return model
    
    def grid_search(self, param_grid, cv=5, scoring='balanced_accuracy', verbose=1, n_jobs=-1):
        """
        Perform Grid Search for hyperparameter tuning.
        
        Parameters:
        -----------
        param_grid : dict
            Dictionary with parameters names as keys and lists of parameter settings
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='balanced_accuracy'
            Scoring metric
        verbose : int, default=1
            Verbosity level
        n_jobs : int, default=-1
            Number of jobs to run in parallel
            
        Returns:
        --------
        best_model : RandomForestClassifier
            Best model found
        """
        print("="*70)
        print("GRID SEARCH HYPERPARAMETER TUNING")
        print("="*70)
        print(f"Parameter grid: {param_grid}")
        print(f"Cross-validation folds: {cv}")
        print(f"Scoring metric: {scoring}")
        print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
        print("\nStarting grid search...")
        
        # Create base model
        base_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=n_jobs
        )
        
        # Create cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
            return_train_score=True
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Store results
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.search_results = pd.DataFrame(grid_search.cv_results_)
        
        print(f"\n✓ Grid search completed!")
        print(f"  Best parameters: {self.best_params}")
        print(f"  Best CV score ({scoring}): {self.best_score:.4f}")
        
        return self.best_model
    
    def random_search(self, param_distributions, n_iter=50, cv=5, 
                     scoring='balanced_accuracy', verbose=1, n_jobs=-1, random_state=None):
        """
        Perform Random Search for hyperparameter tuning.
        
        Parameters:
        -----------
        param_distributions : dict
            Dictionary with parameters names as keys and distributions/lists
        n_iter : int, default=50
            Number of parameter settings sampled
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='balanced_accuracy'
            Scoring metric
        verbose : int, default=1
            Verbosity level
        n_jobs : int, default=-1
            Number of jobs to run in parallel
        random_state : int, optional
            Random state (uses self.random_state if None)
            
        Returns:
        --------
        best_model : RandomForestClassifier
            Best model found
        """
        if random_state is None:
            random_state = self.random_state
        
        print("="*70)
        print("RANDOM SEARCH HYPERPARAMETER TUNING")
        print("="*70)
        print(f"Parameter distributions: {param_distributions}")
        print(f"Number of iterations: {n_iter}")
        print(f"Cross-validation folds: {cv}")
        print(f"Scoring metric: {scoring}")
        print("\nStarting random search...")
        
        # Create base model
        base_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=n_jobs
        )
        
        # Create cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=random_state,
            return_train_score=True
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        # Store results
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.search_results = pd.DataFrame(random_search.cv_results_)
        
        print(f"\n✓ Random search completed!")
        print(f"  Best parameters: {self.best_params}")
        print(f"  Best CV score ({scoring}): {self.best_score:.4f}")
        
        return self.best_model
    
    def compare_before_after(self, baseline_model, threshold=0.5):
        """
        Compare model performance before and after hyperparameter tuning.
        
        Parameters:
        -----------
        baseline_model : RandomForestClassifier
            Baseline model (before tuning)
        threshold : float, default=0.5
            Classification threshold
            
        Returns:
        --------
        comparison_df : DataFrame
            Comparison of metrics before and after tuning
        """
        if self.best_model is None:
            print("No tuned model found. Run grid_search() or random_search() first.")
            return None
        
        print("\n" + "="*70)
        print("BEFORE vs AFTER TUNING COMPARISON")
        print("="*70)
        
        # Evaluate baseline model
        print("\nEvaluating baseline model (before tuning)...")
        self.before_tuning_metrics = self.evaluate_model(
            baseline_model, self.X_test, self.y_test, threshold=threshold
        )
        
        # Evaluate tuned model
        print("Evaluating tuned model (after tuning)...")
        self.after_tuning_metrics = self.evaluate_model(
            self.best_model, self.X_test, self.y_test, threshold=threshold
        )
        
        # Create comparison DataFrame
        comparison_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
                      'F1-Score (Weighted)', 'Balanced Accuracy', 'AUC-ROC'],
            'Before Tuning': [
                self.before_tuning_metrics['accuracy'],
                self.before_tuning_metrics['precision'],
                self.before_tuning_metrics['recall'],
                self.before_tuning_metrics['f1_score'],
                self.before_tuning_metrics['f1_weighted'],
                self.before_tuning_metrics['balanced_accuracy'],
                self.before_tuning_metrics['roc_auc']
            ],
            'After Tuning': [
                self.after_tuning_metrics['accuracy'],
                self.after_tuning_metrics['precision'],
                self.after_tuning_metrics['recall'],
                self.after_tuning_metrics['f1_score'],
                self.after_tuning_metrics['f1_weighted'],
                self.after_tuning_metrics['balanced_accuracy'],
                self.after_tuning_metrics['roc_auc']
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['Improvement'] = comparison_df['After Tuning'] - comparison_df['Before Tuning']
        comparison_df['Improvement %'] = (comparison_df['Improvement'] / comparison_df['Before Tuning'] * 100).round(2)
        
        # Display results
        print("\n" + "-"*70)
        print(f"{'Metric':<25} {'Before':<15} {'After':<15} {'Improvement':<15}")
        print("-"*70)
        for _, row in comparison_df.iterrows():
            print(f"{row['Metric']:<25} {row['Before Tuning']:<15.4f} {row['After Tuning']:<15.4f} "
                  f"{row['Improvement']:+.4f} ({row['Improvement %']:+.2f}%)")
        print("-"*70)
        
        return comparison_df
    
    def plot_comparison(self, comparison_df, figsize=(12, 6)):
        """
        Plot comparison of metrics before and after tuning.
        
        Parameters:
        -----------
        comparison_df : DataFrame
            Comparison DataFrame from compare_before_after()
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if comparison_df is None:
            print("No comparison data available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Bar chart comparison
        ax1 = axes[0]
        x = np.arange(len(comparison_df))
        width = 0.35
        
        ax1.bar(x - width/2, comparison_df['Before Tuning'], width, 
               label='Before Tuning', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, comparison_df['After Tuning'], width, 
               label='After Tuning', alpha=0.8, color='coral')
        
        ax1.set_xlabel('Metrics', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Model Performance: Before vs After Tuning', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comparison_df['Metric'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Improvement percentages
        ax2 = axes[1]
        colors = ['green' if x > 0 else 'red' for x in comparison_df['Improvement %']]
        ax2.barh(comparison_df['Metric'], comparison_df['Improvement %'], color=colors, alpha=0.7)
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Improvement (%)', fontsize=12)
        ax2.set_title('Percentage Improvement After Tuning', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, top_n=15):
        """
        Get feature importance from the tuned model.
        
        Parameters:
        -----------
        top_n : int, default=15
            Number of top features to return
            
        Returns:
        --------
        importance_df : DataFrame
            DataFrame with feature importance
        """
        if self.best_model is None:
            print("No tuned model found. Run grid_search() or random_search() first.")
            return None
        
        importance = self.best_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def print_summary(self):
        """
        Print summary of hyperparameter tuning results.
        """
        if self.best_model is None:
            print("No tuning results available.")
            return
        
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("="*70)
        print(f"\nBest Parameters Found:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV Score: {self.best_score:.4f}")
        
        if self.after_tuning_metrics:
            print(f"\nTest Set Performance (After Tuning):")
            print(f"  Accuracy: {self.after_tuning_metrics['accuracy']:.4f}")
            print(f"  Precision: {self.after_tuning_metrics['precision']:.4f}")
            print(f"  Recall: {self.after_tuning_metrics['recall']:.4f}")
            print(f"  F1-Score: {self.after_tuning_metrics['f1_score']:.4f}")
            print(f"  Balanced Accuracy: {self.after_tuning_metrics['balanced_accuracy']:.4f}")
            print(f"  AUC-ROC: {self.after_tuning_metrics['roc_auc']:.4f}")


