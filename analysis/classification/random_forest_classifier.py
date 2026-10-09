"""
Random Forest Classification Module
Provides functions for training and evaluating Random Forest classifier for heart disease prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')


class RandomForestClassifierAnalyzer:
    """
    A class for performing Random Forest classification analysis.
    """
    
    def __init__(self, X, y, feature_names=None, random_state=42):
        """
        Initialize the Random Forest Classifier Analyzer.
        
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
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.cv_scores = None
        self.metrics = {}
        self.optimal_threshold = 0.5
    
    def validate_data(self):
        """Validate input data for common issues."""
        issues = []
        
        # Check for NaN values (only works on float arrays)
        if self.X.dtype in [np.float32, np.float64]:
            if np.isnan(self.X).any():
                issues.append("X contains NaN values")
        if self.y.dtype in [np.float32, np.float64]:
            if np.isnan(self.y).any():
                issues.append("y contains NaN values")
        
        # Check for infinite values (only works on float arrays)
        if self.X.dtype in [np.float32, np.float64]:
            if np.isinf(self.X).any():
                issues.append("X contains infinite values")
        if self.y.dtype in [np.float32, np.float64]:
            if np.isinf(self.y).any():
                issues.append("y contains infinite values")
        
        # Check data types
        if self.X.dtype not in [np.float32, np.float64, np.int32, np.int64]:
            issues.append(f"X has unexpected dtype: {self.X.dtype}")
        
        # Check target values
        unique_y = np.unique(self.y)
        if len(unique_y) != 2:
            issues.append(f"Expected binary classification, found {len(unique_y)} classes: {unique_y}")
        if not all(y_val in [0, 1] for y_val in unique_y):
            issues.append(f"Target values should be 0 and 1, found: {unique_y}")
        
        # Check class distribution
        unique, counts = np.unique(self.y, return_counts=True)
        class_ratio = counts.min() / counts.max()
        if class_ratio < 0.1:
            issues.append(f"Severe class imbalance: ratio = {class_ratio:.3f}")
        
        if issues:
            print("⚠ Data Validation Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✓ Data validation passed")
            return True
        
    def split_data(self, test_size=0.2, stratify=True):
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        test_size : float, default=0.2
            Proportion of dataset to include in test split
        stratify : bool, default=True
            Whether to stratify split based on target
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Train-test split
        """
        # Validate data before splitting
        self.validate_data()
        
        stratify_param = self.y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"\nData Split:")
        print(f"  Training set: {self.X_train.shape[0]} samples ({self.X_train.shape[1]} features)")
        print(f"  Test set: {self.X_test.shape[0]} samples ({self.X_test.shape[1]} features)")
        print(f"\nTraining set class distribution:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(self.y_train)*100:.2f}%)")
        
        print(f"\nTest set class distribution:")
        unique_test, counts_test = np.unique(self.y_test, return_counts=True)
        for label, count in zip(unique_test, counts_test):
            print(f"  Class {label}: {count} ({count/len(self.y_test)*100:.2f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self, n_estimators=100, max_depth=None, min_samples_split=2, 
              min_samples_leaf=1, max_features='sqrt', class_weight='balanced', 
              verbose=True):
        """
        Train Random Forest classifier.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of trees
        min_samples_split : int, default=2
            Minimum samples required to split a node
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node
        max_features : str or int, default='sqrt'
            Number of features to consider for best split
        class_weight : str or dict, default='balanced'
            Weights for classes to handle imbalance. 'balanced' automatically adjusts.
        verbose : bool, default=True
            Whether to print progress
            
        Returns:
        --------
        model : RandomForestClassifier
            Trained model
        """
        if self.X_train is None:
            print("Data not split yet. Calling split_data() first...")
            self.split_data()
        
        # Check for class imbalance
        unique, counts = np.unique(self.y_train, return_counts=True)
        class_ratio = counts.min() / counts.max()
        
        if verbose:
            print(f"\nTraining Random Forest Classifier...")
            print(f"  n_estimators: {n_estimators}")
            print(f"  max_depth: {max_depth}")
            print(f"  min_samples_split: {min_samples_split}")
            print(f"  min_samples_leaf: {min_samples_leaf}")
            print(f"  max_features: {max_features}")
            print(f"  class_weight: {class_weight}")
            print(f"  Class ratio (minority/majority): {class_ratio:.3f}")
            if class_ratio < 0.3:
                print(f"  ⚠ Warning: Severe class imbalance detected.")
        
        # Auto-adjust class_weight if severe imbalance
        if class_weight == 'auto' and class_ratio < 0.3:
            class_weight = 'balanced'
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        if verbose:
            print("Training completed!")
            # Show training accuracy
            train_pred = self.model.predict(self.X_train)
            train_acc = accuracy_score(self.y_train, train_pred)
            print(f"Training accuracy: {train_acc:.4f}")
        
        return self.model
    
    def predict(self, threshold=None):
        """
        Make predictions on test set.
        
        Parameters:
        -----------
        threshold : float, optional
            Classification threshold. If None, uses optimal_threshold or default 0.5
            
        Returns:
        --------
        y_pred : array
            Predicted labels
        y_pred_proba : array
            Predicted probabilities
        """
        if self.model is None:
            print("Model not trained yet. Call train() first.")
            return None, None
        
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Use specified threshold, optimal threshold, or default 0.5
        if threshold is None:
            threshold = getattr(self, 'optimal_threshold', 0.5)
            if threshold is None:
                threshold = 0.5
        
        self.y_pred = (self.y_pred_proba >= threshold).astype(int)
        
        return self.y_pred, self.y_pred_proba
    
    def find_optimal_threshold(self, metric='f1', thresholds=None):
        """
        Find optimal classification threshold based on specified metric.
        
        Parameters:
        -----------
        metric : str, default='f1'
            Metric to optimize: 'f1', 'f1_weighted', 'recall', 'precision', 'balanced_accuracy'
        thresholds : array-like, optional
            Thresholds to test. If None, tests 100 thresholds from 0.1 to 0.9
            
        Returns:
        --------
        optimal_threshold : float
            Optimal threshold value
        best_score : float
            Best score achieved
        results_df : DataFrame
            Results for all tested thresholds
        """
        if self.model is None:
            print("Model not trained yet. Call train() first.")
            return None, None, None
        
        if self.y_pred_proba is None:
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 100)
        
        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            acc = accuracy_score(self.y_test, y_pred_thresh)
            prec = precision_score(self.y_test, y_pred_thresh, zero_division=0)
            rec = recall_score(self.y_test, y_pred_thresh, zero_division=0)
            f1 = f1_score(self.y_test, y_pred_thresh, zero_division=0)
            
            # Balanced accuracy
            from sklearn.metrics import balanced_accuracy_score
            bal_acc = balanced_accuracy_score(self.y_test, y_pred_thresh)
            
            # Weighted F1
            f1_weighted = f1_score(self.y_test, y_pred_thresh, average='weighted', zero_division=0)
            
            # Calculate score based on metric
            if metric == 'f1':
                score = f1
            elif metric == 'f1_weighted':
                score = f1_weighted
            elif metric == 'recall':
                score = rec
            elif metric == 'precision':
                score = prec
            elif metric == 'balanced_accuracy':
                score = bal_acc
            else:
                score = f1
            
            results.append({
                'threshold': threshold,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'f1_weighted': f1_weighted,
                'balanced_accuracy': bal_acc,
                'score': score
            })
        
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold
        best_idx = results_df['score'].idxmax()
        self.optimal_threshold = results_df.loc[best_idx, 'threshold']
        best_score = results_df.loc[best_idx, 'score']
        
        print(f"\nOptimal Threshold Search (optimizing for {metric}):")
        print(f"  Optimal threshold: {self.optimal_threshold:.4f}")
        print(f"  Best {metric} score: {best_score:.4f}")
        print(f"  At optimal threshold:")
        print(f"    Accuracy: {results_df.loc[best_idx, 'accuracy']:.4f}")
        print(f"    Precision: {results_df.loc[best_idx, 'precision']:.4f}")
        print(f"    Recall: {results_df.loc[best_idx, 'recall']:.4f}")
        print(f"    F1-Score: {results_df.loc[best_idx, 'f1_score']:.4f}")
        
        return self.optimal_threshold, best_score, results_df
    
    def plot_threshold_analysis(self, metric='f1', figsize=(14, 5), results_df=None):
        """
        Plot threshold analysis showing how metrics change with threshold.
        
        Parameters:
        -----------
        metric : str, default='f1'
            Metric to optimize
        figsize : tuple, default=(14, 5)
            Figure size
        results_df : DataFrame, optional
            Pre-computed threshold results. If None, will recalculate.
        """
        if self.model is None:
            print("Model not trained yet. Call train() first.")
            return
        
        # Store the current optimal threshold before potentially recalculating
        current_optimal = getattr(self, 'optimal_threshold', 0.5)
        
        if results_df is None:
            _, _, results_df = self.find_optimal_threshold(metric=metric)
            # Restore the optimal threshold if it was already set
            if hasattr(self, 'optimal_threshold') and current_optimal != 0.5:
                # Only restore if the new one is significantly different and we want to keep the original
                pass  # Actually, let's use the newly calculated one
        
        if results_df is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: All metrics vs threshold
        ax1 = axes[0]
        ax1.plot(results_df['threshold'], results_df['accuracy'], label='Accuracy', linewidth=2)
        ax1.plot(results_df['threshold'], results_df['precision'], label='Precision', linewidth=2)
        ax1.plot(results_df['threshold'], results_df['recall'], label='Recall', linewidth=2)
        ax1.plot(results_df['threshold'], results_df['f1_score'], label='F1-Score', linewidth=2)
        ax1.axvline(self.optimal_threshold, color='red', linestyle='--', 
                    label=f'Optimal ({self.optimal_threshold:.3f})', linewidth=2)
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Metrics vs Classification Threshold', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Optimized metric vs threshold
        ax2 = axes[1]
        ax2.plot(results_df['threshold'], results_df['score'], 'g-', linewidth=2, label=f'{metric.upper()}')
        ax2.axvline(self.optimal_threshold, color='red', linestyle='--', 
                    label=f'Optimal ({self.optimal_threshold:.3f})', linewidth=2)
        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel(f'{metric.upper()} Score', fontsize=12)
        ax2.set_title(f'{metric.upper()} Optimization', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, cv=5, threshold=None, use_optimal_threshold=True):
        """
        Evaluate model performance using multiple metrics.
        
        Parameters:
        -----------
        cv : int, default=5
            Number of cross-validation folds
        threshold : float, optional
            Classification threshold. If None and use_optimal_threshold=True, uses optimal threshold
        use_optimal_threshold : bool, default=True
            Whether to use optimal threshold if it has been calculated
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        if self.model is None:
            print("Model not trained yet. Call train() first.")
            return None
        
        # Use optimal threshold if available and requested
        if use_optimal_threshold and threshold is None and hasattr(self, 'optimal_threshold'):
            threshold = self.optimal_threshold
            if threshold is not None:
                print(f"Using optimal threshold: {threshold:.4f}")
        
        if self.y_pred is None or threshold is not None:
            self.predict(threshold=threshold)
        
        # Validate predictions
        if len(self.y_pred) != len(self.y_test):
            print(f"Error: Prediction length ({len(self.y_pred)}) doesn't match test length ({len(self.y_test)})")
            return None
        
        # Check for valid predictions
        unique_pred = np.unique(self.y_pred)
        unique_test = np.unique(self.y_test)
        
        # Calculate metrics
        try:
            from sklearn.metrics import balanced_accuracy_score
            self.metrics = {
                'accuracy': accuracy_score(self.y_test, self.y_pred),
                'precision': precision_score(self.y_test, self.y_pred, zero_division=0),
                'recall': recall_score(self.y_test, self.y_pred, zero_division=0),
                'f1_score': f1_score(self.y_test, self.y_pred, zero_division=0),
                'f1_weighted': f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0),
                'balanced_accuracy': balanced_accuracy_score(self.y_test, self.y_pred),
                'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba),
                'threshold_used': threshold if threshold is not None else self.optimal_threshold
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
        
        # Cross-validation
        try:
            cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                self.model, 
                self.X_train, 
                self.y_train, 
                cv=cv_fold, 
                scoring='accuracy'
            )
            self.cv_scores = cv_scores
            
            self.metrics['cv_mean'] = cv_scores.mean()
            self.metrics['cv_std'] = cv_scores.std()
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            self.cv_scores = None
            self.metrics['cv_mean'] = None
            self.metrics['cv_std'] = None
        
        return self.metrics
    
    def print_evaluation(self):
        """Print evaluation metrics."""
        if not self.metrics:
            self.evaluate()
        
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        if 'threshold_used' in self.metrics:
            print(f"Classification Threshold: {self.metrics['threshold_used']:.4f}")
        print(f"Accuracy:           {self.metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:  {self.metrics.get('balanced_accuracy', 'N/A'):.4f}")
        print(f"Precision:          {self.metrics['precision']:.4f}")
        print(f"Recall:             {self.metrics['recall']:.4f}")
        print(f"F1-Score:           {self.metrics['f1_score']:.4f}")
        print(f"F1-Score (Weighted): {self.metrics.get('f1_weighted', 'N/A'):.4f}")
        print(f"AUC-ROC:            {self.metrics['roc_auc']:.4f}")
        if self.metrics.get('cv_mean') is not None:
            print(f"\nCross-Validation (5-fold):")
            print(f"  Mean Accuracy: {self.metrics['cv_mean']:.4f} (+/- {self.metrics['cv_std']*2:.4f})")
        print("="*60)
    
    def plot_confusion_matrix(self, figsize=(8, 6)):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        figsize : tuple, default=(8, 6)
            Figure size
        """
        if self.y_pred is None:
            self.predict()
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            cbar=True,
            xticklabels=['No Heart Disease', 'Heart Disease'],
            yticklabels=['No Heart Disease', 'Heart Disease']
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred, 
                                    target_names=['No Heart Disease', 'Heart Disease']))
    
    def plot_roc_curve(self, figsize=(8, 6)):
        """
        Plot ROC curve.
        
        Parameters:
        -----------
        figsize : tuple, default=(8, 6)
            Figure size
        """
        if self.y_pred_proba is None:
            self.predict()
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        auc_score = self.metrics.get('roc_auc', roc_auc_score(self.y_test, self.y_pred_proba))
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, top_n=None):
        """
        Get feature importance from the trained model.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to return
            
        Returns:
        --------
        importance_df : DataFrame
            DataFrame with feature importance
        """
        if self.model is None:
            print("Model not trained yet. Call train() first.")
            return None
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def plot_feature_importance(self, top_n=15, figsize=(10, 8)):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int, default=15
            Number of top features to display
        figsize : tuple, default=(10, 8)
            Figure size
        """
        importance_df = self.get_feature_importance(top_n=top_n)
        
        if importance_df is None:
            return
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(importance_df)), importance_df['Importance'].values, alpha=0.8)
        plt.yticks(range(len(importance_df)), importance_df['Feature'].values)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {len(importance_df)} Features by Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def plot_evaluation_summary(self, figsize=(14, 5)):
        """
        Plot summary of evaluation metrics.
        
        Parameters:
        -----------
        figsize : tuple, default=(14, 5)
            Figure size
        """
        if not self.metrics:
            self.evaluate()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Metrics bar chart
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        metric_values = [self.metrics[m] for m in metrics_to_plot]
        
        ax1 = axes[0]
        bars = ax1.bar(metric_names, metric_values, alpha=0.8, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: CV scores distribution
        if self.cv_scores is not None:
            ax2 = axes[1]
            ax2.boxplot([self.cv_scores], labels=['CV Scores'], patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.set_title(f'Cross-Validation Scores (Mean: {self.metrics["cv_mean"]:.4f})', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()

