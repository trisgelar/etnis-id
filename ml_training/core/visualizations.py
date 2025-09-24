#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive visualization module for ethnicity detection model performance analysis
Based on original notebook analysis and SOLID principles
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from kneed import KneeLocator

# Import SciencePlots for publication-quality figures
try:
    import scienceplots
    plt.style.use(['science', 'ieee', 'grid'])  # IEEE style for academic papers
    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False
    print("Warning: SciencePlots not available. Using default matplotlib styles.")

from .interfaces import ILogger, IProgressTracker
from .utils import TrainingLogger
from .config import get_viz_config


class ModelVisualizer:
    """Comprehensive model performance visualization and analysis"""
    
    def __init__(self, logger: Optional[ILogger] = None, output_dir: str = None,
                 style: str = None):
        """
        Initialize model visualizer
        
        Args:
            logger: Logger instance for logging
            output_dir: Directory to save visualization outputs (uses config if None)
            style: SciencePlots style (uses config if None)
        """
        # Get visualization configuration
        viz_config = get_viz_config()
        
        # Use configuration values if not provided
        self.output_dir = output_dir or viz_config.output_dir
        self.style = style or viz_config.style
        self.logger = logger or TrainingLogger('model_visualizer')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set matplotlib style for publication-quality plots
        if SCIENCEPLOTS_AVAILABLE:
            try:
                plt.style.use(['science', style, 'grid'])
                self.logger.info(f"Using SciencePlots style: {style}")
            except Exception as e:
                self.logger.warning(f"Could not set SciencePlots style: {e}")
                plt.style.use('default')
        else:
            plt.style.use('default')
            self.logger.warning("SciencePlots not available, using default style")
        
        # Set color palette for beautiful plots
        sns.set_palette("husl")
        
        # Set figure parameters for publication quality
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14
        })
        
        self.logger.info("ModelVisualizer initialized with publication-quality settings")
    
    def plot_confusion_matrix_detailed(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     fold_info: Optional[Dict] = None, 
                                     save_path: Optional[str] = None,
                                     normalize: bool = False) -> plt.Figure:
        """
        Create detailed confusion matrix visualization with publication-quality styling
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            fold_info: Information about which fold this is
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            matplotlib figure
        """
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            classes = np.unique(np.concatenate([y_true, y_pred]))
            
            # Normalize if requested
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.3f'
                cmap = 'Blues'
            else:
                fmt = 'd'
                cmap = 'Blues'
            
            # Create figure with better proportions
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot confusion matrix with beautiful styling
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            if normalize:
                cbar.set_label('Normalized Count', rotation=270, labelpad=20)
            else:
                cbar.set_label('Count', rotation=270, labelpad=20)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=12, fontweight='bold')
            
            # Customize plot
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha='right')
            ax.set_yticklabels(classes)
            
            title = "Confusion Matrix"
            if fold_info:
                title += f" - Fold {fold_info.get('fold', 'Unknown')}"
            if normalize:
                title += " (Normalized)"
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
            
            # Add accuracy and other metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            # Create info box
            info_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                   verticalalignment='top')
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                self.logger.info(f"Confusion matrix saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating confusion matrix: {e}")
            raise
    
    def plot_feature_importance(self, model: Any, feature_names: Optional[List[str]] = None,
                              top_n: int = 20, save_path: Optional[str] = None,
                              feature_type_colors: bool = True) -> plt.Figure:
        """
        Plot feature importance from Random Forest model with beautiful styling
        
        Args:
            model: Trained Random Forest model
            feature_names: Names of features (optional)
            top_n: Number of top features to show
            save_path: Path to save the plot
            feature_type_colors: Whether to color-code GLCM vs Color features
            
        Returns:
            matplotlib figure
        """
        try:
            if not hasattr(model, 'feature_importances_'):
                raise ValueError("Model does not have feature_importances_ attribute")
            
            # Get feature importance
            importances = model.feature_importances_
            
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            # Get top N features
            indices = np.argsort(importances)[::-1][:top_n]
            top_importances = importances[indices]
            top_names = [feature_names[i] for i in indices]
            
            # Create plot with better styling
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Color coding for feature types
            if feature_type_colors:
                colors = []
                for name in top_names:
                    if 'GLCM' in name or name.startswith('Feature_') and int(name.split('_')[1]) < 20:
                        colors.append('#1f77b4')  # Blue for GLCM
                    else:
                        colors.append('#ff7f0e')  # Orange for Color
            else:
                colors = plt.cm.viridis(np.linspace(0, 1, len(top_importances)))
            
            # Horizontal bar plot with gradient colors
            bars = ax.barh(range(len(top_importances)), top_importances, color=colors, alpha=0.8)
            
            # Customize plot
            ax.set_yticks(range(len(top_importances)))
            ax.set_yticklabels(top_names, fontsize=10)
            ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, top_importances)):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_axisbelow(True)
            
            # Add legend for feature types
            if feature_type_colors:
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='#1f77b4', label='GLCM Features'),
                                 Patch(facecolor='#ff7f0e', label='Color Features')]
                ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
            
            # Add statistics
            total_importance = np.sum(importances)
            top_importance = np.sum(top_importances)
            percentage = (top_importance / total_importance) * 100
            
            stats_text = f'Top {top_n} features explain {percentage:.1f}% of total importance'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
                   verticalalignment='top')
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                self.logger.info(f"Feature importance plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {e}")
            raise
    
    def plot_cross_validation_results(self, cv_scores: np.ndarray, 
                                    k_values: Optional[List[int]] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cross-validation results with knee detection
        
        Args:
            cv_scores: Cross-validation scores
            k_values: K-fold values (optional)
            save_path: Path to save the plot
            
        Returns:
            matplotlib figure
        """
        try:
            if k_values is None:
                k_values = list(range(2, len(cv_scores) + 2))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot CV scores
            ax.plot(k_values, cv_scores, marker='o', linewidth=2, markersize=8, label='CV Accuracy')
            
            # Find knee point
            try:
                kl = KneeLocator(k_values, cv_scores, S=1.0, curve='concave', direction='increasing')
                if kl.knee:
                    knee_k = kl.knee
                    knee_score = cv_scores[k_values.index(knee_k)]
                    ax.plot(knee_k, knee_score, 'ro', markersize=12, label=f'Optimal K = {knee_k}')
                    ax.axvline(x=knee_k, color='red', linestyle='--', alpha=0.7)
            except Exception as e:
                self.logger.warning(f"Could not find knee point: {e}")
            
            # Customize plot
            ax.set_xlabel('K-Fold Value', fontsize=12)
            ax.set_ylabel('Cross-Validation Accuracy', fontsize=12)
            ax.set_title('Cross-Validation Results with Optimal K Detection', fontsize=16, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            max_score = np.max(cv_scores)
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            stats_text = f'Max: {max_score:.4f}\nMean: {mean_score:.4f}\nStd: {std_score:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
                   verticalalignment='top')
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"CV results plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating CV results plot: {e}")
            raise
    
    def plot_class_distribution(self, labels: np.ndarray, 
                              title: str = "Class Distribution",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot class distribution
        
        Args:
            labels: Class labels
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            matplotlib figure
        """
        try:
            # Count classes
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            bars = ax1.bar(unique_labels, counts, color=sns.color_palette("husl", len(unique_labels)))
            ax1.set_xlabel('Ethnicity Classes', fontsize=12)
            ax1.set_ylabel('Number of Samples', fontsize=12)
            ax1.set_title(f'{title} - Bar Chart', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(count), ha='center', va='bottom', fontsize=10)
            
            # Pie chart
            ax2.pie(counts, labels=unique_labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'{title} - Pie Chart', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Class distribution plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating class distribution plot: {e}")
            raise
    
    def plot_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            
        Returns:
            matplotlib figure
        """
        try:
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            # Get classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))
            
            # Overall metrics
            ax1 = plt.subplot(2, 2, 1)
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [accuracy, precision, recall, f1]
            bars = ax1.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            ax1.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Score', fontsize=12)
            ax1.set_ylim(0, 1.1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            # Per-class metrics
            ax2 = plt.subplot(2, 2, 2)
            classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            precisions = [report[cls]['precision'] for cls in classes]
            recalls = [report[cls]['recall'] for cls in classes]
            f1_scores = [report[cls]['f1-score'] for cls in classes]
            
            x = np.arange(len(classes))
            width = 0.25
            
            ax2.bar(x - width, precisions, width, label='Precision', alpha=0.8)
            ax2.bar(x, recalls, width, label='Recall', alpha=0.8)
            ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
            
            ax2.set_xlabel('Classes', fontsize=12)
            ax2.set_ylabel('Score', fontsize=12)
            ax2.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(classes, rotation=45)
            ax2.legend()
            ax2.set_ylim(0, 1.1)
            
            # Support (number of samples per class)
            ax3 = plt.subplot(2, 2, 3)
            supports = [report[cls]['support'] for cls in classes]
            bars = ax3.bar(classes, supports, color='lightblue')
            ax3.set_xlabel('Classes', fontsize=12)
            ax3.set_ylabel('Number of Samples', fontsize=12)
            ax3.set_title('Support (Number of Samples per Class)', fontsize=14, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, support in zip(bars, supports):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(support), ha='center', va='bottom', fontsize=10)
            
            # Metrics summary table
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('tight')
            ax4.axis('off')
            
            # Create summary table
            summary_data = [
                ['Overall Accuracy', f'{accuracy:.4f}'],
                ['Weighted Precision', f'{precision:.4f}'],
                ['Weighted Recall', f'{recall:.4f}'],
                ['Weighted F1-Score', f'{f1:.4f}'],
                ['Total Samples', str(len(y_true))],
                ['Number of Classes', str(len(classes))]
            ]
            
            table = ax4.table(cellText=summary_data,
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Performance metrics plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating performance metrics plot: {e}")
            raise
    
    def analyze_wrong_predictions(self, X_test: np.ndarray, y_true: np.ndarray, 
                                y_pred: np.ndarray, test_indices: np.ndarray,
                                save_path: Optional[str] = None) -> Dict:
        """
        Analyze wrong predictions in detail
        
        Args:
            X_test: Test features
            y_true: True labels
            y_pred: Predicted labels
            test_indices: Original indices of test samples
            save_path: Path to save analysis
            
        Returns:
            Dictionary with wrong prediction analysis
        """
        try:
            # Find wrong predictions
            wrong_mask = y_true != y_pred
            wrong_indices = np.where(wrong_mask)[0]
            
            if len(wrong_indices) == 0:
                self.logger.info("No wrong predictions found!")
                return {"wrong_count": 0}
            
            # Create analysis dictionary
            analysis = {
                "wrong_count": len(wrong_indices),
                "total_samples": len(y_true),
                "error_rate": len(wrong_indices) / len(y_true),
                "wrong_predictions": []
            }
            
            # Analyze each wrong prediction
            for idx in wrong_indices:
                wrong_pred = {
                    "test_index": idx,
                    "original_index": test_indices[idx] if idx < len(test_indices) else None,
                    "true_label": y_true[idx],
                    "predicted_label": y_pred[idx],
                    "confidence": None  # Will be filled if model supports it
                }
                analysis["wrong_predictions"].append(wrong_pred)
            
            # Create visualization
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            # Show up to 8 wrong predictions
            num_to_show = min(8, len(wrong_indices))
            
            for i in range(num_to_show):
                idx = wrong_indices[i]
                ax = axes[i]
                
                # Create a simple visualization (placeholder for actual images)
                # In real implementation, you would load and display the actual image
                ax.text(0.5, 0.5, f'True: {y_true[idx]}\nPred: {y_pred[idx]}', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                ax.set_title(f'Wrong Prediction {i+1}')
                ax.axis('off')
            
            # Hide unused subplots
            for i in range(num_to_show, 8):
                axes[i].axis('off')
            
            plt.suptitle(f'Wrong Predictions Analysis ({len(wrong_indices)} total)', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Wrong predictions analysis saved to {save_path}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing wrong predictions: {e}")
            raise
    
    def create_comprehensive_report(self, model: Any, X: np.ndarray, y: np.ndarray,
                                  cv_folds: int = 6, save_dir: Optional[str] = None) -> Dict:
        """
        Create comprehensive model performance report
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Labels
            cv_folds: Number of CV folds
            save_dir: Directory to save all visualizations
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                self.logger.info(f"Creating comprehensive report in {save_dir}")
            
            # Initialize results dictionary
            results = {
                "model_info": {
                    "type": type(model).__name__,
                    "parameters": getattr(model, 'get_params', lambda: {})()
                },
                "data_info": {
                    "n_samples": len(X),
                    "n_features": X.shape[1] if len(X.shape) > 1 else 1,
                    "n_classes": len(np.unique(y)),
                    "classes": list(np.unique(y))
                }
            }
            
            # 1. Class distribution
            self.logger.info("Creating class distribution plot...")
            class_dist_fig = self.plot_class_distribution(y, "Training Data Class Distribution")
            if save_dir:
                class_dist_fig.savefig(os.path.join(save_dir, "class_distribution.png"), 
                                     dpi=300, bbox_inches='tight')
            plt.close(class_dist_fig)
            
            # 2. Cross-validation analysis
            self.logger.info("Performing cross-validation analysis...")
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            results["cv_results"] = {
                "scores": cv_scores.tolist(),
                "mean": float(np.mean(cv_scores)),
                "std": float(np.std(cv_scores)),
                "min": float(np.min(cv_scores)),
                "max": float(np.max(cv_scores))
            }
            
            # Plot CV results
            cv_fig = self.plot_cross_validation_results(cv_scores, 
                                                      k_values=list(range(1, cv_folds + 1)))
            if save_dir:
                cv_fig.savefig(os.path.join(save_dir, "cross_validation_results.png"), 
                              dpi=300, bbox_inches='tight')
            plt.close(cv_fig)
            
            # 3. Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                self.logger.info("Creating feature importance plot...")
                feature_names = [f'GLCM_{i}' if i < 20 else f'Color_{i-20}' 
                               for i in range(X.shape[1])]
                feature_fig = self.plot_feature_importance(model, feature_names)
                if save_dir:
                    feature_fig.savefig(os.path.join(save_dir, "feature_importance.png"), 
                                       dpi=300, bbox_inches='tight')
                plt.close(feature_fig)
            
            # 4. Detailed CV analysis with confusion matrices
            self.logger.info("Creating detailed CV analysis...")
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            all_predictions = []
            all_true_labels = []
            fold_results = []
            
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                # Train and predict
                model.fit(X[train_idx], y[train_idx])
                y_pred = model.predict(X[test_idx])
                y_true_fold = y[test_idx]
                
                # Store results
                all_predictions.extend(y_pred)
                all_true_labels.extend(y_true_fold)
                
                # Calculate metrics for this fold
                fold_accuracy = accuracy_score(y_true_fold, y_pred)
                fold_results.append({
                    "fold": fold + 1,
                    "accuracy": float(fold_accuracy),
                    "n_test_samples": len(y_true_fold)
                })
                
                # Create confusion matrix for this fold
                cm_fig = self.plot_confusion_matrix_detailed(
                    y_true_fold, y_pred, 
                    fold_info={"fold": fold + 1}
                )
                if save_dir:
                    cm_fig.savefig(os.path.join(save_dir, f"confusion_matrix_fold_{fold+1}.png"), 
                                  dpi=300, bbox_inches='tight')
                plt.close(cm_fig)
            
            results["fold_results"] = fold_results
            
            # 5. Overall performance metrics
            self.logger.info("Creating overall performance metrics...")
            perf_fig = self.plot_performance_metrics(
                np.array(all_true_labels), np.array(all_predictions)
            )
            if save_dir:
                perf_fig.savefig(os.path.join(save_dir, "performance_metrics.png"), 
                                dpi=300, bbox_inches='tight')
            plt.close(perf_fig)
            
            # 6. Wrong predictions analysis
            self.logger.info("Analyzing wrong predictions...")
            wrong_analysis = self.analyze_wrong_predictions(
                X, np.array(all_true_labels), np.array(all_predictions),
                np.arange(len(X))
            )
            results["wrong_predictions_analysis"] = wrong_analysis
            
            self.logger.info("Comprehensive report completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive report: {e}")
            raise
    
    def save_results_to_excel(self, results: Dict, save_path: str):
        """
        Save analysis results to Excel file
        
        Args:
            results: Results dictionary from comprehensive_report
            save_path: Path to save Excel file
        """
        try:
            import xlsxwriter
            
            workbook = xlsxwriter.Workbook(save_path)
            
            # Model info sheet
            model_sheet = workbook.add_worksheet('Model Info')
            model_sheet.write(0, 0, 'Parameter')
            model_sheet.write(0, 1, 'Value')
            
            row = 1
            for key, value in results["model_info"]["parameters"].items():
                model_sheet.write(row, 0, key)
                model_sheet.write(row, 1, str(value))
                row += 1
            
            # CV results sheet
            cv_sheet = workbook.add_worksheet('CV Results')
            cv_sheet.write(0, 0, 'Fold')
            cv_sheet.write(0, 1, 'Accuracy')
            cv_sheet.write(0, 2, 'Test Samples')
            
            row = 1
            for fold_result in results["fold_results"]:
                cv_sheet.write(row, 0, fold_result["fold"])
                cv_sheet.write(row, 1, fold_result["accuracy"])
                cv_sheet.write(row, 2, fold_result["n_test_samples"])
                row += 1
            
            # Summary sheet
            summary_sheet = workbook.add_worksheet('Summary')
            summary_data = [
                ['Overall CV Accuracy', results["cv_results"]["mean"]],
                ['CV Standard Deviation', results["cv_results"]["std"]],
                ['Min CV Accuracy', results["cv_results"]["min"]],
                ['Max CV Accuracy', results["cv_results"]["max"]],
                ['Total Samples', results["data_info"]["n_samples"]],
                ['Number of Features', results["data_info"]["n_features"]],
                ['Number of Classes', results["data_info"]["n_classes"]],
                ['Wrong Predictions', results["wrong_predictions_analysis"]["wrong_count"]],
                ['Error Rate', results["wrong_predictions_analysis"]["error_rate"]]
            ]
            
            for row, (metric, value) in enumerate(summary_data):
                summary_sheet.write(row, 0, metric)
                summary_sheet.write(row, 1, value)
            
            workbook.close()
            self.logger.info(f"Results saved to Excel: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results to Excel: {e}")
            raise


# Factory function for creating visualizer
def create_model_visualizer(logger: Optional[ILogger] = None, 
                          output_dir: str = "logs") -> ModelVisualizer:
    """
    Factory function to create ModelVisualizer instance
    
    Args:
        logger: Logger instance
        output_dir: Output directory for visualizations
        
    Returns:
        ModelVisualizer instance
    """
    return ModelVisualizer(logger, output_dir)
