#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Trainers Implementation
Single Responsibility Principle - Each trainer handles one type of model
Open/Closed Principle - Easy to extend with new model types
Dependency Inversion - Depends on sklearn abstractions
"""

import numpy as np
from abc import ABC
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from .interfaces import IModelTrainer, ILogger, IProgressTracker
from .config import get_model_config, get_training_config


class BaseModelTrainer(IModelTrainer, ABC):
    """Base class for model trainers"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None):
        """
        Initialize base model trainer
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
        """
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.model_info = {}
        self.trained_model = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info.copy()


class RandomForestTrainer(BaseModelTrainer):
    """Random Forest Model Trainer"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None,
                 n_estimators: int = None, random_state: int = None, **kwargs):
        """
        Initialize Random Forest trainer
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
            n_estimators: Number of trees in the forest (uses config if None)
            random_state: Random state for reproducibility (uses config if None)
            **kwargs: Additional parameters for RandomForestClassifier
        """
        super().__init__(logger, progress_tracker)
        
        # Get configuration
        model_config = get_model_config()
        
        # Use configuration values if not provided
        self.n_estimators = n_estimators or model_config.n_estimators
        self.random_state = random_state or model_config.random_state
        
        # Merge additional parameters with configuration
        self.additional_params = {
            'max_depth': model_config.max_depth,
            'min_samples_split': model_config.min_samples_split,
            'min_samples_leaf': model_config.min_samples_leaf,
            'max_features': model_config.max_features,
            'class_weight': model_config.class_weight,
            'n_jobs': model_config.n_jobs,
            'bootstrap': model_config.bootstrap,
            'oob_score': model_config.oob_score,
            'verbose': model_config.verbose,
            **kwargs  # Override with any provided kwargs
        }
        
        # Remove None values
        self.additional_params = {k: v for k, v in self.additional_params.items() if v is not None}
        
        self.model_info = {
            'type': 'RandomForest',
            'algorithm': 'Random Forest Classifier',
            'n_estimators': n_estimators,
            'random_state': random_state,
            'additional_params': kwargs,
            'description': 'Ensemble method using multiple decision trees'
        }
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> BaseEstimator:
        """
        Train Random Forest model
        
        Args:
            features: Feature matrix (N, num_features)
            labels: Label array (N,)
            
        Returns:
            Trained Random Forest model
        """
        self.logger.info("Training Random Forest model...")
        
        if self.progress_tracker:
            self.progress_tracker.start_task("Model Training", 1)
        
        try:
            # Shuffle data for better training
            features_shuffled, labels_shuffled = shuffle(features, labels, random_state=220)
            
            # Create and train model
            self.trained_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                **self.additional_params
            )
            
            self.trained_model.fit(features_shuffled, labels_shuffled)
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(1)
                self.progress_tracker.complete_task()
            
            # Update model info
            self.model_info.update({
                'training_samples': len(features),
                'num_features': features.shape[1],
                'num_classes': len(np.unique(labels)),
                'feature_importance_available': True,
                'trained': True
            })
            
            self.logger.info("Random Forest model training completed")
            return self.trained_model
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {e}")
            raise
    
    def cross_validate(self, features: np.ndarray, labels: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            features: Feature matrix (N, num_features)
            labels: Label array (N,)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with CV results
        """
        self.logger.info(f"Running {cv_folds}-fold cross validation...")
        
        try:
            # Shuffle data
            features_shuffled, labels_shuffled = shuffle(features, labels, random_state=220)
            
            # Create model for CV
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                **self.additional_params
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, features_shuffled, labels_shuffled, 
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=220),
                scoring='accuracy'
            )
            
            # Calculate statistics
            mean_accuracy = np.mean(cv_scores) * 100
            std_accuracy = np.std(cv_scores) * 100
            
            cv_results = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'cv_scores': cv_scores * 100,  # Convert to percentage
                'cv_folds': cv_folds,
                'min_accuracy': np.min(cv_scores) * 100,
                'max_accuracy': np.max(cv_scores) * 100
            }
            
            self.logger.info(f"Cross-validation completed:")
            self.logger.info(f"   Mean accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
            self.logger.info(f"   Individual fold scores: {cv_scores * 100}")
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            raise


class SVMTrainer(BaseModelTrainer):
    """SVM Model Trainer"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None,
                 C: float = 0.1, gamma: float = 1, kernel: str = 'poly', **kwargs):
        """
        Initialize SVM trainer
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
            C: Regularization parameter
            gamma: Kernel coefficient
            kernel: Kernel type
            **kwargs: Additional parameters for SVC
        """
        super().__init__(logger, progress_tracker)
        
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.additional_params = kwargs
        
        self.model_info = {
            'type': 'SVM',
            'algorithm': 'Support Vector Classifier',
            'C': C,
            'gamma': gamma,
            'kernel': kernel,
            'additional_params': kwargs,
            'description': 'Support Vector Machine classifier'
        }
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> BaseEstimator:
        """
        Train SVM model
        
        Args:
            features: Feature matrix (N, num_features)
            labels: Label array (N,)
            
        Returns:
            Trained SVM model
        """
        self.logger.info("Training SVM model...")
        
        if self.progress_tracker:
            self.progress_tracker.start_task("SVM Training", 1)
        
        try:
            # Shuffle data for better training
            features_shuffled, labels_shuffled = shuffle(features, labels, random_state=220)
            
            # Create and train model
            self.trained_model = SVC(
                C=self.C,
                gamma=self.gamma,
                kernel=self.kernel,
                **self.additional_params
            )
            
            self.trained_model.fit(features_shuffled, labels_shuffled)
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(1)
                self.progress_tracker.complete_task()
            
            # Update model info
            self.model_info.update({
                'training_samples': len(features),
                'num_features': features.shape[1],
                'num_classes': len(np.unique(labels)),
                'feature_importance_available': False,
                'trained': True
            })
            
            self.logger.info("SVM model training completed")
            return self.trained_model
            
        except Exception as e:
            self.logger.error(f"Error training SVM model: {e}")
            raise
    
    def cross_validate(self, features: np.ndarray, labels: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            features: Feature matrix (N, num_features)
            labels: Label array (N,)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with CV results
        """
        self.logger.info(f"Running {cv_folds}-fold cross validation...")
        
        try:
            # Shuffle data
            features_shuffled, labels_shuffled = shuffle(features, labels, random_state=220)
            
            # Create model for CV
            model = SVC(
                C=self.C,
                gamma=self.gamma,
                kernel=self.kernel,
                **self.additional_params
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, features_shuffled, labels_shuffled, 
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=220),
                scoring='accuracy'
            )
            
            # Calculate statistics
            mean_accuracy = np.mean(cv_scores) * 100
            std_accuracy = np.std(cv_scores) * 100
            
            cv_results = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'cv_scores': cv_scores * 100,  # Convert to percentage
                'cv_folds': cv_folds,
                'min_accuracy': np.min(cv_scores) * 100,
                'max_accuracy': np.max(cv_scores) * 100
            }
            
            self.logger.info(f"Cross-validation completed:")
            self.logger.info(f"   Mean accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
            self.logger.info(f"   Individual fold scores: {cv_scores * 100}")
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            raise


class ModelFactory:
    """Factory for creating model trainers"""
    
    @staticmethod
    def create_trainer(trainer_type: str, logger: ILogger, 
                      progress_tracker: IProgressTracker = None, **kwargs) -> IModelTrainer:
        """
        Create model trainer instance
        
        Args:
            trainer_type: Type of trainer ('random_forest' or 'svm')
            logger: Logger instance
            progress_tracker: Progress tracker instance
            **kwargs: Additional parameters for the trainer
            
        Returns:
            Model trainer instance
            
        Raises:
            ValueError: If trainer_type is not supported
        """
        trainer_type = trainer_type.lower()
        
        if trainer_type == 'random_forest':
            return RandomForestTrainer(logger, progress_tracker, **kwargs)
        elif trainer_type == 'svm':
            return SVMTrainer(logger, progress_tracker, **kwargs)
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")
    
    @staticmethod
    def get_available_trainers() -> list:
        """Get list of available trainer types"""
        return ['random_forest', 'svm']
