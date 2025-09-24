#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Validation Module for Ethnicity Detection
Follows SOLID principles with proper separation of concerns
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .interfaces import ILogger, IProgressTracker
from .config import get_cv_config, get_model_config


class CrossValidationConfig:
    """Configuration for cross-validation (Single Responsibility Principle)"""
    
    def __init__(self, 
                 n_folds: int = None,
                 test_size: float = None,
                 random_state: int = None,
                 cv_scoring: str = None):
        """
        Initialize cross-validation configuration
        
        Args:
            n_folds: Number of CV folds (uses config if None)
            test_size: Test set size ratio (uses config if None)
            random_state: Random state for reproducibility (uses config if None)
            cv_scoring: Scoring metric for CV (uses config if None)
        """
        # Get configuration from environment
        config = get_cv_config()
        
        self.n_folds = n_folds or config.n_folds
        self.test_size = test_size or config.test_size
        self.random_state = random_state or config.random_state
        self.cv_scoring = cv_scoring or config.scoring


class ModelConfig:
    """Model configuration (Single Responsibility Principle)"""
    
    def __init__(self, 
                 n_estimators: List[int] = None,
                 max_depth: List[int] = None,
                 min_samples_split: List[int] = None,
                 min_samples_leaf: List[int] = None,
                 class_weight: List[str] = None):
        """
        Initialize model configuration
        
        Args:
            n_estimators: List of n_estimators values to test (uses config if None)
            max_depth: List of max_depth values to test (uses config if None)
            min_samples_split: List of min_samples_split values to test (uses config if None)
            min_samples_leaf: List of min_samples_leaf values to test (uses config if None)
            class_weight: List of class_weight values to test (uses config if None)
        """
        # Get configuration from environment
        config = get_model_config()
        
        self.n_estimators = n_estimators or [config.n_estimators // 2, config.n_estimators, config.n_estimators * 2]
        self.max_depth = max_depth or [None, 10, 20, 30]
        self.min_samples_split = min_samples_split or [2, 5, 10]
        self.min_samples_leaf = min_samples_leaf or [1, 2, 4]
        self.class_weight = class_weight or [None, 'balanced']


class CrossValidationResults:
    """Results container for cross-validation (Single Responsibility Principle)"""
    
    def __init__(self):
        """Initialize results container"""
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        self.scaler = None
        self.config = None
    
    def add_result(self, model_name: str, result: Dict[str, Any]) -> None:
        """
        Add cross-validation result
        
        Args:
            model_name: Name of the model
            result: Result dictionary with metrics
        """
        self.results[model_name] = result
        
        # Update best model if this is better
        if result['cv_mean'] > self.best_score:
            self.best_score = result['cv_mean']
            self.best_model = result['model']
            self.best_model_name = model_name
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all results"""
        return {
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'num_models_tested': len(self.results),
            'all_results': self.results
        }


class CrossValidationEngine:
    """Main cross-validation engine (Single Responsibility Principle)"""
    
    def __init__(self, 
                 logger: ILogger, 
                 progress_tracker: Optional[IProgressTracker] = None,
                 config: Optional[CrossValidationConfig] = None):
        """
        Initialize cross-validation engine
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
            config: Cross-validation configuration
        """
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.config = config or CrossValidationConfig()
        self.results = CrossValidationResults()
        
        self.logger.info("CrossValidationEngine initialized")
    
    def create_stratified_kfold(self) -> StratifiedKFold:
        """Create stratified k-fold splitter"""
        return StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
    
    def create_model_configs(self, model_config: ModelConfig) -> List[Dict[str, Any]]:
        """
        Create model configurations for testing
        
        Args:
            model_config: Model configuration
            
        Returns:
            List of model configuration dictionaries
        """
        configs = []
        
        # Create different model configurations
        for n_est in model_config.n_estimators:
            for max_d in model_config.max_depth:
                for min_split in model_config.min_samples_split:
                    for min_leaf in model_config.min_samples_leaf:
                        for class_w in model_config.class_weight:
                            config = {
                                'n_estimators': n_est,
                                'max_depth': max_d,
                                'min_samples_split': min_split,
                                'min_samples_leaf': min_leaf,
                                'class_weight': class_w,
                                'random_state': self.config.random_state
                            }
                            configs.append(config)
        
        return configs
    
    def test_model_config(self, 
                         X: np.ndarray, 
                         y: np.ndarray, 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a single model configuration
        
        Args:
            X: Feature matrix
            y: Target labels
            config: Model configuration
            
        Returns:
            Results dictionary
        """
        # Create model
        model = RandomForestClassifier(**config)
        
        # Create CV splitter
        cv = self.create_stratified_kfold()
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y, 
            cv=cv, 
            scoring=self.config.cv_scoring
        )
        
        # Train-test split for detailed analysis
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state, 
            stratify=y
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        result = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'y_test': y_test,
            'y_pred': y_pred,
            'model': model,
            'feature_importances': model.feature_importances_,
            'config': config
        }
        
        return result
    
    def run_cross_validation(self, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           model_config: Optional[ModelConfig] = None) -> CrossValidationResults:
        """
        Run complete cross-validation process
        
        Args:
            X: Feature matrix
            y: Target labels
            model_config: Model configuration
            
        Returns:
            Cross-validation results
        """
        self.logger.info("Starting cross-validation process")
        
        # Use default model config if not provided
        if model_config is None:
            model_config = ModelConfig()
        
        # Create model configurations
        configs = self.create_model_configs(model_config)
        
        self.logger.info(f"Testing {len(configs)} model configurations")
        
        if self.progress_tracker:
            self.progress_tracker.start_task("Cross-Validation", len(configs))
        
        # Test each configuration
        for i, config in enumerate(configs):
            model_name = f"RF_{config['n_estimators']}_{config['max_depth']}_{config['class_weight']}"
            
            self.logger.info(f"Testing configuration {i+1}/{len(configs)}: {model_name}")
            
            try:
                result = self.test_model_config(X, y, config)
                self.results.add_result(model_name, result)
                
                self.logger.info(f"  CV Accuracy: {result['cv_mean']:.3f} (+/- {result['cv_std'] * 2:.3f})")
                self.logger.info(f"  Test Accuracy: {result['test_accuracy']:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error testing configuration {model_name}: {e}")
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        self.logger.info(f"Cross-validation completed. Best model: {self.results.best_model_name}")
        self.logger.info(f"Best CV score: {self.results.best_score:.3f}")
        
        return self.results
    
    def perform_hyperparameter_tuning(self, 
                                    X: np.ndarray, 
                                    y: np.ndarray,
                                    param_grid: Optional[Dict[str, List]] = None) -> Tuple[Any, Dict[str, Any], float]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X: Feature matrix
            y: Target labels
            param_grid: Parameter grid for tuning
            
        Returns:
            Tuple of (best_estimator, best_params, best_score)
        """
        self.logger.info("Starting hyperparameter tuning")
        
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced']
            }
        
        # Create base model
        base_model = RandomForestClassifier(random_state=self.config.random_state)
        
        # Create CV splitter
        cv = self.create_stratified_kfold()
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=self.config.cv_scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


class FeatureScaler:
    """Feature scaling utility (Single Responsibility Principle)"""
    
    def __init__(self, logger: ILogger):
        """
        Initialize feature scaler
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform features
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        self.logger.info("Fitting and transforming features with StandardScaler")
        
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        self.logger.info(f"Features scaled: {X.shape} -> {X_scaled.shape}")
        self.logger.info(f"Scaled features - Mean: {np.mean(X_scaled):.6f}, Std: {np.std(X_scaled):.6f}")
        
        return X_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        return self.scaler.transform(X)
    
    def save_scaler(self, filepath: str) -> None:
        """
        Save fitted scaler
        
        Args:
            filepath: Path to save scaler
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        self.logger.info(f"Scaler saved to: {filepath}")
    
    @classmethod
    def load_scaler(cls, filepath: str, logger: ILogger) -> 'FeatureScaler':
        """
        Load fitted scaler
        
        Args:
            filepath: Path to load scaler from
            logger: Logger instance
            
        Returns:
            FeatureScaler instance with loaded scaler
        """
        scaler_instance = cls(logger)
        
        with open(filepath, 'rb') as f:
            scaler_instance.scaler = pickle.load(f)
        
        scaler_instance.is_fitted = True
        logger.info(f"Scaler loaded from: {filepath}")
        
        return scaler_instance


class CrossValidationManager:
    """Main manager for cross-validation operations (Facade Pattern)"""
    
    def __init__(self, 
                 logger: ILogger, 
                 progress_tracker: Optional[IProgressTracker] = None):
        """
        Initialize cross-validation manager
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
        """
        self.logger = logger
        self.progress_tracker = progress_tracker
        
        # Initialize components
        self.cv_config = CrossValidationConfig()
        self.model_config = ModelConfig()
        self.cv_engine = CrossValidationEngine(logger, progress_tracker, self.cv_config)
        self.feature_scaler = FeatureScaler(logger)
        
        self.logger.info("CrossValidationManager initialized")
    
    def run_complete_cv_pipeline(self, 
                               X: np.ndarray, 
                               y: np.ndarray,
                               output_dir: str = "logs/cv_results") -> CrossValidationResults:
        """
        Run complete cross-validation pipeline
        
        Args:
            X: Feature matrix
            y: Target labels
            output_dir: Directory to save results
            
        Returns:
            Cross-validation results
        """
        self.logger.info("Starting complete CV pipeline")
        
        # 1. Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # 2. Run cross-validation
        results = self.cv_engine.run_cross_validation(X_scaled, y, self.model_config)
        
        # 3. Store scaler in results
        results.scaler = self.feature_scaler
        results.config = self.cv_config
        
        # 4. Save results
        self.save_results(results, output_dir)
        
        self.logger.info("Complete CV pipeline finished")
        
        return results
    
    def save_results(self, results: CrossValidationResults, output_dir: str) -> None:
        """
        Save cross-validation results
        
        Args:
            results: Cross-validation results
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best model
        if results.best_model:
            model_path = os.path.join(output_dir, 'best_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(results.best_model, f)
            
            self.logger.info(f"Best model saved to: {model_path}")
        
        # Save scaler
        if results.scaler:
            scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
            results.scaler.save_scaler(scaler_path)
        
        # Save results summary
        summary_path = os.path.join(output_dir, 'cv_results_summary.pkl')
        with open(summary_path, 'wb') as f:
            pickle.dump(results.get_summary(), f)
        
        self.logger.info(f"Results summary saved to: {summary_path}")
    
    def load_results(self, output_dir: str) -> CrossValidationResults:
        """
        Load cross-validation results
        
        Args:
            output_dir: Directory containing results
            
        Returns:
            Cross-validation results
        """
        import os
        
        # Load results summary
        summary_path = os.path.join(output_dir, 'cv_results_summary.pkl')
        with open(summary_path, 'rb') as f:
            summary = pickle.load(f)
        
        # Load best model
        model_path = os.path.join(output_dir, 'best_model.pkl')
        with open(model_path, 'rb') as f:
            best_model = pickle.load(f)
        
        # Load scaler
        scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
        scaler = FeatureScaler.load_scaler(scaler_path, self.logger)
        
        # Create results object
        results = CrossValidationResults()
        results.results = summary['all_results']
        results.best_model = best_model
        results.best_model_name = summary['best_model_name']
        results.best_score = summary['best_score']
        results.scaler = scaler
        
        self.logger.info(f"Results loaded from: {output_dir}")
        
        return results
