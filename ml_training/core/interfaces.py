#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Interfaces for Ethnicity Detection Training System
Following SOLID Principles - Interface Segregation & Dependency Inversion
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator


class IDataLoader(ABC):
    """Interface for data loading operations"""
    
    @abstractmethod
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load data from specified path
        
        Args:
            data_path: Path to the dataset directory
            
        Returns:
            Tuple of (images, labels, metadata)
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: np.ndarray, labels: np.ndarray) -> bool:
        """
        Validate loaded data
        
        Args:
            data: Image data array
            labels: Label array
            
        Returns:
            True if data is valid, False otherwise
        """
        pass


class IImagePreprocessor(ABC):
    """Interface for image preprocessing operations"""
    
    @abstractmethod
    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess images for feature extraction
        
        Args:
            images: Input images array
            
        Returns:
            Preprocessed images
        """
        pass
    
    @abstractmethod
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about preprocessing parameters
        
        Returns:
            Dictionary with preprocessing parameters
        """
        pass


class IFeatureExtractor(ABC):
    """Interface for feature extraction operations"""
    
    @abstractmethod
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from preprocessed images
        
        Args:
            images: Preprocessed images array
            
        Returns:
            Feature matrix
        """
        pass
    
    @abstractmethod
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about extracted features
        
        Returns:
            Dictionary with feature information
        """
        pass


class IModelTrainer(ABC):
    """Interface for model training operations"""
    
    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> BaseEstimator:
        """
        Train the model
        
        Args:
            features: Feature matrix
            labels: Label array
            
        Returns:
            Trained model
        """
        pass
    
    @abstractmethod
    def cross_validate(self, features: np.ndarray, labels: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            features: Feature matrix
            labels: Label array
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with CV results
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        
        Returns:
            Dictionary with model information
        """
        pass


class IModelSaver(ABC):
    """Interface for model saving operations"""
    
    @abstractmethod
    def save_model(self, model: BaseEstimator, file_path: str) -> bool:
        """
        Save model to file
        
        Args:
            model: Trained model
            file_path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_model(self, file_path: str) -> Optional[BaseEstimator]:
        """
        Load model from file
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Loaded model or None if failed
        """
        pass


class ITrainingPipeline(ABC):
    """Interface for complete training pipeline"""
    
    @abstractmethod
    def run_pipeline(self, data_path: str, output_path: str) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            data_path: Path to training data
            output_path: Path to save the trained model
            
        Returns:
            Dictionary with training results
        """
        pass
    
    @abstractmethod
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline
        
        Returns:
            Dictionary with pipeline information
        """
        pass


class ILogger(ABC):
    """Interface for logging operations"""
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Log info message"""
        pass
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """Log warning message"""
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Log error message"""
        pass
    
    @abstractmethod
    def debug(self, message: str) -> None:
        """Log debug message"""
        pass


class IProgressTracker(ABC):
    """Interface for progress tracking"""
    
    @abstractmethod
    def start_task(self, task_name: str, total_steps: int) -> None:
        """Start tracking a new task"""
        pass
    
    @abstractmethod
    def update_progress(self, completed_steps: int) -> None:
        """Update progress for current task"""
        pass
    
    @abstractmethod
    def complete_task(self) -> None:
        """Mark current task as complete"""
        pass
    
    @abstractmethod
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information"""
        pass
