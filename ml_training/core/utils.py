#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Classes for Training System (Windows Compatible)
Single Responsibility Principle - Each utility has one responsibility
Windows-compatible version without Unicode emojis
"""

import os
import pickle
import logging
import numpy as np
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator
from .interfaces import IModelSaver, ILogger, IProgressTracker
from .config import get_logging_config


class ModelSaver(IModelSaver):
    """Handles saving and loading of trained models"""
    
    def __init__(self, logger: ILogger):
        """
        Initialize model saver
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
    
    def save_model(self, model: BaseEstimator, file_path: str) -> bool:
        """
        Save model to file
        
        Args:
            model: Trained model
            file_path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Saving model to {file_path}...")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save model using pickle
            with open(file_path, 'wb') as file:
                pickle.dump(model, file)
            
            # Verify file was created
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                self.logger.info(f"Model saved successfully! File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                return True
            else:
                self.logger.error("Model file was not created")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, file_path: str) -> Optional[BaseEstimator]:
        """
        Load model from file
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Loaded model or None if failed
        """
        try:
            self.logger.info(f"Loading model from {file_path}...")
            
            if not os.path.exists(file_path):
                self.logger.error(f"Model file not found: {file_path}")
                return None
            
            # Load model using pickle
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            
            self.logger.info(f"Model loaded successfully: {type(model)}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None


class TrainingLogger(ILogger):
    """Enhanced logger for training system (Windows compatible)"""
    
    def __init__(self, name: str = 'ethnicity_training', log_file: str = None, 
                 console_level: int = None, file_level: int = None):
        """
        Initialize training logger
        
        Args:
            name: Logger name
            log_file: Path to log file (optional, uses config if None)
            console_level: Console logging level (uses config if None)
            file_level: File logging level (uses config if None)
        """
        # Get logging configuration
        log_config = get_logging_config()
        
        # Use configuration values if not provided
        if console_level is None:
            console_level = getattr(logging, log_config.log_level.upper(), logging.INFO)
        if file_level is None:
            file_level = getattr(logging, log_config.log_level.upper(), logging.DEBUG)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Set UTF-8 encoding for console handler
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8')
        
        self.logger.addHandler(console_handler)
        
        # Set up log file path (use config if not specified)
        if log_file is None:
            log_file = log_config.log_file
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # File handler (always create file handler if enabled in config)
        if log_config.file_output:
            file_handler = logging.FileHandler(log_file, encoding=log_config.encoding)
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter(log_config.log_format, datefmt=log_config.log_date_format)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler (only if enabled in config)
        if not log_config.console_output:
            # Remove console handler if disabled
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
        
        # Store log file path for reference
        self.log_file = log_file
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)


class ProgressTracker(IProgressTracker):
    """Simple progress tracker for training operations (Windows compatible)"""
    
    def __init__(self, logger: ILogger = None):
        """
        Initialize progress tracker
        
        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger
        self.current_task = None
        self.total_steps = 0
        self.completed_steps = 0
    
    def start_task(self, task_name: str, total_steps: int) -> None:
        """Start tracking a new task"""
        self.current_task = task_name
        self.total_steps = total_steps
        self.completed_steps = 0
        
        if self.logger:
            self.logger.info(f"Starting: {task_name} ({total_steps} steps)")
    
    def update_progress(self, completed_steps: int) -> None:
        """Update progress for current task"""
        self.completed_steps = completed_steps
        
        if self.total_steps > 0:
            percentage = (completed_steps / self.total_steps) * 100
            
            if self.logger and completed_steps % max(1, self.total_steps // 10) == 0:
                self.logger.debug(f"   Progress: {completed_steps}/{self.total_steps} ({percentage:.1f}%)")
    
    def complete_task(self) -> None:
        """Mark current task as complete"""
        if self.logger:
            self.logger.info(f"Completed: {self.current_task}")
        
        self.current_task = None
        self.total_steps = 0
        self.completed_steps = 0
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information"""
        return {
            'current_task': self.current_task,
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps,
            'percentage': (self.completed_steps / self.total_steps * 100) if self.total_steps > 0 else 0
        }


class TrainingConfig:
    """Configuration class for training parameters"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize training configuration
        
        Args:
            config_dict: Dictionary with configuration parameters
        """
        # Default configuration
        self.defaults = {
            'dataset_path': 'dataset_periorbital',
            'model_output_path': 'model_ml/pickle_model.pkl',
            'model_type': 'random_forest',
            'cv_folds': 6,
            'random_state': 220,
            'n_estimators': 200,
            'image_size': (400, 200),
            'glcm_distances': [1],
            'glcm_angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
            'glcm_levels': 256,
            'color_bins': 16,
            'color_channels': [1, 2],  # S and V channels for HSV
            'log_file': 'training.log'
        }
        
        # Update with provided config
        self.config = self.defaults.copy()
        if config_dict:
            self.config.update(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.config.copy()
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        required_keys = ['dataset_path', 'model_output_path', 'model_type']
        
        for key in required_keys:
            if key not in self.config:
                return False
        
        # Validate model type
        valid_model_types = ['random_forest', 'svm']
        if self.config['model_type'] not in valid_model_types:
            return False
        
        return True
