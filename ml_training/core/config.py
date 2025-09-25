#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module for Ethnicity Detection System
Uses python-dotenv for environment-based configuration management
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv


class Config:
    """Main configuration class using environment variables"""
    
    def __init__(self, env_file: str = '.env'):
        """
        Initialize configuration
        
        Args:
            env_file: Path to .env file
        """
        # Load environment variables from .env file
        # override=True ensures temporary/test env files take precedence
        load_dotenv(env_file, override=True)
        
        # Initialize all configuration sections
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration sections"""
        self.dataset = DatasetConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.cross_validation = CrossValidationConfig()
        self.feature_extraction = FeatureExtractionConfig()
        self.logging = LoggingConfig()
        self.server = ServerConfig()
        self.visualization = VisualizationConfig()
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as dictionary"""
        return {
            'dataset': self.dataset.get_config(),
            'model': self.model.get_config(),
            'training': self.training.get_config(),
            'cross_validation': self.cross_validation.get_config(),
            'feature_extraction': self.feature_extraction.get_config(),
            'logging': self.logging.get_config(),
            'server': self.server.get_config(),
            'visualization': self.visualization.get_config()
        }
    
    def print_config(self):
        """Print all configurations"""
        print("🔧 CONFIGURATION OVERVIEW")
        print("=" * 50)
        
        for section_name, config in self.get_all_configs().items():
            print(f"\n📁 {section_name.upper()}:")
            for key, value in config.items():
                print(f"   {key}: {value}")


class BaseConfig:
    """Base configuration class with common functionality"""
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration as dictionary - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _get_env_var(self, key: str, default: Any = None, var_type: type = str) -> Any:
        """
        Get environment variable with type conversion
        
        Args:
            key: Environment variable key
            default: Default value if not found
            var_type: Type to convert to
            
        Returns:
            Converted environment variable value
        """
        value = os.getenv(key, default)
        
        if value is None:
            return default
        
        # If value is already the correct type, return it
        if isinstance(value, var_type):
            return value
        
        try:
            if var_type == bool:
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ('true', '1', 'yes', 'on')
            elif var_type == list:
                # Handle comma-separated lists
                if isinstance(value, list):
                    return value
                return [item.strip() for item in str(value).split(',') if item.strip()]
            elif var_type == dict:
                # Handle key=value,key2=value2 format
                if isinstance(value, dict):
                    return value
                result = {}
                for pair in str(value).split(','):
                    if '=' in pair:
                        k, v = pair.split('=', 1)
                        result[k.strip()] = v.strip()
                return result
            elif var_type == tuple:
                # Handle comma-separated tuples
                if isinstance(value, tuple):
                    return value
                return tuple(item.strip() for item in str(value).split(',') if item.strip())
            else:
                return var_type(value)
        except (ValueError, TypeError, AttributeError):
            return default


class DatasetConfig(BaseConfig):
    """Dataset configuration"""
    
    def __init__(self):
        """Initialize dataset configuration"""
        self.data_dir = self._get_env_var('DATASET_DIR', '../dataset/dataset_periorbital')
        self.holistic_dir = self._get_env_var('DATASET_HOLISTIC_DIR', '../dataset/dataset_holistik')
        self.periorbital_dir = self._get_env_var('DATASET_PERIORBITAL_DIR', '../dataset/dataset_periorbital')
        self.ethnicities = self._get_env_var('ETHNICITIES', ['Banjar', 'Bugis', 'Javanese', 'Malay', 'Sundanese'], list)
        self.image_extensions = self._get_env_var('IMAGE_EXTENSIONS', ['.jpg', '.jpeg', '.png', '.bmp'], list)
        self.max_images_per_class = self._get_env_var('MAX_IMAGES_PER_CLASS', 1000, int)
        self.train_split = self._get_env_var('TRAIN_SPLIT', 0.8, float)
        self.val_split = self._get_env_var('VAL_SPLIT', 0.1, float)
        self.test_split = self._get_env_var('TEST_SPLIT', 0.1, float)
        self.random_seed = self._get_env_var('DATASET_RANDOM_SEED', 42, int)
    
    def get_config(self) -> Dict[str, Any]:
        """Get dataset configuration"""
        return {
            'data_dir': self.data_dir,
            'holistic_dir': self.holistic_dir,
            'periorbital_dir': self.periorbital_dir,
            'ethnicities': self.ethnicities,
            'image_extensions': self.image_extensions,
            'max_images_per_class': self.max_images_per_class,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'random_seed': self.random_seed
        }


class ModelConfig(BaseConfig):
    """Model configuration"""
    
    def __init__(self):
        """Initialize model configuration"""
        self.model_type = self._get_env_var('MODEL_TYPE', 'RandomForest')
        self.model_path = self._get_env_var('MODEL_PATH', 'model_ml/pickle_model.pkl')
        self.n_estimators = self._get_env_var('MODEL_N_ESTIMATORS', 200, int)
        self.max_depth = self._get_env_var('MODEL_MAX_DEPTH', None, int)
        self.min_samples_split = self._get_env_var('MODEL_MIN_SAMPLES_SPLIT', 2, int)
        self.min_samples_leaf = self._get_env_var('MODEL_MIN_SAMPLES_LEAF', 1, int)
        self.max_features = self._get_env_var('MODEL_MAX_FEATURES', 'sqrt', str)
        self.class_weight = self._get_env_var('MODEL_CLASS_WEIGHT', None, str)
        self.random_state = self._get_env_var('MODEL_RANDOM_STATE', 42, int)
        self.n_jobs = self._get_env_var('MODEL_N_JOBS', -1, int)
        self.bootstrap = self._get_env_var('MODEL_BOOTSTRAP', True, bool)
        self.oob_score = self._get_env_var('MODEL_OOB_SCORE', False, bool)
        self.verbose = self._get_env_var('MODEL_VERBOSE', 0, int)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'verbose': self.verbose
        }


class TrainingConfig(BaseConfig):
    """Training configuration"""
    
    def __init__(self):
        """Initialize training configuration"""
        self.batch_size = self._get_env_var('TRAINING_BATCH_SIZE', 32, int)
        self.epochs = self._get_env_var('TRAINING_EPOCHS', 100, int)
        self.learning_rate = self._get_env_var('TRAINING_LEARNING_RATE', 0.001, float)
        self.early_stopping_patience = self._get_env_var('TRAINING_EARLY_STOPPING_PATIENCE', 10, int)
        self.save_best_only = self._get_env_var('TRAINING_SAVE_BEST_ONLY', True, bool)
        self.monitor_metric = self._get_env_var('TRAINING_MONITOR_METRIC', 'val_accuracy', str)
        self.mode = self._get_env_var('TRAINING_MODE', 'max', str)
        self.verbose = self._get_env_var('TRAINING_VERBOSE', 1, int)
        self.random_seed = self._get_env_var('TRAINING_RANDOM_SEED', 42, int)
    
    def get_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'early_stopping_patience': self.early_stopping_patience,
            'save_best_only': self.save_best_only,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode,
            'verbose': self.verbose,
            'random_seed': self.random_seed
        }


class CrossValidationConfig(BaseConfig):
    """Cross-validation configuration"""
    
    def __init__(self):
        """Initialize cross-validation configuration"""
        self.n_folds = self._get_env_var('CV_N_FOLDS', 6, int)
        self.test_size = self._get_env_var('CV_TEST_SIZE', 0.2, float)
        self.random_state = self._get_env_var('CV_RANDOM_STATE', 42, int)
        self.scoring = self._get_env_var('CV_SCORING', 'accuracy', str)
        self.shuffle = self._get_env_var('CV_SHUFFLE', True, bool)
        self.stratify = self._get_env_var('CV_STRATIFY', True, bool)
        self.return_train_score = self._get_env_var('CV_RETURN_TRAIN_SCORE', False, bool)
        self.n_jobs = self._get_env_var('CV_N_JOBS', -1, int)
        self.verbose = self._get_env_var('CV_VERBOSE', 0, int)
        self.pre_dispatch = self._get_env_var('CV_PRE_DISPATCH', '2*n_jobs', str)
    
    def get_config(self) -> Dict[str, Any]:
        """Get cross-validation configuration"""
        return {
            'n_folds': self.n_folds,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'scoring': self.scoring,
            'shuffle': self.shuffle,
            'stratify': self.stratify,
            'return_train_score': self.return_train_score,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
            'pre_dispatch': self.pre_dispatch
        }


class FeatureExtractionConfig(BaseConfig):
    """Feature extraction configuration"""
    
    def __init__(self):
        """Initialize feature extraction configuration"""
        # GLCM Configuration
        self.glc_distances = self._get_env_var('GLCM_DISTANCES', [1], list)
        # Coerce to ints if provided as strings
        try:
            self.glc_distances = [int(x) for x in self.glc_distances]
        except Exception:
            pass

        self.glc_angles = self._get_env_var('GLCM_ANGLES', [0, 45, 90, 135], list)
        try:
            self.glc_angles = [int(x) for x in self.glc_angles]
        except Exception:
            pass
        self.glc_levels = self._get_env_var('GLCM_LEVELS', 256, int)
        self.glc_symmetric = self._get_env_var('GLCM_SYMMETRIC', True, bool)
        self.glc_normed = self._get_env_var('GLCM_NORMED', True, bool)
        
        # Color Histogram Configuration
        self.color_bins = self._get_env_var('COLOR_BINS', 16, int)
        self.color_channels = self._get_env_var('COLOR_CHANNELS', [1, 2], list)  # S and V for HSV
        try:
            self.color_channels = [int(x) for x in self.color_channels]
        except Exception:
            pass
        self.color_space = self._get_env_var('COLOR_SPACE', 'HSV', str)
        
        # Image Preprocessing
        self.image_size = self._get_env_var('IMAGE_SIZE', (256, 256), tuple)
        try:
            self.image_size = tuple(int(x) for x in self.image_size)
        except Exception:
            pass
        self.resize_method = self._get_env_var('RESIZE_METHOD', 'cv2', str)
        self.normalize = self._get_env_var('NORMALIZE_IMAGES', True, bool)
        
        # Feature Scaling
        self.scale_features = self._get_env_var('SCALE_FEATURES', True, bool)
        self.scaler_type = self._get_env_var('SCALER_TYPE', 'StandardScaler', str)
        
        # LBP parameters
        self.lbp_p = self._get_env_var('LBP_P', [8, 16, 24], list)
        self.lbp_r = self._get_env_var('LBP_R', [1.0, 2.0, 3.0], list)
        self.lbp_method = self._get_env_var('LBP_METHOD', 'uniform', str)
        self.lbp_bins = self._get_env_var('LBP_BINS', 10, int)
        
        # Paper LBP parameters
        self.paper_lbp_radii = self._get_env_var('PAPER_LBP_RADII', [1, 2, 3], list)
        self.paper_lbp_points = self._get_env_var('PAPER_LBP_POINTS', [8, 16, 24], list)
        self.paper_lbp_method = self._get_env_var('PAPER_LBP_METHOD', 'uniform', str)
        self.paper_lbp_grid_size = self._get_env_var('PAPER_LBP_GRID_SIZE', [4, 4], list)
        
        # HOG parameters
        self.hog_orientations = self._get_env_var('HOG_ORIENTATIONS', 9, int)
        self.hog_pixels_per_cell = self._get_env_var('HOG_PIXELS_PER_CELL', [8, 8], list)
        self.hog_cells_per_block = self._get_env_var('HOG_CELLS_PER_BLOCK', [2, 2], list)
        
        # LBP Variants parameters
        # MB-LBP
        self.mb_lbp_block_size = self._get_env_var('MB_LBP_BLOCK_SIZE', [2, 3], list)
        
        # MQLBP
        self.mqlbp_L = self._get_env_var('MQLBP_L', 2, int)
        self.mqlbp_thresholds = self._get_env_var('MQLBP_THRESHOLDS', [0.1, 0.2], list)
        
        # d-LBP
        self.dlbp_radius1 = self._get_env_var('DLBP_RADIUS1', 1, int)
        self.dlbp_radius2 = self._get_env_var('DLBP_RADIUS2', 3, int)
        self.dlbp_n_points = self._get_env_var('DLBP_N_POINTS', 8, int)
        
        # RedDLBP
        self.reddlbp_radius = self._get_env_var('REDDLBP_RADIUS', 2, int)
    
    def get_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration"""
        return {
            'glcm_distances': self.glc_distances,
            'glcm_angles': self.glc_angles,
            'glcm_levels': self.glc_levels,
            'glcm_symmetric': self.glc_symmetric,
            'glcm_normed': self.glc_normed,
            'color_bins': self.color_bins,
            'color_channels': self.color_channels,
            'color_space': self.color_space,
            'image_size': self.image_size,
            'resize_method': self.resize_method,
            'normalize': self.normalize,
            'scale_features': self.scale_features,
            'scaler_type': self.scaler_type
        }


class LoggingConfig(BaseConfig):
    """Logging configuration"""
    
    def __init__(self):
        """Initialize logging configuration"""
        self.log_level = self._get_env_var('LOG_LEVEL', 'INFO', str)
        self.log_file = self._get_env_var('LOG_FILE', 'logs/training.log', str)
        self.log_format = self._get_env_var('LOG_FORMAT', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s', str)
        self.log_date_format = self._get_env_var('LOG_DATE_FORMAT', '%Y-%m-%d %H:%M:%S', str)
        self.max_log_size = self._get_env_var('MAX_LOG_SIZE', 10 * 1024 * 1024, int)  # 10MB
        self.backup_count = self._get_env_var('LOG_BACKUP_COUNT', 5, int)
        self.console_output = self._get_env_var('LOG_CONSOLE_OUTPUT', True, bool)
        self.file_output = self._get_env_var('LOG_FILE_OUTPUT', True, bool)
        self.encoding = self._get_env_var('LOG_ENCODING', 'utf-8', str)
    
    def get_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'log_level': self.log_level,
            'log_file': self.log_file,
            'log_format': self.log_format,
            'log_date_format': self.log_date_format,
            'max_log_size': self.max_log_size,
            'backup_count': self.backup_count,
            'console_output': self.console_output,
            'file_output': self.file_output,
            'encoding': self.encoding
        }


class ServerConfig(BaseConfig):
    """Server configuration"""
    
    def __init__(self):
        """Initialize server configuration"""
        self.host = self._get_env_var('SERVER_HOST', 'localhost', str)
        self.port = self._get_env_var('SERVER_PORT', 8080, int)
        self.debug = self._get_env_var('SERVER_DEBUG', False, bool)
        self.max_connections = self._get_env_var('SERVER_MAX_CONNECTIONS', 100, int)
        self.timeout = self._get_env_var('SERVER_TIMEOUT', 30, int)
        self.keep_alive = self._get_env_var('SERVER_KEEP_ALIVE', True, bool)
        self.cors_enabled = self._get_env_var('SERVER_CORS_ENABLED', True, bool)
        self.cors_origins = self._get_env_var('SERVER_CORS_ORIGINS', '*', str)
    
    def get_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return {
            'host': self.host,
            'port': self.port,
            'debug': self.debug,
            'max_connections': self.max_connections,
            'timeout': self.timeout,
            'keep_alive': self.keep_alive,
            'cors_enabled': self.cors_enabled,
            'cors_origins': self.cors_origins
        }


class VisualizationConfig(BaseConfig):
    """Visualization configuration"""
    
    def __init__(self):
        """Initialize visualization configuration"""
        self.output_dir = self._get_env_var('VIZ_OUTPUT_DIR', 'logs/analysis', str)
        self.figure_dpi = self._get_env_var('VIZ_FIGURE_DPI', 300, int)
        self.figure_size = self._get_env_var('VIZ_FIGURE_SIZE', (12, 8), tuple)
        self.style = self._get_env_var('VIZ_STYLE', 'default', str)
        self.color_palette = self._get_env_var('VIZ_COLOR_PALETTE', 'husl', str)
        self.font_size = self._get_env_var('VIZ_FONT_SIZE', 12, int)
        self.save_format = self._get_env_var('VIZ_SAVE_FORMAT', 'png', str)
        self.show_plots = self._get_env_var('VIZ_SHOW_PLOTS', False, bool)
        self.science_plots = self._get_env_var('VIZ_SCIENCE_PLOTS', True, bool)
        self.background_color = self._get_env_var('VIZ_BACKGROUND_COLOR', 'white', str)
    
    def get_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        return {
            'output_dir': self.output_dir,
            'figure_dpi': self.figure_dpi,
            'figure_size': self.figure_size,
            'style': self.style,
            'color_palette': self.color_palette,
            'font_size': self.font_size,
            'save_format': self.save_format,
            'show_plots': self.show_plots,
            'science_plots': self.science_plots,
            'background_color': self.background_color
        }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return config


def reload_config(env_file: str = '.env') -> Config:
    """Reload configuration from environment file"""
    global config
    config = Config(env_file)
    return config


# Convenience functions for quick access
def get_dataset_config() -> DatasetConfig:
    """Get dataset configuration"""
    return config.dataset


def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return config.model


def get_training_config() -> TrainingConfig:
    """Get training configuration"""
    return config.training


def get_cv_config() -> CrossValidationConfig:
    """Get cross-validation configuration"""
    return config.cross_validation


def get_feature_config() -> FeatureExtractionConfig:
    """Get feature extraction configuration"""
    return config.feature_extraction


def get_logging_config() -> LoggingConfig:
    """Get logging configuration"""
    return config.logging


def get_server_config() -> ServerConfig:
    """Get server configuration"""
    return config.server


def get_viz_config() -> VisualizationConfig:
    """Get visualization configuration"""
    return config.visualization
