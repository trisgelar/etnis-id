#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Training Pipeline
Orchestrates the complete training process following SOLID principles
"""

import numpy as np
from typing import Dict, Any, Optional
from .interfaces import ITrainingPipeline, IDataLoader, IModelTrainer, IModelSaver, ILogger, IProgressTracker
from .data_loader import EthnicityDataLoader
from .preprocessors import GLCMPreprocessor, ColorHistogramPreprocessor, PreprocessingPipeline
from .feature_extractors import GLCFeatureExtractor, ColorHistogramFeatureExtractor, CombinedFeatureExtractor
from .model_trainers import ModelFactory
from .utils import ModelSaver
from .config import get_config, get_dataset_config, get_model_config, get_training_config


class EthnicityTrainingPipeline(ITrainingPipeline):
    """Complete training pipeline for ethnicity detection"""
    
    def __init__(self, logger: ILogger, 
                 progress_tracker: IProgressTracker = None, custom_config: Dict[str, Any] = None):
        """
        Initialize training pipeline
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
            custom_config: Custom configuration overrides (optional)
        """
        self.logger = logger
        self.progress_tracker = progress_tracker
        
        # Get configuration from environment
        self.config = get_config()
        self.dataset_config = get_dataset_config()
        self.model_config = get_model_config()
        self.training_config = get_training_config()
        
        # Apply custom configuration overrides if provided
        if custom_config:
            self._apply_custom_config(custom_config)
        
        # Initialize components
        self.data_loader = None
        self.preprocessing_pipeline = None
        self.feature_extractor = None
        self.model_trainer = None
        self.model_saver = None
        
        # Training results
        self.training_results = {}
        
        self._initialize_components()
    
    def _apply_custom_config(self, custom_config: Dict[str, Any]):
        """Apply custom configuration overrides"""
        for section, values in custom_config.items():
            if hasattr(self, f"{section}_config"):
                config_obj = getattr(self, f"{section}_config")
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
                        self.logger.info(f"Override: {section}.{key} = {value}")
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        self.logger.info("Initializing training pipeline components...")
        
        # Data loader
        self.data_loader = EthnicityDataLoader(self.logger)
        
        # Preprocessing pipeline
        self.preprocessing_pipeline = PreprocessingPipeline(self.logger, self.progress_tracker)
        
        # Add preprocessors
        self.preprocessing_pipeline.add_preprocessor(
            GLCMPreprocessor(self.logger, self.progress_tracker)
        ).add_preprocessor(
            ColorHistogramPreprocessor(self.logger, self.progress_tracker)
        )
        
        # Feature extractors
        self.feature_extractor = CombinedFeatureExtractor(self.logger, self.progress_tracker)
        
        # Add feature extractors
        self.feature_extractor.add_extractor(
            GLCFeatureExtractor(
                self.logger, 
                self.progress_tracker,
                distances=self.config.get('glcm_distances'),
                angles=self.config.get('glcm_angles'),
                levels=self.config.get('glcm_levels')
            )
        ).add_extractor(
            ColorHistogramFeatureExtractor(
                self.logger,
                self.progress_tracker,
                bins=self.config.get('color_bins'),
                channels=self.config.get('color_channels')
            )
        )
        
        # Model trainer
        trainer_type = self.config.get('model_type')
        trainer_params = self._get_trainer_params(trainer_type)
        self.model_trainer = ModelFactory.create_trainer(
            trainer_type, self.logger, self.progress_tracker, **trainer_params
        )
        
        # Model saver
        self.model_saver = ModelSaver(self.logger)
        
        self.logger.info("Pipeline components initialized")
    
    def _get_trainer_params(self, trainer_type: str) -> Dict[str, Any]:
        """Get parameters for specific trainer type"""
        if trainer_type == 'random_forest':
            return {
                'n_estimators': self.config.get('n_estimators'),
                'random_state': self.config.get('random_state')
            }
        elif trainer_type == 'svm':
            return {
                'random_state': self.config.get('random_state')
            }
        else:
            return {}
    
    def run_pipeline(self, data_path: str, output_path: str) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            data_path: Path to training data
            output_path: Path to save the trained model
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("STARTING ETHNICITY DETECTION TRAINING PIPELINE")
        self.logger.info("=" * 70)
        
        try:
            # Phase 1: Load Data
            self.logger.info("\nPHASE 1: LOADING DATA")
            self.logger.info("-" * 30)
            
            images, labels, metadata = self.data_loader.load_data(data_path)
            self.training_results['data_metadata'] = metadata
            
            # Phase 2: Preprocessing
            self.logger.info("\nPHASE 2: PREPROCESSING")
            self.logger.info("-" * 30)
            
            # Resize images first
            from .preprocessors import ResizePreprocessor
            resize_preprocessor = ResizePreprocessor(
                self.config.get('image_size'), self.logger, self.progress_tracker
            )
            resized_images = resize_preprocessor.preprocess(images)
            
            # GLCM preprocessing
            glcm_images = self.preprocessing_pipeline.preprocessors[0].preprocess(resized_images)
            
            # Color preprocessing
            color_images = self.preprocessing_pipeline.preprocessors[1].preprocess(resized_images)
            
            preprocessed_data = {
                'glcm': glcm_images,
                'color': color_images
            }
            
            self.training_results['preprocessing_info'] = self.preprocessing_pipeline.get_pipeline_info()
            
            # Phase 3: Feature Extraction
            self.logger.info("\nPHASE 3: FEATURE EXTRACTION")
            self.logger.info("-" * 30)
            
            features = self.feature_extractor.extract_features(preprocessed_data)
            self.training_results['feature_info'] = self.feature_extractor.get_combined_feature_info()
            
            # Phase 4: Model Training
            self.logger.info("\nPHASE 4: MODEL TRAINING")
            self.logger.info("-" * 30)
            
            # Cross-validation
            cv_results = self.model_trainer.cross_validate(
                features, labels, self.config.get('cv_folds')
            )
            self.training_results['cross_validation'] = cv_results
            
            # Train final model
            trained_model = self.model_trainer.train(features, labels)
            self.training_results['model_info'] = self.model_trainer.get_model_info()
            
            # Phase 5: Save Model
            self.logger.info("\nPHASE 5: SAVING MODEL")
            self.logger.info("-" * 30)
            
            save_success = self.model_saver.save_model(trained_model, output_path)
            self.training_results['model_saved'] = save_success
            
            if not save_success:
                raise RuntimeError("Failed to save model")
            
            # Phase 6: Final Summary
            self.logger.info("\nTRAINING COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 70)
            
            self._log_final_summary()
            
            return self.training_results
            
        except Exception as e:
            self.logger.error(f"\nTraining pipeline failed: {e}")
            self.training_results['error'] = str(e)
            raise
    
    def _log_final_summary(self):
        """Log final training summary"""
        cv_results = self.training_results.get('cross_validation', {})
        model_info = self.training_results.get('model_info', {})
        feature_info = self.training_results.get('feature_info', {})
        data_metadata = self.training_results.get('data_metadata', {})
        
        self.logger.info(f"Final Model Performance:")
        self.logger.info(f"   - Mean CV Accuracy: {cv_results.get('mean_accuracy', 0):.2f}%")
        self.logger.info(f"   - Standard Deviation: {cv_results.get('std_accuracy', 0):.2f}%")
        self.logger.info(f"   - Features used: {feature_info.get('total_features', 0)}")
        self.logger.info(f"   - Model type: {model_info.get('algorithm', 'Unknown')}")
        self.logger.info(f"   - Training samples: {data_metadata.get('total_images', 0)}")
        self.logger.info(f"   - Number of classes: {data_metadata.get('num_classes', 0)}")
        self.logger.info(f"   - Supported ethnicities: {list(data_metadata.get('label_map', {}).values())}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline"""
        return {
            'pipeline_type': 'EthnicityTrainingPipeline',
            'components': {
                'data_loader': type(self.data_loader).__name__,
                'preprocessors': len(self.preprocessing_pipeline.preprocessors),
                'feature_extractors': len(self.feature_extractor.extractors),
                'model_trainer': type(self.model_trainer).__name__,
                'model_saver': type(self.model_saver).__name__
            },
            'configuration': self.config.to_dict(),
            'training_results': self.training_results
        }


class PipelineFactory:
    """Factory for creating training pipelines"""
    
    @staticmethod
    def create_pipeline(logger: ILogger = None, 
                       progress_tracker: IProgressTracker = None,
                       custom_config: Dict[str, Any] = None) -> ITrainingPipeline:
        """
        Create training pipeline instance
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
            custom_config: Custom configuration overrides (optional)
            
        Returns:
            Training pipeline instance
        """
        if logger is None:
            from .utils import TrainingLogger
            logger = TrainingLogger('pipeline_factory')
        
        return EthnicityTrainingPipeline(logger, progress_tracker, custom_config)
