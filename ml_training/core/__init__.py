#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Module for Ethnicity Detection Training System
"""

from .interfaces import (
    IDataLoader, IImagePreprocessor, IFeatureExtractor, 
    IModelTrainer, IModelSaver, ITrainingPipeline, 
    ILogger, IProgressTracker
)

from .data_loader import EthnicityDataLoader
from .preprocessors import GLCMPreprocessor, ColorHistogramPreprocessor, PreprocessingPipeline
from .feature_extractors import GLCFeatureExtractor, ColorHistogramFeatureExtractor, CombinedFeatureExtractor
from .model_trainers import RandomForestTrainer, SVMTrainer, ModelFactory
from .utils import ModelSaver, TrainingLogger, ProgressTracker, TrainingConfig
from .training_pipeline import EthnicityTrainingPipeline, PipelineFactory

__all__ = [
    # Interfaces
    'IDataLoader', 'IImagePreprocessor', 'IFeatureExtractor',
    'IModelTrainer', 'IModelSaver', 'ITrainingPipeline',
    'ILogger', 'IProgressTracker',
    
    # Implementations
    'EthnicityDataLoader',
    'GLCMPreprocessor', 'ColorHistogramPreprocessor', 'PreprocessingPipeline',
    'GLCFeatureExtractor', 'ColorHistogramFeatureExtractor', 'CombinedFeatureExtractor',
    'RandomForestTrainer', 'SVMTrainer', 'ModelFactory',
    'ModelSaver', 'TrainingLogger', 'ProgressTracker', 'TrainingConfig',
    'EthnicityTrainingPipeline', 'PipelineFactory'
]
