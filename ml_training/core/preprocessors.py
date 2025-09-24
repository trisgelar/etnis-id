#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Preprocessors Implementation
Single Responsibility Principle - Each preprocessor handles one type of preprocessing
Open/Closed Principle - Easy to extend with new preprocessors
"""

import cv2
import numpy as np
from abc import ABC
from typing import Dict, Any
from .interfaces import IImagePreprocessor, ILogger, IProgressTracker
from .config import get_feature_config


class BasePreprocessor(IImagePreprocessor, ABC):
    """Base class for image preprocessors"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None):
        """
        Initialize base preprocessor
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
        """
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.preprocessing_info = {}
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get preprocessing information"""
        return self.preprocessing_info.copy()


class GLCMPreprocessor(BasePreprocessor):
    """Preprocessor for GLCM feature extraction - converts RGB to Grayscale"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None):
        """Initialize GLCM preprocessor"""
        super().__init__(logger, progress_tracker)
        self.preprocessing_info = {
            'type': 'GLCM',
            'description': 'RGB to Grayscale conversion for texture analysis',
            'output_channels': 1,
            'color_space': 'GRAYSCALE'
        }
    
    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """
        Convert RGB images to grayscale for GLCM analysis
        
        Args:
            images: Input RGB images array (N, H, W, 3)
            
        Returns:
            Grayscale images array (N, H, W)
        """
        self.logger.info("GLCM Preprocessing: Converting RGB to Grayscale...")
        
        if self.progress_tracker:
            self.progress_tracker.start_task("GLCM Preprocessing", len(images))
        
        grayscale_images = []
        
        for i, img in enumerate(images):
            # Convert BGR to Grayscale (OpenCV uses BGR by default)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grayscale_images.append(gray_img)
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        result = np.array(grayscale_images)
        self.logger.info(f"GLCM preprocessing complete: {len(result)} images converted to grayscale")
        
        # Update preprocessing info
        self.preprocessing_info.update({
            'processed_count': len(result),
            'input_shape': images.shape,
            'output_shape': result.shape
        })
        
        return result


class ColorHistogramPreprocessor(BasePreprocessor):
    """Preprocessor for Color Histogram feature extraction - converts RGB to HSV"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None):
        """Initialize Color Histogram preprocessor"""
        super().__init__(logger, progress_tracker)
        
        # Get configuration
        config = get_feature_config()
        
        self.preprocessing_info = {
            'type': 'ColorHistogram',
            'description': f'RGB to {config.color_space} conversion for color analysis',
            'output_channels': 3,
            'color_space': config.color_space
        }
        
        # Store configuration for reference
        self.config = config
    
    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """
        Convert RGB images to HSV for color histogram analysis
        
        Args:
            images: Input RGB images array (N, H, W, 3)
            
        Returns:
            HSV images array (N, H, W, 3)
        """
        self.logger.info("Color Preprocessing: Converting RGB to HSV...")
        
        if self.progress_tracker:
            self.progress_tracker.start_task("Color Preprocessing", len(images))
        
        hsv_images = []
        
        for i, img in enumerate(images):
            # Convert BGR to HSV (OpenCV uses BGR by default)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_images.append(hsv_img)
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        result = np.array(hsv_images)
        self.logger.info(f"Color preprocessing complete: {len(result)} images converted to HSV")
        
        # Update preprocessing info
        self.preprocessing_info.update({
            'processed_count': len(result),
            'input_shape': images.shape,
            'output_shape': result.shape
        })
        
        return result


class ResizePreprocessor(BasePreprocessor):
    """Preprocessor for image resizing"""
    
    def __init__(self, target_size: tuple, logger: ILogger, progress_tracker: IProgressTracker = None):
        """
        Initialize resize preprocessor
        
        Args:
            target_size: Target size (width, height)
            logger: Logger instance
            progress_tracker: Progress tracker instance
        """
        super().__init__(logger, progress_tracker)
        self.target_size = target_size
        self.preprocessing_info = {
            'type': 'Resize',
            'description': f'Resize images to {target_size}',
            'target_size': target_size,
            'interpolation': cv2.INTER_AREA
        }
    
    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """
        Resize images to target size
        
        Args:
            images: Input images array
            
        Returns:
            Resized images array
        """
        self.logger.info(f"Resizing images to {self.target_size}...")
        
        if self.progress_tracker:
            self.progress_tracker.start_task("Image Resizing", len(images))
        
        resized_images = []
        
        for i, img in enumerate(images):
            # Resize image
            resized_img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            resized_images.append(resized_img)
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        result = np.array(resized_images)
        self.logger.info(f"Resizing complete: {len(result)} images resized")
        
        # Update preprocessing info
        self.preprocessing_info.update({
            'processed_count': len(result),
            'input_shape': images.shape,
            'output_shape': result.shape
        })
        
        return result


class PreprocessingPipeline:
    """Pipeline for chaining multiple preprocessors"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None):
        """
        Initialize preprocessing pipeline
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
        """
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.preprocessors = []
    
    def add_preprocessor(self, preprocessor: IImagePreprocessor) -> 'PreprocessingPipeline':
        """
        Add preprocessor to pipeline
        
        Args:
            preprocessor: Preprocessor to add
            
        Returns:
            Self for method chaining
        """
        self.preprocessors.append(preprocessor)
        return self
    
    def process(self, images: np.ndarray) -> np.ndarray:
        """
        Process images through all preprocessors in sequence
        
        Args:
            images: Input images
            
        Returns:
            Processed images
        """
        self.logger.info(f"Starting preprocessing pipeline with {len(self.preprocessors)} steps...")
        
        current_images = images
        
        for i, preprocessor in enumerate(self.preprocessors):
            self.logger.info(f"Step {i+1}/{len(self.preprocessors)}: {preprocessor.__class__.__name__}")
            current_images = preprocessor.preprocess(current_images)
        
        self.logger.info("Preprocessing pipeline completed")
        return current_images
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about all preprocessors in pipeline"""
        return {
            'num_preprocessors': len(self.preprocessors),
            'preprocessors': [p.get_preprocessing_info() for p in self.preprocessors]
        }
