#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Extractors Implementation
Single Responsibility Principle - Each extractor handles one type of feature
Open/Closed Principle - Easy to extend with new feature extractors
"""

import numpy as np
import cv2
from abc import ABC
from typing import Dict, Any
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import shannon_entropy
from .interfaces import IFeatureExtractor, ILogger, IProgressTracker
from .config import get_feature_config


class BaseFeatureExtractor(IFeatureExtractor, ABC):
    """Base class for feature extractors"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None):
        """
        Initialize base feature extractor
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
        """
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.feature_info = {}
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature extraction information"""
        return self.feature_info.copy()


class GLCFeatureExtractor(BaseFeatureExtractor):
    """GLCM (Gray Level Co-occurrence Matrix) Feature Extractor"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None,
                 distances: list = None, angles: list = None, levels: int = None):
        """
        Initialize GLCM feature extractor
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
            distances: List of distances for GLCM (uses config if None)
            angles: List of angles for GLCM (uses config if None)
            levels: Number of gray levels (uses config if None)
        """
        super().__init__(logger, progress_tracker)
        
        # Get configuration
        config = get_feature_config()
        
        # Use configuration values if not provided and ensure numeric types
        raw_distances = distances if distances is not None else config.glc_distances
        # Convert to integers when provided as strings
        self.distances = [int(d) for d in raw_distances]
        self.levels = int(levels) if levels is not None else int(config.glc_levels)
        
        # Convert angles from degrees to radians if needed
        if angles is None:
            angles = config.glc_angles
        
        # Convert degrees to radians if angles are in degrees; coerce from strings if needed
        self.angles = []
        for angle in angles:
            try:
                numeric_angle = float(angle)
            except (TypeError, ValueError):
                # Fallback: skip invalid items
                continue
            if numeric_angle <= 2 * np.pi:  # Assume radians if <= 2π
                self.angles.append(numeric_angle)
            else:
                self.angles.append(np.radians(numeric_angle))
        
        self.feature_info = {
            'type': 'GLCM',
            'description': 'Gray Level Co-occurrence Matrix features',
            'distances': self.distances,
            'angles': self.angles,
            'levels': self.levels,
            'num_angles': len(self.angles),
            'haralick_features': ['contrast', 'homogeneity', 'correlation', 'energy']
        }
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract GLCM features from grayscale images
        
        Args:
            images: Grayscale images array (N, H, W)
            
        Returns:
            GLCM features array (N, num_features)
        """
        self.logger.info("Extracting GLCM features...")
        
        if self.progress_tracker:
            self.progress_tracker.start_task("GLCM Feature Extraction", len(images))
        
        features = []
        
        for i, img in enumerate(images):
            if i % 100 == 0 and i > 0:  # Progress indicator for large datasets
                self.logger.debug(f"   Processing image {i+1}/{len(images)}")
            
            # Resize image if too large (for efficiency)
            if img.shape[0] > 256 or img.shape[1] > 256:
                img = cv2.resize(img, (256, 256))
            
            # Extract GLCM features
            img_features = self._extract_glcm_features(img)
            features.append(img_features)
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        result = np.array(features)
        self.logger.info(f"GLCM features extracted: {len(result)} samples, {len(result[0])} features each")
        
        # Update feature info
        self.feature_info.update({
            'extracted_count': len(result),
            'feature_dimension': len(result[0]) if len(result) > 0 else 0,
            'input_shape': images.shape,
            'output_shape': result.shape
        })
        
        return result
    
    def _extract_glcm_features(self, img: np.ndarray) -> np.ndarray:
        """Extract GLCM features from single image"""
        try:
            # Calculate GLCM
            glcm = graycomatrix(
                img,
                distances=self.distances,
                angles=self.angles,
                levels=self.levels,
                symmetric=True,
                normed=True
            )
            
            # Extract Haralick features (matching notebook: energy instead of ASM)
            properties = ['contrast', 'homogeneity', 'correlation', 'energy']
            haralick_features = []
            
            for prop in properties:
                feature_values = graycoprops(glcm, prop).ravel()
                haralick_features.extend(feature_values)
            
            # Extract entropy for each angle (average across distances)
            entropy_features = []
            for j in range(len(self.angles)):
                # Average GLCM across distances for angle j
                P_avg = np.mean(glcm[:, :, :, j], axis=2)
                entropy_val = shannon_entropy(P_avg)
                entropy_features.append(entropy_val)
            
            # Combine all features
            all_features = np.concatenate([haralick_features, entropy_features])
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"Error extracting GLCM features: {e}")
            # Return zero features if extraction fails
            num_haralick = len(self.distances) * len(self.angles) * 4  # 4 properties
            num_entropy = len(self.angles)
            return np.zeros(num_haralick + num_entropy)


class ColorHistogramFeatureExtractor(BaseFeatureExtractor):
    """Color Histogram Feature Extractor"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None,
                 bins: int = None, channels: list = None):
        """
        Initialize Color Histogram feature extractor
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
            bins: Number of histogram bins (uses config if None)
            channels: List of channels to extract (uses config if None)
        """
        super().__init__(logger, progress_tracker)
        
        # Get configuration
        config = get_feature_config()
        
        # Use configuration values if not provided and ensure numeric types
        self.bins = int(bins) if bins is not None else int(config.color_bins)
        raw_channels = channels if channels is not None else config.color_channels
        self.channels = [int(c) for c in raw_channels]
        
        self.feature_info = {
            'type': 'ColorHistogram',
            'description': 'Color histogram features from HSV channels',
            'bins': self.bins,
            'channels': self.channels,
            'channels_description': {1: 'Saturation', 2: 'Value'} if channels == [1, 2] else 'Custom channels'
        }
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features from HSV images
        
        Args:
            images: HSV images array (N, H, W, 3)
            
        Returns:
            Color histogram features array (N, num_features)
        """
        self.logger.info("Extracting Color Histogram features...")
        
        if self.progress_tracker:
            self.progress_tracker.start_task("Color Feature Extraction", len(images))
        
        features = []
        
        for i, img in enumerate(images):
            if i % 100 == 0 and i > 0:  # Progress indicator for large datasets
                self.logger.debug(f"   Processing image {i+1}/{len(images)}")
            
            # Extract color histogram features
            img_features = self._extract_color_features(img)
            features.append(img_features)
            
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        
        result = np.array(features)
        self.logger.info(f"Color features extracted: {len(result)} samples, {len(result[0])} features each")
        
        # Update feature info
        self.feature_info.update({
            'extracted_count': len(result),
            'feature_dimension': len(result[0]) if len(result) > 0 else 0,
            'input_shape': images.shape,
            'output_shape': result.shape
        })
        
        return result
    
    def _extract_color_features(self, img: np.ndarray) -> np.ndarray:
        """Extract color histogram features from single image"""
        try:
            features = []
            
            # Extract histogram for each specified channel
            for channel in self.channels:
                hist = cv2.calcHist([img], [channel], None, [self.bins], [0, 256])
                # Normalize histogram to sum to 1 (L1), matching notebook scaling
                if hist.sum() > 0:
                    hist = hist / hist.sum()
                features.extend(hist.flatten())
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting color features: {e}")
            # Return zero features if extraction fails
            return np.zeros(self.bins * len(self.channels))


class CombinedFeatureExtractor:
    """Combines multiple feature extractors"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None):
        """
        Initialize combined feature extractor
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
        """
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.extractors = []
    
    def add_extractor(self, extractor: IFeatureExtractor) -> 'CombinedFeatureExtractor':
        """
        Add feature extractor
        
        Args:
            extractor: Feature extractor to add
            
        Returns:
            Self for method chaining
        """
        self.extractors.append(extractor)
        return self
    
    def extract_features(self, preprocessed_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract features using all extractors and combine results
        
        Args:
            preprocessed_data: Dictionary with preprocessed data for each extractor
            
        Returns:
            Combined features array
        """
        self.logger.info(f"Combining features from {len(self.extractors)} extractors...")
        
        all_features = []
        
        for i, extractor in enumerate(self.extractors):
            extractor_name = extractor.__class__.__name__
            self.logger.info(f"Extractor {i+1}/{len(self.extractors)}: {extractor_name}")
            
            # Get appropriate preprocessed data for this extractor
            if hasattr(extractor, '_get_preprocessed_data'):
                data_key = extractor._get_preprocessed_data()
            else:
                # Default mapping based on extractor type
                if 'GLCM' in extractor_name or 'LBP' in extractor_name:
                    data_key = 'glcm'  # Both GLCM and LBP need grayscale images
                elif 'Color' in extractor_name:
                    data_key = 'color'
                else:
                    data_key = list(preprocessed_data.keys())[i % len(preprocessed_data)]
            
            if data_key not in preprocessed_data:
                self.logger.error(f"Required data key '{data_key}' not found in preprocessed data")
                raise KeyError(f"Required data key '{data_key}' not found")
            
            # Extract features
            features = extractor.extract_features(preprocessed_data[data_key])
            all_features.append(features)
            
            self.logger.info(f"   {extractor_name}: {features.shape[1]} features")
        
        # Combine all features
        if len(all_features) > 1:
            combined_features = np.concatenate(all_features, axis=1)
        else:
            combined_features = all_features[0]
        
        self.logger.info(f"Combined features: {combined_features.shape[0]} samples, {combined_features.shape[1]} total features")
        
        return combined_features
    
    def get_combined_feature_info(self) -> Dict[str, Any]:
        """Get information about all extractors and combined features"""
        extractor_info = []
        total_features = 0
        
        for extractor in self.extractors:
            info = extractor.get_feature_info()
            extractor_info.append(info)
            total_features += info.get('feature_dimension', 0)
        
        return {
            'num_extractors': len(self.extractors),
            'total_features': total_features,
            'extractors': extractor_info
        }


class LBPFeatureExtractor(BaseFeatureExtractor):
    """Local Binary Pattern (LBP) Feature Extractor"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None,
                 P: int = 8, R: float = 1.0, method: str = 'uniform', bins: int = None):
        super().__init__(logger, progress_tracker)
        # For 'uniform', histogram length is P + 2
        self.P = int(P)
        self.R = float(R)
        self.method = method
        self.bins = int(bins) if bins is not None else (self.P + 2)
        self.feature_info = {
            'type': 'LBP',
            'description': 'Local Binary Pattern histogram features',
            'P': self.P,
            'R': self.R,
            'method': self.method,
            'bins': self.bins,
            'feature_dimension': self.bins
        }
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        self.logger.info("Extracting LBP features...")
        if self.progress_tracker:
            self.progress_tracker.start_task("LBP Feature Extraction", len(images))
        features = []
        for i, img in enumerate(images):
            # Expect grayscale images (H, W)
            lbp = local_binary_pattern(img, self.P, self.R, method=self.method)
            hist, _ = np.histogram(lbp.ravel(), bins=self.bins, range=(0, self.bins), density=True)
            features.append(hist.astype(np.float32))
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        result = np.vstack(features)
        self.feature_info.update({
            'extracted_count': len(result),
            'input_shape': images.shape,
            'output_shape': result.shape
        })
        return result


class HOGFeatureExtractor(BaseFeatureExtractor):
    """Histogram of Oriented Gradients (HOG) Feature Extractor"""
    
    def __init__(self, logger: ILogger, progress_tracker: IProgressTracker = None,
                 pixels_per_cell: tuple = (8, 8), cells_per_block: tuple = (2, 2), orientations: int = 9):
        super().__init__(logger, progress_tracker)
        self.pixels_per_cell = tuple(pixels_per_cell)
        self.cells_per_block = tuple(cells_per_block)
        self.orientations = int(orientations)
        self.feature_info = {
            'type': 'HOG',
            'description': 'Histogram of Oriented Gradients features',
            'pixels_per_cell': self.pixels_per_cell,
            'cells_per_block': self.cells_per_block,
            'orientations': self.orientations
        }
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        self.logger.info("Extracting HOG features...")
        if self.progress_tracker:
            self.progress_tracker.start_task("HOG Feature Extraction", len(images))
        features = []
        for i, img in enumerate(images):
            # Expect grayscale images (H, W)
            feat = hog(img, orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm='L2-Hys', feature_vector=True)
            features.append(feat.astype(np.float32))
            if self.progress_tracker:
                self.progress_tracker.update_progress(i + 1)
        if self.progress_tracker:
            self.progress_tracker.complete_task()
        result = np.vstack(features)
        # Record feature dimension after first extraction
        if result.size > 0:
            self.feature_info.update({'feature_dimension': result.shape[1]})
        self.feature_info.update({
            'extracted_count': len(result),
            'input_shape': images.shape,
            'output_shape': result.shape
        })
        return result
