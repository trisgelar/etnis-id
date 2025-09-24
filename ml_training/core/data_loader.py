#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loader Implementation
Single Responsibility Principle - Only handles data loading
"""

import os
import cv2
import numpy as np
from typing import Tuple, Dict, Any, List
from .interfaces import IDataLoader, ILogger
from .config import get_dataset_config


class EthnicityDataLoader(IDataLoader):
    """Concrete implementation for loading ethnicity detection data"""
    
    def __init__(self, logger: ILogger, supported_formats: List[str] = None):
        """
        Initialize data loader
        
        Args:
            logger: Logger instance
            supported_formats: List of supported image formats (uses config if None)
        """
        self.logger = logger
        
        # Get configuration
        config = get_dataset_config()
        
        # Use configuration values if not provided
        self.supported_formats = supported_formats or config.image_extensions
        
        # Create label map from configuration ethnicities
        self.ethnicities = config.ethnicities
        self.label_map = {i: ethnicity for i, ethnicity in enumerate(self.ethnicities)}
        
        # Store configuration for reference
        self.config = config
        
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load ethnicity data from directory structure
        
        Args:
            data_path: Path to dataset directory
            
        Returns:
            Tuple of (images, labels, metadata)
        """
        self.logger.info(f"Loading data from: {data_path}")
        
        if not os.path.exists(data_path):
            error_msg = f"Dataset directory '{data_path}' does not exist!"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        images, labels, paths, names, folders = self._load_images_from_directory(data_path)
        
        if len(images) == 0:
            error_msg = "No valid images found in dataset!"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate loaded data
        if not self.validate_data(images, labels):
            error_msg = "Loaded data validation failed!"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create metadata
        metadata = self._create_metadata(images, labels, paths, names, folders)
        
        self.logger.info(f"Successfully loaded {len(images)} images from {len(np.unique(labels))} classes")
        return images, labels, metadata
    
    def validate_data(self, data: np.ndarray, labels: np.ndarray) -> bool:
        """
        Validate loaded data
        
        Args:
            data: Image data array
            labels: Label array
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check basic structure
            if data is None or labels is None:
                self.logger.error("Data or labels is None")
                return False
            
            if len(data) != len(labels):
                self.logger.error(f"Data length ({len(data)}) != labels length ({len(labels)})")
                return False
            
            if len(data) == 0:
                self.logger.error("No data loaded")
                return False
            
            # Check image dimensions
            if len(data.shape) != 4 or data.shape[3] != 3:
                self.logger.error(f"Invalid image shape: {data.shape}")
                return False
            
            # Check label range
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                self.logger.error("Need at least 2 classes")
                return False
            
            if not all(0 <= label < len(self.label_map) for label in unique_labels):
                self.logger.error(f"Labels out of range: {unique_labels}")
                return False
            
            self.logger.debug("Data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False
    
    def _load_images_from_directory(self, data_path: str) -> Tuple[List, List, List, List, List]:
        """Load images from directory structure"""
        images = []
        labels = []
        paths = []
        names = []
        folders = []
        
        # Get class directories
        classes = sorted([d for d in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, d))])
        
        if not classes:
            self.logger.warning("No class directories found")
            return images, labels, paths, names, folders
        
        self.logger.info(f"Found classes: {classes}")
        
        # Load images from each class
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(data_path, class_name)
            class_images = self._load_images_from_class(class_path, class_name, class_idx)
            
            for img_data in class_images:
                images.append(img_data['image'])
                labels.append(img_data['label'])
                paths.append(img_data['path'])
                names.append(img_data['name'])
                folders.append(img_data['folder'])
        
        return images, labels, paths, names, folders
    
    def _load_images_from_class(self, class_path: str, class_name: str, class_idx: int) -> List[Dict]:
        """Load images from a single class directory"""
        class_images = []
        
        # Get all image files in class directory
        image_files = [f for f in os.listdir(class_path) 
                      if any(f.lower().endswith(ext) for ext in self.supported_formats)]
        
        self.logger.info(f"{class_name}: {len(image_files)} images")
        
        for img_name in image_files:
            img_path = os.path.join(class_path, img_name)
            
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    self.logger.warning(f"Could not load image: {img_path}")
                    continue
                
                # Resize to standard size
                image = cv2.resize(image, (400, 200))
                
                class_images.append({
                    'image': image,
                    'label': class_idx,
                    'path': img_path,
                    'name': img_name,
                    'folder': class_name
                })
                
            except Exception as e:
                self.logger.warning(f"Error loading {img_path}: {e}")
                continue
        
        return class_images
    
    def _create_metadata(self, images: np.ndarray, labels: np.ndarray, 
                        paths: List, names: List, folders: List) -> Dict[str, Any]:
        """Create metadata dictionary"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        class_distribution = {}
        for label, count in zip(unique_labels, counts):
            class_name = self.label_map.get(label, f"Class_{label}")
            class_distribution[class_name] = count
        
        return {
            'total_images': len(images),
            'num_classes': len(unique_labels),
            'class_distribution': class_distribution,
            'image_shape': images[0].shape if len(images) > 0 else None,
            'supported_formats': self.supported_formats,
            'label_map': self.label_map,
            'paths': paths,
            'names': names,
            'folders': folders
        }
