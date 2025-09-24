#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Entry Point for Ethnicity Model Training
Clean, SOLID-compliant training system
"""

import sys
import os
from typing import Dict, Any

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.utils import TrainingConfig, TrainingLogger, ProgressTracker
from core.training_pipeline import PipelineFactory


def create_default_config() -> TrainingConfig:
    """Create default training configuration"""
    config_dict = {
        'dataset_path': 'dataset_periorbital',
        'model_output_path': 'model_ml/pickle_model.pkl',
        'model_type': 'random_forest',
        'cv_folds': 6,
        'random_state': 220,
        'n_estimators': 200,
        'image_size': (400, 200),
        'glcm_distances': [1],
        'glcm_angles': [0, 3.14159/4, 3.14159/2, 3*3.14159/4],  # 0, 45, 90, 135 degrees
        'glcm_levels': 256,
        'color_bins': 16,
        'color_channels': [1, 2],  # S and V channels for HSV
        'log_file': 'training.log'
    }
    return TrainingConfig(config_dict)


def main():
    """Main training function"""
    print("INDONESIAN ETHNICITY DETECTION - SOLID TRAINING SYSTEM")
    print("=" * 70)
    
    try:
        # Create configuration
        config = create_default_config()
        
        # Validate configuration
        if not config.validate():
            print("ERROR: Invalid configuration!")
            return False
        
        # Create logger
        logger = TrainingLogger('ethnicity_training', config.get('log_file'))
        
        # Create progress tracker
        progress_tracker = ProgressTracker(logger)
        
        # Create training pipeline
        pipeline = PipelineFactory.create_pipeline(config, logger, progress_tracker)
        
        # Get paths from config
        data_path = config.get('dataset_path')
        output_path = config.get('model_output_path')
        
        # Check if dataset exists
        if not os.path.exists(data_path):
            logger.error(f"Dataset directory '{data_path}' not found!")
            logger.info("\nTo use this training system:")
            logger.info("1. Create a dataset directory with the following structure:")
            logger.info("   dataset_periorbital/")
            logger.info("   ├── Bugis/")
            logger.info("   ├── Sunda/")
            logger.info("   ├── Malay/")
            logger.info("   ├── Jawa/")
            logger.info("   └── Banjar/")
            logger.info("2. Place face images in each ethnicity folder")
            logger.info("3. Run this script again")
            return False
        
        # Run training pipeline
        results = pipeline.run_pipeline(data_path, output_path)
        
        # Check if training was successful
        if results.get('model_saved', False):
            logger.info("\nTraining completed successfully!")
            logger.info(f"Model saved to: {output_path}")
            logger.info("Your ML server will automatically use the new model!")
            return True
        else:
            logger.error("Training failed!")
            return False
            
    except KeyboardInterrupt:
        print("\nWARNING: Training cancelled by user")
        return False
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        sys.exit(1)
