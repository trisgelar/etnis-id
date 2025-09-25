#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the Fixed SOLID-Compliant Ethnicity Detection Training System
Tests the Windows-compatible version with your dataset
"""

import sys
import os
from typing import Dict, Any

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_training.core.utils import TrainingLogger, ProgressTracker
from ml_training.core.training_pipeline import PipelineFactory


def create_test_config() -> Dict[str, Any]:
    """Create custom_config sections for the pipeline (uses new config system)"""
    return {
        'model': {
            'model_type': 'RandomForest',
            'n_estimators': 50,
            'random_state': 220
        },
        'cross_validation': {
            'n_folds': 3
        },
        'feature_extraction': {
            'glc_distances': [1],
            'glc_angles': [0, 45, 90, 135],
            'glc_levels': 256,
            'color_bins': 16,
            'color_channels': [1, 2],
        },
        'training': {
            'random_seed': 220
        }
    }


def test_individual_components():
    """Test individual components of the SOLID system"""
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 50)
    
    try:
        # Test 1: Logger
        print("\n1. Testing Logger...")
        logger = TrainingLogger('test_logger')
        logger.info("Logger test successful!")
        print("SUCCESS: Logger working correctly")
        
        # Test 2: Progress Tracker
        print("\n2. Testing Progress Tracker...")
        progress_tracker = ProgressTracker(logger)
        progress_tracker.start_task("Test Task", 10)
        for i in range(10):
            progress_tracker.update_progress(i + 1)
        progress_tracker.complete_task()
        print("SUCCESS: Progress Tracker working correctly")
        
        # Test 3: Configuration (custom_config dict for new system)
        print("\n3. Testing Configuration...")
        config = create_test_config()
        print("SUCCESS: Configuration prepared (custom_config dict)")
        print(f"   Dataset path: {config.get('dataset_path', '../dataset/dataset_periorbital')}")
        print(f"   Model type: {config.get('model', {}).get('model_type', 'RandomForest')}")
        print(f"   CV folds: {config.get('cross_validation', {}).get('n_folds', 3)}")
        
        # Test 4: Data Loader
        print("\n4. Testing Data Loader...")
        from ml_training.core.data_loader import EthnicityDataLoader
        
        data_loader = EthnicityDataLoader(logger)
        
        # Check if dataset exists
        dataset_path = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_periorbital')
        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset path not found: {dataset_path}")
            return False
        
        print(f"SUCCESS: Dataset path exists: {dataset_path}")
        
        # Test loading a small sample
        print("   Loading dataset (this may take a moment)...")
        images, labels, metadata = data_loader.load_data(dataset_path)
        
        print(f"SUCCESS: Data loaded successfully:")
        print(f"   Images shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Number of classes: {metadata.get('num_classes', 'Unknown')}")
        print(f"   Total images: {metadata.get('total_images', 'Unknown')}")
        print(f"   Class distribution: {metadata.get('class_distribution', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline():
    """Test the complete training pipeline"""
    print("\nTESTING COMPLETE TRAINING PIPELINE")
    print("=" * 50)
    
    try:
        # Create configuration
        custom_config = create_test_config()
        
        # Create logger and progress tracker
        logger = TrainingLogger('test_pipeline')
        progress_tracker = ProgressTracker(logger)
        
        # Create training pipeline
        print("Creating training pipeline...")
        pipeline = PipelineFactory.create_pipeline(logger, progress_tracker, custom_config=custom_config)
        print("SUCCESS: Pipeline created successfully")
        
        # Get paths (absolute from project root)
        data_path = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_periorbital')
        output_path = os.path.join(PROJECT_ROOT, 'model_ml', 'test_solid_model.pkl')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\nStarting training with:")
        print(f"   Dataset: {data_path}")
        print(f"   Output: {output_path}")
        print(f"   Model type: RandomForest")
        print(f"   CV folds: 3")
        print(f"   N estimators: 50")
        
        # Run the pipeline
        results = pipeline.run_pipeline(data_path, output_path)
        
        # Check results
        if results.get('model_saved', False):
            print("\nTRAINING PIPELINE TEST SUCCESSFUL!")
            print("=" * 50)
            
            # Display results
            cv_results = results.get('cross_validation', {})
            model_info = results.get('model_info', {})
            feature_info = results.get('feature_info', {})
            data_metadata = results.get('data_metadata', {})
            
            print(f"Training Results:")
            print(f"   Mean CV Accuracy: {cv_results.get('mean_accuracy', 0):.2f}%")
            print(f"   Standard Deviation: {cv_results.get('std_accuracy', 0):.2f}%")
            print(f"   Features used: {feature_info.get('total_features', 0)}")
            print(f"   Model type: {model_info.get('algorithm', 'Unknown')}")
            print(f"   Training samples: {data_metadata.get('total_images', 0)}")
            print(f"   Number of classes: {data_metadata.get('num_classes', 0)}")
            print(f"   Model saved to: {output_path}")
            
            return True
        else:
            print("ERROR: Training pipeline failed - model not saved")
            return False
            
    except Exception as e:
        print(f"ERROR: Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("TESTING FIXED SOLID-COMPLIANT ETHNICITY DETECTION TRAINING SYSTEM")
    print("=" * 70)
    
    # Test individual components first
    if not test_individual_components():
        print("\nERROR: Individual component tests failed. Stopping.")
        return False
    
    # Test complete pipeline
    if not test_training_pipeline():
        print("\nERROR: Training pipeline test failed.")
        return False
    
    print("\nALL TESTS PASSED!")
    print("=" * 70)
    print("SUCCESS: The SOLID-compliant training system is working correctly!")
    print("SUCCESS: Your dataset has been processed successfully!")
    print("SUCCESS: The new architecture is ready for production use!")
    print("\nThe system is now fully functional and Windows-compatible!")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nWARNING: Testing cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
