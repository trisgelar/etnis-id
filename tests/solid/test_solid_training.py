#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for SOLID-Compliant Ethnicity Detection Training System
Tests the new architecture with the actual dataset
"""

import sys
import os
from typing import Dict, Any

# Add ml_training core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_training', 'core'))

from ml_training.core.utils import TrainingConfig, TrainingLogger, ProgressTracker
from ml_training.core.training_pipeline import PipelineFactory


def create_test_config() -> TrainingConfig:
    """Create test configuration for your dataset"""
    config_dict = {
        # Use your actual dataset path
        'dataset_path': 'dataset/dataset_periorbital',
        'model_output_path': 'model_ml/test_solid_model.pkl',
        'model_type': 'random_forest',
        'cv_folds': 5,  # Reduced for faster testing
        'random_state': 220,
        'n_estimators': 100,  # Reduced for faster testing
        'image_size': (400, 200),
        'glcm_distances': [1],
        'glcm_angles': [0, 3.14159/4, 3.14159/2, 3*3.14159/4],  # 0, 45, 90, 135 degrees
        'glcm_levels': 256,
        'color_bins': 16,
        'color_channels': [1, 2],  # S and V channels for HSV
        'log_file': 'test_training.log'
    }
    return TrainingConfig(config_dict)


def test_individual_components():
    """Test individual components of the SOLID system"""
    print("🧪 TESTING INDIVIDUAL COMPONENTS")
    print("=" * 50)
    
    try:
        # Test 1: Logger
        print("\n1️⃣ Testing Logger...")
        logger = TrainingLogger('test_logger')
        logger.info("Logger test successful!")
        print("✅ Logger working correctly")
        
        # Test 2: Progress Tracker
        print("\n2️⃣ Testing Progress Tracker...")
        progress_tracker = ProgressTracker(logger)
        progress_tracker.start_task("Test Task", 10)
        for i in range(10):
            progress_tracker.update_progress(i + 1)
        progress_tracker.complete_task()
        print("✅ Progress Tracker working correctly")
        
        # Test 3: Configuration
        print("\n3️⃣ Testing Configuration...")
        config = create_test_config()
        if config.validate():
            print("✅ Configuration valid")
            print(f"   Dataset path: {config.get('dataset_path')}")
            print(f"   Model type: {config.get('model_type')}")
            print(f"   CV folds: {config.get('cv_folds')}")
        else:
            print("❌ Configuration invalid")
            return False
        
        # Test 4: Data Loader
        print("\n4️⃣ Testing Data Loader...")
        from ml_training.core.data_loader import EthnicityDataLoader
        
        data_loader = EthnicityDataLoader(logger)
        
        # Check if dataset exists
        dataset_path = config.get('dataset_path')
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return False
        
        print(f"✅ Dataset path exists: {dataset_path}")
        
        # Test loading a small sample
        print("   Loading dataset (this may take a moment)...")
        images, labels, metadata = data_loader.load_data(dataset_path)
        
        print(f"✅ Data loaded successfully:")
        print(f"   Images shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Number of classes: {metadata.get('num_classes', 'Unknown')}")
        print(f"   Total images: {metadata.get('total_images', 'Unknown')}")
        print(f"   Class distribution: {metadata.get('class_distribution', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline():
    """Test the complete training pipeline"""
    print("\n🚀 TESTING COMPLETE TRAINING PIPELINE")
    print("=" * 50)
    
    try:
        # Create configuration
        config = create_test_config()
        
        # Create logger and progress tracker
        logger = TrainingLogger('test_pipeline', config.get('log_file'))
        progress_tracker = ProgressTracker(logger)
        
        # Create training pipeline
        print("Creating training pipeline...")
        pipeline = PipelineFactory.create_pipeline(config, logger, progress_tracker)
        print("✅ Pipeline created successfully")
        
        # Get paths
        data_path = config.get('dataset_path')
        output_path = config.get('model_output_path')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n📊 Starting training with:")
        print(f"   Dataset: {data_path}")
        print(f"   Output: {output_path}")
        print(f"   Model type: {config.get('model_type')}")
        print(f"   CV folds: {config.get('cv_folds')}")
        
        # Run the pipeline
        results = pipeline.run_pipeline(data_path, output_path)
        
        # Check results
        if results.get('model_saved', False):
            print("\n🎉 TRAINING PIPELINE TEST SUCCESSFUL!")
            print("=" * 50)
            
            # Display results
            cv_results = results.get('cross_validation', {})
            model_info = results.get('model_info', {})
            feature_info = results.get('feature_info', {})
            data_metadata = results.get('data_metadata', {})
            
            print(f"📊 Training Results:")
            print(f"   Mean CV Accuracy: {cv_results.get('mean_accuracy', 0):.2f}%")
            print(f"   Standard Deviation: {cv_results.get('std_accuracy', 0):.2f}%")
            print(f"   Features used: {feature_info.get('total_features', 0)}")
            print(f"   Model type: {model_info.get('algorithm', 'Unknown')}")
            print(f"   Training samples: {data_metadata.get('total_images', 0)}")
            print(f"   Number of classes: {data_metadata.get('num_classes', 0)}")
            
            return True
        else:
            print("❌ Training pipeline failed - model not saved")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🧪 TESTING SOLID-COMPLIANT ETHNICITY DETECTION TRAINING SYSTEM")
    print("=" * 70)
    
    # Test individual components first
    if not test_individual_components():
        print("\n❌ Individual component tests failed. Stopping.")
        return False
    
    # Test complete pipeline
    if not test_training_pipeline():
        print("\n❌ Training pipeline test failed.")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print("✅ The SOLID-compliant training system is working correctly!")
    print("✅ Your dataset has been processed successfully!")
    print("✅ The new architecture is ready for production use!")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Testing cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
