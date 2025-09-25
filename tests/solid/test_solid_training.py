#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for SOLID-Compliant Ethnicity Detection Training System
Tests the new architecture with the actual dataset
"""

import sys
import os
import time
import json
from typing import Dict, Any
from datetime import datetime

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_training.core.utils import TrainingLogger, ProgressTracker
from ml_training.core.training_pipeline import PipelineFactory


def create_test_config() -> Dict[str, Any]:
    """Create custom_config sections for the new config-based pipeline"""
    return {
        'model': {
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'random_state': 220,
        },
        'cross_validation': {
            'n_folds': 5,
        },
        'feature_extraction': {
            'glc_distances': [1],
            'glc_angles': [0, 45, 90, 135],
            'glc_levels': 256,
            'color_bins': 16,
            'color_channels': [1, 2],
        }
    }


def test_individual_components(timestamp: str):
    """Test individual components of the SOLID system"""
    print("🧪 TESTING INDIVIDUAL COMPONENTS")
    print("=" * 50)
    
    try:
        # Test 1: Logger
        print("\n1️⃣ Testing Logger...")
        logger = TrainingLogger(f'test_logger_{timestamp}')
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
        print("✅ Configuration prepared (custom_config dict)")
        print(f"   Model type: {config.get('model',{}).get('model_type')}")
        print(f"   CV folds: {config.get('cross_validation',{}).get('n_folds')}")
        
        # Test 4: Data Loader
        print("\n4️⃣ Testing Data Loader...")
        from ml_training.core.data_loader import EthnicityDataLoader
        
        data_loader = EthnicityDataLoader(logger)
        
        # Check if dataset exists
        dataset_path = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_periorbital')
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


def test_training_pipeline(timestamp: str, output_dir: str):
    """Test the complete training pipeline"""
    print("\n🚀 TESTING COMPLETE TRAINING PIPELINE")
    print("=" * 50)
    
    try:
        # Create configuration
        custom_config = create_test_config()
        
        # Create logger and progress tracker with dynamic naming
        logger = TrainingLogger(f'test_pipeline_{timestamp}')
        progress_tracker = ProgressTracker(logger)
        
        # Create training pipeline
        print("Creating training pipeline...")
        pipeline = PipelineFactory.create_pipeline(logger, progress_tracker, custom_config=custom_config)
        print("✅ Pipeline created successfully")
        
        # Get paths with dynamic naming
        data_path = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_periorbital')
        output_path = os.path.join(output_dir, f'test_solid_model_{timestamp}.pkl')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n📊 Starting training with:")
        print(f"   Dataset: {data_path}")
        print(f"   Output: {output_path}")
        print(f"   Model type: {custom_config.get('model',{}).get('model_type')}")
        print(f"   CV folds: {custom_config.get('cross_validation',{}).get('n_folds')}")
        
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
            
            # Phase 6: Hold-out Test Evaluation
            print("\n📈 PHASE 6: TEST EVALUATION RESULTS")
            print("=" * 50)
            test_eval = results.get('test_evaluation', {})
            if test_eval:
                print(f"   Test accuracy: {test_eval.get('test_accuracy', 0):.2f}%")
                print(f"   Train size: {test_eval.get('train_size', 0)}")
                print(f"   Test size: {test_eval.get('test_size', 0)}")
            else:
                print("   No test evaluation found in results.")

            # Phase 7: Analysis Artifacts Summary
            print("\n🗂️ PHASE 7: ANALYSIS ARTIFACTS (Saved in logs/analysis)")
            print("=" * 50)
            analysis_dir = os.path.join(PROJECT_ROOT, 'logs', 'analysis')
            print(f"   Directory: {analysis_dir}")
            print("   Files expected:")
            print("     - classification_report.txt (CV predictions)")
            print("     - confusion_matrix.png / confusion_matrix.csv (CV predictions)")
            print("     - classification_report_test.txt (hold-out test)")
            print("     - confusion_matrix_test.png / confusion_matrix_test.csv (hold-out test)")
            print("   Note: Previous low-confidence/overfit analysis artifacts are preserved for comparison.")
            
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
    
    # Create dynamic timestamp and output directory for reproducibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, 'logs', 'solid_training_test', f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Results will be saved to: {output_dir}")
    
    start_time = time.time()
    test_results = {
        'timestamp': timestamp,
        'output_dir': output_dir,
        'individual_components_test': False,
        'training_pipeline_test': False,
        'total_runtime_seconds': 0,
        'total_runtime_minutes': 0,
        'success': False
    }
    
    try:
        # Test individual components first
        print(f"\n📊 Progress: 1/2 - Testing individual components...")
        if not test_individual_components(timestamp):
            print("\n❌ Individual component tests failed. Stopping.")
            test_results['individual_components_test'] = False
            return False
        else:
            test_results['individual_components_test'] = True
            print("✅ Individual component tests passed!")
        
        # Test complete pipeline
        print(f"\n📊 Progress: 2/2 - Testing training pipeline...")
        if not test_training_pipeline(timestamp, output_dir):
            print("\n❌ Training pipeline test failed.")
            test_results['training_pipeline_test'] = False
            return False
        else:
            test_results['training_pipeline_test'] = True
            print("✅ Training pipeline test passed!")
        
        # Calculate total runtime
        total_time = time.time() - start_time
        test_results['total_runtime_seconds'] = total_time
        test_results['total_runtime_minutes'] = total_time / 60
        test_results['success'] = True
        
        # Save test results
        results_file = os.path.join(output_dir, f'solid_training_test_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print(f"⏱️  Total runtime: {total_time/60:.1f} minutes")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Test results: {results_file}")
        print("✅ The SOLID-compliant training system is working correctly!")
        print("✅ Your dataset has been processed successfully!")
        print("✅ The new architecture is ready for production use!")
        
        return True
        
    except Exception as e:
        # Save failed test results
        total_time = time.time() - start_time
        test_results['total_runtime_seconds'] = total_time
        test_results['total_runtime_minutes'] = total_time / 60
        test_results['error'] = str(e)
        
        results_file = os.path.join(output_dir, f'solid_training_test_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n💥 Test failed with error: {e}")
        return False


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
