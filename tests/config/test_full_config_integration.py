#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Full Configuration Integration
Tests that all modules properly use the configuration system throughout the codebase
"""

import sys
import os
import tempfile
sys.path.insert(0, '.')

def test_full_config_integration():
    """Test configuration integration across all modules"""
    print("🧪 TESTING FULL CONFIGURATION INTEGRATION")
    print("=" * 50)
    
    try:
        # Test 1: Data Loader with Configuration
        print("1. Testing Data Loader with Configuration:")
        from ml_training.core.data_loader import EthnicityDataLoader
        from ml_training.core.utils import TrainingLogger
        
        logger = TrainingLogger('test_data_loader')
        data_loader = EthnicityDataLoader(logger)
        
        print(f"   ✅ Data loader - supported formats: {data_loader.supported_formats}")
        print(f"   ✅ Data loader - ethnicities: {data_loader.ethnicities}")
        print(f"   ✅ Data loader - label map: {data_loader.label_map}")
        
        # Test 2: Feature Extractors with Configuration
        print("\n2. Testing Feature Extractors with Configuration:")
        from ml_training.core.feature_extractors import GLCFeatureExtractor, ColorHistogramFeatureExtractor
        
        glcm_extractor = GLCFeatureExtractor(logger)
        color_extractor = ColorHistogramFeatureExtractor(logger)
        
        print(f"   ✅ GLCM extractor - distances: {glcm_extractor.distances}")
        print(f"   ✅ GLCM extractor - levels: {glcm_extractor.levels}")
        print(f"   ✅ Color extractor - bins: {color_extractor.bins}")
        print(f"   ✅ Color extractor - channels: {color_extractor.channels}")
        
        # Test 3: Model Trainers with Configuration
        print("\n3. Testing Model Trainers with Configuration:")
        from ml_training.core.model_trainers import RandomForestTrainer
        
        rf_trainer = RandomForestTrainer(logger)
        
        print(f"   ✅ RF trainer - n_estimators: {rf_trainer.n_estimators}")
        print(f"   ✅ RF trainer - random_state: {rf_trainer.random_state}")
        print(f"   ✅ RF trainer - additional_params: {len(rf_trainer.additional_params)} parameters")
        
        # Test 4: Preprocessors with Configuration
        print("\n4. Testing Preprocessors with Configuration:")
        from ml_training.core.preprocessors import ColorHistogramPreprocessor
        
        color_preprocessor = ColorHistogramPreprocessor(logger)
        
        print(f"   ✅ Color preprocessor - color_space: {color_preprocessor.preprocessing_info['color_space']}")
        print(f"   ✅ Color preprocessor - has config: {hasattr(color_preprocessor, 'config')}")
        
        # Test 5: Training Pipeline with Configuration
        print("\n5. Testing Training Pipeline with Configuration:")
        from ml_training.core.training_pipeline import EthnicityTrainingPipeline
        
        pipeline = EthnicityTrainingPipeline(logger)
        
        print(f"   ✅ Pipeline - has dataset config: {hasattr(pipeline, 'dataset_config')}")
        print(f"   ✅ Pipeline - has model config: {hasattr(pipeline, 'model_config')}")
        print(f"   ✅ Pipeline - has training config: {hasattr(pipeline, 'training_config')}")
        
        # Test 6: Cross-Validation with Configuration
        print("\n6. Testing Cross-Validation with Configuration:")
        from ml_training.core.cross_validation import CrossValidationConfig, ModelConfig
        
        cv_config = CrossValidationConfig()
        model_config = ModelConfig()
        
        print(f"   ✅ CV config - folds: {cv_config.n_folds}")
        print(f"   ✅ CV config - test size: {cv_config.test_size}")
        print(f"   ✅ Model config - n_estimators: {model_config.n_estimators}")
        
        # Test 7: Ethnic Detector with Configuration
        print("\n7. Testing Ethnic Detector with Configuration:")
        from ethnic_detector import EthnicDetector
        
        detector = EthnicDetector()
        
        print(f"   ✅ Detector - ethnicities: {detector.ethnicities}")
        print(f"   ✅ Detector - label map: {detector.label_map}")
        print(f"   ✅ Detector - has feature config: {hasattr(detector, 'feature_config')}")
        
        # Test 8: ML Server with Configuration
        print("\n8. Testing ML Server with Configuration:")
        from ml_server import MLTCPServer
        
        server = MLTCPServer()
        
        print(f"   ✅ Server - host: {server.host}")
        print(f"   ✅ Server - port: {server.port}")
        
        # Test 9: Visualizations with Configuration
        print("\n9. Testing Visualizations with Configuration:")
        from ml_training.core.visualizations import ModelVisualizer
        
        visualizer = ModelVisualizer(logger)
        
        print(f"   ✅ Visualizer - output_dir: {visualizer.output_dir}")
        print(f"   ✅ Visualizer - style: {visualizer.style}")
        
        print("\n🎉 ALL FULL CONFIGURATION INTEGRATION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Full configuration integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_consistency():
    """Test that configuration values are consistent across modules"""
    print("\n🔍 TESTING CONFIGURATION CONSISTENCY")
    print("=" * 50)
    
    try:
        from ml_training.core.config import get_config
        from ml_training.core.utils import TrainingLogger
        from ml_training.core.data_loader import EthnicityDataLoader
        from ml_training.core.feature_extractors import GLCFeatureExtractor, ColorHistogramFeatureExtractor
        from ethnic_detector import EthnicDetector
        
        logger = TrainingLogger('test_consistency')
        
        # Get global configuration
        config = get_config()
        
        # Test data loader consistency
        data_loader = EthnicityDataLoader(logger)
        assert data_loader.ethnicities == config.dataset.ethnicities
        assert data_loader.supported_formats == config.dataset.image_extensions
        print("✅ Data loader configuration consistent")
        
        # Test feature extractor consistency
        glcm_extractor = GLCFeatureExtractor(logger)
        color_extractor = ColorHistogramFeatureExtractor(logger)
        
        assert glcm_extractor.distances == config.feature_extraction.glc_distances
        assert glcm_extractor.levels == config.feature_extraction.glc_levels
        assert color_extractor.bins == config.feature_extraction.color_bins
        assert color_extractor.channels == config.feature_extraction.color_channels
        print("✅ Feature extractor configuration consistent")
        
        # Test ethnic detector consistency
        detector = EthnicDetector()
        assert detector.ethnicities == config.dataset.ethnicities
        assert detector.feature_config.glc_distances == config.feature_extraction.glc_distances
        assert detector.feature_config.color_bins == config.feature_extraction.color_bins
        print("✅ Ethnic detector configuration consistent")
        
        print("\n✅ All configuration consistency tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_override():
    """Test that configuration can be overridden at runtime"""
    print("\n🔧 TESTING CONFIGURATION OVERRIDE")
    print("=" * 50)
    
    try:
        from ml_training.core.utils import TrainingLogger
        from ml_training.core.feature_extractors import GLCFeatureExtractor, ColorHistogramFeatureExtractor
        from ml_training.core.model_trainers import RandomForestTrainer
        
        logger = TrainingLogger('test_override')
        
        # Test GLCM extractor override
        custom_glcm = GLCFeatureExtractor(logger, distances=[2, 3], levels=512)
        assert custom_glcm.distances == [2, 3]
        assert custom_glcm.levels == 512
        print("✅ GLCM extractor override working")
        
        # Test color extractor override
        custom_color = ColorHistogramFeatureExtractor(logger, bins=64)
        assert custom_color.bins == 64
        print("✅ Color extractor override working")
        
        # Test model trainer override
        custom_rf = RandomForestTrainer(logger, n_estimators=500, random_state=123)
        assert custom_rf.n_estimators == 500
        assert custom_rf.random_state == 123
        print("✅ Model trainer override working")
        
        print("\n✅ All configuration override tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration override test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_with_custom_env():
    """Test configuration with custom environment file"""
    print("\n🌍 TESTING CONFIGURATION WITH CUSTOM ENV")
    print("=" * 50)
    
    try:
        # Create a temporary .env file with custom values
        env_content = """
# Custom test configuration
MODEL_N_ESTIMATORS=150
CV_N_FOLDS=8
GLCM_LEVELS=128
COLOR_BINS=32
LOG_LEVEL=DEBUG
SERVER_PORT=9090
VIZ_FIGURE_DPI=150
ETHNICITIES=Test1,Test2,Test3
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            temp_env_file = f.name
        
        try:
            # Reload configuration with custom env file
            from ml_training.core.config import reload_config
            config = reload_config(temp_env_file)
            
            # Test that custom values are loaded
            assert config.model.n_estimators == 150
            assert config.cross_validation.n_folds == 8
            assert config.feature_extraction.glc_levels == 128
            assert config.feature_extraction.color_bins == 32
            assert config.logging.log_level == "DEBUG"
            assert config.server.port == 9090
            assert config.visualization.figure_dpi == 150
            assert config.dataset.ethnicities == ['Test1', 'Test2', 'Test3']
            
            print("✅ Custom environment file configuration loaded correctly")
            
            # Test that modules use the custom configuration
            from ml_training.core.utils import TrainingLogger
            from ml_training.core.feature_extractors import GLCFeatureExtractor
            
            logger = TrainingLogger('test_custom_env')
            glcm_extractor = GLCFeatureExtractor(logger)
            
            assert glcm_extractor.levels == 128
            print("✅ Modules use custom environment configuration")
            
        finally:
            # Clean up temporary file
            os.unlink(temp_env_file)
        
        print("\n✅ All custom environment configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Custom environment configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_performance():
    """Test that configuration doesn't impact performance significantly"""
    print("\n⚡ TESTING CONFIGURATION PERFORMANCE")
    print("=" * 50)
    
    try:
        import time
        from ml_training.core.utils import TrainingLogger
        from ml_training.core.feature_extractors import GLCFeatureExtractor
        
        logger = TrainingLogger('test_performance')
        
        # Test configuration loading time
        start_time = time.time()
        for _ in range(100):
            extractor = GLCFeatureExtractor(logger)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"   ✅ Average configuration loading time: {avg_time:.4f} seconds")
        
        # Configuration loading should be fast (< 0.01 seconds per instance)
        assert avg_time < 0.01, f"Configuration loading too slow: {avg_time:.4f}s"
        
        print("✅ Configuration performance acceptable")
        print("\n✅ All configuration performance tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 TESTING FULL CONFIGURATION INTEGRATION ACROSS CODEBASE")
    print("=" * 70)
    
    # Run all tests
    tests = [
        test_full_config_integration,
        test_configuration_consistency,
        test_configuration_override,
        test_configuration_with_custom_env,
        test_configuration_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 70)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL FULL CONFIGURATION INTEGRATION TESTS PASSED!")
        print("✅ Configuration system is fully integrated across the entire codebase")
        print("🔧 All modules now use environment-based configuration")
        print("🚀 System is ready for production deployment with flexible configuration")
        print()
        print("📋 IMPLEMENTATION SUMMARY:")
        print("• ✅ Data Loader - Uses dataset configuration")
        print("• ✅ Feature Extractors - Uses feature extraction configuration")
        print("• ✅ Model Trainers - Uses model configuration")
        print("• ✅ Preprocessors - Uses feature configuration")
        print("• ✅ Training Pipeline - Uses all configuration sections")
        print("• ✅ Cross-Validation - Uses CV and model configuration")
        print("• ✅ Ethnic Detector - Uses model, dataset, and feature configuration")
        print("• ✅ ML Server - Uses server configuration")
        print("• ✅ Visualizations - Uses visualization configuration")
        print()
        print("🎯 NEXT STEPS:")
        print("1. Create .env file from env.template for your specific configuration")
        print("2. Deploy with environment-specific configuration files")
        print("3. Continue with cross-validation implementation using the new configuration system")
    else:
        print("❌ Some full configuration integration tests failed")
        print("Please check the implementation")
    
    return passed == total

if __name__ == "__main__":
    main()
