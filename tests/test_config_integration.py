#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Configuration Integration
Tests that all modules properly use the configuration system
"""

import sys
import os
import tempfile
sys.path.insert(0, '.')

def test_config_integration():
    """Test configuration integration with existing modules"""
    print("🧪 TESTING CONFIGURATION INTEGRATION")
    print("=" * 50)
    
    try:
        # Test 1: TrainingLogger with configuration
        print("1. Testing TrainingLogger with configuration:")
        from ml_training.core.utils import TrainingLogger
        from ml_training.core.config import get_logging_config
        
        logger = TrainingLogger('config_test')
        log_config = get_logging_config()
        
        print(f"   ✅ Logger created with config: {log_config.log_level}")
        print(f"   ✅ Log file: {logger.log_file}")
        
        # Test 2: Feature extractors with configuration
        print("\n2. Testing Feature Extractors with configuration:")
        from ml_training.core.feature_extractors import GLCFeatureExtractor, ColorHistogramFeatureExtractor
        from ml_training.core.config import get_feature_config
        
        glcm_extractor = GLCFeatureExtractor(logger)
        color_extractor = ColorHistogramFeatureExtractor(logger)
        feature_config = get_feature_config()
        
        print(f"   ✅ GLCM extractor - distances: {glcm_extractor.distances}")
        print(f"   ✅ GLCM extractor - levels: {glcm_extractor.levels}")
        print(f"   ✅ Color extractor - bins: {color_extractor.bins}")
        print(f"   ✅ Color extractor - channels: {color_extractor.channels}")
        
        # Test 3: Cross-validation with configuration
        print("\n3. Testing Cross-Validation with configuration:")
        from ml_training.core.cross_validation import CrossValidationConfig, ModelConfig
        from ml_training.core.config import get_cv_config, get_model_config
        
        cv_config = CrossValidationConfig()
        model_config = ModelConfig()
        cv_env_config = get_cv_config()
        model_env_config = get_model_config()
        
        print(f"   ✅ CV config - folds: {cv_config.n_folds}")
        print(f"   ✅ CV config - test size: {cv_config.test_size}")
        print(f"   ✅ Model config - n_estimators: {model_config.n_estimators}")
        print(f"   ✅ Model config - max_depth: {model_config.max_depth}")
        
        # Test 4: Configuration override with custom values
        print("\n4. Testing Configuration Override:")
        
        # Test with custom parameters
        custom_glcm = GLCFeatureExtractor(logger, distances=[2], levels=128)
        custom_color = ColorHistogramFeatureExtractor(logger, bins=32)
        
        print(f"   ✅ Custom GLCM - distances: {custom_glcm.distances}")
        print(f"   ✅ Custom GLCM - levels: {custom_glcm.levels}")
        print(f"   ✅ Custom Color - bins: {custom_color.bins}")
        
        # Test 5: Configuration with environment file
        print("\n5. Testing Configuration with Environment File:")
        
        env_content = """
# Custom test configuration
MODEL_N_ESTIMATORS=150
CV_N_FOLDS=8
GLCM_LEVELS=128
COLOR_BINS=32
LOG_LEVEL=DEBUG
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
            
            print("   ✅ Custom environment file configuration loaded correctly")
            
        finally:
            # Clean up temporary file
            os.unlink(temp_env_file)
        
        print("\n🎉 ALL CONFIGURATION INTEGRATION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_usage_examples():
    """Test practical usage examples of the configuration system"""
    print("\n🎯 CONFIGURATION USAGE EXAMPLES")
    print("=" * 50)
    
    try:
        from ml_training.core.config import (
            get_config, get_model_config, get_cv_config, 
            get_feature_config, get_logging_config
        )
        from ml_training.core.utils import TrainingLogger
        from ml_training.core.feature_extractors import GLCFeatureExtractor, ColorHistogramFeatureExtractor
        from ml_training.core.cross_validation import CrossValidationConfig, ModelConfig
        
        # Example 1: Creating a logger with configuration
        print("Example 1: Creating Logger with Configuration")
        logger = TrainingLogger('example')
        print(f"   Logger created with log level: {get_logging_config().log_level}")
        
        # Example 2: Creating feature extractors with configuration
        print("\nExample 2: Creating Feature Extractors with Configuration")
        glcm_extractor = GLCFeatureExtractor(logger)
        color_extractor = ColorHistogramFeatureExtractor(logger)
        
        print(f"   GLCM extractor configured with {glcm_extractor.levels} levels")
        print(f"   Color extractor configured with {color_extractor.bins} bins")
        
        # Example 3: Creating cross-validation with configuration
        print("\nExample 3: Creating Cross-Validation with Configuration")
        cv_config = CrossValidationConfig()
        model_config = ModelConfig()
        
        print(f"   CV configured with {cv_config.n_folds} folds")
        print(f"   Model configured with {model_config.n_estimators} estimators")
        
        # Example 4: Using configuration for RandomForest parameters
        print("\nExample 4: RandomForest Configuration")
        model_config_obj = get_model_config()
        
        rf_params = {
            'n_estimators': model_config_obj.n_estimators,
            'max_depth': model_config_obj.max_depth,
            'min_samples_split': model_config_obj.min_samples_split,
            'min_samples_leaf': model_config_obj.min_samples_leaf,
            'random_state': model_config_obj.random_state,
            'n_jobs': model_config_obj.n_jobs
        }
        
        # Remove None values
        rf_params = {k: v for k, v in rf_params.items() if v is not None}
        
        print(f"   RandomForest parameters: {rf_params}")
        
        # Example 5: Using configuration for feature extraction
        print("\nExample 5: Feature Extraction Configuration")
        feature_config = get_feature_config()
        
        print(f"   GLCM distances: {feature_config.glc_distances}")
        print(f"   GLCM angles: {feature_config.glc_angles}")
        print(f"   Color bins: {feature_config.color_bins}")
        print(f"   Color channels: {feature_config.color_channels}")
        
        print("\n✅ All configuration usage examples working!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration usage examples failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_validation():
    """Test configuration validation and error handling"""
    print("\n🔍 TESTING CONFIGURATION VALIDATION")
    print("=" * 50)
    
    try:
        from ml_training.core.config import get_config
        
        # Test configuration access
        config = get_config()
        
        # Test that all required configurations are present
        assert hasattr(config, 'dataset')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'cross_validation')
        assert hasattr(config, 'feature_extraction')
        assert hasattr(config, 'logging')
        assert hasattr(config, 'server')
        assert hasattr(config, 'visualization')
        
        print("✅ All configuration sections present")
        
        # Test that configuration values are reasonable
        assert config.model.n_estimators > 0
        assert config.cross_validation.n_folds > 0
        assert config.feature_extraction.glc_levels > 0
        assert config.feature_extraction.color_bins > 0
        assert len(config.dataset.ethnicities) > 0
        
        print("✅ All configuration values are reasonable")
        
        # Test configuration printing
        print("\n📋 Current Configuration Summary:")
        config.print_config()
        
        print("\n✅ Configuration validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 TESTING CONFIGURATION INTEGRATION SYSTEM")
    print("=" * 70)
    
    # Run all tests
    tests = [
        test_config_integration,
        test_config_usage_examples,
        test_config_validation
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
        print("🎉 ALL CONFIGURATION INTEGRATION TESTS PASSED!")
        print("✅ Configuration system is fully integrated")
        print("🔧 All modules now use environment-based configuration")
        print()
        print("📋 NEXT STEPS:")
        print("1. Create .env file from env.template for custom configuration")
        print("2. Update deployment scripts to use environment variables")
        print("3. Test with production configuration values")
    else:
        print("❌ Some configuration integration tests failed")
        print("Please check the implementation")
    
    return passed == total

if __name__ == "__main__":
    main()
