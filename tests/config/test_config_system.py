#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Configuration System
Tests the new python-dotenv based configuration system
"""

import sys
import os
import tempfile
sys.path.insert(0, '.')

def test_config_system():
    """Test the configuration system"""
    print("🧪 TESTING CONFIGURATION SYSTEM")
    print("=" * 50)
    
    try:
        # Import the configuration module
        from ml_training.core.config import (
            Config, DatasetConfig, ModelConfig, TrainingConfig,
            CrossValidationConfig, FeatureExtractionConfig,
            LoggingConfig, ServerConfig, VisualizationConfig
        )
        
        print("✅ Configuration modules imported successfully")
        
        # Test individual configuration classes
        print("\n📊 Testing Individual Configuration Classes:")
        
        # Test DatasetConfig
        dataset_config = DatasetConfig()
        assert dataset_config.data_dir is not None
        assert len(dataset_config.ethnicities) == 5
        print("✅ DatasetConfig working")
        
        # Test ModelConfig
        model_config = ModelConfig()
        assert model_config.n_estimators > 0
        assert model_config.random_state is not None
        print("✅ ModelConfig working")
        
        # Test TrainingConfig
        training_config = TrainingConfig()
        assert training_config.batch_size > 0
        assert 0 < training_config.learning_rate < 1
        print("✅ TrainingConfig working")
        
        # Test CrossValidationConfig
        cv_config = CrossValidationConfig()
        assert cv_config.n_folds > 0
        assert 0 < cv_config.test_size < 1
        print("✅ CrossValidationConfig working")
        
        # Test FeatureExtractionConfig
        feature_config = FeatureExtractionConfig()
        assert feature_config.glc_levels > 0
        assert len(feature_config.glc_angles) > 0
        print("✅ FeatureExtractionConfig working")
        
        # Test LoggingConfig
        logging_config = LoggingConfig()
        assert logging_config.log_level is not None
        assert logging_config.log_file is not None
        print("✅ LoggingConfig working")
        
        # Test ServerConfig
        server_config = ServerConfig()
        assert server_config.port > 0
        assert server_config.host is not None
        print("✅ ServerConfig working")
        
        # Test VisualizationConfig
        viz_config = VisualizationConfig()
        assert viz_config.figure_dpi > 0
        assert viz_config.output_dir is not None
        print("✅ VisualizationConfig working")
        
        print("\n🎉 ALL CONFIGURATION CLASSES WORKING!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_with_env_file():
    """Test configuration with custom environment file"""
    print("\n🔧 TESTING CONFIGURATION WITH ENV FILE")
    print("=" * 50)
    
    try:
        from ml_training.core.config import Config
        
        # Create a temporary .env file
        env_content = """
# Test configuration
DATASET_DIR=test_dataset
MODEL_N_ESTIMATORS=100
CV_N_FOLDS=5
GLCM_LEVELS=128
LOG_LEVEL=DEBUG
SERVER_PORT=9090
VIZ_FIGURE_DPI=150
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            temp_env_file = f.name
        
        try:
            # Load configuration with custom env file
            config = Config(temp_env_file)
            
            # Test that custom values are loaded
            assert config.dataset.data_dir == "test_dataset"
            assert config.model.n_estimators == 100
            assert config.cross_validation.n_folds == 5
            assert config.feature_extraction.glc_levels == 128
            assert config.logging.log_level == "DEBUG"
            assert config.server.port == 9090
            assert config.visualization.figure_dpi == 150
            
            print("✅ Custom environment file configuration working")
            
            # Test configuration printing
            print("\n📋 Configuration Overview:")
            config.print_config()
            
            return True
            
        finally:
            # Clean up temporary file
            os.unlink(temp_env_file)
        
    except Exception as e:
        print(f"❌ Environment file test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_getters():
    """Test configuration getter functions"""
    print("\n🔍 TESTING CONFIGURATION GETTERS")
    print("=" * 50)
    
    try:
        from ml_training.core.config import (
            get_config, get_dataset_config, get_model_config,
            get_training_config, get_cv_config, get_feature_config,
            get_logging_config, get_server_config, get_viz_config
        )
        
        # Test global config getter
        config = get_config()
        assert config is not None
        print("✅ Global config getter working")
        
        # Test individual config getters
        dataset_config = get_dataset_config()
        assert dataset_config is not None
        print("✅ Dataset config getter working")
        
        model_config = get_model_config()
        assert model_config is not None
        print("✅ Model config getter working")
        
        training_config = get_training_config()
        assert training_config is not None
        print("✅ Training config getter working")
        
        cv_config = get_cv_config()
        assert cv_config is not None
        print("✅ CV config getter working")
        
        feature_config = get_feature_config()
        assert feature_config is not None
        print("✅ Feature config getter working")
        
        logging_config = get_logging_config()
        assert logging_config is not None
        print("✅ Logging config getter working")
        
        server_config = get_server_config()
        assert server_config is not None
        print("✅ Server config getter working")
        
        viz_config = get_viz_config()
        assert viz_config is not None
        print("✅ Visualization config getter working")
        
        print("\n🎉 ALL CONFIGURATION GETTERS WORKING!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration getters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """Test configuration integration with existing modules"""
    print("\n🔗 TESTING CONFIGURATION INTEGRATION")
    print("=" * 50)
    
    try:
        from ml_training.core.config import get_config
        from ml_training.core.utils import TrainingLogger
        
        # Get configuration
        config = get_config()
        
        # Test with TrainingLogger using config
        logger = TrainingLogger('config_test')
        
        # Test that we can access all configuration sections
        dataset_config = config.dataset.get_config()
        model_config = config.model.get_config()
        cv_config = config.cross_validation.get_config()
        
        logger.info(f"Dataset config loaded: {len(dataset_config)} parameters")
        logger.info(f"Model config loaded: {len(model_config)} parameters")
        logger.info(f"CV config loaded: {len(cv_config)} parameters")
        
        # Test configuration values
        assert 'ethnicities' in dataset_config
        assert 'n_estimators' in model_config
        assert 'n_folds' in cv_config
        
        print("✅ Configuration integration working")
        print(f"   Dataset parameters: {len(dataset_config)}")
        print(f"   Model parameters: {len(model_config)}")
        print(f"   CV parameters: {len(cv_config)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_config_usage():
    """Demo how to use the configuration system"""
    print("\n🎯 CONFIGURATION USAGE DEMO")
    print("=" * 50)
    
    try:
        from ml_training.core.config import (
            get_config, get_model_config, get_cv_config,
            get_feature_config
        )
        
        print("📋 Example: Using Configuration in Your Code")
        print()
        
        # Example 1: Using global config
        print("1. Using global configuration:")
        config = get_config()
        print(f"   Model type: {config.model.model_type}")
        print(f"   CV folds: {config.cross_validation.n_folds}")
        print(f"   Dataset ethnicities: {config.dataset.ethnicities}")
        
        # Example 2: Using specific config getters
        print("\n2. Using specific configuration getters:")
        model_config = get_model_config()
        cv_config = get_cv_config()
        feature_config = get_feature_config()
        
        print(f"   Model n_estimators: {model_config.n_estimators}")
        print(f"   CV test size: {cv_config.test_size}")
        print(f"   GLCM levels: {feature_config.glc_levels}")
        
        # Example 3: Configuration for RandomForest
        print("\n3. Configuration for RandomForest:")
        rf_params = {
            'n_estimators': model_config.n_estimators,
            'max_depth': model_config.max_depth,
            'min_samples_split': model_config.min_samples_split,
            'min_samples_leaf': model_config.min_samples_leaf,
            'random_state': model_config.random_state,
            'n_jobs': model_config.n_jobs
        }
        
        # Remove None values
        rf_params = {k: v for k, v in rf_params.items() if v is not None}
        
        print(f"   RandomForest parameters: {rf_params}")
        
        # Example 4: Configuration for Cross-Validation
        print("\n4. Configuration for Cross-Validation:")
        cv_params = {
            'n_splits': cv_config.n_folds,
            'shuffle': cv_config.shuffle,
            'random_state': cv_config.random_state
        }
        print(f"   CV parameters: {cv_params}")
        
        print("\n✅ Configuration usage demo completed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration usage demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 TESTING PYTHON-DOTENV CONFIGURATION SYSTEM")
    print("=" * 70)
    
    # Run all tests
    tests = [
        test_config_system,
        test_config_with_env_file,
        test_config_getters,
        test_config_integration,
        demo_config_usage
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
        print("🎉 ALL CONFIGURATION TESTS PASSED!")
        print("✅ Configuration system is working correctly")
        print("🔧 Ready for integration with existing modules")
        print()
        print("📋 NEXT STEPS:")
        print("1. Create .env file from env.template")
        print("2. Update existing modules to use configuration")
        print("3. Test with real environment variables")
    else:
        print("❌ Some configuration tests failed")
        print("Please check the implementation")
    
    return passed == total

if __name__ == "__main__":
    main()
