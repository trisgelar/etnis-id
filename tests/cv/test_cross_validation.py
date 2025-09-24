#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Cross-Validation Module
Tests the new SOLID-based cross-validation system
"""

import sys
import os
import numpy as np
sys.path.insert(0, '.')

def test_cross_validation_module():
    """Test the cross-validation module"""
    print("🧪 TESTING CROSS-VALIDATION MODULE")
    print("=" * 50)
    
    try:
        # Import the new cross-validation module
        from ml_training.core.cross_validation import (
            CrossValidationConfig, ModelConfig, CrossValidationResults,
            CrossValidationEngine, FeatureScaler, CrossValidationManager
        )
        from ml_training.core.utils import TrainingLogger
        
        print("✅ All imports successful")
        
        # Test configuration classes
        print("\n📊 Testing Configuration Classes:")
        
        # Test CrossValidationConfig
        cv_config = CrossValidationConfig(n_folds=6, test_size=0.2)
        assert cv_config.n_folds == 6
        assert cv_config.test_size == 0.2
        print("✅ CrossValidationConfig working")
        
        # Test ModelConfig
        model_config = ModelConfig(n_estimators=[100, 200])
        assert len(model_config.n_estimators) == 2
        print("✅ ModelConfig working")
        
        # Test CrossValidationResults
        results = CrossValidationResults()
        assert results.best_score == 0.0
        print("✅ CrossValidationResults working")
        
        # Test FeatureScaler
        logger = TrainingLogger('test_cv')
        scaler = FeatureScaler(logger)
        
        # Create test data
        X_test = np.random.randn(100, 10)
        X_scaled = scaler.fit_transform(X_test)
        
        assert X_scaled.shape == X_test.shape
        assert np.isclose(np.mean(X_scaled), 0.0, atol=1e-10)
        assert np.isclose(np.std(X_scaled), 1.0, atol=1e-10)
        print("✅ FeatureScaler working")
        
        # Test CrossValidationEngine
        cv_engine = CrossValidationEngine(logger, config=cv_config)
        assert cv_engine.config.n_folds == 6
        print("✅ CrossValidationEngine working")
        
        # Test CrossValidationManager
        cv_manager = CrossValidationManager(logger)
        assert cv_manager.cv_config.n_folds == 6
        print("✅ CrossValidationManager working")
        
        print("\n🎉 ALL CROSS-VALIDATION MODULE TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_validation_integration():
    """Test cross-validation with synthetic data"""
    print("\n🔗 TESTING CROSS-VALIDATION INTEGRATION")
    print("=" * 50)
    
    try:
        from ml_training.core.cross_validation import (
            CrossValidationManager, ModelConfig
        )
        from ml_training.core.utils import TrainingLogger
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 200
        n_features = 15
        
        # Create features with some pattern
        X = np.random.randn(n_samples, n_features)
        
        # Create labels with some pattern (5 classes)
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            if i < 40:
                y[i] = 0  # Class 0
            elif i < 80:
                y[i] = 1  # Class 1
            elif i < 120:
                y[i] = 2  # Class 2
            elif i < 160:
                y[i] = 3  # Class 3
            else:
                y[i] = 4  # Class 4
        
        print(f"✅ Synthetic dataset created: {X.shape}, {y.shape}")
        
        # Test cross-validation manager
        logger = TrainingLogger('test_cv_integration')
        cv_manager = CrossValidationManager(logger)
        
        # Create simpler model config for testing
        model_config = ModelConfig(
            n_estimators=[50, 100],  # Fewer estimators for speed
            max_depth=[None, 5],
            min_samples_split=[2, 5],
            min_samples_leaf=[1, 2],
            class_weight=[None, 'balanced']
        )
        cv_manager.model_config = model_config
        
        # Run cross-validation
        print("🚀 Running cross-validation...")
        results = cv_manager.cv_engine.run_cross_validation(X, y, model_config)
        
        # Check results
        assert len(results.results) > 0
        assert results.best_score > 0
        assert results.best_model is not None
        
        print(f"✅ Cross-validation completed successfully")
        print(f"   Best model: {results.best_model_name}")
        print(f"   Best score: {results.best_score:.3f}")
        print(f"   Models tested: {len(results.results)}")
        
        print("\n🎉 CROSS-VALIDATION INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_scaling():
    """Test feature scaling functionality"""
    print("\n🔍 TESTING FEATURE SCALING")
    print("=" * 50)
    
    try:
        from ml_training.core.cross_validation import FeatureScaler
        from ml_training.core.utils import TrainingLogger
        
        logger = TrainingLogger('test_scaling')
        scaler = FeatureScaler(logger)
        
        # Create test data with different scales
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        # Add different scales to different features
        X[:, 0] *= 1000  # Large scale
        X[:, 1] *= 0.001  # Small scale
        X[:, 2] += 500  # Large offset
        
        print(f"Original data - Mean: {np.mean(X):.2f}, Std: {np.std(X):.2f}")
        print(f"Original data - Min: {np.min(X):.2f}, Max: {np.max(X):.2f}")
        
        # Scale features
        X_scaled = scaler.fit_transform(X)
        
        print(f"Scaled data - Mean: {np.mean(X_scaled):.6f}, Std: {np.std(X_scaled):.6f}")
        print(f"Scaled data - Min: {np.min(X_scaled):.2f}, Max: {np.max(X_scaled):.2f}")
        
        # Verify scaling
        assert np.isclose(np.mean(X_scaled), 0.0, atol=1e-10)
        assert np.isclose(np.std(X_scaled), 1.0, atol=1e-10)
        
        # Test transform on new data
        X_new = np.random.randn(50, 5)
        X_new_scaled = scaler.transform(X_new)
        
        assert X_new_scaled.shape == X_new.shape
        print("✅ Transform on new data working")
        
        print("\n🎉 FEATURE SCALING TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Feature scaling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 TESTING SOLID CROSS-VALIDATION MODULE")
    print("=" * 70)
    
    # Run all tests
    tests = [
        test_cross_validation_module,
        test_feature_scaling,
        test_cross_validation_integration
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
        print("🎉 ALL TESTS PASSED! Cross-validation module is working correctly.")
        print("✅ Ready for production use with proper SOLID architecture.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()
