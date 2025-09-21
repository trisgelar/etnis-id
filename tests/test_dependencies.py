#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Dependencies & Environment
Verifikasi semua library terinstall dengan benar untuk ML ethnic detection
"""

import sys
import os
import importlib

def test_dependencies():
    """Test semua dependencies yang diperlukan"""
    print("=" * 60)
    print("🧪 TESTING DEPENDENCIES & ENVIRONMENT")
    print("=" * 60)
    
    dependencies = [
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("skimage", "scikit-image"),
        ("PIL", "Pillow"),
        ("scipy", "scipy"),
        ("pickle", "Built-in"),
        ("json", "Built-in"),
        ("socket", "Built-in"),
        ("threading", "Built-in"),
        ("base64", "Built-in"),
        ("time", "Built-in")
    ]
    
    results = {}
    all_good = True
    
    for import_name, package_name in dependencies:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package_name:<15}: {version}")
            results[package_name] = {"status": "OK", "version": version}
        except ImportError as e:
            print(f"❌ {package_name:<15}: NOT INSTALLED - {e}")
            results[package_name] = {"status": "MISSING", "error": str(e)}
            all_good = False
        except Exception as e:
            print(f"⚠️  {package_name:<15}: ERROR - {e}")
            results[package_name] = {"status": "ERROR", "error": str(e)}
            all_good = False
    
    print("\n" + "=" * 60)
    print(f"📍 Python Version: {sys.version}")
    print(f"📁 Python Path: {sys.executable}")
    print(f"📂 Current Directory: {os.getcwd()}")
    
    return all_good, results

def test_specific_imports():
    """Test import spesifik yang digunakan di ethnic_detector"""
    print("\n" + "=" * 60)
    print("🔬 TESTING SPECIFIC IMPORTS")
    print("=" * 60)
    
    specific_imports = [
        "from skimage.feature import graycomatrix, graycoprops",
        "from skimage.measure import shannon_entropy",
        "from sklearn.ensemble import RandomForestClassifier",
        "import cv2",
        "import numpy as np",
        "from PIL import Image",
        "from io import BytesIO",
        "import base64",
        "import pickle"
    ]
    
    all_good = True
    
    for import_statement in specific_imports:
        try:
            exec(import_statement)
            print(f"✅ {import_statement}")
        except Exception as e:
            print(f"❌ {import_statement} - ERROR: {e}")
            all_good = False
    
    return all_good

def test_model_file():
    """Test keberadaan dan validitas model file"""
    print("\n" + "=" * 60)
    print("🤖 TESTING ML MODEL FILE")
    print("=" * 60)
    
    model_path = "model_ml/pickle_model.pkl"
    
    # Check file existence
    if not os.path.exists(model_path):
        print(f"❌ Model file tidak ditemukan: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path)
    print(f"✅ Model file ditemukan: {model_path}")
    print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # Try to load model
    try:
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model berhasil dimuat: {type(model)}")
        
        # Check model attributes
        if hasattr(model, 'predict'):
            print("✅ Model memiliki method predict()")
        else:
            print("❌ Model TIDAK memiliki method predict()")
            return False
            
        if hasattr(model, 'predict_proba'):
            print("✅ Model memiliki method predict_proba()")
        else:
            print("⚠️  Model tidak memiliki method predict_proba()")
        
        # Test with dummy data
        import numpy as np
        dummy_features = np.random.rand(1, 52)  # 20 GLCM + 32 Color features
        
        try:
            prediction = model.predict(dummy_features)
            print(f"✅ Test prediction berhasil: {prediction}")
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(dummy_features)
                print(f"✅ Test prediction probabilities berhasil: shape {probabilities.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error dalam test prediction: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def main():
    """Main testing function"""
    print("🚀 STARTING COMPREHENSIVE DEPENDENCY TEST")
    print(f"⏰ Timestamp: {__import__('datetime').datetime.now()}")
    
    # Test dependencies
    deps_ok, deps_results = test_dependencies()
    
    # Test specific imports
    imports_ok = test_specific_imports()
    
    # Test model file
    model_ok = test_model_file()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY HASIL TEST")
    print("=" * 60)
    
    print(f"Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"Specific Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"ML Model: {'✅ PASS' if model_ok else '❌ FAIL'}")
    
    overall_status = deps_ok and imports_ok and model_ok
    print(f"\n🎯 OVERALL STATUS: {'✅ READY FOR ML PROCESSING' if overall_status else '❌ SYSTEM NOT READY'}")
    
    if not overall_status:
        print("\n💡 RECOMMENDED ACTIONS:")
        if not deps_ok:
            print("   - Run: pip install -r requirements.txt")
        if not imports_ok:
            print("   - Check library versions compatibility")
        if not model_ok:
            print("   - Check model_ml/pickle_model.pkl file")
    
    return overall_status

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test dibatalkan oleh user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)