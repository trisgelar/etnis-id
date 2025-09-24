#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test to verify everything is working
"""

import sys
import os
sys.path.insert(0, '.')

print("🔍 QUICK SYSTEM TEST")
print("=" * 30)

try:
    print("1. Testing imports...")
    from ethnic_detector import EthnicDetector
    print("   ✅ EthnicDetector imported")
    
    print("2. Testing model loading...")
    detector = EthnicDetector()
    print("   ✅ Model loaded successfully")
    
    print("3. Testing prediction...")
    import numpy as np
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    prediction, confidence, status = detector.predict_ethnicity(test_image)
    print(f"   ✅ Prediction: {prediction} (Confidence: {confidence:.1f}%)")
    
    print("4. Testing SciencePlots...")
    try:
        import scienceplots
        print("   ✅ SciencePlots available")
    except ImportError:
        print("   ⚠️ SciencePlots not available")
    
    print("\n✅ ALL TESTS PASSED!")
    print("🎯 System is ready for analysis")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
