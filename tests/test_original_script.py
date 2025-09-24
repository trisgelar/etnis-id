#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the original working script with your dataset
"""

import sys
import os

# Add the ml_training directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_training'))

# Import the original working script
from script_training import load_data, preprocessing_glcm, preprocessing_color, glcm_extraction, color_extraction, crossVal

def test_original_script():
    """Test the original script with your dataset"""
    print("TESTING ORIGINAL TRAINING SCRIPT")
    print("=" * 50)
    
    try:
        # Test data loading
        print("1. Testing data loading...")
        dataset_path = "../dataset/dataset_periorbital"
        
        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset path not found: {dataset_path}")
            return False
        
        print(f"SUCCESS: Dataset path exists: {dataset_path}")
        
        # Load data
        print("   Loading dataset...")
        data, label, idx, img_name, fld = load_data(dataset_path)
        
        if data is None:
            print("ERROR: Failed to load data")
            return False
        
        print(f"SUCCESS: Data loaded successfully!")
        print(f"   Data shape: {data.shape}")
        print(f"   Labels shape: {label.shape}")
        print(f"   Unique labels: {len(set(label))}")
        print(f"   Label distribution: {dict(zip(*np.unique(label, return_counts=True)))}")
        
        # Test preprocessing
        print("\n2. Testing preprocessing...")
        print("   GLCM preprocessing...")
        glcm_prep = preprocessing_glcm(data)
        print(f"   GLCM preprocessed shape: {glcm_prep.shape}")
        
        print("   Color preprocessing...")
        color_prep = preprocessing_color(data)
        print(f"   Color preprocessed shape: {color_prep.shape}")
        
        # Test feature extraction
        print("\n3. Testing feature extraction...")
        print("   GLCM feature extraction...")
        glcm_feat = glcm_extraction(glcm_prep)
        print(f"   GLCM features shape: {glcm_feat.shape}")
        
        print("   Color feature extraction...")
        color_feat = color_extraction(color_prep)
        print(f"   Color features shape: {color_feat.shape}")
        
        # Combine features
        print("\n4. Testing feature combination...")
        feature = np.concatenate((glcm_feat, color_feat), axis=1)
        print(f"   Combined features shape: {feature.shape}")
        
        # Test cross-validation
        print("\n5. Testing cross-validation...")
        cv_scores = crossVal(5, feature, label)  # 5-fold CV
        print(f"   Cross-validation scores: {cv_scores}")
        print(f"   Mean accuracy: {np.mean(cv_scores):.2f}%")
        print(f"   Std deviation: {np.std(cv_scores):.2f}%")
        
        print("\nALL TESTS PASSED!")
        print("=" * 50)
        print("SUCCESS: Original script works correctly with your dataset!")
        print("SUCCESS: Ready to proceed with full training!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import numpy as np
    success = test_original_script()
    sys.exit(0 if success else 1)
