#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Training Test with Your Dataset
Simple test to validate training works with your dataset
"""

import sys
import os
import numpy as np

# Add the ml_training directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_training'))

def run_training_test():
    """Run training test with your dataset"""
    print("RUNNING TRAINING TEST WITH YOUR DATASET")
    print("=" * 60)
    
    try:
        # Import training functions
        from script_training import (
            load_data, preprocessing_glcm, preprocessing_color, 
            glcm_extraction, color_extraction, crossVal,
            label_map
        )
        
        # Dataset path
        dataset_path = "../dataset/dataset_periorbital"
        
        print(f"Dataset path: {dataset_path}")
        print(f"Label mapping: {label_map}")
        print()
        
        # Step 1: Load data
        print("STEP 1: Loading data...")
        data, label, idx, img_name, fld = load_data(dataset_path)
        
        if data is None:
            print("ERROR: Failed to load data!")
            return False
        
        print(f"SUCCESS: Loaded {len(data)} images")
        print(f"Data shape: {data.shape}")
        print(f"Label distribution:")
        unique_labels, counts = np.unique(label, return_counts=True)
        for label_val, count in zip(unique_labels, counts):
            ethnicity = label_map.get(label_val, f"Unknown_{label_val}")
            print(f"  {ethnicity}: {count} images")
        print()
        
        # Step 2: Preprocessing
        print("STEP 2: Preprocessing...")
        print("  Converting to grayscale for GLCM...")
        glcm_prep = preprocessing_glcm(data)
        print(f"  GLCM preprocessed shape: {glcm_prep.shape}")
        
        print("  Converting to HSV for color features...")
        color_prep = preprocessing_color(data)
        print(f"  Color preprocessed shape: {color_prep.shape}")
        print()
        
        # Step 3: Feature extraction
        print("STEP 3: Feature extraction...")
        print("  Extracting GLCM features...")
        glcm_feat = glcm_extraction(glcm_prep)
        print(f"  GLCM features shape: {glcm_feat.shape}")
        
        print("  Extracting color features...")
        color_feat = color_extraction(color_prep)
        print(f"  Color features shape: {color_feat.shape}")
        print()
        
        # Step 4: Combine features
        print("STEP 4: Combining features...")
        feature = np.concatenate((glcm_feat, color_feat), axis=1)
        print(f"Combined features shape: {feature.shape}")
        print(f"GLCM features: {glcm_feat.shape[1]}")
        print(f"Color features: {color_feat.shape[1]}")
        print(f"Total features: {feature.shape[1]}")
        print()
        
        # Step 5: Cross-validation test
        print("STEP 5: Cross-validation test...")
        print("  Running 5-fold cross-validation...")
        cv_scores = crossVal(5, feature, label)
        
        print("CROSS-VALIDATION RESULTS:")
        print(f"  Individual fold scores: {cv_scores}")
        print(f"  Mean accuracy: {np.mean(cv_scores):.2f}%")
        print(f"  Standard deviation: {np.std(cv_scores):.2f}%")
        print(f"  Min accuracy: {np.min(cv_scores):.2f}%")
        print(f"  Max accuracy: {np.max(cv_scores):.2f}%")
        print()
        
        # Summary
        print("TRAINING TEST SUMMARY")
        print("=" * 60)
        print("SUCCESS: All components working correctly!")
        print(f"SUCCESS: Dataset loaded successfully ({len(data)} images)")
        print(f"SUCCESS: Features extracted ({feature.shape[1]} total features)")
        print(f"SUCCESS: Cross-validation completed (Mean: {np.mean(cv_scores):.2f}%)")
        print()
        print("READY FOR FULL TRAINING!")
        print("You can now run the complete training script.")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_training_test()
    if success:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Run: python ml_training/script_training.py")
        print("2. Or run the SOLID-compliant system (after fixing Unicode issues)")
        print("="*60)
    
    sys.exit(0 if success else 1)
