#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Cross-Validation Test
Processes fewer images for quick testing and validation
"""

import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

def load_small_sample(data_dir, images_per_ethnicity=20):
    """Load a small sample for fast testing"""
    print(f"📂 Loading small sample ({images_per_ethnicity} images per ethnicity)...")
    
    X, y = [], []
    ethnicities = os.listdir(data_dir)
    print(f"Found ethnicities: {ethnicities}")
    
    for ethnicity in ethnicities:
        ethnicity_path = os.path.join(data_dir, ethnicity)
        if not os.path.isdir(ethnicity_path):
            continue
            
        image_files = [f for f in os.listdir(ethnicity_path) if f.lower().endswith('.jpg')][:images_per_ethnicity]
        print(f"Loading {len(image_files)} images for {ethnicity}")
        
        for img_file in image_files:
            img_path = os.path.join(ethnicity_path, img_file)
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    X.append(image)
                    y.append(ethnicity)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
    
    print(f"✅ Total loaded: {len(X)} images")
    return X, np.array(y)

def extract_features_fast(data):
    """Extract GLCM and color features quickly"""
    print("🔧 Extracting features...")
    
    glcm_features = []
    color_features = []
    
    for i, image in enumerate(data):
        # Convert to grayscale for GLCM
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # GLCM features (simplified)
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3/4*(np.pi)]
        
        try:
            glcm = graycomatrix(gray, distances=distances, angles=angles, 
                              symmetric=True, normed=True, levels=256)
            
            properties = ['contrast', 'homogeneity', 'correlation', 'ASM']
            feats = np.hstack([graycoprops(glcm, prop).ravel() for prop in properties])
            entropy = [shannon_entropy(glcm[:,:,:,idx]) for idx in range(glcm.shape[3])]
            glcm_feat = np.concatenate((entropy, feats), axis=0)
            glcm_features.append(glcm_feat)
        except:
            # Fallback: simple features if GLCM fails
            glcm_features.append(np.zeros(20))
        
        # Color features (HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv], [1], None, [16], [0, 256])  # S channel
        hist2 = cv2.calcHist([hsv], [2], None, [16], [0, 256])  # V channel
        color_feat = np.concatenate((hist1, hist2)).flatten()
        color_features.append(color_feat)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(data)} images")
    
    glcm_features = np.array(glcm_features)
    color_features = np.array(color_features)
    combined_features = np.concatenate((glcm_features, color_features), axis=1)
    
    print(f"✅ Features extracted:")
    print(f"   GLCM: {glcm_features.shape}")
    print(f"   Color: {color_features.shape}")
    print(f"   Combined: {combined_features.shape}")
    
    return combined_features

def run_fast_cv(features, labels):
    """Run fast cross-validation"""
    print("🔄 Running cross-validation...")
    
    # Shuffle data
    X, y = shuffle(features, labels, random_state=220)
    
    # Create model
    clf = RandomForestClassifier(n_estimators=100, random_state=0)  # Reduced for speed
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=6)
    scores = cross_val_score(clf, X, y, cv=cv)
    
    return scores

def main():
    """Main function"""
    print("🚀 FAST CROSS-VALIDATION TEST")
    print("=" * 40)
    
    try:
        # Load small sample
        dataset_path = "dataset/dataset_periorbital"
        data, labels = load_small_sample(dataset_path, images_per_ethnicity=30)
        
        # Show class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\n📊 Class Distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"   {label}: {count}")
        
        # Extract features
        features = extract_features_fast(data)
        
        # Run cross-validation
        scores = run_fast_cv(features, labels)
        
        # Results
        print(f"\n📈 CROSS-VALIDATION RESULTS:")
        print("=" * 40)
        for i, score in enumerate(scores):
            print(f"Fold {i+1}: {score*100:.2f}%")
        
        mean_accuracy = np.mean(scores) * 100
        std_accuracy = np.std(scores) * 100
        
        print(f"\n📊 Mean accuracy: {mean_accuracy:.2f}%")
        print(f"📊 Standard deviation: {std_accuracy:.2f}%")
        
        # Check target
        target = 98.65
        if mean_accuracy >= target:
            print(f"🎯 TARGET ACHIEVED! {mean_accuracy:.2f}% >= {target}%")
        else:
            print(f"⚠️  Target not reached: {mean_accuracy:.2f}% < {target}%")
            print(f"   Difference: {target - mean_accuracy:.2f}%")
        
        print(f"\n🎉 FAST TEST COMPLETED!")
        print(f"📊 Processed {len(data)} images")
        print(f"📊 Used {features.shape[1]} features")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
