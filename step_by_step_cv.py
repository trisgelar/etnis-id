#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step-by-Step Cross-Validation System
Simple implementation with clear output
"""

import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy

def main():
    print("🚀 STEP-BY-STEP CROSS-VALIDATION SYSTEM")
    print("=" * 50)
    
    # Step 1: Load a small sample first
    print("📂 Step 1: Loading dataset sample...")
    dataset_path = "dataset/dataset_periorbital"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    # Load just a few images per ethnicity for testing
    X, y = [], []
    ethnicities = os.listdir(dataset_path)
    print(f"Found ethnicities: {ethnicities}")
    
    max_per_ethnicity = 50  # Limit for testing
    
    for ethnicity in ethnicities[:2]:  # Just test with 2 ethnicities first
        ethnicity_path = os.path.join(dataset_path, ethnicity)
        if not os.path.isdir(ethnicity_path):
            continue
            
        image_files = [f for f in os.listdir(ethnicity_path) if f.lower().endswith('.jpg')][:max_per_ethnicity]
        print(f"Loading {len(image_files)} images for {ethnicity}")
        
        for img_file in image_files:
            img_path = os.path.join(ethnicity_path, img_file)
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    X.append(image)
                    y.append(ethnicity)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Loaded {len(X)} images total")
    print(f"Classes: {np.unique(y)}")
    
    # Step 2: Preprocessing
    print("\n🔧 Step 2: Preprocessing...")
    print("Converting to grayscale...")
    gray_images = []
    for image in X:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray)
    
    print("Converting to HSV...")
    hsv_images = []
    for image in X:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_images.append(hsv)
    
    # Step 3: Feature extraction
    print("\n🔧 Step 3: Feature extraction...")
    print("Extracting GLCM features...")
    
    glcm_features = []
    for i, gray_img in enumerate(gray_images):
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3/4*(np.pi)]
        
        glcm = greycomatrix(gray_img, 
                           distances=distances, 
                           angles=angles, 
                           symmetric=True, 
                           normed=True, 
                           levels=256)
        
        properties = ['contrast', 'homogeneity', 'correlation', 'ASM']
        feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
        
        entropy = [shannon_entropy(glcm[:,:,:,idx]) for idx in range(glcm.shape[3])]
        feat = np.concatenate((entropy, feats), axis=0)
        
        glcm_features.append(feat)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(gray_images)} images")
    
    glcm_features = np.array(glcm_features)
    print(f"GLCM features shape: {glcm_features.shape}")
    
    print("Extracting color features...")
    color_features = []
    for i, hsv_img in enumerate(hsv_images):
        hist1 = cv2.calcHist([hsv_img], [1], None, [16], [0, 256])
        hist2 = cv2.calcHist([hsv_img], [2], None, [16], [0, 256])
        feature = np.concatenate((hist1, hist2))
        arr = np.array(feature).flatten()
        color_features.append(arr)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(hsv_images)} images")
    
    color_features = np.array(color_features)
    print(f"Color features shape: {color_features.shape}")
    
    # Step 4: Combine features
    print("\n🔗 Step 4: Combining features...")
    combined_features = np.concatenate((glcm_features, color_features), axis=1)
    print(f"Combined features shape: {combined_features.shape}")
    
    # Step 5: Cross-validation
    print("\n🔄 Step 5: Cross-validation...")
    X, y = shuffle(combined_features, y, random_state=220)
    
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    cv = StratifiedKFold(n_splits=6)
    
    scores = cross_val_score(clf, X, y, cv=cv)
    
    print("Cross-validation scores:")
    for i, score in enumerate(scores):
        print(f"  Fold {i+1}: {score*100:.2f}%")
    
    mean_accuracy = np.mean(scores) * 100
    print(f"\nMean accuracy: {mean_accuracy:.2f}%")
    
    # Step 6: Train final model
    print("\n🎯 Step 6: Training final model...")
    final_model = RandomForestClassifier(n_estimators=200, random_state=0)
    final_model.fit(combined_features, y)
    
    print("✅ Model trained successfully!")
    print(f"📊 Final accuracy: {mean_accuracy:.2f}%")
    
    print("\n🎉 STEP-BY-STEP SYSTEM COMPLETED!")

if __name__ == "__main__":
    main()

