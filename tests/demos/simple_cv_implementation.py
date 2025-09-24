#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Cross-Validation Implementation
Direct implementation without complex configuration system
"""

import os
import sys
import numpy as np
import cv2
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy

def load_data(data_dir):
    """Load data exactly like the original notebook"""
    m = 0
    X, y, idx, name, fl = [], [], [], [], []
    labels = os.listdir(data_dir)
    print(f"Found labels: {labels}")
    
    for label in labels:
        datas_path = os.path.join(data_dir, label)
        datas = os.listdir(datas_path)
        fl.append(datas)
        print(f"Processing {label}: {len(datas)} folders")
        
        for data in datas:
            data_path = os.path.join(datas_path, data)
            image_name = os.listdir(data_path)
            name.append(image_name)
            
            for img in image_name:
                img_path = os.path.join(data_path, img)
                image = cv2.imread(img_path)
                if image is not None:
                    X.append(image)
                    y.append(label)
                    idx.append(m)
                    m += 1
    
    return np.array(X), np.array(y), np.array(idx), name, fl

def preprocessing_glcm(data):
    """Convert RGB to Grayscale for GLCM"""
    grays = []
    for i in range(len(data)):
        img = cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY)
        grays.append(img)
    return grays

def preprocessing_color(array):
    """Convert RGB to HSV for Color Histogram"""
    preprocessed = []
    for i in range(len(array)):
        img = array[i].copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        preprocessed.append(img)
    return preprocessed

def glcm_extraction(data):
    """Extract GLCM features exactly like notebook"""
    feature = []
    for i in range(0, len(data)):
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3/4*(np.pi)]
        
        glcm = greycomatrix(data[i], 
                            distances=distances, 
                            angles=angles, 
                            symmetric=True, 
                            normed=True, levels=256)
        
        properties = ['contrast','homogeneity','correlation','ASM']
        feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
        
        entropy = [shannon_entropy(glcm[:,:,:,idx]) for idx in range(glcm.shape[3])]
        feat = np.concatenate((entropy,feats),axis=0)
        feature.append(feat)
    return feature

def color_extraction(img):
    """Extract color histogram features exactly like notebook"""
    features = []
    for i in range(np.shape(img)[0]):
        hist1 = cv2.calcHist([img[i]], [1], None, [16], [0, 256])
        hist2 = cv2.calcHist([img[i]], [2], None, [16], [0, 256])
        fitur = np.concatenate((hist1,hist2))
        arr = np.array(fitur).flatten()
        features.append(arr)
    return features

def crossVal(K, X, y):
    """Cross validation exactly like notebook"""
    X, y = shuffle(X, y, random_state=220)
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    cv = StratifiedKFold(n_splits=K)
    scores = cross_val_score(clf, X, y, cv=cv)
    return scores

def main():
    """Main function"""
    print("🚀 SIMPLE CROSS-VALIDATION IMPLEMENTATION")
    print("=" * 50)
    
    # Dataset path
    dataset_path = "dataset/dataset_periorbital"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    print(f"📂 Loading dataset from: {dataset_path}")
    
    try:
        # Load data
        data, label, idx, img_name, fld = load_data(dataset_path)
        print(f"✅ Loaded {len(data)} images")
        
        # Show class distribution
        unique_labels, counts = np.unique(label, return_counts=True)
        print("📊 Class distribution:")
        for label_name, count in zip(unique_labels, counts):
            print(f"   {label_name}: {count}")
        
        # Preprocessing
        print("🔧 Preprocessing images...")
        glcm_prep = preprocessing_glcm(data)
        color_prep = preprocessing_color(data)
        
        # Feature extraction
        print("🔧 Extracting GLCM features...")
        glcm_feat = glcm_extraction(glcm_prep)
        print(f"   GLCM features shape: {np.array(glcm_feat).shape}")
        
        print("🔧 Extracting color features...")
        color_feat = color_extraction(color_prep)
        print(f"   Color features shape: {np.array(color_feat).shape}")
        
        # Combine features
        feature = np.concatenate((glcm_feat, color_feat), axis=1)
        print(f"✅ Combined features shape: {feature.shape}")
        
        # Cross-validation
        print("🔄 Running cross-validation...")
        k_folds = 6  # Optimal from notebook
        scores = crossVal(k_folds, feature, label)
        
        print(f"📈 Cross-validation results:")
        for i, score in enumerate(scores):
            print(f"   Fold {i+1}: {score*100:.4f}%")
        
        mean_accuracy = np.mean(scores) * 100
        std_accuracy = np.std(scores) * 100
        
        print(f"📊 Mean accuracy: {mean_accuracy:.4f}%")
        print(f"📊 Standard deviation: {std_accuracy:.4f}%")
        
        # Check target
        target = 98.65
        if mean_accuracy >= target:
            print(f"🎯 TARGET ACHIEVED! {mean_accuracy:.2f}% >= {target}%")
        else:
            print(f"⚠️  Target not reached: {mean_accuracy:.2f}% < {target}%")
        
        # Train final model
        print("🎯 Training final model...")
        X, y = shuffle(feature, label, random_state=220)
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
        clf.fit(X, y)
        
        # Save model
        model_path = "model_ml/pickle_model_cv_simple.pkl"
        os.makedirs("model_ml", exist_ok=True)
        
        with open(model_path, 'wb') as file:
            pickle.dump(clf, file)
        
        print(f"✅ Model saved to: {model_path}")
        
        print("\n🎉 CROSS-VALIDATION COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

