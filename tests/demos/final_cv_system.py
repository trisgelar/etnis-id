#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Cross-Validation System
Corrected implementation for the actual dataset structure
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
import time

def load_data_corrected(data_dir):
    """Load data with corrected structure - images directly in ethnicity folders"""
    X, y = [], []
    
    print(f"Loading data from: {data_dir}")
    
    # Get all ethnicity folders
    ethnicities = os.listdir(data_dir)
    print(f"Found ethnicities: {ethnicities}")
    
    for ethnicity in ethnicities:
        ethnicity_path = os.path.join(data_dir, ethnicity)
        if not os.path.isdir(ethnicity_path):
            continue
            
        # Get all images in this ethnicity folder
        image_files = [f for f in os.listdir(ethnicity_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
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
    
    print(f"Total loaded: {len(X)} images")
    return np.array(X), np.array(y)

def preprocessing_glcm(data):
    """Convert RGB to Grayscale for GLCM"""
    print("Converting to grayscale...")
    grays = []
    for i, image in enumerate(data):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)} images")
    return grays

def preprocessing_color(data):
    """Convert RGB to HSV for Color Histogram"""
    print("Converting to HSV...")
    hsv_images = []
    for i, image in enumerate(data):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_images.append(hsv)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)} images")
    return hsv_images

def glcm_extraction(data):
    """Extract GLCM features exactly like notebook"""
    print("Extracting GLCM features...")
    features = []
    
    # Parameters from notebook
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3/4*(np.pi)]
    
    for i, gray_img in enumerate(data):
        # GLCM computation
        glcm = greycomatrix(gray_img, 
                           distances=distances, 
                           angles=angles, 
                           symmetric=True, 
                           normed=True, 
                           levels=256)
        
        # Haralick features
        properties = ['contrast', 'homogeneity', 'correlation', 'ASM']
        feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
        
        # Entropy features
        entropy = [shannon_entropy(glcm[:,:,:,idx]) for idx in range(glcm.shape[3])]
        feat = np.concatenate((entropy, feats), axis=0)
        
        features.append(feat)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)} images")
    
    return np.array(features)

def color_extraction(hsv_images):
    """Extract color histogram features exactly like notebook"""
    print("Extracting color histogram features...")
    features = []
    
    for i, hsv_img in enumerate(hsv_images):
        # S and V channel histograms (channels 1 and 2)
        hist1 = cv2.calcHist([hsv_img], [1], None, [16], [0, 256])  # S channel
        hist2 = cv2.calcHist([hsv_img], [2], None, [16], [0, 256])  # V channel
        
        # Concatenate and flatten
        feature = np.concatenate((hist1, hist2))
        arr = np.array(feature).flatten()
        
        features.append(arr)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(hsv_images)} images")
    
    return np.array(features)

def run_cross_validation(features, labels):
    """Run cross-validation exactly like notebook"""
    print("Running cross-validation...")
    
    # Parameters from notebook
    k_folds = 6
    n_estimators = 200
    random_state = 0
    
    # Shuffle data exactly like notebook
    X, y = shuffle(features, labels, random_state=220)
    
    # Create model exactly like notebook
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    
    # Create CV splitter exactly like notebook
    cv = StratifiedKFold(n_splits=k_folds)
    
    # Run cross-validation
    scores = cross_val_score(clf, X, y, cv=cv)
    
    return scores, clf

def create_confusion_matrix(clf, X, y, ethnicities):
    """Create detailed confusion matrix"""
    cv = StratifiedKFold(n_splits=6)
    
    # Get predictions for confusion matrix
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=ethnicities)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ethnicities, yticklabels=ethnicities)
    plt.title('Confusion Matrix - Cross-Validation Results')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('logs/confusion_matrix_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm, all_y_true, all_y_pred

def main():
    """Main function"""
    print("🚀 FINAL CROSS-VALIDATION SYSTEM")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs('logs', exist_ok=True)
    os.makedirs('model_ml', exist_ok=True)
    
    try:
        # Step 1: Load dataset
        dataset_path = "dataset/dataset_periorbital"
        data, labels = load_data_corrected(dataset_path)
        
        # Show class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\n📊 Class Distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"   {label}: {count}")
        
        # Step 2: Preprocessing
        print("\n🔧 Preprocessing...")
        gray_images = preprocessing_glcm(data)
        hsv_images = preprocessing_color(data)
        
        # Step 3: Feature extraction
        print("\n🔧 Feature Extraction...")
        glcm_features = glcm_extraction(gray_images)
        color_features = color_extraction(hsv_images)
        
        print(f"GLCM features shape: {glcm_features.shape}")
        print(f"Color features shape: {color_features.shape}")
        
        # Step 4: Combine features
        combined_features = np.concatenate((glcm_features, color_features), axis=1)
        print(f"Combined features shape: {combined_features.shape}")
        
        # Step 5: Cross-validation
        print("\n🔄 Cross-Validation...")
        scores, model = run_cross_validation(combined_features, labels)
        
        # Results
        print("\n📈 Cross-Validation Results:")
        for i, score in enumerate(scores):
            print(f"   Fold {i+1}: {score*100:.4f}%")
        
        mean_accuracy = np.mean(scores) * 100
        std_accuracy = np.std(scores) * 100
        
        print(f"\n📊 Mean accuracy: {mean_accuracy:.4f}%")
        print(f"📊 Standard deviation: {std_accuracy:.4f}%")
        
        # Check target
        target = 98.65
        if mean_accuracy >= target:
            print(f"🎯 TARGET ACHIEVED! {mean_accuracy:.2f}% >= {target}%")
        else:
            print(f"⚠️  Target not reached: {mean_accuracy:.2f}% < {target}%")
        
        # Step 6: Train final model and create confusion matrix
        print("\n📊 Creating confusion matrix...")
        X, y = shuffle(combined_features, labels, random_state=220)
        cm, y_true, y_pred = create_confusion_matrix(model, X, y, unique_labels)
        
        # Step 7: Train final model on all data
        print("\n🎯 Training final model...")
        final_model = RandomForestClassifier(n_estimators=200, random_state=0)
        final_model.fit(combined_features, labels)
        
        # Step 8: Save model
        model_path = "model_ml/pickle_model_cv_final.pkl"
        with open(model_path, 'wb') as file:
            pickle.dump(final_model, file)
        
        print(f"✅ Model saved to: {model_path}")
        
        # Step 9: Create performance summary
        print("\n📋 PERFORMANCE SUMMARY:")
        print("=" * 50)
        print(f"📈 Final Accuracy: {mean_accuracy:.4f}%")
        print(f"📈 Standard Deviation: {std_accuracy:.4f}%")
        print(f"🎯 Target Accuracy: {target}%")
        print(f"📊 Total Images: {len(data)}")
        print(f"📊 Total Features: {combined_features.shape[1]}")
        print(f"📊 GLCM Features: {glcm_features.shape[1]}")
        print(f"📊 Color Features: {color_features.shape[1]}")
        print(f"⏱️  Total Time: {time.time() - start_time:.2f} seconds")
        
        if mean_accuracy >= target:
            print("\n🎉 SUCCESS! Model achieved target accuracy!")
            print("✅ Ready for deployment!")
        else:
            print(f"\n⚠️  Model accuracy below target by {target - mean_accuracy:.2f}%")
            print("🔧 Consider further optimization")
        
        print("\n🎉 CROSS-VALIDATION SYSTEM COMPLETED!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

