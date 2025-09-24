#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indonesian Ethnicity Recognition - Fixed Training Script
Pengenalan Etnis Indonesia Berdasarkan Citra Wajah Menggunakan GLCM dan Color Histogram

Fixed version that works locally without Google Colab dependencies
"""

import os
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.utils import shuffle
from skimage.measure import shannon_entropy
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Mapping label ke nama suku
label_map = {0: "Bugis", 1: "Sunda", 2: "Malay", 3: "Jawa", 4: "Banjar"}

def load_data(data_dir):
    """Load image data from directory structure"""
    print(f"📁 Loading data from: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory {data_dir} does not exist!")
        print("💡 Please create the directory structure:")
        print("   dataset_periorbital/")
        print("   ├── Bugis/")
        print("   ├── Sunda/")
        print("   ├── Malay/")
        print("   ├── Jawa/")
        print("   └── Banjar/")
        return None, None, None, None, None
    
    data = []
    label = []
    idx = []
    name = []
    fld = []

    classes = sorted(os.listdir(data_dir))  # Sort for consistent ordering
    print(f"📂 Found classes: {classes}")

    for i, cls in enumerate(classes):
        class_path = os.path.join(data_dir, cls)
        if not os.path.isdir(class_path):
            continue
            
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"📸 {cls}: {len(images)} images")

        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Could not load image: {img_path}")
                continue

            # Resize ke ukuran standar
            img = cv2.resize(img, (400, 200))

            data.append(img)
            label.append(i)
            idx.append(img_path)
            name.append(img_name)
            fld.append(cls)

    if len(data) == 0:
        print("❌ No valid images found!")
        return None, None, None, None, None
        
    print(f"✅ Total loaded: {len(data)} images")
    return np.array(data), np.array(label), np.array(idx), np.array(name), np.array(fld)

def preprocessing_glcm(data):
    """Mengubah citra RGB menjadi grayscale untuk GLCM"""
    print("🔄 Preprocessing GLCM: Converting RGB to Grayscale...")
    grays = []
    for i in range(len(data)):
        img = cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY)
        grays.append(img)
    print(f"✅ Converted {len(grays)} images to grayscale")
    return grays

def preprocessing_color(array):
    """Mengubah citra RGB menjadi HSV untuk Color Histogram"""
    print("🔄 Preprocessing Color: Converting RGB to HSV...")
    preprocessed = []
    for i in range(len(array)):
        img = array[i].copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        preprocessed.append(img)
    print(f"✅ Converted {len(preprocessed)} images to HSV")
    return preprocessed

def glcm_extraction(data):
    """Ekstraksi fitur GLCM"""
    print("🧮 Extracting GLCM features...")
    distances = [1]     # Jarak dari satu tetangga ke tetangga lain
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]   # Orientasi 0, 45, 90, dan 135 derajat
    levels = 256
    symmetric = True
    normed = True
    
    features = []

    for i, img in enumerate(data):
        if i % 100 == 0:  # Progress indicator
            print(f"   Processing image {i+1}/{len(data)}")
            
        # Resize image jika terlalu besar
        if img.shape[0] > 256 or img.shape[1] > 256:
            img = cv2.resize(img, (256, 256))
        
        # Calculate GLCM
        glcm = graycomatrix(img, distances=distances, angles=angles, 
                           levels=levels, symmetric=symmetric, normed=normed)
        
        # Haralick Features
        properties = ['contrast', 'homogeneity', 'correlation', 'ASM']
        feats = np.hstack([graycoprops(glcm, prop).ravel() for prop in properties])

        # Entropy feature
        entropy = [shannon_entropy(glcm[:, :, :, idx]) for idx in range(glcm.shape[3])]
        feat = np.concatenate((entropy, feats), axis=0)
        features.append(feat)

    print(f"✅ GLCM features extracted: {len(features)} samples, {len(features[0])} features each")
    return np.array(features)

def color_extraction(img_array):
    """Ekstraksi fitur Color Histogram"""
    print("🎨 Extracting Color Histogram features...")
    features = []

    for i, img in enumerate(img_array):
        if i % 100 == 0:  # Progress indicator
            print(f"   Processing image {i+1}/{len(img_array)}")
            
        hist1 = cv2.calcHist([img], [1], None, [16], [0, 256])  # Channel 1 (S)
        hist2 = cv2.calcHist([img], [2], None, [16], [0, 256])  # Channel 2 (V)
        fitur = np.concatenate((hist1, hist2))
        arr = np.array(fitur).flatten()
        features.append(arr)
    
    print(f"✅ Color features extracted: {len(features)} samples, {len(features[0])} features each")
    return features

def crossVal(K, X, y):
    """Cross Validation menggunakan Random Forest"""
    print(f"🔄 Running {K}-fold cross validation...")
    X, y = shuffle(X, y, random_state=220)
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    cv = StratifiedKFold(n_splits=K)
    scores = cross_val_score(clf, X, y, cv=cv)
    return scores

def train_final_model(X, y):
    """Train final model dengan k optimal"""
    print("🤖 Training final Random Forest model...")
    X, y = shuffle(X, y, random_state=220)
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X, y)
    return clf

def save_model(model, filename="model_ml/pickle_model.pkl"):
    """Save trained model to file"""
    print(f"💾 Saving model to {filename}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"✅ Model saved successfully!")

def main():
    """Main training function"""
    print("🚀 STARTING INDONESIAN ETHNICITY RECOGNITION TRAINING")
    print("=" * 70)
    
    # Check if dataset exists
    dataset_path = "dataset_periorbital"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory '{dataset_path}' not found!")
        print("\n💡 To use this training script:")
        print("1. Create a dataset directory with the following structure:")
        print("   dataset_periorbital/")
        print("   ├── Bugis/")
        print("   ├── Sunda/")
        print("   ├── Malay/")
        print("   ├── Jawa/")
        print("   └── Banjar/")
        print("2. Place face images in each ethnicity folder")
        print("3. Run this script again")
        return False
    
    try:
        # Load Data
        print("\n📊 PHASE 1: LOADING DATA")
        print("-" * 30)
        data, label, idx, img_name, fld = load_data(dataset_path)
        
        if data is None:
            return False
        
        print(f"Total Data: {len(data)}")
        unique_labels, counts = np.unique(label, return_counts=True)
        print("Data per class:")
        for i, count in enumerate(counts):
            print(f"  {label_map[i]}: {count} images")
        
        # Preprocessing
        print("\n🔄 PHASE 2: PREPROCESSING")
        print("-" * 30)
        glcm_prep = preprocessing_glcm(data)
        color_prep = preprocessing_color(data)
        
        # Feature Extraction
        print("\n🧮 PHASE 3: FEATURE EXTRACTION")
        print("-" * 30)
        glcm_feat = glcm_extraction(glcm_prep)
        color_feat = color_extraction(color_prep)
        
        # Combine Features
        print("\n🔗 PHASE 4: COMBINING FEATURES")
        print("-" * 30)
        feature = np.concatenate((glcm_feat, color_feat), axis=1)
        print(f"Combined features shape: {feature.shape}")
        print(f"GLCM features: {glcm_feat.shape[1]}")
        print(f"Color features: {len(color_feat[0])}")
        print(f"Total features: {feature.shape[1]}")
        
        # Model Training and Evaluation
        print("\n🎯 PHASE 5: MODEL TRAINING")
        print("-" * 30)
        
        # Cross validation dengan k optimal (6)
        print("🔄 Running 6-fold cross validation...")
        cv_scores = crossVal(6, feature, label)
        mean_accuracy = np.mean(cv_scores) * 100
        std_accuracy = np.std(cv_scores) * 100
        
        print(f"✅ Cross-validation results:")
        print(f"   Mean accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
        print(f"   Individual fold scores: {cv_scores * 100}")
        
        # Train final model
        final_model = train_final_model(feature, label)
        
        # Save model
        print("\n💾 PHASE 6: SAVING MODEL")
        print("-" * 30)
        save_model(final_model)
        
        # Final summary
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📊 Final Model Performance:")
        print(f"   - Mean CV Accuracy: {mean_accuracy:.2f}%")
        print(f"   - Standard Deviation: {std_accuracy:.2f}%")
        print(f"   - Features used: {feature.shape[1]} (GLCM: {glcm_feat.shape[1]}, Color: {len(color_feat[0])})")
        print(f"   - Model saved to: model_ml/pickle_model.pkl")
        print(f"   - Supported ethnicities: {list(label_map.values())}")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Training script completed successfully!")
        else:
            print("\n❌ Training script failed!")
    except KeyboardInterrupt:
        print("\n⚠️ Training cancelled by user")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
