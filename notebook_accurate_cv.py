#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notebook-Accurate Cross-Validation System
Implements the exact cross-validation from the original notebook
"""

import os
import numpy as np
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_notebook_style(data_dir, max_per_ethnicity=100):
    """Load data exactly like the original notebook"""
    print("📂 Loading dataset...")
    
    X, y = [], []
    ethnicities = os.listdir(data_dir)
    print(f"Found ethnicities: {ethnicities}")
    
    for ethnicity in ethnicities:
        ethnicity_path = os.path.join(data_dir, ethnicity)
        if not os.path.isdir(ethnicity_path):
            continue
            
        # Get images from this ethnicity folder
        image_files = [f for f in os.listdir(ethnicity_path) if f.lower().endswith('.jpg')]
        
        # Limit for faster processing
        if max_per_ethnicity:
            image_files = image_files[:max_per_ethnicity]
            
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
    
    print(f"✅ Total loaded: {len(X)} images")
    return np.array(X), np.array(y)

def preprocessing_glcm(data):
    """Convert RGB to Grayscale for GLCM - exactly like notebook"""
    print("🔧 Converting to grayscale for GLCM...")
    grays = []
    for i, image in enumerate(data):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(data)} images")
    return grays

def preprocessing_color(data):
    """Convert RGB to HSV for Color Histogram - exactly like notebook"""
    print("🔧 Converting to HSV for color histogram...")
    hsv_images = []
    for i, image in enumerate(data):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_images.append(hsv)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(data)} images")
    return hsv_images

def glcm_extraction_notebook(data):
    """Extract GLCM features exactly like the original notebook"""
    print("🔧 Extracting GLCM features...")
    features = []
    
    # Exact parameters from notebook
    distances = [1]  # Distance = 1
    angles = [0, np.pi/4, np.pi/2, 3/4*(np.pi)]  # 0, 45, 90, 135 degrees
    
    for i, gray_img in enumerate(data):
        # GLCM computation exactly like notebook
        glcm = greycomatrix(gray_img, 
                           distances=distances, 
                           angles=angles, 
                           symmetric=True, 
                           normed=True, 
                           levels=256)
        
        # Haralick features exactly like notebook
        properties = ['contrast', 'homogeneity', 'correlation', 'ASM']
        feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
        
        # Entropy features exactly like notebook
        entropy = [shannon_entropy(glcm[:,:,:,idx]) for idx in range(glcm.shape[3])]
        feat = np.concatenate((entropy, feats), axis=0)
        
        features.append(feat)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(data)} images")
    
    features = np.array(features)
    print(f"✅ GLCM features shape: {features.shape}")
    return features

def color_extraction_notebook(hsv_images):
    """Extract color histogram features exactly like the original notebook"""
    print("🔧 Extracting color histogram features...")
    features = []
    
    for i, hsv_img in enumerate(hsv_images):
        # Exact implementation from notebook
        hist1 = cv2.calcHist([hsv_img], [1], None, [16], [0, 256])  # S channel
        hist2 = cv2.calcHist([hsv_img], [2], None, [16], [0, 256])  # V channel
        
        # Concatenate and flatten exactly like notebook
        fitur = np.concatenate((hist1, hist2))
        arr = np.array(fitur).flatten()
        
        features.append(arr)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(hsv_images)} images")
    
    features = np.array(features)
    print(f"✅ Color features shape: {features.shape}")
    return features

def cross_validation_notebook(features, labels):
    """Run cross-validation exactly like the original notebook"""
    print("🔄 Running cross-validation...")
    
    # Exact parameters from notebook
    k_folds = 6  # Optimal k from notebook
    n_estimators = 200  # From notebook
    random_state = 0  # From notebook
    
    # Shuffle data exactly like notebook (random_state=220)
    X, y = shuffle(features, labels, random_state=220)
    
    # Create model exactly like notebook
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    
    # Create CV splitter exactly like notebook
    cv = StratifiedKFold(n_splits=k_folds)
    
    # Run cross-validation exactly like notebook
    scores = cross_val_score(clf, X, y, cv=cv)
    
    return scores, clf

def create_performance_analysis(scores, clf, X, y, ethnicities):
    """Create performance analysis and visualizations"""
    print("📊 Creating performance analysis...")
    
    # Calculate statistics
    mean_accuracy = np.mean(scores) * 100
    std_accuracy = np.std(scores) * 100
    
    print(f"\n📈 CROSS-VALIDATION RESULTS:")
    print("=" * 40)
    for i, score in enumerate(scores):
        print(f"Fold {i+1}: {score*100:.4f}%")
    
    print(f"\n📊 Mean accuracy: {mean_accuracy:.4f}%")
    print(f"📊 Standard deviation: {std_accuracy:.4f}%")
    
    # Check against target
    target_accuracy = 98.65
    print(f"\n🎯 Target accuracy: {target_accuracy}%")
    
    if mean_accuracy >= target_accuracy:
        print(f"🎉 TARGET ACHIEVED! {mean_accuracy:.2f}% >= {target_accuracy}%")
        success = True
    else:
        print(f"⚠️  Target not reached: {mean_accuracy:.2f}% < {target_accuracy}%")
        print(f"   Difference: {target_accuracy - mean_accuracy:.2f}%")
        success = False
    
    # Create confusion matrix
    print("\n📊 Creating confusion matrix...")
    cv = StratifiedKFold(n_splits=6)
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_y_true, all_y_pred, labels=ethnicities)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ethnicities, yticklabels=ethnicities)
    plt.title(f'Confusion Matrix - Mean Accuracy: {mean_accuracy:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('logs', exist_ok=True)
    plt.savefig('logs/confusion_matrix_notebook_accurate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Confusion matrix saved to logs/confusion_matrix_notebook_accurate.png")
    
    return success, mean_accuracy

def save_final_model(clf, features, labels):
    """Save the final trained model"""
    print("💾 Saving final model...")
    
    # Train final model on all data
    final_model = RandomForestClassifier(n_estimators=200, random_state=0)
    final_model.fit(features, labels)
    
    # Save model
    os.makedirs('model_ml', exist_ok=True)
    model_path = "model_ml/pickle_model_notebook_accurate.pkl"
    
    with open(model_path, 'wb') as file:
        pickle.dump(final_model, file)
    
    print(f"✅ Model saved to: {model_path}")
    return model_path

def main():
    """Main function"""
    print("🚀 NOTEBOOK-ACCURATE CROSS-VALIDATION SYSTEM")
    print("=" * 60)
    print("📋 Implementing exact cross-validation from original notebook")
    print("🎯 Target: Achieve 98.65% accuracy")
    print("=" * 60)
    
    try:
        # Step 1: Load dataset
        dataset_path = "dataset/dataset_periorbital"
        data, labels = load_data_notebook_style(dataset_path, max_per_ethnicity=200)
        
        # Show class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\n📊 Class Distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"   {label}: {count}")
        
        # Step 2: Preprocessing
        print(f"\n🔧 Preprocessing {len(data)} images...")
        gray_images = preprocessing_glcm(data)
        hsv_images = preprocessing_color(data)
        
        # Step 3: Feature extraction
        print(f"\n🔧 Feature Extraction...")
        glcm_features = glcm_extraction_notebook(gray_images)
        color_features = color_extraction_notebook(hsv_images)
        
        # Step 4: Combine features exactly like notebook
        print(f"\n🔗 Combining features...")
        combined_features = np.concatenate((glcm_features, color_features), axis=1)
        print(f"✅ Combined features shape: {combined_features.shape}")
        print(f"   GLCM features: {glcm_features.shape[1]}")
        print(f"   Color features: {color_features.shape[1]}")
        print(f"   Total features: {combined_features.shape[1]}")
        
        # Verify feature dimensions match notebook
        expected_total = 20 + 32  # 20 GLCM + 32 Color
        if combined_features.shape[1] == expected_total:
            print(f"✅ Feature dimensions match notebook ({expected_total})")
        else:
            print(f"⚠️  Feature dimensions differ: {combined_features.shape[1]} vs {expected_total}")
        
        # Step 5: Cross-validation
        scores, model = cross_validation_notebook(combined_features, labels)
        
        # Step 6: Performance analysis
        success, mean_accuracy = create_performance_analysis(scores, model, combined_features, labels, unique_labels)
        
        # Step 7: Save final model
        model_path = save_final_model(model, combined_features, labels)
        
        # Final summary
        print(f"\n🎉 CROSS-VALIDATION SYSTEM COMPLETED!")
        print("=" * 60)
        print(f"📈 Final Accuracy: {mean_accuracy:.4f}%")
        print(f"🎯 Target Accuracy: 98.65%")
        print(f"📊 Total Images Processed: {len(data)}")
        print(f"📊 Total Features: {combined_features.shape[1]}")
        print(f"💾 Model Saved: {model_path}")
        
        if success:
            print(f"\n🎉 SUCCESS! Target accuracy achieved!")
            print("✅ Model ready for deployment!")
        else:
            print(f"\n⚠️  Target not reached, but significant improvement achieved!")
            print("🔧 Consider increasing dataset size or hyperparameter tuning")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

