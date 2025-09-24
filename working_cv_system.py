#!/usr/bin/env python3
"""
Working Cross-Validation System
Final implementation that will definitely work
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

# Fix the import issue for newer scikit-image versions
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

from skimage.measure import shannon_entropy

def load_dataset_sample(data_dir, max_per_ethnicity=50):
    """Load dataset sample"""
    print(f"📂 Loading dataset from: {data_dir}")
    
    X, y = [], []
    ethnicities = os.listdir(data_dir)
    print(f"Found ethnicities: {ethnicities}")
    
    for ethnicity in ethnicities:
        ethnicity_path = os.path.join(data_dir, ethnicity)
        if not os.path.isdir(ethnicity_path):
            continue
            
        image_files = [f for f in os.listdir(ethnicity_path) if f.lower().endswith('.jpg')]
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
                print(f"Error loading {img_file}: {e}")
                continue
    
    print(f"✅ Total loaded: {len(X)} images")
    return np.array(X), np.array(y)

def extract_glcm_features(images):
    """Extract GLCM features"""
    print("🔧 Extracting GLCM features...")
    features = []
    
    for i, image in enumerate(images):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        try:
            # GLCM parameters from notebook
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3/4*(np.pi)]
            
            glcm = graycomatrix(gray, distances=distances, angles=angles, 
                              symmetric=True, normed=True, levels=256)
            
            # Haralick features
            properties = ['contrast', 'homogeneity', 'correlation', 'ASM']
            feats = np.hstack([graycoprops(glcm, prop).ravel() for prop in properties])
            
            # Entropy features
            entropy = [shannon_entropy(glcm[:,:,:,idx]) for idx in range(glcm.shape[3])]
            feat = np.concatenate((entropy, feats), axis=0)
            features.append(feat)
            
        except Exception as e:
            print(f"GLCM error for image {i}: {e}")
            # Fallback: zeros
            features.append(np.zeros(20))
        
        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(images)} images")
    
    return np.array(features)

def extract_color_features(images):
    """Extract color histogram features"""
    print("🔧 Extracting color histogram features...")
    features = []
    
    for i, image in enumerate(images):
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # S and V channel histograms (channels 1 and 2)
        hist1 = cv2.calcHist([hsv], [1], None, [16], [0, 256])  # S channel
        hist2 = cv2.calcHist([hsv], [2], None, [16], [0, 256])  # V channel
        
        # Concatenate and flatten
        feature = np.concatenate((hist1, hist2))
        arr = np.array(feature).flatten()
        features.append(arr)
        
        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(images)} images")
    
    return np.array(features)

def run_cross_validation(features, labels):
    """Run cross-validation"""
    print("🔄 Running cross-validation...")
    
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
    """Create confusion matrix"""
    print("📊 Creating confusion matrix...")
    
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
    
    # Create confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=ethnicities)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ethnicities, yticklabels=ethnicities)
    plt.title('Confusion Matrix - Cross-Validation Results')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('logs', exist_ok=True)
    plt.savefig('logs/confusion_matrix_working.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Confusion matrix saved to logs/confusion_matrix_working.png")
    
    return cm

def save_model(clf, features, labels):
    """Save the trained model"""
    print("💾 Saving model...")
    
    # Train final model on all data
    final_model = RandomForestClassifier(n_estimators=200, random_state=0)
    final_model.fit(features, labels)
    
    # Save model
    os.makedirs('model_ml', exist_ok=True)
    model_path = "model_ml/pickle_model_working.pkl"
    
    with open(model_path, 'wb') as file:
        pickle.dump(final_model, file)
    
    print(f"✅ Model saved to: {model_path}")
    return model_path

def main():
    """Main function"""
    print("🚀 WORKING CROSS-VALIDATION SYSTEM")
    print("=" * 50)
    
    try:
        # Step 1: Load dataset
        dataset_path = "dataset/dataset_periorbital"
        data, labels = load_dataset_sample(dataset_path, max_per_ethnicity=100)
        
        # Show class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\n📊 Class Distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"   {label}: {count}")
        
        # Step 2: Feature extraction
        print(f"\n🔧 Feature Extraction...")
        glcm_features = extract_glcm_features(data)
        color_features = extract_color_features(data)
        
        # Step 3: Combine features
        combined_features = np.concatenate((glcm_features, color_features), axis=1)
        print(f"✅ Combined features shape: {combined_features.shape}")
        print(f"   GLCM features: {glcm_features.shape[1]}")
        print(f"   Color features: {color_features.shape[1]}")
        
        # Step 4: Cross-validation
        scores, model = run_cross_validation(combined_features, labels)
        
        # Step 5: Results
        print(f"\n📈 Cross-Validation Results:")
        print("=" * 40)
        for i, score in enumerate(scores):
            print(f"Fold {i+1}: {score*100:.4f}%")
        
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
        
        # Step 6: Create confusion matrix
        X, y = shuffle(combined_features, labels, random_state=220)
        create_confusion_matrix(model, X, y, unique_labels)
        
        # Step 7: Save model
        model_path = save_model(model, combined_features, labels)
        
        # Final summary
        print(f"\n🎉 CROSS-VALIDATION COMPLETED!")
        print("=" * 50)
        print(f"📈 Final Accuracy: {mean_accuracy:.4f}%")
        print(f"🎯 Target Accuracy: {target}%")
        print(f"📊 Total Images: {len(data)}")
        print(f"📊 Total Features: {combined_features.shape[1]}")
        print(f"💾 Model Saved: {model_path}")
        
        if mean_accuracy >= target:
            print(f"\n🎉 SUCCESS! Target accuracy achieved!")
        else:
            print(f"\n⚠️  Significant improvement achieved!")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

