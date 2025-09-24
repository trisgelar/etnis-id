#!/usr/bin/env python3

print("🚀 SIMPLE CROSS-VALIDATION TEST")
print("=" * 40)

import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

print("✅ Imports successful")

# Load just 2 images per ethnicity for very quick test
print("\n📂 Loading minimal dataset...")
dataset_path = "dataset/dataset_periorbital"
ethnicities = os.listdir(dataset_path)
print(f"Found ethnicities: {ethnicities}")

X, y = [], []
for ethnicity in ethnicities[:2]:  # Just 2 ethnicities
    ethnicity_path = os.path.join(dataset_path, ethnicity)
    image_files = [f for f in os.listdir(ethnicity_path) if f.lower().endswith('.jpg')][:2]  # Just 2 images
    print(f"Loading {len(image_files)} images for {ethnicity}")
    
    for img_file in image_files:
        img_path = os.path.join(ethnicity_path, img_file)
        image = cv2.imread(img_path)
        if image is not None:
            # Simple feature: resize and flatten
            image_resized = cv2.resize(image, (32, 32))
            X.append(image_resized.flatten())
            y.append(ethnicity)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} images")
print(f"Feature shape: {X.shape}")
print(f"Classes: {np.unique(y)}")

# Simple cross-validation
print("\n🔄 Running cross-validation...")
X, y = shuffle(X, y, random_state=42)

clf = RandomForestClassifier(n_estimators=10, random_state=42)
scores = cross_val_score(clf, X, y, cv=3)

print("Cross-validation scores:")
for i, score in enumerate(scores):
    print(f"  Fold {i+1}: {score*100:.2f}%")

mean_accuracy = np.mean(scores) * 100
print(f"\n📊 Mean accuracy: {mean_accuracy:.2f}%")

print("\n🎉 SIMPLE TEST COMPLETED!")

