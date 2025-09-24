#!/usr/bin/env python3

import os
import cv2
import numpy as np

print("Testing dataset access...")

# Check if dataset directory exists
dataset_path = "dataset/dataset_periorbital"
print(f"Dataset path: {dataset_path}")
print(f"Exists: {os.path.exists(dataset_path)}")

if os.path.exists(dataset_path):
    # List directories
    labels = os.listdir(dataset_path)
    print(f"Labels found: {labels}")
    
    # Check first label directory
    if labels:
        first_label = labels[0]
        label_path = os.path.join(dataset_path, first_label)
        print(f"First label path: {label_path}")
        
        if os.path.exists(label_path):
            subdirs = os.listdir(label_path)
            print(f"Subdirectories in {first_label}: {len(subdirs)}")
            
            if subdirs:
                first_subdir = subdirs[0]
                subdir_path = os.path.join(label_path, first_subdir)
                print(f"First subdir path: {subdir_path}")
                
                if os.path.exists(subdir_path):
                    images = os.listdir(subdir_path)
                    print(f"Images in first subdir: {len(images)}")
                    
                    if images:
                        first_image = images[0]
                        image_path = os.path.join(subdir_path, first_image)
                        print(f"First image path: {image_path}")
                        
                        # Try to load image
                        image = cv2.imread(image_path)
                        if image is not None:
                            print(f"Image loaded successfully: {image.shape}")
                        else:
                            print("Failed to load image")
                    else:
                        print("No images found in subdirectory")
                else:
                    print("Subdirectory does not exist")
            else:
                print("No subdirectories found")
        else:
            print("Label directory does not exist")
    else:
        print("No labels found")

print("Dataset access test completed.")

