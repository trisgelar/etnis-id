#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test to debug the SOLID system
"""

import sys
import os

print("Starting simple test...")

# Add ml_training core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_training', 'core'))

try:
    print("1. Testing imports...")
    from ml_training.core.utils import TrainingLogger, ProgressTracker
    print("   SUCCESS: Utils imported")
    
    from ml_training.core.data_loader import EthnicityDataLoader
    print("   SUCCESS: Data loader imported")
    
    from ml_training.core.training_pipeline import PipelineFactory
    print("   SUCCESS: Pipeline factory imported")
    
    print("\n2. Testing basic functionality...")
    logger = TrainingLogger('test')
    logger.info("Test log message")
    print("   SUCCESS: Logger working")
    
    progress_tracker = ProgressTracker(logger)
    progress_tracker.start_task("Test", 5)
    progress_tracker.complete_task()
    print("   SUCCESS: Progress tracker working")
    
    print("\n3. Testing data loader...")
    data_loader = EthnicityDataLoader(logger)
    dataset_path = "../dataset/dataset_periorbital"
    
    if os.path.exists(dataset_path):
        print(f"   SUCCESS: Dataset path exists: {dataset_path}")
        
        # Try loading just a few images
        print("   Loading dataset...")
        images, labels, metadata = data_loader.load_data(dataset_path)
        
        print(f"   SUCCESS: Loaded {len(images)} images")
        print(f"   Images shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Metadata: {metadata}")
        
    else:
        print(f"   ERROR: Dataset path not found: {dataset_path}")
    
    print("\nALL TESTS COMPLETED!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("Test finished.")
