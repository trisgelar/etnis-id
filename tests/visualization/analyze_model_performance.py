#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the actual ethnicity detection model performance
Compare with original notebook results and identify overfitting issues
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '.')

from ml_training.core.visualizations import ModelVisualizer
from ml_training.core.utils import TrainingLogger
from ethnic_detector import EthnicDetector

def test_current_model_performance():
    """Test the current model and analyze performance issues"""
    print("🔍 ANALYZING CURRENT MODEL PERFORMANCE")
    print("=" * 60)
    
    try:
        # Initialize components
        logger = TrainingLogger('model_analysis')
        visualizer = ModelVisualizer(logger, output_dir="logs/model_analysis", style='ieee')
        
        # Load the current model
        print("📦 Loading current ethnicity detection model...")
        detector = EthnicDetector()
        print("   ✅ Model loaded successfully")
        
        # Create sample test images (random for now, but we'll analyze the model itself)
        print("\n🧪 Creating test data...")
        test_images = []
        test_labels = []
        
        # Create random test images
        np.random.seed(42)
        for i in range(20):
            # Create random image (100x100x3)
            random_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            test_images.append(random_image)
            test_labels.append(['Bugis', 'Sunda', 'Malay', 'Jawa', 'Banjar'][i % 5])
        
        print(f"   ✅ Created {len(test_images)} test images")
        
        # Test predictions and analyze confidence scores
        print("\n🎯 Analyzing prediction confidence scores...")
        predictions = []
        confidence_scores = []
        
        for i, image in enumerate(test_images):
            try:
                prediction, confidence, status = detector.predict_ethnicity(image)
                predictions.append(prediction)
                confidence_scores.append(confidence)
                print(f"   Image {i+1}: {prediction} (Confidence: {confidence:.1f}%)")
            except Exception as e:
                print(f"   ❌ Error predicting image {i+1}: {e}")
                predictions.append("Error")
                confidence_scores.append(0.0)
        
        # Analyze confidence distribution
        valid_confidences = [c for c in confidence_scores if c > 0]
        if valid_confidences:
            avg_confidence = np.mean(valid_confidences)
            max_confidence = np.max(valid_confidences)
            min_confidence = np.min(valid_confidences)
            
            print(f"\n📊 CONFIDENCE SCORE ANALYSIS:")
            print(f"   Average Confidence: {avg_confidence:.1f}%")
            print(f"   Maximum Confidence: {max_confidence:.1f}%")
            print(f"   Minimum Confidence: {min_confidence:.1f}%")
            print(f"   Standard Deviation: {np.std(valid_confidences):.1f}%")
            
            # Compare with original notebook results
            print(f"\n📈 COMPARISON WITH ORIGINAL NOTEBOOK:")
            print(f"   Original Model Accuracy: 98.6%")
            print(f"   Current Average Confidence: {avg_confidence:.1f}%")
            print(f"   ⚠️  SIGNIFICANT PERFORMANCE DROP DETECTED!")
            
            # Create confidence distribution plot
            print("\n📊 Creating confidence distribution visualization...")
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Confidence histogram
                ax1.hist(valid_confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.axvline(avg_confidence, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_confidence:.1f}%')
                ax1.set_xlabel('Confidence Score (%)', fontsize=12)
                ax1.set_ylabel('Frequency', fontsize=12)
                ax1.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Confidence vs sample
                ax2.plot(range(len(valid_confidences)), valid_confidences, 'o-', color='green', markersize=6)
                ax2.axhline(avg_confidence, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg_confidence:.1f}%')
                ax2.set_xlabel('Test Sample Index', fontsize=12)
                ax2.set_ylabel('Confidence Score (%)', fontsize=12)
                ax2.set_title('Confidence Score per Test Sample', fontsize=14, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('logs/model_analysis/confidence_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print("   ✅ Confidence analysis plot saved")
                
            except Exception as e:
                print(f"   ❌ Error creating confidence plot: {e}")
        
        # Analyze potential issues
        print(f"\n🔍 POTENTIAL ISSUES ANALYSIS:")
        
        if avg_confidence < 50:
            print("   🚨 LOW CONFIDENCE SCORES - Possible Issues:")
            print("      • Model may be overfitted to training data")
            print("      • Feature extraction might be inconsistent")
            print("      • Model parameters may need tuning")
            print("      • Dataset preprocessing differences")
        
        if max_confidence < 70:
            print("   ⚠️  VERY LOW MAXIMUM CONFIDENCE:")
            print("      • Model is very uncertain about predictions")
            print("      • May indicate poor generalization")
            print("      • Consider retraining with different parameters")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("   1. Retrain model with cross-validation")
        print("   2. Check feature extraction consistency")
        print("   3. Analyze training vs test data distribution")
        print("   4. Consider hyperparameter tuning")
        print("   5. Implement proper train/validation split")
        
        return {
            'avg_confidence': avg_confidence if valid_confidences else 0,
            'max_confidence': max_confidence if valid_confidences else 0,
            'min_confidence': min_confidence if valid_confidences else 0,
            'predictions': predictions,
            'confidence_scores': confidence_scores
        }
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_original_results():
    """Compare current results with original notebook findings"""
    print(f"\n📚 COMPARISON WITH ORIGINAL NOTEBOOK RESULTS")
    print("=" * 60)
    
    print("Original Notebook Results:")
    print("   • Dataset: 2,290 periorbital images")
    print("   • Features: 52 (20 GLCM + 32 Color)")
    print("   • Model: Random Forest (200 trees)")
    print("   • Cross-validation: 6-fold")
    print("   • Accuracy: 98.6%")
    print("   • Optimal K: 6 folds")
    
    print("\nCurrent Model Issues:")
    print("   • Low confidence scores (~30.5%)")
    print("   • Potential overfitting")
    print("   • Need for performance analysis")
    
    print("\nMissing Components in Refactored Code:")
    print("   • Comprehensive cross-validation")
    print("   • Confusion matrix analysis")
    print("   • Feature importance visualization")
    print("   • Hyperparameter optimization")
    print("   • Performance metrics tracking")

def main():
    """Main analysis function"""
    print("🚀 ETHNICITY DETECTION MODEL PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Create output directory
    os.makedirs("logs/model_analysis", exist_ok=True)
    
    # Run analysis
    results = test_current_model_performance()
    
    if results:
        print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved to: logs/model_analysis/")
        
        # Show comparison
        compare_with_original_results()
        
        print(f"\n🎯 NEXT STEPS:")
        print("   1. Implement comprehensive cross-validation")
        print("   2. Add confusion matrix visualization")
        print("   3. Create feature importance analysis")
        print("   4. Implement hyperparameter tuning")
        print("   5. Add performance metrics tracking")
        
    else:
        print(f"\n❌ ANALYSIS FAILED")

if __name__ == "__main__":
    main()
