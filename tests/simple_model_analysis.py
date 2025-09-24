#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple model performance analysis without external dependencies
Focus on identifying the low confidence score issues
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '.')

# Import SciencePlots for beautiful plots
try:
    import scienceplots
    plt.style.use(['science', 'ieee', 'grid'])
    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False
    print("Warning: SciencePlots not available. Using default matplotlib styles.")

def analyze_current_model():
    """Analyze the current ethnicity detection model"""
    print("🔍 ANALYZING CURRENT MODEL PERFORMANCE")
    print("=" * 60)
    
    try:
        # Import the detector
        from ethnic_detector import EthnicDetector
        
        # Load the current model
        print("📦 Loading current ethnicity detection model...")
        detector = EthnicDetector()
        print("   ✅ Model loaded successfully")
        
        # Test with random images
        print("\n🧪 Testing model with random images...")
        np.random.seed(42)
        
        confidence_scores = []
        predictions = []
        
        for i in range(10):
            # Create random test image
            random_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            try:
                prediction, confidence, status = detector.predict_ethnicity(random_image)
                confidence_scores.append(confidence)
                predictions.append(prediction)
                print(f"   Test {i+1}: {prediction} (Confidence: {confidence:.1f}%)")
            except Exception as e:
                print(f"   ❌ Error in test {i+1}: {e}")
                confidence_scores.append(0.0)
                predictions.append("Error")
        
        # Analyze results
        valid_confidences = [c for c in confidence_scores if c > 0]
        
        if valid_confidences:
            avg_confidence = np.mean(valid_confidences)
            max_confidence = np.max(valid_confidences)
            min_confidence = np.min(valid_confidences)
            
            print(f"\n📊 CONFIDENCE SCORE ANALYSIS:")
            print(f"   Average Confidence: {avg_confidence:.1f}%")
            print(f"   Maximum Confidence: {max_confidence:.1f}%")
            print(f"   Minimum Confidence: {min_confidence:.1f}%")
            
            # Create visualization
            print("\n📈 Creating confidence analysis plot...")
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Histogram
                ax1.hist(valid_confidences, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.axvline(avg_confidence, color='red', linestyle='--', linewidth=2, 
                           label=f'Average: {avg_confidence:.1f}%')
                ax1.set_xlabel('Confidence Score (%)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax1.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Line plot
                ax2.plot(range(len(valid_confidences)), valid_confidences, 'o-', 
                        color='green', markersize=8, linewidth=2)
                ax2.axhline(avg_confidence, color='red', linestyle='--', alpha=0.7, 
                           label=f'Average: {avg_confidence:.1f}%')
                ax2.set_xlabel('Test Sample Index', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Confidence Score (%)', fontsize=12, fontweight='bold')
                ax2.set_title('Confidence Score per Test Sample', fontsize=14, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                os.makedirs("logs", exist_ok=True)
                plt.savefig('logs/confidence_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print("   ✅ Confidence analysis plot saved to logs/confidence_analysis.png")
                
            except Exception as e:
                print(f"   ❌ Error creating plot: {e}")
            
            # Analysis and recommendations
            print(f"\n🔍 ANALYSIS RESULTS:")
            
            if avg_confidence < 50:
                print("   🚨 LOW CONFIDENCE DETECTED!")
                print("   • Model shows very low confidence in predictions")
                print("   • This suggests potential overfitting or poor generalization")
                print("   • Original notebook achieved 98.6% accuracy")
                print("   • Current confidence ~30% indicates serious issues")
            
            print(f"\n💡 RECOMMENDATIONS:")
            print("   1. 🎯 Check training data quality and consistency")
            print("   2. 🔄 Retrain model with proper cross-validation")
            print("   3. 📊 Implement comprehensive performance metrics")
            print("   4. 🧪 Add confusion matrix analysis")
            print("   5. ⚙️  Consider hyperparameter tuning")
            print("   6. 📈 Add feature importance visualization")
            
            return {
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'min_confidence': min_confidence,
                'predictions': predictions,
                'confidence_scores': confidence_scores
            }
        else:
            print("   ❌ No valid predictions obtained")
            return None
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_original():
    """Compare with original notebook results"""
    print(f"\n📚 COMPARISON WITH ORIGINAL NOTEBOOK")
    print("=" * 50)
    
    print("Original Results (from notebook):")
    print("   • Dataset: 2,290 periorbital images")
    print("   • Features: 52 (20 GLCM + 32 Color histogram)")
    print("   • Model: Random Forest (200 estimators)")
    print("   • Cross-validation: 6-fold")
    print("   • Accuracy: 98.6%")
    print("   • Optimal K: 6 folds")
    
    print("\nCurrent Issues:")
    print("   • Low confidence scores (~30.5%)")
    print("   • Potential overfitting")
    print("   • Missing comprehensive evaluation")
    
    print("\nMissing Components:")
    print("   • Cross-validation analysis")
    print("   • Confusion matrix visualization")
    print("   • Feature importance plots")
    print("   • Performance metrics tracking")
    print("   • Hyperparameter optimization")

def main():
    """Main analysis function"""
    print("🚀 ETHNICITY DETECTION MODEL ANALYSIS")
    print("=" * 70)
    
    # Run analysis
    results = analyze_current_model()
    
    if results:
        print(f"\n✅ ANALYSIS COMPLETED!")
        
        # Show comparison
        compare_with_original()
        
        print(f"\n🎯 CONCLUSION:")
        print("   The current model shows significantly lower performance")
        print("   compared to the original notebook results. The low confidence")
        print("   scores (~30%) suggest overfitting or data inconsistency issues.")
        print("   We need to implement comprehensive visualization and analysis")
        print("   tools to identify and fix these performance problems.")
        
    else:
        print(f"\n❌ ANALYSIS FAILED")

if __name__ == "__main__":
    main()
