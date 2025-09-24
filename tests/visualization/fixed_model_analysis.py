#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed model performance analysis with error handling
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '.')

# Set up SciencePlots for beautiful publication-quality plots
try:
    import scienceplots
    plt.style.use(['science', 'ieee', 'grid'])
    SCIENCEPLOTS_AVAILABLE = True
    print("✅ SciencePlots loaded successfully")
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False
    print("⚠️ SciencePlots not available. Using default matplotlib styles.")

def safe_model_prediction(detector, image):
    """Safely predict ethnicity with error handling"""
    try:
        # Ensure image is in correct format
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be 3D with 3 channels")
        
        prediction, confidence, status = detector.predict_ethnicity(image)
        
        # Get probabilities safely
        try:
            # Reshape image for model prediction
            image_reshaped = image.reshape(1, -1)
            probabilities = detector.model.predict_proba(image_reshaped)[0]
        except Exception as e:
            print(f"   Warning: Could not get probabilities: {e}")
            probabilities = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Default uniform distribution
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'status': status,
            'probabilities': probabilities
        }
    except Exception as e:
        print(f"   Error in prediction: {e}")
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'status': 'Error',
            'probabilities': np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        }

def create_simple_diagnostic_plots():
    """Create simple diagnostic plots with error handling"""
    print("🔍 CREATING MODEL DIAGNOSTIC PLOTS")
    print("=" * 50)
    
    try:
        from ethnic_detector import EthnicDetector
        
        # Load model
        print("📦 Loading ethnicity detection model...")
        detector = EthnicDetector()
        print("   ✅ Model loaded successfully")
        
        # Test with simple random images
        print("📊 Testing with random images...")
        test_cases = []
        
        np.random.seed(42)
        for i in range(5):
            print(f"   Testing image {i+1}...")
            try:
                # Create random test image
                random_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                
                # Get prediction
                result = safe_model_prediction(detector, random_image)
                
                test_cases.append({
                    'type': f'Random_{i+1}',
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
                
                print(f"      Result: {result['prediction']} (Confidence: {result['confidence']:.1f}%)")
                
            except Exception as e:
                print(f"      Error in test {i+1}: {e}")
                test_cases.append({
                    'type': f'Random_{i+1}',
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'probabilities': np.array([0.2, 0.2, 0.2, 0.2, 0.2])
                })
        
        # Test with uniform color images
        print("📊 Testing with uniform color images...")
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        
        for i, color in enumerate(colors):
            print(f"   Testing uniform color {color}...")
            try:
                uniform_image = np.full((100, 100, 3), color, dtype=np.uint8)
                result = safe_model_prediction(detector, uniform_image)
                
                test_cases.append({
                    'type': f'Uniform_{color[0]}',
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
                
                print(f"      Result: {result['prediction']} (Confidence: {result['confidence']:.1f}%)")
                
            except Exception as e:
                print(f"      Error in uniform test {i+1}: {e}")
                test_cases.append({
                    'type': f'Uniform_{color[0]}',
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'probabilities': np.array([0.2, 0.2, 0.2, 0.2, 0.2])
                })
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   Total test cases: {len(test_cases)}")
        
        # Calculate statistics
        valid_confidences = [tc['confidence'] for tc in test_cases if tc['confidence'] > 0]
        if valid_confidences:
            avg_confidence = np.mean(valid_confidences)
            max_confidence = np.max(valid_confidences)
            min_confidence = np.min(valid_confidences)
            
            print(f"   Average confidence: {avg_confidence:.1f}%")
            print(f"   Max confidence: {max_confidence:.1f}%")
            print(f"   Min confidence: {min_confidence:.1f}%")
            
            # Create simple visualization
            print("📈 Creating visualization...")
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Confidence scores
                test_names = [tc['type'] for tc in test_cases]
                confidences = [tc['confidence'] for tc in test_cases]
                
                bars = ax1.bar(range(len(test_names)), confidences, color='skyblue', alpha=0.7)
                ax1.set_xlabel('Test Case', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
                ax1.set_title('Confidence Scores by Test Case', fontsize=14, fontweight='bold')
                ax1.set_xticks(range(len(test_names)))
                ax1.set_xticklabels([f"T{i+1}" for i in range(len(test_names))])
                ax1.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, conf in zip(bars, confidences):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{conf:.1f}%', ha='center', va='bottom', fontsize=10)
                
                # Probability distribution
                ethnicities = ['Bugis', 'Sunda', 'Malay', 'Jawa', 'Banjar']
                all_probs = np.array([tc['probabilities'] for tc in test_cases])
                avg_probs = np.mean(all_probs, axis=0)
                
                bars = ax2.bar(ethnicities, avg_probs, color=['red', 'green', 'blue', 'orange', 'purple'], alpha=0.7)
                ax2.set_xlabel('Ethnicity', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Average Probability', fontsize=12, fontweight='bold')
                ax2.set_title('Average Probability by Ethnicity', fontsize=14, fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, prob in zip(bars, avg_probs):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
                
                plt.tight_layout()
                
                # Save the plot
                os.makedirs("logs", exist_ok=True)
                plt.savefig('logs/model_analysis_fixed.png', dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print("   ✅ Visualization saved to logs/model_analysis_fixed.png")
                
            except Exception as e:
                print(f"   ❌ Error creating visualization: {e}")
            
            return {
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'min_confidence': min_confidence,
                'test_cases': test_cases,
                'ethnicities': ethnicities,
                'avg_probabilities': avg_probs
            }
        else:
            print("   ❌ No valid predictions obtained")
            return None
            
    except Exception as e:
        print(f"❌ ERROR in diagnostic plots: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_summary_report(results):
    """Create a summary report of the analysis"""
    if not results:
        print("❌ No results to summarize")
        return
    
    print(f"\n📋 MODEL ANALYSIS SUMMARY")
    print("=" * 50)
    
    print(f"🎯 CONFIDENCE ANALYSIS:")
    print(f"   Average Confidence: {results['avg_confidence']:.1f}%")
    print(f"   Maximum Confidence: {results['max_confidence']:.1f}%")
    print(f"   Minimum Confidence: {results['min_confidence']:.1f}%")
    
    print(f"\n📊 PROBABILITY DISTRIBUTION:")
    for ethnicity, prob in zip(results['ethnicities'], results['avg_probabilities']):
        print(f"   {ethnicity}: {prob:.3f}")
    
    print(f"\n🔍 ISSUES DETECTED:")
    if results['avg_confidence'] < 50:
        print("   🚨 LOW CONFIDENCE SCORES")
        print("      • Model shows very low confidence in predictions")
        print("      • Suggests overfitting or poor generalization")
    
    if results['max_confidence'] < 70:
        print("   ⚠️ VERY LOW MAXIMUM CONFIDENCE")
        print("      • Model is highly uncertain about all predictions")
        print("      • May indicate serious training issues")
    
    # Check if probabilities are too uniform
    prob_std = np.std(results['avg_probabilities'])
    if prob_std < 0.1:
        print("   📊 UNIFORM PROBABILITY DISTRIBUTION")
        print("      • Model assigns similar probabilities to all classes")
        print("      • Suggests poor feature discrimination")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("   1. 🔄 Retrain model with proper cross-validation")
    print("   2. 📊 Implement confusion matrix analysis")
    print("   3. 🔍 Add feature importance visualization")
    print("   4. ⚙️ Perform hyperparameter optimization")
    print("   5. 📈 Monitor training/validation curves")
    
    print(f"\n📚 COMPARISON WITH ORIGINAL NOTEBOOK:")
    print("   Original Results: 98.6% accuracy with 6-fold CV")
    print(f"   Current Results: {results['avg_confidence']:.1f}% average confidence")
    print("   ⚠️ Significant performance degradation detected!")

def main():
    """Main function"""
    print("🚀 FIXED MODEL PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    try:
        # Create diagnostic plots
        results = create_simple_diagnostic_plots()
        
        if results:
            # Create summary report
            create_summary_report(results)
            
            print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"📁 Results saved to logs/ directory")
            
        else:
            print(f"\n❌ ANALYSIS FAILED")
            
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
