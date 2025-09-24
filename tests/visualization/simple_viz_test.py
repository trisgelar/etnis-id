#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple visualization test without LaTeX dependencies
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '.')

# Set up matplotlib without LaTeX
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_simple_visualization():
    """Create simple visualization without LaTeX dependencies"""
    print("🎨 CREATING SIMPLE VISUALIZATION")
    print("=" * 40)
    
    try:
        from ethnic_detector import EthnicDetector
        
        # Load model
        print("📦 Loading model...")
        detector = EthnicDetector()
        
        # Test predictions
        print("🧪 Testing predictions...")
        test_results = []
        
        np.random.seed(42)
        for i in range(5):
            # Create test image
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Get prediction
            prediction, confidence, status = detector.predict_ethnicity(test_image)
            test_results.append({
                'test': f'Test {i+1}',
                'prediction': prediction,
                'confidence': confidence
            })
            
            print(f"   {test_results[-1]['test']}: {prediction} ({confidence:.1f}%)")
        
        # Create simple plot
        print("📊 Creating visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence scores
        test_names = [r['test'] for r in test_results]
        confidences = [r['confidence'] for r in test_results]
        
        bars = ax1.bar(test_names, confidences, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Test Case')
        ax1.set_ylabel('Confidence (%)')
        ax1.set_title('Model Confidence Scores')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, conf in zip(bars, confidences):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{conf:.1f}%', ha='center', va='bottom')
        
        # Predictions distribution
        predictions = [r['prediction'] for r in test_results]
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        ax2.bar(pred_counts.keys(), pred_counts.values(), 
               color=['red', 'green', 'blue', 'orange', 'purple'][:len(pred_counts)])
        ax2.set_xlabel('Predicted Ethnicity')
        ax2.set_ylabel('Count')
        ax2.set_title('Prediction Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs("logs", exist_ok=True)
        plt.savefig('logs/simple_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualization saved to logs/simple_model_analysis.png")
        
        # Print summary
        avg_confidence = np.mean(confidences)
        print(f"\n📊 SUMMARY:")
        print(f"   Average Confidence: {avg_confidence:.1f}%")
        print(f"   Unique Predictions: {len(set(predictions))}")
        
        if avg_confidence < 50:
            print(f"   ⚠️ LOW CONFIDENCE - Model may be overfitted")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 SIMPLE VISUALIZATION TEST")
    print("=" * 50)
    
    success = create_simple_visualization()
    
    if success:
        print("\n✅ SUCCESS! Visualization created without errors")
        print("📁 Check logs/simple_model_analysis.png")
    else:
        print("\n❌ FAILED")

if __name__ == "__main__":
    main()
