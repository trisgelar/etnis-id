#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test ML Model Direct
Test langsung model ML tanpa TCP untuk memastikan ethnic detection bekerja
"""

import numpy as np
import time
import base64
from io import BytesIO
from PIL import Image
import os
import sys

# Import ethnic detector
try:
    from ethnic_detector import EthnicDetector
except ImportError as e:
    print(f"❌ Cannot import EthnicDetector: {e}")
    sys.exit(1)

class MLModelTester:
    def __init__(self):
        self.detector = None
        self.test_results = {}
    
    def setup(self):
        """Initialize ML detector"""
        print("🤖 Initializing ML Ethnic Detector...")
        
        try:
            self.detector = EthnicDetector()
            
            if self.detector.model is None:
                print("❌ ML Model tidak dapat dimuat!")
                return False
            else:
                print("✅ ML Model berhasil dimuat!")
                return True
                
        except Exception as e:
            print(f"❌ Error initializing detector: {e}")
            return False
    
    def create_test_images(self):
        """Buat berbagai jenis test images"""
        print("\n🎨 Creating test images...")
        
        test_images = {}
        
        # 1. Random noise image
        print("  📷 Creating random noise image...")
        random_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        test_images['random'] = random_img
        
        # 2. Gradient image (more structured)
        print("  📷 Creating gradient image...")
        gradient_img = np.zeros((300, 300, 3), dtype=np.uint8)
        for i in range(300):
            for j in range(300):
                gradient_img[i, j, 0] = min(255, int(i * 255 / 300))  # Red gradient
                gradient_img[i, j, 1] = min(255, int(j * 255 / 300))  # Green gradient
                gradient_img[i, j, 2] = min(255, int((i+j) * 255 / 600))  # Blue combined
        test_images['gradient'] = gradient_img
        
        # 3. Pattern image (geometric patterns)
        print("  📷 Creating pattern image...")
        pattern_img = np.zeros((300, 300, 3), dtype=np.uint8)
        for i in range(300):
            for j in range(300):
                # Create checkerboard-like pattern
                if (i // 20 + j // 20) % 2 == 0:
                    pattern_img[i, j] = [200, 150, 100]  # Skin-like color
                else:
                    pattern_img[i, j] = [180, 120, 80]   # Darker skin-like color
        test_images['pattern'] = pattern_img
        
        # 4. Face-like structured image
        print("  📷 Creating face-like structure...")
        face_img = np.full((300, 300, 3), [190, 140, 100], dtype=np.uint8)  # Base skin color
        
        # Add "eyes" (darker regions)
        face_img[80:120, 80:120] = [100, 70, 50]   # Left eye
        face_img[80:120, 180:220] = [100, 70, 50]  # Right eye
        
        # Add "nose" (slightly different color)
        face_img[140:180, 135:165] = [170, 120, 90]
        
        # Add "mouth"
        face_img[220:240, 120:180] = [120, 80, 60]
        
        test_images['face_like'] = face_img
        
        print(f"✅ Created {len(test_images)} test images")
        return test_images
    
    def test_feature_extraction(self, test_images):
        """Test feature extraction process"""
        print("\n🧮 TESTING FEATURE EXTRACTION")
        print("="*50)
        
        extraction_results = {}
        
        for img_name, img_array in test_images.items():
            print(f"\n📊 Testing {img_name} image...")
            
            try:
                start_time = time.time()
                features = self.detector.extract_features(img_array)
                extraction_time = time.time() - start_time
                
                if features is not None:
                    print(f"✅ Feature extraction successful:")
                    print(f"   - Shape: {features.shape}")
                    print(f"   - Time: {extraction_time:.3f}s")
                    print(f"   - Min value: {features.min():.4f}")
                    print(f"   - Max value: {features.max():.4f}")
                    print(f"   - Mean: {features.mean():.4f}")
                    
                    extraction_results[img_name] = {
                        'success': True,
                        'shape': features.shape,
                        'time': extraction_time,
                        'features': features
                    }
                else:
                    print(f"❌ Feature extraction failed")
                    extraction_results[img_name] = {'success': False}
                    
            except Exception as e:
                print(f"❌ Error in feature extraction: {e}")
                extraction_results[img_name] = {'success': False, 'error': str(e)}
        
        return extraction_results
    
    def test_predictions(self, test_images):
        """Test ethnic predictions"""
        print("\n🎯 TESTING ETHNIC PREDICTIONS")
        print("="*50)
        
        prediction_results = {}
        
        for img_name, img_array in test_images.items():
            print(f"\n🔍 Predicting {img_name} image...")
            
            try:
                start_time = time.time()
                ethnicity, confidence, message = self.detector.predict_ethnicity(img_array)
                prediction_time = time.time() - start_time
                
                if ethnicity:
                    print(f"✅ Prediction successful:")
                    print(f"   - Ethnicity: {ethnicity}")
                    print(f"   - Confidence: {confidence:.1f}%")
                    print(f"   - Time: {prediction_time:.3f}s")
                    print(f"   - Message: {message}")
                    
                    prediction_results[img_name] = {
                        'success': True,
                        'ethnicity': ethnicity,
                        'confidence': confidence,
                        'time': prediction_time,
                        'message': message
                    }
                else:
                    print(f"❌ Prediction failed: {message}")
                    prediction_results[img_name] = {
                        'success': False,
                        'message': message
                    }
                    
            except Exception as e:
                print(f"❌ Error in prediction: {e}")
                prediction_results[img_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return prediction_results
    
    def test_base64_workflow(self, test_images):
        """Test workflow dengan base64 encoding (seperti dari Godot)"""
        print("\n📦 TESTING BASE64 WORKFLOW")
        print("="*50)
        
        base64_results = {}
        
        for img_name, img_array in test_images.items():
            print(f"\n📤 Testing base64 workflow for {img_name}...")
            
            try:
                # Convert to PIL Image
                pil_img = Image.fromarray(img_array)
                
                # Convert to base64 (simulate Godot)
                buffer = BytesIO()
                pil_img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                print(f"   📏 Base64 length: {len(img_base64)} chars")
                
                # Test prediction with base64
                start_time = time.time()
                ethnicity, confidence, message = self.detector.predict_ethnicity(img_base64)
                process_time = time.time() - start_time
                
                if ethnicity:
                    print(f"✅ Base64 prediction successful:")
                    print(f"   - Ethnicity: {ethnicity}")
                    print(f"   - Confidence: {confidence:.1f}%")
                    print(f"   - Processing time: {process_time:.3f}s")
                    
                    base64_results[img_name] = {
                        'success': True,
                        'ethnicity': ethnicity,
                        'confidence': confidence,
                        'time': process_time,
                        'base64_length': len(img_base64)
                    }
                else:
                    print(f"❌ Base64 prediction failed: {message}")
                    base64_results[img_name] = {
                        'success': False,
                        'message': message
                    }
                    
            except Exception as e:
                print(f"❌ Error in base64 workflow: {e}")
                base64_results[img_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return base64_results
    
    def test_consistency(self, test_images, iterations=3):
        """Test consistency of predictions"""
        print(f"\n🔄 TESTING PREDICTION CONSISTENCY ({iterations}x)")
        print("="*50)
        
        consistency_results = {}
        
        # Test with one image multiple times
        test_img_name = 'face_like'  # Use the most structured image
        if test_img_name not in test_images:
            test_img_name = list(test_images.keys())[0]
        
        test_img = test_images[test_img_name]
        print(f"🎯 Testing consistency with {test_img_name} image...")
        
        predictions = []
        times = []
        
        for i in range(iterations):
            print(f"   🔄 Iteration {i+1}/{iterations}")
            
            try:
                start_time = time.time()
                ethnicity, confidence, message = self.detector.predict_ethnicity(test_img)
                process_time = time.time() - start_time
                
                if ethnicity:
                    predictions.append({
                        'ethnicity': ethnicity,
                        'confidence': confidence,
                        'time': process_time
                    })
                    times.append(process_time)
                    print(f"      Result: {ethnicity} ({confidence:.1f}%) in {process_time:.3f}s")
                else:
                    print(f"      ❌ Failed: {message}")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        # Analyze consistency
        if predictions:
            ethnicities = [p['ethnicity'] for p in predictions]
            confidences = [p['confidence'] for p in predictions]
            
            unique_ethnicities = list(set(ethnicities))
            most_common = max(set(ethnicities), key=ethnicities.count)
            consistency_rate = ethnicities.count(most_common) / len(ethnicities) * 100
            
            avg_confidence = sum(confidences) / len(confidences)
            avg_time = sum(times) / len(times)
            
            print(f"\n📊 Consistency Analysis:")
            print(f"   - Successful predictions: {len(predictions)}/{iterations}")
            print(f"   - Unique results: {unique_ethnicities}")
            print(f"   - Most common: {most_common}")
            print(f"   - Consistency rate: {consistency_rate:.1f}%")
            print(f"   - Average confidence: {avg_confidence:.1f}%")
            print(f"   - Average time: {avg_time:.3f}s")
            
            consistency_results = {
                'success_rate': len(predictions) / iterations * 100,
                'consistency_rate': consistency_rate,
                'most_common': most_common,
                'avg_confidence': avg_confidence,
                'avg_time': avg_time,
                'all_predictions': predictions
            }
        else:
            print("❌ No successful predictions for consistency test")
            consistency_results = {'success_rate': 0}
        
        return consistency_results

def main():
    """Main testing function"""
    print("🚀 STARTING ML MODEL DIRECT TEST")
    print("="*60)
    
    tester = MLModelTester()
    
    # Setup
    if not tester.setup():
        print("❌ Setup failed. Cannot continue.")
        return False
    
    # Create test images
    test_images = tester.create_test_images()
    
    # Run tests
    print("\n" + "="*60)
    print("🧪 RUNNING ML TESTS")
    print("="*60)
    
    try:
        # Test 1: Feature extraction
        extraction_results = tester.test_feature_extraction(test_images)
        
        # Test 2: Predictions
        prediction_results = tester.test_predictions(test_images)
        
        # Test 3: Base64 workflow
        base64_results = tester.test_base64_workflow(test_images)
        
        # Test 4: Consistency
        consistency_results = tester.test_consistency(test_images, 5)
        
        # Summary
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        
        # Feature extraction summary
        extraction_success = sum(1 for r in extraction_results.values() if r.get('success', False))
        print(f"Feature Extraction: {extraction_success}/{len(test_images)} successful")
        
        # Prediction summary
        prediction_success = sum(1 for r in prediction_results.values() if r.get('success', False))
        print(f"Predictions: {prediction_success}/{len(test_images)} successful")
        
        # Base64 summary
        base64_success = sum(1 for r in base64_results.values() if r.get('success', False))
        print(f"Base64 Workflow: {base64_success}/{len(test_images)} successful")
        
        # Consistency summary
        consistency_rate = consistency_results.get('consistency_rate', 0)
        print(f"Prediction Consistency: {consistency_rate:.1f}%")
        
        # Overall assessment
        all_features_ok = extraction_success == len(test_images)
        all_predictions_ok = prediction_success == len(test_images)
        all_base64_ok = base64_success == len(test_images)
        consistency_ok = consistency_rate >= 80  # 80% consistency threshold
        
        overall_success = all_features_ok and all_predictions_ok and all_base64_ok and consistency_ok
        
        print(f"\n🎯 OVERALL ML STATUS: {'✅ EXCELLENT' if overall_success else '⚠️ NEEDS ATTENTION'}")
        
        if overall_success:
            print("\n🎉 ML Model is working perfectly!")
            print("   ✅ Feature extraction working")
            print("   ✅ Predictions working") 
            print("   ✅ Base64 workflow working")
            print("   ✅ Predictions are consistent")
            print("\n🎯 Ready for production use!")
        else:
            print("\n💡 Issues detected:")
            if not all_features_ok:
                print("   ⚠️ Feature extraction issues")
            if not all_predictions_ok:
                print("   ⚠️ Prediction issues")
            if not all_base64_ok:
                print("   ⚠️ Base64 workflow issues")
            if not consistency_ok:
                print("   ⚠️ Prediction consistency issues")
        
        return overall_success
        
    except Exception as e:
        print(f"\n💥 Fatal error during testing: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)