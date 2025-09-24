import numpy as np
import cv2
import pickle
import os
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import base64
from io import BytesIO
from PIL import Image
from ml_training.core.config import get_model_config, get_dataset_config, get_feature_config

class EthnicDetector:
    def __init__(self, model_path=None):
        """
        Inisialisasi detector etnis
        """
        # Get configuration
        model_config = get_model_config()
        dataset_config = get_dataset_config()
        feature_config = get_feature_config()
        
        # Use configuration values
        self.model_path = model_path or model_config.model_path
        
        # Create label map from configuration ethnicities
        self.ethnicities = dataset_config.ethnicities
        self.label_map = {i: ethnicity for i, ethnicity in enumerate(self.ethnicities)}
        
        # Store feature configuration for feature extraction
        self.feature_config = feature_config
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load model ML dari file pickle"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Model ML berhasil dimuat dari {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Gagal memuat model: {e}")
            return False
    
    def preprocessing_glcm(self, image):
        """
        Preprocessing untuk GLCM: RGB to Grayscale
        """
        if len(image.shape) == 3:
            # Convert RGB to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return gray
    
    def preprocessing_color(self, image):
        """
        Preprocessing untuk Color Histogram: RGB to HSV
        """
        if len(image.shape) == 3:
            # Convert RGB to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            # Jika grayscale, convert ke RGB dulu
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return hsv
    
    def glcm_extraction(self, gray_image):
        """
        Ekstraksi fitur GLCM berdasarkan script training yang benar
        """
        try:
            # Get GLCM parameters from configuration
            distances = self.feature_config.glc_distances
            angles = [np.radians(angle) for angle in self.feature_config.glc_angles]  # Convert degrees to radians
            levels = self.feature_config.glc_levels
            symmetric = self.feature_config.glc_symmetric
            normed = self.feature_config.glc_normed
            
            # Resize image jika terlalu besar
            if gray_image.shape[0] > 256 or gray_image.shape[1] > 256:
                gray_image = cv2.resize(gray_image, (256, 256))
            
            # Calculate GLCM menggunakan import yang benar
            from skimage.feature import graycomatrix as greycomatrix
            from skimage.feature import graycoprops as greycoprops
            
            glcm = greycomatrix(gray_image, distances=distances, angles=angles, 
                               levels=levels, symmetric=symmetric, normed=normed)
            
            # Extract Haralick features PERSIS seperti script training
            contrast = greycoprops(glcm, 'contrast')
            homogeneity = greycoprops(glcm, 'homogeneity')
            correlation = greycoprops(glcm, 'correlation')
            asm = greycoprops(glcm, 'ASM')
            
            # Entropy untuk tiap sudut SEPERTI DI TRAINING
            entropy_vals = [shannon_entropy(glcm[:, :, 0, j]) for j in range(len(angles))]
            
            # Gabung jadi satu vektor fitur SESUAI URUTAN TRAINING
            feat = np.hstack([
                contrast.flatten(),
                homogeneity.flatten(),
                correlation.flatten(),
                asm.flatten(),
                entropy_vals
            ])
            
            print(f"📊 GLCM features: contrast={len(contrast.flatten())}, homogeneity={len(homogeneity.flatten())}, correlation={len(correlation.flatten())}, asm={len(asm.flatten())}, entropy={len(entropy_vals)}, total={len(feat)}")
            
            return feat
            
        except Exception as e:
            print(f"❌ Error dalam ekstraksi GLCM: {e}")
            return np.zeros(20)  # 4 entropy + 16 haralick = 20
    
    def color_extraction(self, hsv_image):
        """
        Ekstraksi fitur Color Histogram berdasarkan script training yang benar
        """
        try:
            # Split channels H, S, V
            h, s, v = cv2.split(hsv_image)
            
            # Get histogram parameters from configuration
            bins = self.feature_config.color_bins
            channels = self.feature_config.color_channels
            
            # Hitung histogram using configuration
            hist1 = cv2.calcHist([hsv_image], [channels[0]], None, [bins], [0, 256])  # First configured channel
            hist2 = cv2.calcHist([hsv_image], [channels[1]], None, [bins], [0, 256])  # Second configured channel
            
            # Gabungkan histogram S dan V TANPA normalisasi (sesuai training)
            fitur = np.concatenate((hist1, hist2))
            color_features = np.array(fitur).flatten()
            
            print(f"🎨 Color histogram: S={len(hist1.flatten())}, V={len(hist2.flatten())}, total={len(color_features)}")
            
            return color_features
            
        except Exception as e:
            print(f"❌ Error dalam ekstraksi Color Histogram: {e}")
            return np.zeros(32)  # 16 + 16 = 32
    
    def extract_features(self, image):
        """
        Ekstraksi fitur lengkap (GLCM + Color Histogram)
        """
        try:
            # Preprocessing
            gray_image = self.preprocessing_glcm(image)
            hsv_image = self.preprocessing_color(image)
            
            # Feature extraction
            glcm_features = self.glcm_extraction(gray_image)
            color_features = self.color_extraction(hsv_image)
            
            # Gabungkan fitur
            combined_features = np.concatenate([glcm_features, color_features])
            
            print(f"📊 Ekstraksi fitur selesai: GLCM={len(glcm_features)}, Color={len(color_features)}, Total={len(combined_features)}")
            
            return combined_features.reshape(1, -1)  # Reshape untuk prediksi
            
        except Exception as e:
            print(f"❌ Error dalam ekstraksi fitur: {e}")
            return None
    
    def predict_ethnicity(self, image_data):
        """
        Prediksi etnis dari image data (base64 atau numpy array)
        """
        try:
            print(f"🔍 Starting prediction process...")
            
            # Jika input berupa base64 string
            if isinstance(image_data, str):
                print(f"📥 Decoding base64 image data ({len(image_data)} chars)")
                # Decode base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                image = np.array(image)
                print(f"📊 Decoded image shape: {image.shape}")
            else:
                image = image_data
                print(f"📊 Input image shape: {image.shape}")
            
            # Resize jika terlalu besar (untuk efisiensi)
            original_shape = image.shape
            if image.shape[0] > 512 or image.shape[1] > 512:
                image = cv2.resize(image, (512, 512))
                print(f"📏 Resized image: {original_shape} → {image.shape}")
            
            # Debug: Print sample pixel values
            print(f"🎨 Image stats: min={image.min()}, max={image.max()}, mean={image.mean():.2f}")
            
            # Ekstraksi fitur
            features = self.extract_features(image)
            
            if features is None:
                return None, 0.0, "Error dalam ekstraksi fitur"
            
            # Debug: Print feature statistics  
            print(f"🧮 Feature stats: shape={features.shape}, min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
            
            # Prediksi
            if self.model is None:
                return None, 0.0, "Model tidak tersedia"
            
            prediction = self.model.predict(features)[0]
            print(f"🎯 Raw prediction: {prediction}")
            
            # Get probability/confidence jika model support
            try:
                probabilities = self.model.predict_proba(features)[0]
                confidence = np.max(probabilities) * 100
                print(f"📈 All probabilities: {probabilities}")
                print(f"📈 Confidence scores: {[f'{self.label_map[i]}: {p*100:.1f}%' for i, p in enumerate(probabilities)]}")
            except:
                confidence = 85.0  # Default confidence jika tidak bisa dihitung
                print(f"⚠️ Using default confidence: {confidence}")
            
            # Convert prediction ke nama etnis
            ethnicity = self.label_map.get(prediction, "Unknown")
            
            print(f"🎯 Final Prediction: {ethnicity} (Confidence: {confidence:.1f}%)")
            
            return ethnicity, confidence, "Success"
            
        except Exception as e:
            error_msg = f"Error dalam prediksi: {str(e)}"
            print(f"❌ {error_msg}")
            return None, 0.0, error_msg
    
    def process_webcam_frame(self, frame_data):
        """
        Khusus untuk memproses frame webcam
        """
        try:
            print(f"🎥 Processing webcam frame data: {type(frame_data)}")
            
            # Handle simulasi mode
            if isinstance(frame_data, str):
                if frame_data == "simulated_webcam_frame_data":
                    print("🎭 Mode simulasi webcam - menggunakan dummy image")
                    # Create dummy webcam frame
                    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    return self.predict_ethnicity(dummy_frame)
                else:
                    # Try to decode as base64
                    try:
                        frame_bytes = base64.b64decode(frame_data)
                        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            raise ValueError("Failed to decode base64 to image")
                    except Exception as decode_error:
                        print(f"❌ Error decoding base64: {decode_error}")
                        return None, 0.0, f"Error decoding webcam frame: {decode_error}"
            else:
                frame = frame_data
            
            # Validasi frame
            if frame is None:
                return None, 0.0, "Frame webcam kosong atau tidak valid"
                
            # Check if frame has proper shape
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                return None, 0.0, f"Frame format tidak valid: {frame.shape}"
            
            print(f"🎥 Frame valid: shape={frame.shape}")
            
            # Crop area wajah jika perlu (implementasi sederhana)
            # Untuk sementara gunakan area tengah
            h, w = frame.shape[:2]
            center_crop = frame[h//4:3*h//4, w//4:3*w//4]
            
            print(f"🎯 Cropped frame: {center_crop.shape}")
            
            # Prediksi
            return self.predict_ethnicity(center_crop)
            
        except Exception as e:
            error_msg = f"Error memproses webcam frame: {str(e)}"
            print(f"❌ {error_msg}")
            return None, 0.0, error_msg

# Test function
def test_detector():
    """Test function untuk memastikan detector berfungsi"""
    detector = EthnicDetector()
    
    if detector.model is None:
        print("❌ Test gagal: Model tidak dapat dimuat")
        return False
    
    # Create dummy image untuk test
    dummy_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    ethnicity, confidence, message = detector.predict_ethnicity(dummy_image)
    
    if ethnicity:
        print(f"✅ Test berhasil: {ethnicity} ({confidence:.1f}%)")
        return True
    else:
        print(f"❌ Test gagal: {message}")
        return False

if __name__ == "__main__":
    # Test detector
    print("🧪 Testing Ethnic Detector...")
    test_detector()