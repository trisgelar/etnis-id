#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Cross-Validation System - Final Implementation
Implements the exact cross-validation system from the original notebook
to achieve 98.65% accuracy with proper feature extraction and model training
"""

import os
import sys
import numpy as np
import cv2
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
sys.path.insert(0, '.')

# Import our configuration system
from ml_training.core.config import get_config
from ml_training.core.utils import TrainingLogger, ProgressTracker

class CompleteCVSystem:
    """Complete Cross-Validation System matching original notebook"""
    
    def __init__(self):
        """Initialize the system with configuration"""
        # Get configuration
        self.config = get_config()
        self.dataset_config = self.config.dataset
        self.model_config = self.config.model
        self.cv_config = self.config.cross_validation
        self.feature_config = self.config.feature_extraction
        
        # Initialize logging
        self.logger = TrainingLogger('complete_cv_system')
        self.progress_tracker = ProgressTracker('CV System')
        
        # Create output directory
        self.output_dir = Path('logs/cv_final_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.data = None
        self.labels = None
        self.features = None
        self.model = None
        self.cv_results = None
        
        self.logger.info("🚀 Complete Cross-Validation System initialized")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
    
    def load_dataset(self) -> bool:
        """Load the real ethnicity dataset"""
        self.logger.info("📊 LOADING REAL ETHNICITY DATASET")
        self.logger.info("=" * 50)
        
        try:
            # Dataset path from configuration
            dataset_path = Path(self.dataset_config.periorbital_dir)
            self.logger.info(f"📂 Dataset path: {dataset_path}")
            
            if not dataset_path.exists():
                self.logger.error(f"❌ Dataset path does not exist: {dataset_path}")
                return False
            
            # Load data exactly like the notebook
            data, labels, idx, img_names, folder_list = self._load_data_notebook_style(dataset_path)
            
            self.data = data
            self.labels = labels
            self.indices = idx
            self.img_names = img_names
            self.folder_list = folder_list
            
            self.logger.info(f"✅ Dataset loaded successfully!")
            self.logger.info(f"   📊 Total images: {len(data)}")
            self.logger.info(f"   🏷️  Classes: {list(set(labels))}")
            self.logger.info(f"   📈 Class distribution:")
            
            # Show class distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                self.logger.info(f"      {label}: {count} images")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_data_notebook_style(self, data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, List]:
        """Load data exactly like the original notebook"""
        m = 0
        X, y, idx, name, fl = [], [], [], [], []
        labels = os.listdir(data_dir)
        
        self.logger.info(f"📂 Found {len(labels)} label directories")
        
        for label in labels:
            if label not in self.dataset_config.ethnicities:
                self.logger.warning(f"⚠️  Skipping unknown ethnicity: {label}")
                continue
                
            datas_path = data_dir / label
            if not datas_path.exists():
                self.logger.warning(f"⚠️  Directory does not exist: {datas_path}")
                continue
                
            datas = os.listdir(datas_path)
            fl.append(datas)
            
            self.logger.info(f"📁 Processing {label}: {len(datas)} folders")
            
            for data in datas:
                data_path = datas_path / data
                if not data_path.exists():
                    continue
                    
                image_names = os.listdir(data_path)
                name.append(image_names)
                
                for img in image_names:
                    img_path = data_path / img
                    if not img_path.exists():
                        continue
                        
                    # Load image using cv2 (like notebook)
                    image = cv2.imread(str(img_path))
                    if image is not None:
                        X.append(image)
                        y.append(label)
                        idx.append(m)
                        m += 1
            
            self.logger.info(f"   ✅ Loaded {len([x for x in y if x == label])} images for {label}")
        
        return np.array(X), np.array(y), np.array(idx), name, fl
    
    def preprocess_images(self) -> bool:
        """Preprocess images for GLCM and color histogram extraction"""
        self.logger.info("🔧 PREPROCESSING IMAGES")
        self.logger.info("=" * 50)
        
        try:
            # GLCM preprocessing: RGB to Grayscale
            self.logger.info("📸 Converting RGB to Grayscale for GLCM...")
            self.gray_images = []
            for i, image in enumerate(self.data):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.gray_images.append(gray)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"   Processed {i + 1}/{len(self.data)} images")
            
            # Color preprocessing: RGB to HSV
            self.logger.info("🎨 Converting RGB to HSV for Color Histogram...")
            self.hsv_images = []
            for i, image in enumerate(self.data):
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                self.hsv_images.append(hsv)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"   Processed {i + 1}/{len(self.data)} images")
            
            self.logger.info("✅ Image preprocessing completed!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_glcm_features(self) -> bool:
        """Extract GLCM features exactly like the notebook"""
        self.logger.info("🔧 EXTRACTING GLCM FEATURES")
        self.logger.info("=" * 50)
        
        try:
            # Parameters from notebook
            distances = [1]  # Distance = 1
            angles = [0, np.pi/4, np.pi/2, 3/4*(np.pi)]  # 0, 45, 90, 135 degrees in radians
            
            self.logger.info(f"📊 GLCM Parameters:")
            self.logger.info(f"   Distances: {distances}")
            self.logger.info(f"   Angles: {[f'{a*180/np.pi:.0f}°' for a in angles]}")
            self.logger.info(f"   Levels: 256")
            self.logger.info(f"   Symmetric: True")
            self.logger.info(f"   Normed: True")
            
            glcm_features = []
            
            for i, gray_img in enumerate(self.gray_images):
                # GLCM computation (exactly like notebook)
                glcm = greycomatrix(gray_img, 
                                  distances=distances, 
                                  angles=angles, 
                                  symmetric=True, 
                                  normed=True, 
                                  levels=256)
                
                # Haralick features (exactly like notebook)
                properties = ['contrast', 'homogeneity', 'correlation', 'ASM']
                feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
                
                # Entropy features (exactly like notebook)
                entropy = [shannon_entropy(glcm[:,:,:,idx]) for idx in range(glcm.shape[3])]
                feat = np.concatenate((entropy, feats), axis=0)
                
                glcm_features.append(feat)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"   Processed {i + 1}/{len(self.gray_images)} images")
            
            self.glcm_features = np.array(glcm_features)
            self.logger.info(f"✅ GLCM feature extraction completed!")
            self.logger.info(f"   📊 Feature shape: {self.glcm_features.shape}")
            self.logger.info(f"   📊 Feature length per image: {len(self.glcm_features[0])}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in GLCM feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_color_features(self) -> bool:
        """Extract color histogram features exactly like the notebook"""
        self.logger.info("🎨 EXTRACTING COLOR HISTOGRAM FEATURES")
        self.logger.info("=" * 50)
        
        try:
            # Parameters from notebook
            bins = 16  # 16 bins
            channels = [1, 2]  # S and V channels (indices 1 and 2 in HSV)
            
            self.logger.info(f"📊 Color Histogram Parameters:")
            self.logger.info(f"   Bins: {bins}")
            self.logger.info(f"   Channels: {channels} (S and V from HSV)")
            self.logger.info(f"   Range: [0, 256]")
            
            color_features = []
            
            for i, hsv_img in enumerate(self.hsv_images):
                # Extract histograms for S and V channels (exactly like notebook)
                hist1 = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256])  # S channel
                hist2 = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256])  # V channel
                
                # Concatenate and flatten (exactly like notebook)
                feature = np.concatenate((hist1, hist2))
                arr = np.array(feature).flatten()
                
                color_features.append(arr)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"   Processed {i + 1}/{len(self.hsv_images)} images")
            
            self.color_features = np.array(color_features)
            self.logger.info(f"✅ Color histogram feature extraction completed!")
            self.logger.info(f"   📊 Feature shape: {self.color_features.shape}")
            self.logger.info(f"   📊 Feature length per image: {len(self.color_features[0])}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in color feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def combine_features(self) -> bool:
        """Combine GLCM and color features"""
        self.logger.info("🔗 COMBINING FEATURES")
        self.logger.info("=" * 50)
        
        try:
            # Combine features exactly like notebook
            self.combined_features = np.concatenate((self.glcm_features, self.color_features), axis=1)
            
            self.logger.info(f"✅ Feature combination completed!")
            self.logger.info(f"   📊 GLCM features: {self.glcm_features.shape[1]}")
            self.logger.info(f"   📊 Color features: {self.color_features.shape[1]}")
            self.logger.info(f"   📊 Combined features: {self.combined_features.shape[1]}")
            
            # Verify feature dimensions match notebook
            expected_total = 20 + 32  # 20 GLCM + 32 Color
            if self.combined_features.shape[1] == expected_total:
                self.logger.info(f"✅ Feature dimensions match notebook ({expected_total})")
            else:
                self.logger.warning(f"⚠️  Feature dimensions differ from notebook: {self.combined_features.shape[1]} vs {expected_total}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error combining features: {e}")
            return False
    
    def run_cross_validation(self) -> bool:
        """Run cross-validation exactly like the notebook"""
        self.logger.info("🔄 RUNNING CROSS-VALIDATION")
        self.logger.info("=" * 50)
        
        try:
            # Parameters from notebook
            k_folds = 6  # Optimal k from notebook
            n_estimators = 200  # From notebook
            random_state = 0  # From notebook
            
            self.logger.info(f"📊 Cross-Validation Parameters:")
            self.logger.info(f"   K-folds: {k_folds}")
            self.logger.info(f"   Random Forest n_estimators: {n_estimators}")
            self.logger.info(f"   Random state: {random_state}")
            
            # Shuffle data exactly like notebook
            X, y = shuffle(self.combined_features, self.labels, random_state=220)
            
            # Create model exactly like notebook
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            
            # Create CV splitter exactly like notebook
            cv = StratifiedKFold(n_splits=k_folds)
            
            # Run cross-validation exactly like notebook
            self.logger.info("🚀 Running cross-validation...")
            scores = cross_val_score(model, X, y, cv=cv)
            
            # Calculate mean accuracy
            mean_accuracy = np.mean(scores) * 100
            
            self.cv_scores = scores
            self.mean_accuracy = mean_accuracy
            
            self.logger.info(f"✅ Cross-validation completed!")
            self.logger.info(f"   📈 Individual fold scores:")
            for i, score in enumerate(scores):
                self.logger.info(f"      Fold {i+1}: {score*100:.4f}%")
            self.logger.info(f"   📈 Mean accuracy: {mean_accuracy:.4f}%")
            self.logger.info(f"   📈 Standard deviation: {np.std(scores)*100:.4f}%")
            
            # Check if we achieved target accuracy
            target_accuracy = 98.65
            if mean_accuracy >= target_accuracy:
                self.logger.info(f"🎯 TARGET ACHIEVED! {mean_accuracy:.2f}% >= {target_accuracy}%")
            else:
                self.logger.warning(f"⚠️  Target not reached: {mean_accuracy:.2f}% < {target_accuracy}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in cross-validation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_final_model(self) -> bool:
        """Train final model on all data"""
        self.logger.info("🎯 TRAINING FINAL MODEL")
        self.logger.info("=" * 50)
        
        try:
            # Parameters from notebook
            n_estimators = 200
            random_state = 0
            
            # Shuffle data exactly like notebook
            X, y = shuffle(self.combined_features, self.labels, random_state=220)
            
            # Create and train model exactly like notebook
            self.final_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            self.final_model.fit(X, y)
            
            self.logger.info(f"✅ Final model trained successfully!")
            self.logger.info(f"   📊 Model type: RandomForestClassifier")
            self.logger.info(f"   📊 N estimators: {n_estimators}")
            self.logger.info(f"   📊 Training samples: {len(X)}")
            self.logger.info(f"   📊 Features per sample: {X.shape[1]}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error training final model: {e}")
            return False
    
    def save_model(self) -> bool:
        """Save the trained model"""
        self.logger.info("💾 SAVING MODEL")
        self.logger.info("=" * 50)
        
        try:
            # Save model exactly like notebook
            model_path = self.output_dir / "pickle_model_cv_final.pkl"
            
            with open(model_path, 'wb') as file:
                pickle.dump(self.final_model, file)
            
            self.logger.info(f"✅ Model saved successfully!")
            self.logger.info(f"   📁 Path: {model_path}")
            
            # Also save to the main model directory
            main_model_path = Path("model_ml/pickle_model_cv_final.pkl")
            main_model_path.parent.mkdir(exist_ok=True)
            
            with open(main_model_path, 'wb') as file:
                pickle.dump(self.final_model, file)
            
            self.logger.info(f"   📁 Also saved to: {main_model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving model: {e}")
            return False
    
    def create_visualizations(self) -> bool:
        """Create performance visualizations"""
        self.logger.info("📊 CREATING VISUALIZATIONS")
        self.logger.info("=" * 50)
        
        try:
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Cross-validation scores plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # CV scores bar plot
            fold_numbers = list(range(1, len(self.cv_scores) + 1))
            bars = ax1.bar(fold_numbers, self.cv_scores * 100, color='skyblue', alpha=0.7, edgecolor='navy')
            ax1.set_xlabel('Fold Number')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_title('Cross-Validation Scores by Fold')
            ax1.set_ylim([95, 100])
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, self.cv_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{score*100:.2f}%', ha='center', va='bottom')
            
            # Mean accuracy indicator
            ax1.axhline(y=self.mean_accuracy, color='red', linestyle='--', 
                       label=f'Mean: {self.mean_accuracy:.2f}%')
            ax1.legend()
            
            # 2. Feature importance plot
            feature_importance = self.final_model.feature_importances_
            
            # Separate GLCM and Color features
            n_glcm = len(self.glcm_features[0])
            glcm_importance = feature_importance[:n_glcm]
            color_importance = feature_importance[n_glcm:]
            
            # Plot feature importance
            x_pos = range(len(feature_importance))
            colors = ['lightcoral'] * n_glcm + ['lightblue'] * len(color_importance)
            
            bars = ax2.bar(x_pos, feature_importance, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Feature Index')
            ax2.set_ylabel('Importance')
            ax2.set_title('Feature Importance (GLCM vs Color)')
            ax2.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='lightcoral', alpha=0.7, label='GLCM Features'),
                             Patch(facecolor='lightblue', alpha=0.7, label='Color Features')]
            ax2.legend(handles=legend_elements)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / "cv_performance_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ Visualizations created!")
            self.logger.info(f"   📁 Saved to: {plot_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_complete_system(self) -> bool:
        """Run the complete cross-validation system"""
        self.logger.info("🚀 COMPLETE CROSS-VALIDATION SYSTEM")
        self.logger.info("=" * 70)
        
        try:
            # Step 1: Load dataset
            if not self.load_dataset():
                return False
            
            # Step 2: Preprocess images
            if not self.preprocess_images():
                return False
            
            # Step 3: Extract GLCM features
            if not self.extract_glcm_features():
                return False
            
            # Step 4: Extract color features
            if not self.extract_color_features():
                return False
            
            # Step 5: Combine features
            if not self.combine_features():
                return False
            
            # Step 6: Run cross-validation
            if not self.run_cross_validation():
                return False
            
            # Step 7: Train final model
            if not self.train_final_model():
                return False
            
            # Step 8: Save model
            if not self.save_model():
                return False
            
            # Step 9: Create visualizations
            if not self.create_visualizations():
                return False
            
            self.logger.info("🎉 COMPLETE SYSTEM SUCCESSFUL!")
            self.logger.info("=" * 70)
            self.logger.info(f"📈 Final Accuracy: {self.mean_accuracy:.2f}%")
            self.logger.info(f"🎯 Target Accuracy: 98.65%")
            
            if self.mean_accuracy >= 98.65:
                self.logger.info("✅ TARGET ACHIEVED! Model ready for deployment!")
            else:
                self.logger.info("⚠️  Target not reached, but significant improvement achieved!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Complete system failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    print("🚀 COMPLETE CROSS-VALIDATION SYSTEM - FINAL IMPLEMENTATION")
    print("=" * 70)
    print("📋 This system implements the exact cross-validation from the original notebook")
    print("🎯 Target: Achieve 98.65% accuracy with proper feature extraction")
    print("=" * 70)
    
    # Create and run system
    cv_system = CompleteCVSystem()
    success = cv_system.run_complete_system()
    
    if success:
        print("\n🎉 SYSTEM COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("📁 Results saved to: logs/cv_final_results/")
        print("🤖 New model saved to: model_ml/pickle_model_cv_final.pkl")
        print("📊 Performance analysis saved as PNG")
        print("=" * 70)
    else:
        print("\n❌ SYSTEM FAILED!")
        print("=" * 70)
        print("Please check the error messages above")

if __name__ == "__main__":
    main()

