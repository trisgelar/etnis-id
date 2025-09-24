#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Validation Fix System - Implementing proper validation and feature rebalancing
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
sys.path.insert(0, '.')

# Configure matplotlib
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'text.usetex': False,
    'font.family': 'DejaVu Sans',
    'axes.grid': True,
    'grid.alpha': 0.3
})

class CrossValidationFixSystem:
    """System to fix overfitting through proper cross-validation and feature rebalancing"""
    
    def __init__(self, output_dir="logs/analysis"):
        """Initialize the fix system"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Ethnicity labels
        self.ethnicity_labels = ['Bugis', 'Sunda', 'Malay', 'Jawa', 'Banjar']
        
        # Color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'ethnicities': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
        
        print(f"🔧 CrossValidationFixSystem initialized - Output: {output_dir}")
    
    def load_and_prepare_data(self):
        """Load and prepare the dataset for cross-validation"""
        print("📊 LOADING AND PREPARING DATA")
        print("=" * 50)
        
        try:
            from ml_training.core.data_loader import EthnicityDataLoader
            from ml_training.core.feature_extractors import GLCFeatureExtractor, ColorHistogramFeatureExtractor
            
            # Load data
            data_loader = EthnicityDataLoader()
            images, labels = data_loader.load_all_data()
            
            print(f"✅ Data loaded: {len(images)} images, {len(set(labels))} classes")
            
            # Extract features
            print("🔍 Extracting features...")
            from ml_training.core.utils import TrainingLogger
            logger = TrainingLogger('cross_validation')
            
            glcm_extractor = GLCFeatureExtractor(logger)
            color_extractor = ColorHistogramFeatureExtractor(logger)
            
            all_features = []
            all_labels = []
            
            for i, (image, label) in enumerate(zip(images, labels)):
                if i % 100 == 0:
                    print(f"   Processing image {i+1}/{len(images)}")
                
                # Extract GLCM features
                glcm_features = glcm_extractor.extract_features(image)
                
                # Extract color features  
                color_features = color_extractor.extract_features(image)
                
                # Combine features
                combined_features = np.concatenate([glcm_features, color_features])
                all_features.append(combined_features)
                all_labels.append(label)
            
            X = np.array(all_features)
            y = np.array(all_labels)
            
            print(f"✅ Feature extraction complete:")
            print(f"   Feature matrix shape: {X.shape}")
            print(f"   Labels shape: {y.shape}")
            print(f"   GLCM features: {glcm_features.shape[0]}")
            print(f"   Color features: {color_features.shape[0]}")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def analyze_feature_scaling(self, X):
        """Analyze and fix feature scaling issues"""
        print("\n🔍 ANALYZING FEATURE SCALING")
        print("=" * 50)
        
        # Separate GLCM and Color features
        glcm_features = X[:, :20]  # First 20 are GLCM
        color_features = X[:, 20:]  # Last 32 are Color
        
        print(f"📊 GLCM Features Analysis:")
        print(f"   Mean: {np.mean(glcm_features):.6f}")
        print(f"   Std: {np.std(glcm_features):.6f}")
        print(f"   Min: {np.min(glcm_features):.6f}")
        print(f"   Max: {np.max(glcm_features):.6f}")
        print(f"   Range: {np.max(glcm_features) - np.min(glcm_features):.6f}")
        
        print(f"\n🎨 Color Features Analysis:")
        print(f"   Mean: {np.mean(color_features):.6f}")
        print(f"   Std: {np.std(color_features):.6f}")
        print(f"   Min: {np.min(color_features):.6f}")
        print(f"   Max: {np.max(color_features):.6f}")
        print(f"   Range: {np.max(color_features) - np.min(color_features):.6f}")
        
        # Calculate scaling ratio
        color_range = np.max(color_features) - np.min(color_features)
        glcm_range = np.max(glcm_features) - np.min(glcm_features)
        scaling_ratio = color_range / glcm_range if glcm_range > 0 else float('inf')
        
        print(f"\n⚠️ SCALING ISSUE DETECTED:")
        print(f"   Color features are {scaling_ratio:.1f}x larger in range than GLCM features!")
        print(f"   This explains why the model ignores GLCM features.")
        
        return scaling_ratio
    
    def fix_feature_scaling(self, X):
        """Fix feature scaling issues"""
        print("\n🔧 FIXING FEATURE SCALING")
        print("=" * 50)
        
        # Apply different scaling strategies
        scaler_standard = StandardScaler()
        scaler_minmax = MinMaxScaler()
        
        # Standard scaling (mean=0, std=1)
        X_standard = scaler_standard.fit_transform(X)
        
        # MinMax scaling (0-1 range)
        X_minmax = scaler_minmax.fit_transform(X)
        
        # Separate scaling for GLCM and Color features
        glcm_features = X[:, :20]
        color_features = X[:, 20:]
        
        # Scale GLCM and Color features separately
        glcm_scaler = StandardScaler()
        color_scaler = StandardScaler()
        
        glcm_scaled = glcm_scaler.fit_transform(glcm_features)
        color_scaled = color_scaler.fit_transform(color_features)
        X_separate = np.concatenate([glcm_scaled, color_scaled], axis=1)
        
        print(f"✅ Scaling strategies applied:")
        print(f"   Original shape: {X.shape}")
        print(f"   Standard scaled shape: {X_standard.shape}")
        print(f"   MinMax scaled shape: {X_minmax.shape}")
        print(f"   Separate scaled shape: {X_separate.shape}")
        
        return {
            'original': X,
            'standard': X_standard,
            'minmax': X_minmax,
            'separate': X_separate
        }
    
    def perform_cross_validation(self, X_scaled, y, scaling_name):
        """Perform cross-validation with different scaling approaches"""
        print(f"\n🎯 PERFORMING CROSS-VALIDATION ({scaling_name.upper()})")
        print("=" * 50)
        
        # Create stratified k-fold
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
        
        # Test different models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'RandomForest_Balanced': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced'
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"🧪 Testing {model_name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            
            # Train-test split for detailed analysis
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results[model_name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'y_test': y_test,
                'y_pred': y_pred,
                'model': model
            }
            
            print(f"   CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"   Test Accuracy: {accuracy:.3f}")
        
        return results
    
    def create_confusion_matrix_analysis(self, results, scaling_name):
        """Create comprehensive confusion matrix analysis"""
        print(f"\n📊 CREATING CONFUSION MATRIX ANALYSIS ({scaling_name.upper()})")
        print("=" * 60)
        
        # Create figure with subplots for each model
        n_models = len(results)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for i, (model_name, result) in enumerate(results.items()):
            conf_matrix = result['confusion_matrix']
            
            # Confusion Matrix (top row)
            ax1 = axes[0, i]
            im = ax1.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
            ax1.set_title(f'{model_name}\nConfusion Matrix')
            
            # Add colorbar
            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            
            # Add text annotations
            thresh = conf_matrix.max() / 2.
            for row in range(conf_matrix.shape[0]):
                for col in range(conf_matrix.shape[1]):
                    ax1.text(col, row, conf_matrix[row, col],
                            ha="center", va="center",
                            color="white" if conf_matrix[row, col] > thresh else "black",
                            fontsize=10, fontweight='bold')
            
            # Set labels
            ax1.set_xticks(range(len(self.ethnicity_labels)))
            ax1.set_yticks(range(len(self.ethnicity_labels)))
            ax1.set_xticklabels(self.ethnicity_labels, rotation=45)
            ax1.set_yticklabels(self.ethnicity_labels)
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # Normalized Confusion Matrix (bottom row)
            ax2 = axes[1, i]
            conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            
            im2 = ax2.imshow(conf_matrix_norm, interpolation='nearest', cmap='Blues')
            ax2.set_title(f'{model_name}\nNormalized Confusion Matrix')
            
            # Add colorbar
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # Add text annotations
            thresh = conf_matrix_norm.max() / 2.
            for row in range(conf_matrix_norm.shape[0]):
                for col in range(conf_matrix_norm.shape[1]):
                    ax2.text(col, row, f'{conf_matrix_norm[row, col]:.2f}',
                            ha="center", va="center",
                            color="white" if conf_matrix_norm[row, col] > thresh else "black",
                            fontsize=9)
            
            # Set labels
            ax2.set_xticks(range(len(self.ethnicity_labels)))
            ax2.set_yticks(range(len(self.ethnicity_labels)))
            ax2.set_xticklabels(self.ethnicity_labels, rotation=45)
            ax2.set_yticklabels(self.ethnicity_labels)
            ax2.set_ylabel('True Label')
            ax2.set_xlabel('Predicted Label')
        
        # Main title
        fig.suptitle(f'Confusion Matrix Analysis - {scaling_name.upper()} Scaling', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save plot
        confusion_path = os.path.join(self.output_dir, f'confusion_matrix_{scaling_name}.png')
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Confusion matrix saved to: {confusion_path}")
    
    def create_performance_comparison(self, all_results):
        """Create performance comparison across different scaling approaches"""
        print("\n📈 CREATING PERFORMANCE COMPARISON")
        print("=" * 50)
        
        # Prepare data for comparison
        comparison_data = []
        
        for scaling_name, results in all_results.items():
            for model_name, result in results.items():
                comparison_data.append({
                    'Scaling': scaling_name.title(),
                    'Model': model_name,
                    'CV_Mean': result['cv_mean'],
                    'CV_Std': result['cv_std'],
                    'Test_Accuracy': result['test_accuracy']
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cross-validation comparison
        x_pos = np.arange(len(df))
        bars1 = ax1.bar(x_pos, df['CV_Mean'], yerr=df['CV_Std'], 
                       capsize=5, alpha=0.7, color=self.colors['primary'])
        
        ax1.set_xlabel('Model Configuration')
        ax1.set_ylabel('Cross-Validation Accuracy')
        ax1.set_title('Cross-Validation Performance Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{row['Scaling']}\n{row['Model']}" for _, row in df.iterrows()], 
                           rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars1, df['CV_Mean'], df['CV_Std']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Test accuracy comparison
        bars2 = ax2.bar(x_pos, df['Test_Accuracy'], alpha=0.7, color=self.colors['secondary'])
        
        ax2.set_xlabel('Model Configuration')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Test Set Performance Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{row['Scaling']}\n{row['Model']}" for _, row in df.iterrows()], 
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, accuracy in zip(bars2, df['Test_Accuracy']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        comparison_path = os.path.join(self.output_dir, 'performance_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Performance comparison saved to: {comparison_path}")
        
        # Print summary
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(df.to_string(index=False, float_format='%.3f'))
        
        return df
    
    def save_best_model(self, all_results):
        """Save the best performing model"""
        print("\n💾 SAVING BEST MODEL")
        print("=" * 50)
        
        # Find best model based on CV score
        best_score = 0
        best_config = None
        
        for scaling_name, results in all_results.items():
            for model_name, result in results.items():
                if result['cv_mean'] > best_score:
                    best_score = result['cv_mean']
                    best_config = {
                        'scaling': scaling_name,
                        'model': model_name,
                        'result': result
                    }
        
        if best_config:
            # Save model
            model_path = os.path.join(self.output_dir, 'best_cross_validated_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(best_config['result']['model'], f)
            
            print(f"✅ Best model saved to: {model_path}")
            print(f"📊 Best configuration:")
            print(f"   Scaling: {best_config['scaling']}")
            print(f"   Model: {best_config['model']}")
            print(f"   CV Score: {best_score:.3f}")
            
            return best_config
        else:
            print("❌ No best model found")
            return None

def main():
    """Main function to run the cross-validation fix system"""
    print("🚀 CROSS-VALIDATION FIX SYSTEM")
    print("=" * 70)
    
    # Initialize system
    fix_system = CrossValidationFixSystem()
    
    # Load and prepare data
    X, y = fix_system.load_and_prepare_data()
    
    if X is not None and y is not None:
        # Analyze feature scaling
        scaling_ratio = fix_system.analyze_feature_scaling(X)
        
        # Fix feature scaling
        X_scaled_dict = fix_system.fix_feature_scaling(X)
        
        # Perform cross-validation for each scaling approach
        all_results = {}
        
        for scaling_name, X_scaled in X_scaled_dict.items():
            if scaling_name == 'original':
                continue  # Skip original (we know it has issues)
            
            print(f"\n{'='*70}")
            results = fix_system.perform_cross_validation(X_scaled, y, scaling_name)
            all_results[scaling_name] = results
            
            # Create confusion matrix analysis
            fix_system.create_confusion_matrix_analysis(results, scaling_name)
        
        # Create performance comparison
        comparison_df = fix_system.create_performance_comparison(all_results)
        
        # Save best model
        best_config = fix_system.save_best_model(all_results)
        
        print(f"\n🎉 CROSS-VALIDATION FIX COMPLETED!")
        print(f"📁 Check logs/analysis/ for all generated plots and models")
        
        return all_results, comparison_df, best_config
        
    else:
        print("❌ Failed to load data")
        return None, None, None

if __name__ == "__main__":
    main()
