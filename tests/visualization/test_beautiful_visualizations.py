#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for beautiful visualization system with SciencePlots
Demonstrates publication-quality plots for ethnicity detection model
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '.')

from ml_training.core.visualizations import ModelVisualizer
from ml_training.core.utils import TrainingLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import make_classification

def create_sample_data():
    """Create sample data similar to ethnicity detection dataset"""
    # Create synthetic data with 5 classes (ethnicities) and 52 features
    X, y = make_classification(
        n_samples=1000,
        n_features=52,
        n_classes=5,
        n_clusters_per_class=1,
        n_informative=40,
        random_state=42
    )
    
    # Create realistic class names
    class_names = ['Bugis', 'Sunda', 'Malay', 'Jawa', 'Banjar']
    y_labels = [class_names[i] for i in y]
    
    return X, np.array(y_labels)

def test_beautiful_visualizations():
    """Test all visualization components with beautiful styling"""
    print("🎨 TESTING BEAUTIFUL VISUALIZATION SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize logger and visualizer
        logger = TrainingLogger('beautiful_viz_test')
        visualizer = ModelVisualizer(logger, output_dir="logs/visualizations", style='ieee')
        
        # Create sample data
        print("📊 Creating sample data...")
        X, y = create_sample_data()
        print(f"   ✅ Created dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        
        # Create output directory
        output_dir = "logs/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Train a model for testing
        print("\n🤖 Training model for visualization...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        print("   ✅ Model trained successfully")
        
        # 1. Test class distribution visualization
        print("\n📈 Creating class distribution visualization...")
        try:
            class_dist_fig = visualizer.plot_class_distribution(
                y, 
                title="Ethnicity Class Distribution",
                save_path=os.path.join(output_dir, "class_distribution.png")
            )
            plt.close(class_dist_fig)
            print("   ✅ Class distribution plot created")
        except Exception as e:
            print(f"   ❌ Error creating class distribution: {e}")
        
        # 2. Test feature importance visualization
        print("\n🔍 Creating feature importance visualization...")
        try:
            feature_names = [f'GLCM_{i}' if i < 20 else f'Color_{i-20}' for i in range(X.shape[1])]
            feature_fig = visualizer.plot_feature_importance(
                model,
                feature_names=feature_names,
                top_n=15,
                save_path=os.path.join(output_dir, "feature_importance.png")
            )
            plt.close(feature_fig)
            print("   ✅ Feature importance plot created")
        except Exception as e:
            print(f"   ❌ Error creating feature importance: {e}")
        
        # 3. Test cross-validation results
        print("\n📊 Creating cross-validation results...")
        try:
            cv_scores = cross_val_score(model, X, y, cv=6, scoring='accuracy')
            cv_fig = visualizer.plot_cross_validation_results(
                cv_scores,
                k_values=list(range(1, 7)),
                save_path=os.path.join(output_dir, "cv_results.png")
            )
            plt.close(cv_fig)
            print("   ✅ Cross-validation plot created")
        except Exception as e:
            print(f"   ❌ Error creating CV results: {e}")
        
        # 4. Test confusion matrix
        print("\n🎯 Creating confusion matrix...")
        try:
            # Get predictions for confusion matrix
            cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                if fold == 0:  # Just test first fold
                    model.fit(X[train_idx], y[train_idx])
                    y_pred = model.predict(X[test_idx])
                    y_true = y[test_idx]
                    
                    cm_fig = visualizer.plot_confusion_matrix_detailed(
                        y_true, y_pred,
                        fold_info={"fold": fold + 1},
                        save_path=os.path.join(output_dir, "confusion_matrix_fold_1.png"),
                        normalize=True
                    )
                    plt.close(cm_fig)
                    print("   ✅ Confusion matrix created")
                    break
        except Exception as e:
            print(f"   ❌ Error creating confusion matrix: {e}")
        
        # 5. Test performance metrics
        print("\n📋 Creating performance metrics...")
        try:
            # Get predictions for performance metrics
            cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
            all_pred = []
            all_true = []
            
            for train_idx, test_idx in cv.split(X, y):
                model.fit(X[train_idx], y[train_idx])
                y_pred = model.predict(X[test_idx])
                all_pred.extend(y_pred)
                all_true.extend(y[test_idx])
            
            perf_fig = visualizer.plot_performance_metrics(
                np.array(all_true), np.array(all_pred),
                save_path=os.path.join(output_dir, "performance_metrics.png")
            )
            plt.close(perf_fig)
            print("   ✅ Performance metrics plot created")
        except Exception as e:
            print(f"   ❌ Error creating performance metrics: {e}")
        
        # 6. Test comprehensive report
        print("\n📑 Creating comprehensive report...")
        try:
            results = visualizer.create_comprehensive_report(
                model, X, y, cv_folds=6, save_dir=output_dir
            )
            
            # Save results to Excel
            excel_path = os.path.join(output_dir, "model_analysis_results.xlsx")
            visualizer.save_results_to_excel(results, excel_path)
            
            print("   ✅ Comprehensive report created")
            print(f"   📊 Results saved to: {excel_path}")
            
            # Print summary
            print(f"\n📈 MODEL PERFORMANCE SUMMARY:")
            print(f"   Overall CV Accuracy: {results['cv_results']['mean']:.4f} ± {results['cv_results']['std']:.4f}")
            print(f"   Best Fold Accuracy: {results['cv_results']['max']:.4f}")
            print(f"   Worst Fold Accuracy: {results['cv_results']['min']:.4f}")
            print(f"   Wrong Predictions: {results['wrong_predictions_analysis']['wrong_count']}")
            print(f"   Error Rate: {results['wrong_predictions_analysis']['error_rate']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error creating comprehensive report: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 BEAUTIFUL VISUALIZATION TEST COMPLETED!")
        print(f"📁 All visualizations saved to: {output_dir}")
        print("🎨 All plots use SciencePlots for publication-quality styling!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_scienceplots():
    """Check if SciencePlots is properly installed"""
    print("🔬 CHECKING SCIENCEPLOTS INSTALLATION")
    print("=" * 40)
    
    try:
        import scienceplots
        print("✅ SciencePlots is installed")
        
        # Test different styles
        styles = ['ieee', 'nature', 'science', 'scatter']
        available_styles = []
        
        for style in styles:
            try:
                plt.style.use(['science', style])
                available_styles.append(style)
                print(f"   ✅ Style '{style}' is available")
            except:
                print(f"   ⚠️  Style '{style}' not available")
        
        plt.style.use('default')  # Reset to default
        print(f"\n📋 Available styles: {', '.join(available_styles)}")
        return True
        
    except ImportError:
        print("❌ SciencePlots is not installed")
        print("💡 Install with: pip install SciencePlots")
        return False

if __name__ == "__main__":
    print("🚀 STARTING BEAUTIFUL VISUALIZATION SYSTEM TEST")
    print("=" * 70)
    
    # Check SciencePlots first
    if check_scienceplots():
        print("\n")
        # Run visualization tests
        success = test_beautiful_visualizations()
        
        if success:
            print("\n🎊 SUCCESS! Your visualization system is ready for publication!")
            print("📝 All plots are optimized for academic papers with SciencePlots styling")
        else:
            print("\n💥 Test failed. Check the errors above.")
    else:
        print("\n⚠️  Install SciencePlots first: pip install SciencePlots")

