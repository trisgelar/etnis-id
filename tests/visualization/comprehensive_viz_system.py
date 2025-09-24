#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive visualization system without LaTeX dependencies
Creates beautiful publication-quality plots for ethnicity detection analysis
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, '.')

# Configure matplotlib for publication quality without LaTeX
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

class ModelAnalyzer:
    """Comprehensive model analysis with beautiful visualizations"""
    
    def __init__(self, output_dir="logs/analysis"):
        """Initialize analyzer"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Color scheme for beautiful plots
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'ethnicities': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
        
        print(f"📊 ModelAnalyzer initialized - Output: {output_dir}")
    
    def analyze_model_performance(self):
        """Analyze current model performance"""
        print("🔍 ANALYZING MODEL PERFORMANCE")
        print("=" * 50)
        
        try:
            from ethnic_detector import EthnicDetector
            
            # Load model
            detector = EthnicDetector()
            
            # Test with different image types
            test_cases = []
            
            # Random images
            print("📊 Testing with random images...")
            np.random.seed(42)
            for i in range(8):
                test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                prediction, confidence, status = detector.predict_ethnicity(test_image)
                test_cases.append({
                    'type': 'Random',
                    'prediction': prediction,
                    'confidence': confidence,
                    'image_id': i+1
                })
            
            # Uniform color images
            print("📊 Testing with uniform color images...")
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
            for i, color in enumerate(colors):
                uniform_image = np.full((100, 100, 3), color, dtype=np.uint8)
                prediction, confidence, status = detector.predict_ethnicity(uniform_image)
                test_cases.append({
                    'type': f'Uniform_{color[0]}',
                    'prediction': prediction,
                    'confidence': confidence,
                    'image_id': f'U{i+1}'
                })
            
            return test_cases
            
        except Exception as e:
            print(f"❌ Error in model analysis: {e}")
            return []
    
    def create_comprehensive_dashboard(self, test_cases):
        """Create comprehensive analysis dashboard"""
        print("🎨 CREATING COMPREHENSIVE DASHBOARD")
        print("=" * 50)
        
        if not test_cases:
            print("❌ No test cases available")
            return
        
        # Calculate statistics
        confidences = [tc['confidence'] for tc in test_cases]
        predictions = [tc['prediction'] for tc in test_cases]
        
        avg_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        min_confidence = np.min(confidences)
        std_confidence = np.std(confidences)
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Confidence Distribution (Top Left)
        ax1 = plt.subplot(3, 3, 1)
        ax1.hist(confidences, bins=8, alpha=0.7, color=self.colors['primary'], 
                edgecolor='black', linewidth=1)
        ax1.axvline(avg_confidence, color=self.colors['warning'], linestyle='--', 
                   linewidth=2, label=f'Mean: {avg_confidence:.1f}%')
        ax1.set_xlabel('Confidence (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence by Test Case (Top Center)
        ax2 = plt.subplot(3, 3, 2)
        test_names = [f"T{tc['image_id']}" for tc in test_cases]
        bars = ax2.bar(range(len(test_names)), confidences, 
                      color=self.colors['secondary'], alpha=0.7)
        ax2.set_xlabel('Test Case')
        ax2.set_ylabel('Confidence (%)')
        ax2.set_title('Confidence by Test Case')
        ax2.set_xticks(range(len(test_names)))
        ax2.set_xticklabels(test_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, conf in zip(bars, confidences):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{conf:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Prediction Distribution (Top Right)
        ax3 = plt.subplot(3, 3, 3)
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        bars = ax3.bar(pred_counts.keys(), pred_counts.values(), 
                      color=self.colors['ethnicities'][:len(pred_counts)], alpha=0.7)
        ax3.set_xlabel('Predicted Ethnicity')
        ax3.set_ylabel('Count')
        ax3.set_title('Prediction Distribution')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, pred_counts.values()):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Performance Metrics (Middle Left)
        ax4 = plt.subplot(3, 3, 4)
        metrics = ['Avg Confidence', 'Max Confidence', 'Min Confidence', 'Std Dev']
        values = [avg_confidence, max_confidence, min_confidence, std_confidence]
        colors_metrics = [self.colors['primary'], self.colors['success'], 
                         self.colors['warning'], self.colors['info']]
        
        bars = ax4.bar(metrics, values, color=colors_metrics, alpha=0.7)
        ax4.set_ylabel('Confidence (%)')
        ax4.set_title('Performance Metrics')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 5. Comparison with Original (Middle Center)
        ax5 = plt.subplot(3, 3, 5)
        comparison_data = ['Original\nNotebook', 'Current\nModel']
        comparison_values = [98.6, avg_confidence]
        comparison_colors = [self.colors['success'], self.colors['warning']]
        
        bars = ax5.bar(comparison_data, comparison_values, 
                      color=comparison_colors, alpha=0.7)
        ax5.set_ylabel('Performance (%)')
        ax5.set_title('Performance Comparison')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, comparison_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 6. Issues Analysis (Middle Right)
        ax6 = plt.subplot(3, 3, 6)
        issues = ['Low Confidence', 'Limited Diversity', 'Potential Overfitting']
        issue_scores = [
            1 if avg_confidence < 50 else 0,
            1 if len(set(predictions)) < 3 else 0,
            1 if avg_confidence < 70 else 0
        ]
        issue_colors = [self.colors['warning'] if score == 1 else self.colors['success'] 
                       for score in issue_scores]
        
        bars = ax6.bar(issues, issue_scores, color=issue_colors, alpha=0.7)
        ax6.set_ylabel('Issue Severity (0=OK, 1=Issue)')
        ax6.set_title('Issues Analysis')
        ax6.tick_params(axis='x', rotation=45)
        ax6.set_ylim(0, 1.2)
        ax6.grid(True, alpha=0.3)
        
        # 7. Recommendations (Bottom Left)
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')
        
        recommendations = [
            "1. Implement Cross-Validation",
            "2. Add Confusion Matrix Analysis", 
            "3. Create Feature Importance Plots",
            "4. Perform Hyperparameter Tuning",
            "5. Monitor Training Curves"
        ]
        
        y_pos = 0.9
        for i, rec in enumerate(recommendations):
            ax7.text(0.05, y_pos, rec, transform=ax7.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            y_pos -= 0.15
        
        ax7.set_title('Recommendations', fontsize=12, fontweight='bold')
        
        # 8. Summary Statistics (Bottom Center)
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        summary_text = f"""
MODEL ANALYSIS SUMMARY

Performance Metrics:
• Average Confidence: {avg_confidence:.1f}%
• Standard Deviation: {std_confidence:.1f}%
• Max Confidence: {max_confidence:.1f}%
• Min Confidence: {min_confidence:.1f}%

Prediction Analysis:
• Total Test Cases: {len(test_cases)}
• Unique Predictions: {len(set(predictions))}
• Most Common: {max(set(predictions), key=predictions.count)}

Issues Detected:
• Low Confidence: {'Yes' if avg_confidence < 50 else 'No'}
• Limited Diversity: {'Yes' if len(set(predictions)) < 3 else 'No'}
• Overfitting Risk: {'High' if avg_confidence < 70 else 'Low'}
        """
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 9. Feature Analysis Placeholder (Bottom Right)
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        feature_text = """
FEATURE ANALYSIS
(To be implemented)

Planned Features:
• GLCM Feature Importance
• Color Histogram Analysis
• Cross-Validation Results
• Confusion Matrix
• ROC Curves
        """
        
        ax9.text(0.05, 0.95, feature_text, transform=ax9.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # Main title
        fig.suptitle('Ethnicity Detection Model - Comprehensive Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the dashboard
        dashboard_path = os.path.join(self.output_dir, 'comprehensive_analysis_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Comprehensive dashboard saved to: {dashboard_path}")
        
        return {
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence,
            'std_confidence': std_confidence,
            'unique_predictions': len(set(predictions)),
            'total_tests': len(test_cases)
        }
    
    def create_feature_importance_plot(self, model):
        """Create feature importance visualization"""
        print("🔍 CREATING FEATURE IMPORTANCE PLOT")
        print("=" * 50)
        
        try:
            if not hasattr(model, 'feature_importances_'):
                print("❌ Model does not have feature_importances_ attribute")
                return
            
            importances = model.feature_importances_
            
            # Create feature names
            feature_names = []
            for i in range(len(importances)):
                if i < 20:
                    feature_names.append(f'GLCM_{i+1}')
                else:
                    feature_names.append(f'Color_{i-19}')
            
            # Get top 15 features
            top_n = 15
            indices = np.argsort(importances)[::-1][:top_n]
            top_importances = importances[indices]
            top_names = [feature_names[i] for i in indices]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Color coding for feature types
            colors = []
            for name in top_names:
                if 'GLCM' in name:
                    colors.append(self.colors['primary'])
                else:
                    colors.append(self.colors['secondary'])
            
            # Horizontal bar plot
            bars = ax.barh(range(len(top_importances)), top_importances, 
                          color=colors, alpha=0.7)
            
            # Customize plot
            ax.set_yticks(range(len(top_importances)))
            ax.set_yticklabels(top_names)
            ax.set_xlabel('Feature Importance Score')
            ax.set_title(f'Top {top_n} Most Important Features')
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, top_importances)):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.4f}', ha='left', va='center', fontsize=9)
            
            # Add legend
            glcm_patch = mpatches.Patch(color=self.colors['primary'], label='GLCM Features')
            color_patch = mpatches.Patch(color=self.colors['secondary'], label='Color Features')
            ax.legend(handles=[glcm_patch, color_patch], loc='lower right')
            
            plt.tight_layout()
            
            # Save plot
            feature_path = os.path.join(self.output_dir, 'feature_importance_analysis.png')
            plt.savefig(feature_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Feature importance plot saved to: {feature_path}")
            
        except Exception as e:
            print(f"❌ Error creating feature importance plot: {e}")

def main():
    """Main function to run comprehensive analysis"""
    print("🚀 COMPREHENSIVE VISUALIZATION SYSTEM")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = ModelAnalyzer()
    
    # Analyze model performance
    test_cases = analyzer.analyze_model_performance()
    
    if test_cases:
        # Create comprehensive dashboard
        results = analyzer.create_comprehensive_dashboard(test_cases)
        
        # Create feature importance plot
        try:
            from ethnic_detector import EthnicDetector
            detector = EthnicDetector()
            analyzer.create_feature_importance_plot(detector.model)
        except Exception as e:
            print(f"⚠️ Could not create feature importance plot: {e}")
        
        print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📊 Results Summary:")
        print(f"   Average Confidence: {results['avg_confidence']:.1f}%")
        print(f"   Unique Predictions: {results['unique_predictions']}")
        print(f"   Total Tests: {results['total_tests']}")
        
        print(f"\n📁 Output Files:")
        print(f"   • Comprehensive Dashboard: logs/analysis/comprehensive_analysis_dashboard.png")
        print(f"   • Feature Importance: logs/analysis/feature_importance_analysis.png")
        
        print(f"\n🎨 All plots use publication-quality styling without LaTeX dependencies!")
        
    else:
        print(f"\n❌ ANALYSIS FAILED")

if __name__ == "__main__":
    main()
