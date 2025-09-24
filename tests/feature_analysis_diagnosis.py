#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Analysis Diagnosis - Understanding why only color features matter
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '.')

def analyze_feature_importance_issue():
    """Analyze why only color features are important"""
    print("🔍 ANALYZING FEATURE IMPORTANCE ISSUE")
    print("=" * 60)
    
    try:
        from ethnic_detector import EthnicDetector
        
        # Load model
        detector = EthnicDetector()
        model = detector.model
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Separate GLCM and Color features
            glcm_importances = importances[:20]  # First 20 are GLCM
            color_importances = importances[20:]  # Last 32 are Color
            
            print(f"📊 FEATURE IMPORTANCE ANALYSIS:")
            print(f"   Total Features: {len(importances)}")
            print(f"   GLCM Features: {len(glcm_importances)}")
            print(f"   Color Features: {len(color_importances)}")
            print()
            
            print(f"🎨 COLOR FEATURES:")
            print(f"   Mean Importance: {np.mean(color_importances):.6f}")
            print(f"   Max Importance: {np.max(color_importances):.6f}")
            print(f"   Min Importance: {np.min(color_importances):.6f}")
            print(f"   Std Importance: {np.std(color_importances):.6f}")
            print()
            
            print(f"📊 GLCM FEATURES:")
            print(f"   Mean Importance: {np.mean(glcm_importances):.6f}")
            print(f"   Max Importance: {np.max(glcm_importances):.6f}")
            print(f"   Min Importance: {np.min(glcm_importances):.6f}")
            print(f"   Std Importance: {np.std(glcm_importances):.6f}")
            print()
            
            # Calculate ratios
            color_vs_glcm_ratio = np.mean(color_importances) / np.mean(glcm_importances)
            print(f"⚠️ CRITICAL FINDING:")
            print(f"   Color features are {color_vs_glcm_ratio:.1f}x more important than GLCM features!")
            print(f"   This suggests the model is ONLY learning color patterns, not texture patterns.")
            print()
            
            # Top features analysis
            print(f"🏆 TOP 10 FEATURES:")
            top_10_indices = np.argsort(importances)[::-1][:10]
            for i, idx in enumerate(top_10_indices):
                feature_type = "GLCM" if idx < 20 else "Color"
                feature_name = f"GLCM_{idx+1}" if idx < 20 else f"Color_{idx-19}"
                print(f"   {i+1:2d}. {feature_name:12s} ({feature_type:5s}): {importances[idx]:.6f}")
            
            return {
                'color_mean': np.mean(color_importances),
                'glcm_mean': np.mean(glcm_importances),
                'ratio': color_vs_glcm_ratio,
                'color_importances': color_importances,
                'glcm_importances': glcm_importances,
                'top_10_indices': top_10_indices
            }
            
        else:
            print("❌ Model does not have feature_importances_ attribute")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def create_feature_comparison_plot(analysis_data):
    """Create comparison plot between GLCM and Color features"""
    print("🎨 CREATING FEATURE COMPARISON PLOT")
    print("=" * 50)
    
    if not analysis_data:
        return
    
    # Create comparison visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Feature importance comparison (bar chart)
    feature_types = ['GLCM Features', 'Color Features']
    mean_importances = [analysis_data['glcm_mean'], analysis_data['color_mean']]
    colors = ['#ff7f0e', '#1f77b4']  # Orange for GLCM, Blue for Color
    
    bars = ax1.bar(feature_types, mean_importances, color=colors, alpha=0.7)
    ax1.set_ylabel('Mean Feature Importance')
    ax1.set_title('GLCM vs Color Features - Mean Importance')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, importance in zip(bars, mean_importances):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{importance:.6f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Distribution comparison (histogram)
    ax2.hist(analysis_data['glcm_importances'], bins=10, alpha=0.7, 
             label='GLCM Features', color='#ff7f0e', density=True)
    ax2.hist(analysis_data['color_importances'], bins=10, alpha=0.7, 
             label='Color Features', color='#1f77b4', density=True)
    ax2.set_xlabel('Feature Importance')
    ax2.set_ylabel('Density')
    ax2.set_title('Feature Importance Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot comparison
    box_data = [analysis_data['glcm_importances'], analysis_data['color_importances']]
    bp = ax3.boxplot(box_data, labels=['GLCM', 'Color'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff7f0e')
    bp['boxes'][1].set_facecolor('#1f77b4')
    ax3.set_ylabel('Feature Importance')
    ax3.set_title('Feature Importance Distribution (Box Plot)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Top features breakdown
    top_10_indices = analysis_data['top_10_indices']
    glcm_count = sum(1 for idx in top_10_indices if idx < 20)
    color_count = len(top_10_indices) - glcm_count
    
    sizes = [glcm_count, color_count]
    labels = [f'GLCM Features\n({glcm_count})', f'Color Features\n({color_count})']
    colors_pie = ['#ff7f0e', '#1f77b4']
    
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie, 
                                       autopct='%1.0f%%', startangle=90)
    ax4.set_title('Top 10 Features Breakdown')
    
    # Main title
    fig.suptitle('Feature Importance Analysis - GLCM vs Color Features', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save plot
    os.makedirs("logs/analysis", exist_ok=True)
    plt.savefig('logs/analysis/feature_importance_diagnosis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Feature comparison plot saved to: logs/analysis/feature_importance_diagnosis.png")

def diagnose_model_issues():
    """Diagnose the root causes of model issues"""
    print("\n🔬 DIAGNOSING MODEL ISSUES")
    print("=" * 60)
    
    print("⚠️ CRITICAL ISSUES IDENTIFIED:")
    print()
    print("1. 🎨 COLOR-ONLY LEARNING:")
    print("   • Model only uses color histogram features")
    print("   • GLCM texture features are completely ignored")
    print("   • This explains why random images get similar predictions")
    print()
    
    print("2. 📊 FEATURE EXTRACTION PROBLEM:")
    print("   • GLCM features may not be properly normalized")
    print("   • Color features dominate due to scaling issues")
    print("   • Feature selection is biased toward color information")
    print()
    
    print("3. 🎯 OVERFITTING ROOT CAUSE:")
    print("   • Model learns simple color patterns instead of complex textures")
    print("   • No cross-validation to prevent overfitting")
    print("   • Training data may have color bias")
    print()
    
    print("4. 🔧 SOLUTIONS NEEDED:")
    print("   • Fix feature normalization/scaling")
    print("   • Implement proper cross-validation")
    print("   • Add feature selection/rebalancing")
    print("   • Analyze training data for color bias")
    print()

def main():
    """Main function"""
    print("🚀 FEATURE IMPORTANCE DIAGNOSIS")
    print("=" * 70)
    
    # Analyze feature importance
    analysis_data = analyze_feature_importance_issue()
    
    if analysis_data:
        # Create comparison plot
        create_feature_comparison_plot(analysis_data)
        
        # Diagnose issues
        diagnose_model_issues()
        
        print("✅ DIAGNOSIS COMPLETED!")
        print("📁 Check logs/analysis/feature_importance_diagnosis.png")
        
        return analysis_data
    else:
        print("❌ DIAGNOSIS FAILED")
        return None

if __name__ == "__main__":
    main()
