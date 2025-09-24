#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for SciencePlots functionality
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

print("🔬 TESTING SCIENCEPLOTS")
print("=" * 40)

try:
    import scienceplots
    print("✅ SciencePlots imported successfully")
    
    # Test different styles
    styles = ['ieee', 'nature', 'science', 'grid']
    
    for style in styles:
        try:
            plt.style.use(['science', style])
            print(f"✅ Style '{style}' works")
        except Exception as e:
            print(f"❌ Style '{style}' failed: {e}")
    
    # Create a simple test plot
    print("\n📊 Creating test plot...")
    
    # Set IEEE style
    plt.style.use(['science', 'ieee', 'grid'])
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    ax.set_xlabel('X axis', fontsize=12)
    ax.set_ylabel('Y axis', fontsize=12)
    ax.set_title('SciencePlots Test - IEEE Style', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('logs/scienceplots_test.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Test plot created and saved to logs/scienceplots_test.png")
    print("🎨 SciencePlots is working correctly!")
    
except ImportError:
    print("❌ SciencePlots not installed")
    print("💡 Install with: pip install SciencePlots")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n🎉 SciencePlots test completed successfully!")

