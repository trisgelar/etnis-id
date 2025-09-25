#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBP Variants Comparison Test for Ethnic Detection
Compares all LBP variants + HSV to determine which LBP variant has the best accuracy
for ethnic detection using the SOLID-compliant training system.

LBP Variants tested:
1. KKPaperLBPExtractor - Multi-scale, rotation-invariant LBP with spatial histograms
2. MBLBPExtractor - Multi-Block Local Binary Pattern (Zhang et al. 2007)
3. MBPExtractor - Median Binary Pattern (Hafiane et al.)
4. DLBPExtractor - Divided Local Binary Pattern (Hua et al.)
5. MQLBPExtractor - Multi-quantized Local Binary Pattern (Patel et al. 2016)
6. DLBPExtractor (d-LBP) - Doubled Local Binary Pattern
7. RedDLBPExtractor - Reduced Divided Local Binary Pattern

Each variant is tested with HSV color features for comprehensive comparison.
"""

import sys
import os
import time
import json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_training.core.utils import TrainingLogger, ProgressTracker
from ml_training.core.training_pipeline import PipelineFactory
from ml_training.core.lbp_variants import (
    KKPaperLBPExtractor, MBLBPExtractor, MBPExtractor, 
    DLBPExtractor, MQLBPExtractor, RedDLBPExtractor
)


@dataclass
class LBPTestResult:
    """Data class to store test results for each LBP variant."""
    variant_name: str
    variant_type: str
    description: str
    feature_dimension: int
    mean_cv_accuracy: float
    std_cv_accuracy: float
    test_accuracy: float
    training_time: float
    feature_extraction_time: float
    model_type: str
    total_features: int
    train_size: int
    test_size: int
    num_classes: int
    success: bool
    error_message: str = ""


class LBPVariantsComparator:
    """Main class for comparing LBP variants with HSV for ethnic detection."""
    
    def __init__(self, dataset_path: str, output_dir: str):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Create dynamic logger name with timestamp for reproducibility
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger_name = f'lbp_comparison_{self.timestamp}'
        self.logger = TrainingLogger(self.logger_name)
        self.progress_tracker = ProgressTracker(self.logger)
        self.results: List[LBPTestResult] = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # LBP variants configuration
        self.lbp_variants = {
            'KKPaperLBP': {
                'class': KKPaperLBPExtractor,
                'params': {
                    'radii': [1, 2, 3],
                    'points': [8, 16, 24],
                    'method': 'uniform',
                    'grid_size': (4, 4)
                },
                'description': 'Multi-scale, rotation-invariant LBP with spatial histograms'
            },
            'MBLBP': {
                'class': MBLBPExtractor,
                'params': {
                    'block_size': (2, 3)
                },
                'description': 'Multi-Block Local Binary Pattern (Zhang et al. 2007)'
            },
            'MBP': {
                'class': MBPExtractor,
                'params': {},
                'description': 'Median Binary Pattern (Hafiane et al.)'
            },
            'DLBP': {
                'class': DLBPExtractor,
                'params': {},
                'description': 'Divided Local Binary Pattern (Hua et al.)'
            },
            'MQLBP': {
                'class': MQLBPExtractor,
                'params': {
                    'L': 2,
                    'thresholds': [0.1, 0.2]
                },
                'description': 'Multi-quantized Local Binary Pattern (Patel et al. 2016)'
            },
            'dLBP': {
                'class': DLBPExtractor,  # Note: This is the doubled LBP variant
                'params': {
                    'radius1': 1,
                    'radius2': 3,
                    'n_points': 8
                },
                'description': 'Doubled Local Binary Pattern (two radii neighborhoods)'
            },
            'RedDLBP': {
                'class': RedDLBPExtractor,
                'params': {
                    'radius': 2
                },
                'description': 'Reduced Divided Local Binary Pattern (6 pixels, 2 groups)'
            }
        }
    
    def create_lbp_config(self, variant_name: str, variant_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create configuration for a specific LBP variant."""
        return {
            'model': {
                'model_type': 'RandomForest',
                'n_estimators': 100,
                'random_state': 220,
            },
            'cross_validation': {
                'n_folds': 5,
            },
            'feature_extraction': {
                'glc_distances': [1],
                'glc_angles': [0, 45, 90, 135],
                'glc_levels': 256,
                'color_bins': 16,
                'color_channels': [1, 2],  # HSV channels
                'lbp_variant': variant_name,
                'lbp_params': variant_info['params']
            }
        }
    
    def test_single_lbp_variant(self, variant_name: str, variant_info: Dict[str, Any]) -> LBPTestResult:
        """Test a single LBP variant and return results."""
        self.logger.info(f"🧪 Testing {variant_name}: {variant_info['description']}")
        print(f"\n{'='*60}")
        print(f"🧪 TESTING {variant_name}")
        print(f"📝 {variant_info['description']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Create configuration for this variant
            config = self.create_lbp_config(variant_name, variant_info)
            
            # Create pipeline with LBP variant
            pipeline = PipelineFactory.create_pipeline(
                self.logger, 
                self.progress_tracker, 
                custom_config=config
            )
            
            # Set the LBP variant in the pipeline
            if hasattr(pipeline, 'feature_extractor') and hasattr(pipeline.feature_extractor, 'lbp_extractor'):
                lbp_class = variant_info['class']
                lbp_extractor = lbp_class(
                    self.logger, 
                    self.progress_tracker, 
                    **variant_info['params']
                )
                pipeline.feature_extractor.lbp_extractor = lbp_extractor
            
            # Create output path for this variant with timestamp
            output_path = os.path.join(self.output_dir, f'{variant_name.lower()}_{self.timestamp}_model.pkl')
            
            # Run the pipeline
            feature_start = time.time()
            results = pipeline.run_pipeline(self.dataset_path, output_path)
            feature_end = time.time()
            
            total_time = time.time() - start_time
            feature_time = feature_end - feature_start
            
            if results.get('model_saved', False):
                # Extract results
                cv_results = results.get('cross_validation', {})
                test_eval = results.get('test_evaluation', {})
                model_info = results.get('model_info', {})
                feature_info = results.get('feature_info', {})
                data_metadata = results.get('data_metadata', {})
                
                # Get LBP feature dimension
                lbp_dimension = 0
                if hasattr(pipeline, 'feature_extractor') and hasattr(pipeline.feature_extractor, 'lbp_extractor'):
                    lbp_dimension = pipeline.feature_extractor.lbp_extractor.feature_info.get('feature_dimension', 0)
                
                result = LBPTestResult(
                    variant_name=variant_name,
                    variant_type=variant_info['class'].__name__,
                    description=variant_info['description'],
                    feature_dimension=lbp_dimension,
                    mean_cv_accuracy=cv_results.get('mean_accuracy', 0.0),
                    std_cv_accuracy=cv_results.get('std_accuracy', 0.0),
                    test_accuracy=test_eval.get('test_accuracy', 0.0),
                    training_time=total_time,
                    feature_extraction_time=feature_time,
                    model_type=model_info.get('algorithm', 'Unknown'),
                    total_features=feature_info.get('total_features', 0),
                    train_size=test_eval.get('train_size', 0),
                    test_size=test_eval.get('test_size', 0),
                    num_classes=data_metadata.get('num_classes', 0),
                    success=True
                )
                
                print(f"✅ {variant_name} completed successfully!")
                print(f"   CV Accuracy: {result.mean_cv_accuracy:.2f}% ± {result.std_cv_accuracy:.2f}%")
                print(f"   Test Accuracy: {result.test_accuracy:.2f}%")
                print(f"   Feature Dimension: {result.feature_dimension}")
                print(f"   Training Time: {result.training_time:.2f}s")
                
                return result
            else:
                error_msg = "Model not saved - pipeline failed"
                print(f"❌ {variant_name} failed: {error_msg}")
                
                return LBPTestResult(
                    variant_name=variant_name,
                    variant_type=variant_info['class'].__name__,
                    description=variant_info['description'],
                    feature_dimension=0,
                    mean_cv_accuracy=0.0,
                    std_cv_accuracy=0.0,
                    test_accuracy=0.0,
                    training_time=time.time() - start_time,
                    feature_extraction_time=0.0,
                    model_type='Unknown',
                    total_features=0,
                    train_size=0,
                    test_size=0,
                    num_classes=0,
                    success=False,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Exception during testing: {str(e)}"
            print(f"❌ {variant_name} failed with exception: {error_msg}")
            self.logger.error(f"Error testing {variant_name}: {error_msg}")
            
            return LBPTestResult(
                variant_name=variant_name,
                variant_type=variant_info['class'].__name__,
                description=variant_info['description'],
                feature_dimension=0,
                mean_cv_accuracy=0.0,
                std_cv_accuracy=0.0,
                test_accuracy=0.0,
                training_time=time.time() - start_time,
                feature_extraction_time=0.0,
                model_type='Unknown',
                total_features=0,
                train_size=0,
                test_size=0,
                num_classes=0,
                success=False,
                error_message=error_msg
            )
    
    def run_comparison(self) -> List[LBPTestResult]:
        """Run comparison test for all LBP variants."""
        print("🚀 STARTING LBP VARIANTS COMPARISON FOR ETHNIC DETECTION")
        print("=" * 70)
        print(f"📊 Dataset: {self.dataset_path}")
        print(f"💾 Output Directory: {self.output_dir}")
        print(f"🧪 Testing {len(self.lbp_variants)} LBP variants + HSV")
        print("=" * 70)
        
        self.results = []
        
        for i, (variant_name, variant_info) in enumerate(self.lbp_variants.items(), 1):
            print(f"\n📈 Progress: {i}/{len(self.lbp_variants)} variants")
            
            result = self.test_single_lbp_variant(variant_name, variant_info)
            self.results.append(result)
            
            # Small delay between tests
            time.sleep(1)
        
        return self.results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze and rank the results."""
        print("\n📊 ANALYZING RESULTS")
        print("=" * 50)
        
        # Filter successful results
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        if not successful_results:
            print("❌ No successful tests to analyze!")
            return {}
        
        # Sort by test accuracy (descending)
        accuracy_ranking = sorted(successful_results, key=lambda x: x.test_accuracy, reverse=True)
        
        # Sort by CV accuracy (descending)
        cv_accuracy_ranking = sorted(successful_results, key=lambda x: x.mean_cv_accuracy, reverse=True)
        
        # Sort by training time (ascending - faster is better)
        speed_ranking = sorted(successful_results, key=lambda x: x.training_time)
        
        # Calculate statistics
        accuracies = [r.test_accuracy for r in successful_results]
        cv_accuracies = [r.mean_cv_accuracy for r in successful_results]
        training_times = [r.training_time for r in successful_results]
        
        analysis = {
            'total_variants_tested': len(self.results),
            'successful_tests': len(successful_results),
            'failed_tests': len(failed_results),
            'accuracy_ranking': accuracy_ranking,
            'cv_accuracy_ranking': cv_accuracy_ranking,
            'speed_ranking': speed_ranking,
            'statistics': {
                'test_accuracy': {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'min': np.min(accuracies),
                    'max': np.max(accuracies)
                },
                'cv_accuracy': {
                    'mean': np.mean(cv_accuracies),
                    'std': np.std(cv_accuracies),
                    'min': np.min(cv_accuracies),
                    'max': np.max(cv_accuracies)
                },
                'training_time': {
                    'mean': np.mean(training_times),
                    'std': np.std(training_times),
                    'min': np.min(training_times),
                    'max': np.max(training_times)
                }
            },
            'failed_variants': failed_results
        }
        
        return analysis
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """Print comprehensive analysis report."""
        print("\n🏆 LBP VARIANTS COMPARISON RESULTS")
        print("=" * 70)
        
        print(f"📊 Summary:")
        print(f"   Total variants tested: {analysis['total_variants_tested']}")
        print(f"   Successful tests: {analysis['successful_tests']}")
        print(f"   Failed tests: {analysis['failed_tests']}")
        
        if analysis['failed_tests'] > 0:
            print(f"\n❌ Failed Variants:")
            for result in analysis['failed_variants']:
                print(f"   - {result.variant_name}: {result.error_message}")
        
        print(f"\n📈 Test Accuracy Ranking (Best to Worst):")
        for i, result in enumerate(analysis['accuracy_ranking'], 1):
            print(f"   {i:2d}. {result.variant_name:12s}: {result.test_accuracy:6.2f}% "
                  f"(CV: {result.mean_cv_accuracy:6.2f}% ± {result.std_cv_accuracy:5.2f}%, "
                  f"Time: {result.training_time:6.1f}s, Dim: {result.feature_dimension:4d})")
        
        print(f"\n🎯 Cross-Validation Accuracy Ranking:")
        for i, result in enumerate(analysis['cv_accuracy_ranking'], 1):
            print(f"   {i:2d}. {result.variant_name:12s}: {result.mean_cv_accuracy:6.2f}% ± {result.std_cv_accuracy:5.2f}%")
        
        print(f"\n⚡ Training Speed Ranking (Fastest to Slowest):")
        for i, result in enumerate(analysis['speed_ranking'], 1):
            print(f"   {i:2d}. {result.variant_name:12s}: {result.training_time:6.1f}s")
        
        stats = analysis['statistics']
        print(f"\n📊 Overall Statistics:")
        print(f"   Test Accuracy: {stats['test_accuracy']['mean']:.2f}% ± {stats['test_accuracy']['std']:.2f}% "
              f"(Range: {stats['test_accuracy']['min']:.2f}% - {stats['test_accuracy']['max']:.2f}%)")
        print(f"   CV Accuracy:   {stats['cv_accuracy']['mean']:.2f}% ± {stats['cv_accuracy']['std']:.2f}% "
              f"(Range: {stats['cv_accuracy']['min']:.2f}% - {stats['cv_accuracy']['max']:.2f}%)")
        print(f"   Training Time: {stats['training_time']['mean']:.1f}s ± {stats['training_time']['std']:.1f}s "
              f"(Range: {stats['training_time']['min']:.1f}s - {stats['training_time']['max']:.1f}s)")
        
        # Best performer
        best_result = analysis['accuracy_ranking'][0]
        print(f"\n🏆 BEST PERFORMER: {best_result.variant_name}")
        print(f"   📝 Description: {best_result.description}")
        print(f"   🎯 Test Accuracy: {best_result.test_accuracy:.2f}%")
        print(f"   📊 CV Accuracy: {best_result.mean_cv_accuracy:.2f}% ± {best_result.std_cv_accuracy:.2f}%")
        print(f"   ⚡ Training Time: {best_result.training_time:.1f}s")
        print(f"   🔢 Feature Dimension: {best_result.feature_dimension}")
        print(f"   📦 Model Type: {best_result.model_type}")
    
    def save_results(self, analysis: Dict[str, Any]):
        """Save results to JSON file."""
        results_file = os.path.join(self.output_dir, f'lbp_comparison_results_{self.timestamp}.json')
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append({
                'variant_name': result.variant_name,
                'variant_type': result.variant_type,
                'description': result.description,
                'feature_dimension': result.feature_dimension,
                'mean_cv_accuracy': result.mean_cv_accuracy,
                'std_cv_accuracy': result.std_cv_accuracy,
                'test_accuracy': result.test_accuracy,
                'training_time': result.training_time,
                'feature_extraction_time': result.feature_extraction_time,
                'model_type': result.model_type,
                'total_features': result.total_features,
                'train_size': result.train_size,
                'test_size': result.test_size,
                'num_classes': result.num_classes,
                'success': result.success,
                'error_message': result.error_message
            })
        
        # Create serializable analysis
        serializable_analysis = {
            'total_variants_tested': analysis['total_variants_tested'],
            'successful_tests': analysis['successful_tests'],
            'failed_tests': analysis['failed_tests'],
            'statistics': analysis['statistics'],
            'results': serializable_results,
            'accuracy_ranking': [r.variant_name for r in analysis['accuracy_ranking']],
            'cv_accuracy_ranking': [r.variant_name for r in analysis['cv_accuracy_ranking']],
            'speed_ranking': [r.variant_name for r in analysis['speed_ranking']],
            'failed_variants': [r.variant_name for r in analysis['failed_variants']]
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")


def main():
    """Main function to run LBP variants comparison."""
    print("🧪 LBP VARIANTS COMPARISON FOR ETHNIC DETECTION")
    print("=" * 70)
    
    # Configuration with dynamic timestamp for reproducibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_periorbital')
    output_dir = os.path.join(PROJECT_ROOT, 'logs', 'lbp_comparison', f'run_{timestamp}')
    
    print(f"📁 Results will be saved to: {output_dir}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Please ensure the dataset is available before running the comparison.")
        return False
    
    try:
        start_time = time.time()
        
        # Create comparator
        comparator = LBPVariantsComparator(dataset_path, output_dir)
        
        # Run comparison
        results = comparator.run_comparison()
        
        # Analyze results
        analysis = comparator.analyze_results()
        
        if analysis:
            # Print report
            comparator.print_analysis_report(analysis)
            
            # Save results
            comparator.save_results(analysis)
            
            # Calculate total runtime
            total_time = time.time() - start_time
            
            print("\n🎉 LBP VARIANTS COMPARISON COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"⏱️  Total runtime: {total_time/60:.1f} minutes")
            print(f"📁 Results saved to: {output_dir}")
            print(f"📊 Results file: lbp_comparison_results_{comparator.timestamp}.json")
            print("✅ All LBP variants have been tested and compared")
            print("✅ Results have been analyzed and ranked")
            print("✅ Detailed report has been generated")
            print("✅ Results have been saved to JSON file")
            
            return True
        else:
            print("\n❌ No successful tests to analyze!")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Comparison cancelled by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Testing cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
