#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark different feature sets (GLCM, LBP, HOG, HSV combos) with 6-fold CV
Ensures new texture features appear in top-15 RF importances when combined.
"""

import os
import sys
import numpy as np
import json
import time
from datetime import datetime

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_training.core.utils import TrainingLogger, ProgressTracker
from ml_training.core.training_pipeline import PipelineFactory
from ml_training.core.feature_extractors import (
    CombinedFeatureExtractor, GLCFeatureExtractor, ColorHistogramFeatureExtractor,
    LBPFeatureExtractor, HOGFeatureExtractor
)
# LBP variants removed - use dedicated LBP comparison test instead
# from ml_training.core.lbp_variants import (
#     KKPaperLBPExtractor, MBLBPExtractor, MBPExtractor, DLBPExtractor,
#     MQLBPExtractor, DLBPExtractor as DoubledLBPExtractor, RedDLBPExtractor
# )
from ml_training.core.preprocessors import PreprocessingPipeline, GLCMPreprocessor, ColorHistogramPreprocessor, ResizePreprocessor
from ml_training.core.data_loader import EthnicityDataLoader
from ml_training.core.model_trainers import ModelFactory


def build_features(logger, progress, images, feature_set_name):
    # Preprocess
    resize = ResizePreprocessor((256, 256), logger, progress)
    images_resized = resize.preprocess(images)

    glcm_imgs = GLCMPreprocessor(logger, progress).preprocess(images_resized)
    hsv_imgs = ColorHistogramPreprocessor(logger, progress).preprocess(images_resized)

    # Extract
    combo = CombinedFeatureExtractor(logger, progress)
    preprocessed = {'glcm': glcm_imgs, 'color': hsv_imgs}
    
    # Add extractors based on feature set name
    if 'GLCM' in feature_set_name:
        combo.add_extractor(GLCFeatureExtractor(logger, progress))
    if 'LBP' in feature_set_name and 'Paper' not in feature_set_name and 'MB' not in feature_set_name and 'DLBP' not in feature_set_name and 'MQLBP' not in feature_set_name and 'Doubled' not in feature_set_name and 'Red' not in feature_set_name:
        combo.add_extractor(LBPFeatureExtractor(logger, progress))
    # All LBP variants removed - use dedicated LBP comparison test instead
    # if 'PaperLBP' in feature_set_name:
    #     combo.add_extractor(KKPaperLBPExtractor(logger, progress))
    # if 'MBLBP' in feature_set_name:
    #     combo.add_extractor(MBLBPExtractor(logger, progress))
    # if 'MBP' in feature_set_name:
    #     combo.add_extractor(MBPExtractor(logger, progress))
    # if 'DLBP' in feature_set_name and 'Red' not in feature_set_name and 'Doubled' not in feature_set_name:
    #     combo.add_extractor(DLBPExtractor(logger, progress))
    # if 'MQLBP' in feature_set_name:
    #     combo.add_extractor(MQLBPExtractor(logger, progress))
    # if 'DoubledLBP' in feature_set_name:
    #     combo.add_extractor(DoubledLBPExtractor(logger, progress))
    # if 'RedDLBP' in feature_set_name:
    #     combo.add_extractor(RedDLBPExtractor(logger, progress))
    if 'HOG' in feature_set_name:
        combo.add_extractor(HOGFeatureExtractor(logger, progress))
    if 'HSV' in feature_set_name:
        combo.add_extractor(ColorHistogramFeatureExtractor(logger, progress))

    features = combo.extract_features(preprocessed)
    info = combo.get_combined_feature_info()
    return features, info


def evaluate_feature_set(name, images, labels, output_dir=None):
    # Create dynamic logger name with timestamp for reproducibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace('+', '_').replace('/', '_').replace('\\', '_')
    logger_name = f'feature_set_{safe_name}_{timestamp}'
    
    logger = TrainingLogger(logger_name)
    progress = ProgressTracker(logger)

    # Build features based on name
    feats, info = build_features(logger, progress, images, name)

    # Train with RF (using pipeline defaults)
    trainer = ModelFactory.create_trainer('random_forest', logger, progress)
    cv = trainer.cross_validate(feats, labels, cv_folds=6)
    model = trainer.train(feats, labels)

    # Collect top-15 features if available
    top15 = []
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1][:15]
        # Generate names per extractor blocks
        names = []
        offset = 0
        for ex in info.get('extractors', []):
            dim = int(ex.get('feature_dimension', 0))
            gname = ex.get('type', 'Feature')
            for i in range(dim):
                names.append(f"{gname}_f{i}")
            offset += dim
        if len(names) != len(importances):
            names = [f"f{i}" for i in range(len(importances))]
        top15 = [names[i] for i in order]

    # Create result with additional metadata
    result = {
        'name': name,
        'safe_name': safe_name,
        'timestamp': timestamp,
        'logger_name': logger_name,
        'cv_mean': cv.get('mean_accuracy', 0.0),
        'cv_std': cv.get('std_accuracy', 0.0),
        'top15': top15,
        'feature_info': info,
        'cv_results': cv,
        'model_type': 'RandomForest',
        'cv_folds': 6
    }

    # Save individual results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, f'{safe_name}_{timestamp}_results.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Saved individual results to {result_file}")

    return result


def main():
    print("FEATURE SET BENCHMARKS (GLCM/LBP/HOG/HSV)")
    print("=" * 60)
    
    # Create output directory with timestamp for reproducibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, 'logs', 'feature_sets_comparison', f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Results will be saved to: {output_dir}")
    
    logger = TrainingLogger('feature_sets_loader')
    loader = EthnicityDataLoader(logger)
    data_path = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_periorbital')
    images, labels, _ = loader.load_data(data_path)

    sets = [
        # Individual features
        'GLCM', 'LBP', 'HOG', 'HSV',
        # All LBP variants removed - use dedicated LBP comparison test instead
        # 'MBLBP', 'MBP', 'DLBP', 'MQLBP', 'DoubledLBP', 'RedDLBP',
        # Two-feature combinations
        'GLCM+LBP', 'LBP+HOG', 'GLCM+HOG',
        # Three-feature combinations
        'GLCM+LBP+HOG',
        # All features combined
        'GLCM+LBP+HOG+HSV'
    ]

    results = []
    start_time = time.time()
    
    for i, s in enumerate(sets, 1):
        print(f"\n📊 Progress: {i}/{len(sets)} - Evaluating: {s}")
        res = evaluate_feature_set(s, images, labels, output_dir)
        print(f"   CV: {res['cv_mean']:.2f}% (±{res['cv_std']:.2f}%)")
        if res['top15']:
            print("   Top-15 features include:")
            print("   - " + ", ".join(res['top15'][:10]) + (" ..." if len(res['top15']) > 10 else ""))
        results.append(res)

    # Check if new texture features appear in top-15 when combined
    combined = [r for r in results if r['name'] == 'GLCM+LBP+HOG+HSV']
    if combined:
        top15 = combined[0]['top15']
        has_texture = any(('GLCM_' in f or 'LBP_' in f or 'HOG_' in f) for f in top15)
        print("\nTexture features present in top-15:", has_texture)

    # Calculate total time
    total_time = time.time() - start_time
    
    # Create comprehensive summary
    summary = {
        'run_timestamp': timestamp,
        'total_runtime_seconds': total_time,
        'total_runtime_minutes': total_time / 60,
        'dataset_path': data_path,
        'total_feature_sets': len(sets),
        'feature_sets_tested': [r['name'] for r in results],
        'results': results,
        'analysis': {
            'best_accuracy': max(results, key=lambda x: x['cv_mean']),
            'worst_accuracy': min(results, key=lambda x: x['cv_mean']),
            'accuracy_range': {
                'min': min(r['cv_mean'] for r in results),
                'max': max(r['cv_mean'] for r in results),
                'mean': np.mean([r['cv_mean'] for r in results]),
                'std': np.std([r['cv_mean'] for r in results])
            },
            'texture_features_in_top15': has_texture if combined else None
        }
    }
    
    # Save comprehensive results
    summary_file = os.path.join(output_dir, f'feature_sets_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save ranked results
    ranked_results = sorted(results, key=lambda x: x['cv_mean'], reverse=True)
    ranked_file = os.path.join(output_dir, f'feature_sets_ranked_{timestamp}.json')
    with open(ranked_file, 'w') as f:
        json.dump(ranked_results, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n🏆 FEATURE SETS COMPARISON COMPLETED!")
    print("=" * 60)
    print(f"⏱️  Total runtime: {total_time/60:.1f} minutes")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Summary file: {summary_file}")
    print(f"🏅 Ranked results: {ranked_file}")
    
    print(f"\n🥇 TOP 5 FEATURE SETS BY ACCURACY:")
    for i, result in enumerate(ranked_results[:5], 1):
        print(f"   {i}. {result['name']:20s}: {result['cv_mean']:6.2f}% ± {result['cv_std']:5.2f}%")
    
    print(f"\n📈 ACCURACY STATISTICS:")
    print(f"   Range: {summary['analysis']['accuracy_range']['min']:.2f}% - {summary['analysis']['accuracy_range']['max']:.2f}%")
    print(f"   Mean:  {summary['analysis']['accuracy_range']['mean']:.2f}% ± {summary['analysis']['accuracy_range']['std']:.2f}%")

    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)


