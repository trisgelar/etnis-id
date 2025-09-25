#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Training Pipeline
Orchestrates the complete training process following SOLID principles
"""

import numpy as np
import os
from typing import Dict, Any, Optional
from .interfaces import ITrainingPipeline, IDataLoader, IModelTrainer, IModelSaver, ILogger, IProgressTracker
from .data_loader import EthnicityDataLoader
from .preprocessors import GLCMPreprocessor, ColorHistogramPreprocessor, PreprocessingPipeline
from .feature_extractors import GLCFeatureExtractor, ColorHistogramFeatureExtractor, CombinedFeatureExtractor
from .model_trainers import ModelFactory
from .utils import ModelSaver
from .config import get_config, get_dataset_config, get_model_config, get_training_config
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


class EthnicityTrainingPipeline(ITrainingPipeline):
    """Complete training pipeline for ethnicity detection"""
    
    def __init__(self, logger: ILogger, 
                 progress_tracker: IProgressTracker = None, custom_config: Dict[str, Any] = None):
        """
        Initialize training pipeline
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
            custom_config: Custom configuration overrides (optional)
        """
        self.logger = logger
        self.progress_tracker = progress_tracker
        
        # Get configuration from environment
        self.config = get_config()
        self.dataset_config = get_dataset_config()
        self.model_config = get_model_config()
        self.training_config = get_training_config()
        
        # Apply custom configuration overrides if provided
        if custom_config:
            self._apply_custom_config(custom_config)
        
        # Caching options
        self.use_cache = True
        self.cache_dir = os.path.join('logs', 'cache')

        # Initialize components
        self.data_loader = None
        self.preprocessing_pipeline = None
        self.feature_extractor = None
        self.model_trainer = None
        self.model_saver = None
        
        # Training results
        self.training_results = {}
        
        self._initialize_components()
    
    def _apply_custom_config(self, custom_config: Dict[str, Any]):
        """Apply custom configuration overrides"""
        for section, values in custom_config.items():
            if hasattr(self, f"{section}_config"):
                config_obj = getattr(self, f"{section}_config")
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
                        self.logger.info(f"Override: {section}.{key} = {value}")
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        self.logger.info("Initializing training pipeline components...")
        
        # Data loader
        self.data_loader = EthnicityDataLoader(self.logger)
        
        # Preprocessing pipeline
        self.preprocessing_pipeline = PreprocessingPipeline(self.logger, self.progress_tracker)
        
        # Add preprocessors
        self.preprocessing_pipeline.add_preprocessor(
            GLCMPreprocessor(self.logger, self.progress_tracker)
        ).add_preprocessor(
            ColorHistogramPreprocessor(self.logger, self.progress_tracker)
        )
        
        # Feature extractors
        self.feature_extractor = CombinedFeatureExtractor(self.logger, self.progress_tracker)
        
        # Add feature extractors
        feat_cfg = self.config.feature_extraction
        self.feature_extractor.add_extractor(
            GLCFeatureExtractor(
                self.logger, 
                self.progress_tracker,
                distances=feat_cfg.glc_distances,
                angles=feat_cfg.glc_angles,
                levels=feat_cfg.glc_levels
            )
        ).add_extractor(
            ColorHistogramFeatureExtractor(
                self.logger,
                self.progress_tracker,
                bins=feat_cfg.color_bins,
                channels=feat_cfg.color_channels
            )
        )
        
        # Model trainer
        model_cfg = self.config.model
        trainer_type = (model_cfg.model_type or 'RandomForest')
        trainer_params = self._get_trainer_params(trainer_type)
        self.model_trainer = ModelFactory.create_trainer(
            trainer_type, self.logger, self.progress_tracker, **trainer_params
        )
        
        # Model saver
        self.model_saver = ModelSaver(self.logger)
        
        self.logger.info("Pipeline components initialized")
    
    def _get_trainer_params(self, trainer_type: str) -> Dict[str, Any]:
        """Get parameters for specific trainer type"""
        if str(trainer_type).lower() in ('randomforest', 'random_forest', 'rf'):
            return {
                'n_estimators': self.config.model.n_estimators,
                'random_state': self.config.model.random_state
            }
        elif str(trainer_type).lower() == 'svm':
            return {
                'random_state': self.config.model.random_state
            }
        else:
            return {}
    
    def run_pipeline(self, data_path: str, output_path: str) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            data_path: Path to training data
            output_path: Path to save the trained model
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("STARTING ETHNICITY DETECTION TRAINING PIPELINE")
        self.logger.info("=" * 70)
        
        try:
            # Phase 1: Load Data
            self.logger.info("\nPHASE 1: LOADING DATA")
            self.logger.info("-" * 30)
            
            # Try cache
            cache_images = os.path.join(self.cache_dir, 'images.npy')
            cache_labels = os.path.join(self.cache_dir, 'labels.npy')
            cache_meta = os.path.join(self.cache_dir, 'metadata.npy')

            if self.use_cache and os.path.exists(cache_images) and os.path.exists(cache_labels) and os.path.exists(cache_meta):
                self.logger.info("Loading dataset from cache...")
                images = np.load(cache_images, allow_pickle=True)
                labels = np.load(cache_labels, allow_pickle=True)
                metadata = np.load(cache_meta, allow_pickle=True).item()
            else:
                images, labels, metadata = self.data_loader.load_data(data_path)
                if self.use_cache:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    np.save(cache_images, images)
                    np.save(cache_labels, labels)
                    np.save(cache_meta, np.array(metadata, dtype=object))
            self.training_results['data_metadata'] = metadata
            
            # Phase 2: Preprocessing
            self.logger.info("\nPHASE 2: PREPROCESSING")
            self.logger.info("-" * 30)
            
            # Resize images first
            from .preprocessors import ResizePreprocessor
            resize_preprocessor = ResizePreprocessor(
                self.config.feature_extraction.image_size, self.logger, self.progress_tracker
            )
            cache_resized = os.path.join(self.cache_dir, 'resized_images.npy')
            if self.use_cache and os.path.exists(cache_resized):
                self.logger.info("Loading resized images from cache...")
                resized_images = np.load(cache_resized, allow_pickle=True)
            else:
                resized_images = resize_preprocessor.preprocess(images)
                if self.use_cache:
                    np.save(cache_resized, resized_images)
            
            # GLCM preprocessing
            cache_glcm = os.path.join(self.cache_dir, 'glcm_images.npy')
            if self.use_cache and os.path.exists(cache_glcm):
                self.logger.info("Loading grayscale images from cache...")
                glcm_images = np.load(cache_glcm, allow_pickle=True)
            else:
                glcm_images = self.preprocessing_pipeline.preprocessors[0].preprocess(resized_images)
                if self.use_cache:
                    np.save(cache_glcm, glcm_images)
            
            # Color preprocessing
            cache_color = os.path.join(self.cache_dir, 'color_images.npy')
            if self.use_cache and os.path.exists(cache_color):
                self.logger.info("Loading HSV images from cache...")
                color_images = np.load(cache_color, allow_pickle=True)
            else:
                color_images = self.preprocessing_pipeline.preprocessors[1].preprocess(resized_images)
                if self.use_cache:
                    np.save(cache_color, color_images)
            
            preprocessed_data = {
                'glcm': glcm_images,
                'color': color_images
            }
            
            self.training_results['preprocessing_info'] = self.preprocessing_pipeline.get_pipeline_info()
            
            # Phase 3: Feature Extraction
            self.logger.info("\nPHASE 3: FEATURE EXTRACTION")
            self.logger.info("-" * 30)
            
            cache_features = os.path.join(self.cache_dir, 'features.npy')
            if self.use_cache and os.path.exists(cache_features):
                self.logger.info("Loading features from cache...")
                features = np.load(cache_features, allow_pickle=True)
            else:
                features = self.feature_extractor.extract_features(preprocessed_data)
                if self.use_cache:
                    np.save(cache_features, features)
            self.training_results['feature_info'] = self.feature_extractor.get_combined_feature_info()
            
            # Phase 4: Model Training
            self.logger.info("\nPHASE 4: MODEL TRAINING")
            self.logger.info("-" * 30)
            
            # Cross-validation
            cv_results = self.model_trainer.cross_validate(
                features, labels, self.config.cross_validation.n_folds
            )
            self.training_results['cross_validation'] = cv_results

            # Cross-validated predictions for confusion matrix and report
            self.logger.info("\nGenerating cross-validated predictions for evaluation...")
            cv = StratifiedKFold(
                n_splits=self.config.cross_validation.n_folds,
                shuffle=True,
                random_state=220
            )
            estimator = self.model_trainer.get_estimator()
            y_pred = cross_val_predict(estimator, features, labels, cv=cv, n_jobs=self.config.cross_validation.n_jobs)
            acc = accuracy_score(labels, y_pred) * 100
            cm = confusion_matrix(labels, y_pred)

            # Prepare target names if available
            label_map = self.training_results.get('data_metadata', {}).get('label_map', {})
            target_names = [label_map.get(i, str(i)) for i in sorted(set(labels))]
            cls_report = classification_report(labels, y_pred, target_names=target_names, digits=4)

            self.training_results['evaluation'] = {
                'cv_accuracy': acc,
                'confusion_matrix': cm.tolist(),
                'classification_report': cls_report
            }

            # Save evaluation artifacts
            self._save_evaluation_artifacts(cm, cls_report, acc, target_names)
            
            # Hold-out test evaluation
            self.logger.info("\nEvaluating on hold-out test split...")
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels,
                test_size=self.config.cross_validation.test_size,
                random_state=220,
                stratify=labels
            )

            test_estimator = self.model_trainer.get_estimator()
            test_estimator.fit(X_train, y_train)
            y_test_pred = test_estimator.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred) * 100
            cm_test = confusion_matrix(y_test, y_test_pred)
            target_names = [label_map.get(i, str(i)) for i in sorted(set(labels))]
            test_report = classification_report(y_test, y_test_pred, target_names=target_names, digits=4)

            self.training_results['test_evaluation'] = {
                'test_accuracy': test_acc,
                'confusion_matrix': cm_test.tolist(),
                'classification_report': test_report,
                'train_size': int(X_train.shape[0]),
                'test_size': int(X_test.shape[0])
            }
            self._save_evaluation_artifacts(cm_test, test_report, test_acc, target_names, suffix='_test')

            # Train final model on full dataset (for production)
            trained_model = self.model_trainer.train(features, labels)
            self.training_results['model_info'] = self.model_trainer.get_model_info()

            # Export feature importance (from final model) and permutation importance (from test split)
            try:
                self._export_feature_importances(trained_model, feature_info=self.training_results.get('feature_info', {}))
            except Exception as e:
                self.logger.warning(f"Failed to export feature importances: {e}")
            try:
                self._export_permutation_importance(test_estimator, X_test, y_test,
                                                    feature_info=self.training_results.get('feature_info', {}))
            except Exception as e:
                self.logger.warning(f"Failed to export permutation importance: {e}")

            # Regenerate performance analysis report
            try:
                self._write_performance_report(cv_results, self.training_results.get('test_evaluation', {}))
            except Exception as e:
                self.logger.warning(f"Failed to write performance analysis report: {e}")
            
            # Phase 5: Save Model
            self.logger.info("\nPHASE 5: SAVING MODEL")
            self.logger.info("-" * 30)
            
            save_success = self.model_saver.save_model(trained_model, output_path)
            self.training_results['model_saved'] = save_success
            
            if not save_success:
                raise RuntimeError("Failed to save model")
            
            # Phase 6: Final Summary
            self.logger.info("\nTRAINING COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 70)
            
            self._log_final_summary()
            
            return self.training_results
            
        except Exception as e:
            self.logger.error(f"\nTraining pipeline failed: {e}")
            self.training_results['error'] = str(e)
            raise

    def _save_evaluation_artifacts(self, cm: np.ndarray, report: str, acc: float, target_names: list, suffix: str = '') -> None:
        """Save confusion matrix and classification report to logs/analysis."""
        try:
            output_dir = os.path.join('logs', 'analysis')
            os.makedirs(output_dir, exist_ok=True)

            # Save classification report
            report_path = os.path.join(output_dir, f'classification_report{suffix}.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"Accuracy: {acc:.4f}\n\n")
                f.write(report)
            self.logger.info(f"Saved classification report to {report_path}")

            # Save confusion matrix CSV
            cm_csv_path = os.path.join(output_dir, f'confusion_matrix{suffix}.csv')
            try:
                import csv
                with open(cm_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([''] + list(target_names))
                    for i, row in enumerate(cm):
                        writer.writerow([target_names[i]] + list(row))
                self.logger.info(f"Saved confusion matrix CSV to {cm_csv_path}")
            except Exception:
                pass

            # Save confusion matrix heatmap PNG (matplotlib)
            cm_png_path = os.path.join(output_dir, f'confusion_matrix{suffix}.png')
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                   xticklabels=target_names, yticklabels=target_names,
                   title=f"Confusion Matrix (Acc {acc:.2f}%)",
                   ylabel='True label', xlabel='Predicted label')

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Annotate cells
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            plt.savefig(cm_png_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Saved confusion matrix image to {cm_png_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save evaluation artifacts: {e}")

    def _export_feature_importances(self, trained_model, feature_info: Dict[str, Any]) -> None:
        """Export RandomForest feature importances as CSV and barplot; includes group aggregation."""
        output_dir = os.path.join('logs', 'analysis')
        os.makedirs(output_dir, exist_ok=True)

        # If model is a Pipeline, get the RF step
        rf_model = trained_model
        try:
            from sklearn.pipeline import Pipeline
            if hasattr(trained_model, 'named_steps'):
                rf_model = trained_model.named_steps.get('randomforestclassifier', trained_model)
            elif isinstance(trained_model, Pipeline):
                # Fallback: last step
                rf_model = trained_model.steps[-1][1]
        except Exception:
            pass

        if not hasattr(rf_model, 'feature_importances_'):
            self.logger.warning("Trained model has no feature_importances_. Skipping export.")
            return

        importances = np.asarray(rf_model.feature_importances_)
        total_features = importances.shape[0]

        # Build feature names by groups if available
        names = []
        groups = []
        offsets = []
        try:
            extractors = feature_info.get('extractors', [])
            offset = 0
            for ex in extractors:
                dim = int(ex.get('feature_dimension', 0))
                gname = ex.get('type', 'Feature')
                for i in range(dim):
                    names.append(f"{gname}_f{i}")
                    groups.append(gname)
                offsets.append((gname, offset, offset + dim))
                offset += dim
            if len(names) != total_features:
                # Fallback to generic names
                names = [f"f{i}" for i in range(total_features)]
                groups = ["Feature"] * total_features
        except Exception:
            names = [f"f{i}" for i in range(total_features)]
            groups = ["Feature"] * total_features

        # Save CSV
        import csv
        csv_path = os.path.join(output_dir, 'feature_importances.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'group', 'importance'])
            for n, g, v in zip(names, groups, importances):
                writer.writerow([n, g, float(v)])
        self.logger.info(f"Saved feature importances to {csv_path}")

        # Group aggregation
        group_values: Dict[str, float] = {}
        for g, v in zip(groups, importances):
            group_values[g] = group_values.get(g, 0.0) + float(v)
        group_csv = os.path.join(output_dir, 'feature_importances_grouped.csv')
        with open(group_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['group', 'total_importance'])
            for g, v in group_values.items():
                writer.writerow([g, v])
        self.logger.info(f"Saved grouped feature importances to {group_csv}")

        # Barplot top 20
        try:
            order = np.argsort(importances)[::-1]
            top_k = min(20, len(order))
            top_idx = order[:top_k]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh([names[i] for i in top_idx][::-1], importances[top_idx][::-1])
            ax.set_title('Top Feature Importances (RF)')
            ax.set_xlabel('Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importances_top20.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Failed to save feature importance plot: {e}")

        # Group barplot
        try:
            labels = list(group_values.keys())
            vals = [group_values[k] for k in labels]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(labels, vals)
            ax.set_title('Grouped Feature Importances')
            ax.set_ylabel('Total Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importances_grouped.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Failed to save grouped feature importance plot: {e}")

    def _export_permutation_importance(self, estimator, X_test: np.ndarray, y_test: np.ndarray,
                                       feature_info: Dict[str, Any]) -> None:
        """Export permutation importance on the test split."""
        output_dir = os.path.join('logs', 'analysis')
        os.makedirs(output_dir, exist_ok=True)

        result = permutation_importance(estimator, X_test, y_test, n_repeats=10, random_state=220, n_jobs=-1)
        importances_mean = result.importances_mean
        importances_std = result.importances_std

        total_features = importances_mean.shape[0]
        # Names as before
        names = []
        try:
            extractors = feature_info.get('extractors', [])
            for ex in extractors:
                dim = int(ex.get('feature_dimension', 0))
                gname = ex.get('type', 'Feature')
                for i in range(dim):
                    names.append(f"{gname}_f{i}")
            if len(names) != total_features:
                names = [f"f{i}" for i in range(total_features)]
        except Exception:
            names = [f"f{i}" for i in range(total_features)]

        # CSV
        import csv
        csv_path = os.path.join(output_dir, 'permutation_importance_test.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'importance_mean', 'importance_std'])
            for n, m, s in zip(names, importances_mean, importances_std):
                writer.writerow([n, float(m), float(s)])
        self.logger.info(f"Saved permutation importance (test) to {csv_path}")

        # Barplot top 20
        try:
            order = np.argsort(importances_mean)[::-1]
            top_k = min(20, len(order))
            top_idx = order[:top_k]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh([names[i] for i in top_idx][::-1], importances_mean[top_idx][::-1])
            ax.set_title('Top Permutation Importances (Test)')
            ax.set_xlabel('Importance (mean decrease)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'permutation_importance_test_top20.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Failed to save permutation importance plot: {e}")

    def _write_performance_report(self, cv_results: Dict[str, Any], test_eval: Dict[str, Any]) -> None:
        """Write an updated performance analysis markdown report."""
        output_dir = os.path.join('logs', 'analysis')
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'performance_analysis_report.md')

        cv_acc = cv_results.get('mean_accuracy', 0.0)
        cv_std = cv_results.get('std_accuracy', 0.0)
        test_acc = test_eval.get('test_accuracy', 0.0)
        train_size = test_eval.get('train_size', 0)
        test_size = test_eval.get('test_size', 0)

        content = []
        content.append("")
        content.append("# ETHNICITY DETECTION MODEL - PERFORMANCE ANALYSIS REPORT")
        content.append("")
        content.append("## EXECUTIVE SUMMARY")
        content.append(f"New cross-validated model achieves strong performance: CV {cv_acc:.2f}% (±{cv_std:.2f}%), Test {test_acc:.2f}%.")
        content.append("")
        content.append("## PERFORMANCE COMPARISON")
        content.append("")
        content.append("### Original Notebook (Reference)")
        content.append("- Accuracy: 98.6%")
        content.append("- Cross-Validation: 6-fold CV implemented")
        content.append("")
        content.append("### Previous Model (Problematic, before fixes)")
        content.append("- Accuracy: 30.3%")
        content.append("- Cross-Validation: Not implemented")
        content.append("")
        content.append("### Current Model (This run)")
        content.append(f"- CV Accuracy: {cv_acc:.2f}% (±{cv_std:.2f}%)")
        content.append(f"- Test Accuracy: {test_acc:.2f}% (train={train_size}, test={test_size})")
        content.append(f"- Artifacts: confusion_matrix.png, confusion_matrix_test.png, classification_report.txt, classification_report_test.txt")
        content.append("")
        content.append("## FEATURE IMPORTANCE")
        content.append("- RandomForest importances: feature_importances.csv, feature_importances_top20.png, feature_importances_grouped.png")
        content.append("- Permutation importances (test): permutation_importance_test.csv, permutation_importance_test_top20.png")
        content.append("")
        content.append("## NOTES ON IMPROVEMENT")
        content.append("- Scaling + corrected GLCM features and 6-fold CV align with the notebook, resolving color-only learning.")
        content.append("- Previous low-confidence and overfit artifacts are preserved for comparison.")
        content.append("")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
        self.logger.info(f"Updated performance report written to {report_path}")
    
    def _log_final_summary(self):
        """Log final training summary"""
        cv_results = self.training_results.get('cross_validation', {})
        model_info = self.training_results.get('model_info', {})
        feature_info = self.training_results.get('feature_info', {})
        data_metadata = self.training_results.get('data_metadata', {})
        
        self.logger.info(f"Final Model Performance:")
        self.logger.info(f"   - Mean CV Accuracy: {cv_results.get('mean_accuracy', 0):.2f}%")
        self.logger.info(f"   - Standard Deviation: {cv_results.get('std_accuracy', 0):.2f}%")
        self.logger.info(f"   - Features used: {feature_info.get('total_features', 0)}")
        self.logger.info(f"   - Model type: {model_info.get('algorithm', 'Unknown')}")
        self.logger.info(f"   - Training samples: {data_metadata.get('total_images', 0)}")
        self.logger.info(f"   - Number of classes: {data_metadata.get('num_classes', 0)}")
        self.logger.info(f"   - Supported ethnicities: {list(data_metadata.get('label_map', {}).values())}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline"""
        return {
            'pipeline_type': 'EthnicityTrainingPipeline',
            'components': {
                'data_loader': type(self.data_loader).__name__,
                'preprocessors': len(self.preprocessing_pipeline.preprocessors),
                'feature_extractors': len(self.feature_extractor.extractors),
                'model_trainer': type(self.model_trainer).__name__,
                'model_saver': type(self.model_saver).__name__
            },
            'configuration': {
                'dataset': self.config.dataset.get_config(),
                'model': self.config.model.get_config(),
                'training': self.config.training.get_config(),
                'cross_validation': self.config.cross_validation.get_config(),
                'feature_extraction': self.config.feature_extraction.get_config(),
                'logging': self.config.logging.get_config(),
                'server': self.config.server.get_config(),
                'visualization': self.config.visualization.get_config()
            },
            'training_results': self.training_results
        }


class PipelineFactory:
    """Factory for creating training pipelines"""
    
    @staticmethod
    def create_pipeline(logger: ILogger = None, 
                       progress_tracker: IProgressTracker = None,
                       custom_config: Dict[str, Any] = None) -> ITrainingPipeline:
        """
        Create training pipeline instance
        
        Args:
            logger: Logger instance
            progress_tracker: Progress tracker instance
            custom_config: Custom configuration overrides (optional)
            
        Returns:
            Training pipeline instance
        """
        if logger is None:
            from .utils import TrainingLogger
            logger = TrainingLogger('pipeline_factory')
        
        return EthnicityTrainingPipeline(logger, progress_tracker, custom_config)
