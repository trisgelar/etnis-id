# Configuration System Guide

## Overview

The Ethnicity Detection System now uses `python-dotenv` for comprehensive configuration management. This allows you to configure all aspects of the system through environment variables, making it easy to deploy and maintain across different environments.

## Quick Start

### 1. Create Configuration File

Copy the template file to create your configuration:

```bash
cp env.template .env
```

### 2. Customize Configuration

Edit the `.env` file with your specific values:

```bash
# Example: Change model parameters
MODEL_N_ESTIMATORS=300
MODEL_MAX_DEPTH=20

# Example: Change cross-validation settings
CV_N_FOLDS=10
CV_TEST_SIZE=0.15

# Example: Change feature extraction
GLCM_LEVELS=512
COLOR_BINS=32
```

### 3. Use in Your Code

```python
from ml_training.core.config import get_config, get_model_config

# Get full configuration
config = get_config()
print(f"Model type: {config.model.model_type}")

# Get specific configuration
model_config = get_model_config()
print(f"Number of estimators: {model_config.n_estimators}")
```

## Configuration Sections

### Dataset Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_DIR` | `../dataset/dataset_periorbital` | Main dataset directory |
| `ETHNICITIES` | `Banjar,Bugis,Javanese,Malay,Sundanese` | List of ethnicities |
| `MAX_IMAGES_PER_CLASS` | `1000` | Maximum images per class |
| `TRAIN_SPLIT` | `0.8` | Training set ratio |
| `VAL_SPLIT` | `0.1` | Validation set ratio |
| `TEST_SPLIT` | `0.1` | Test set ratio |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TYPE` | `RandomForest` | Type of model to use |
| `MODEL_N_ESTIMATORS` | `200` | Number of estimators |
| `MODEL_MAX_DEPTH` | (empty) | Maximum tree depth |
| `MODEL_MIN_SAMPLES_SPLIT` | `2` | Minimum samples to split |
| `MODEL_MIN_SAMPLES_LEAF` | `1` | Minimum samples per leaf |
| `MODEL_CLASS_WEIGHT` | (empty) | Class weight strategy |

### Cross-Validation Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CV_N_FOLDS` | `6` | Number of CV folds |
| `CV_TEST_SIZE` | `0.2` | Test set size ratio |
| `CV_SCORING` | `accuracy` | Scoring metric |
| `CV_SHUFFLE` | `true` | Shuffle data before CV |
| `CV_STRATIFY` | `true` | Stratify folds by class |

### Feature Extraction Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GLCM_DISTANCES` | `1` | GLCM distances |
| `GLCM_ANGLES` | `0,45,90,135` | GLCM angles (degrees) |
| `GLCM_LEVELS` | `256` | Number of gray levels |
| `COLOR_BINS` | `16` | Color histogram bins |
| `COLOR_CHANNELS` | `1,2` | Color channels to use |
| `COLOR_SPACE` | `HSV` | Color space |

### Logging Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FILE` | `logs/training.log` | Log file path |
| `LOG_CONSOLE_OUTPUT` | `true` | Enable console output |
| `LOG_FILE_OUTPUT` | `true` | Enable file output |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `localhost` | Server host |
| `SERVER_PORT` | `8080` | Server port |
| `SERVER_DEBUG` | `false` | Debug mode |
| `SERVER_MAX_CONNECTIONS` | `100` | Max connections |

### Visualization Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VIZ_OUTPUT_DIR` | `logs/analysis` | Output directory |
| `VIZ_FIGURE_DPI` | `300` | Figure DPI |
| `VIZ_SCIENCE_PLOTS` | `true` | Use SciencePlots |
| `VIZ_SHOW_PLOTS` | `false` | Show plots |

## Advanced Usage

### Environment-Specific Configuration

Create different configuration files for different environments:

```bash
# Development
cp env.template .env.development

# Production
cp env.template .env.production

# Testing
cp env.template .env.testing
```

Load specific configuration:

```python
from ml_training.core.config import Config

# Load development configuration
dev_config = Config('.env.development')

# Load production configuration
prod_config = Config('.env.production')
```

### Runtime Configuration Override

You can override configuration at runtime:

```python
from ml_training.core.config import get_model_config

# Get default configuration
model_config = get_model_config()

# Override specific parameters
model_config.n_estimators = 500
model_config.max_depth = 15
```

### Configuration Validation

The system automatically validates configuration values:

```python
from ml_training.core.config import get_config

config = get_config()

# All configurations are validated
assert config.model.n_estimators > 0
assert config.cross_validation.n_folds > 0
assert len(config.dataset.ethnicities) > 0
```

## Integration Examples

### Using Configuration in Feature Extractors

```python
from ml_training.core.feature_extractors import GLCFeatureExtractor
from ml_training.core.utils import TrainingLogger

# Logger automatically uses configuration
logger = TrainingLogger('feature_extraction')

# Feature extractors automatically use configuration
glcm_extractor = GLCFeatureExtractor(logger)
# Uses GLCM_DISTANCES, GLCM_ANGLES, GLCM_LEVELS from config

# Override specific parameters
custom_glcm = GLCFeatureExtractor(logger, distances=[2, 3], levels=512)
```

### Using Configuration in Cross-Validation

```python
from ml_training.core.cross_validation import CrossValidationConfig, ModelConfig

# Automatically uses configuration
cv_config = CrossValidationConfig()
model_config = ModelConfig()

# Uses CV_N_FOLDS, CV_TEST_SIZE, MODEL_N_ESTIMATORS, etc.
```

### Using Configuration in Training

```python
from ml_training.core.config import get_model_config
from sklearn.ensemble import RandomForestClassifier

# Get model configuration
model_config = get_model_config()

# Create model with configuration
rf = RandomForestClassifier(
    n_estimators=model_config.n_estimators,
    max_depth=model_config.max_depth,
    min_samples_split=model_config.min_samples_split,
    min_samples_leaf=model_config.min_samples_leaf,
    random_state=model_config.random_state,
    n_jobs=model_config.n_jobs
)
```

## Best Practices

### 1. Use Environment Variables for Secrets

Never commit sensitive information to version control:

```bash
# .env (not committed)
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# .env.template (committed)
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here
```

### 2. Document Configuration Changes

Always document changes to configuration:

```bash
# Configuration change log
# 2024-01-15: Increased MODEL_N_ESTIMATORS from 200 to 300 for better performance
# 2024-01-20: Changed CV_N_FOLDS from 6 to 10 for more robust validation
```

### 3. Use Configuration for A/B Testing

Test different configurations:

```bash
# Configuration A
MODEL_N_ESTIMATORS=200
MODEL_MAX_DEPTH=10

# Configuration B
MODEL_N_ESTIMATORS=300
MODEL_MAX_DEPTH=20
```

### 4. Validate Configuration in CI/CD

Add configuration validation to your CI/CD pipeline:

```python
# test_config.py
def test_production_config():
    config = Config('.env.production')
    
    # Validate critical settings
    assert config.model.n_estimators >= 100
    assert config.cross_validation.n_folds >= 5
    assert config.logging.log_level in ['INFO', 'WARNING', 'ERROR']
```

## Troubleshooting

### Common Issues

1. **Configuration not loaded**: Ensure `.env` file exists and is in the correct location
2. **Invalid values**: Check that numeric values are valid (e.g., positive integers for n_estimators)
3. **Missing variables**: Use the template file to ensure all required variables are present

### Debug Configuration

```python
from ml_training.core.config import get_config

# Print all configuration
config = get_config()
config.print_config()

# Check specific configuration
print(f"Model estimators: {config.model.n_estimators}")
print(f"CV folds: {config.cross_validation.n_folds}")
```

## Migration from Hardcoded Values

If you have existing code with hardcoded values, here's how to migrate:

### Before (Hardcoded)
```python
# Old way
rf = RandomForestClassifier(n_estimators=200, max_depth=10)
cv = StratifiedKFold(n_splits=6)
```

### After (Configuration)
```python
# New way
from ml_training.core.config import get_model_config, get_cv_config

model_config = get_model_config()
cv_config = get_cv_config()

rf = RandomForestClassifier(
    n_estimators=model_config.n_estimators,
    max_depth=model_config.max_depth
)
cv = StratifiedKFold(n_splits=cv_config.n_folds)
```

## Conclusion

The configuration system provides:

- ✅ **Flexibility**: Easy to change parameters without code changes
- ✅ **Environment Management**: Different configs for dev/staging/prod
- ✅ **Maintainability**: Centralized configuration management
- ✅ **Validation**: Automatic validation of configuration values
- ✅ **Documentation**: Self-documenting configuration options

This makes the system more professional, maintainable, and ready for production deployment.
