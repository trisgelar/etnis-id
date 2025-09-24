# 📚 Best Practices Guide - Python ML Development

## 📋 Overview

This guide provides best practices for developing maintainable, professional Python machine learning systems, demonstrated through the ethnicity detection training system refactoring.

## 🎯 Table of Contents

1. [Code Organization](#code-organization)
2. [Error Handling](#error-handling)
3. [Logging Best Practices](#logging-best-practices)
4. [Testing Strategies](#testing-strategies)
5. [Configuration Management](#configuration-management)
6. [Documentation Standards](#documentation-standards)
7. [Performance Optimization](#performance-optimization)
8. [Security Considerations](#security-considerations)

---

## 📁 Code Organization

### **1. Project Structure**

```
project/
├── src/                    # Source code
│   ├── core/              # Core business logic
│   ├── data/              # Data handling
│   ├── models/            # ML models
│   └── utils/             # Utilities
├── tests/                 # Test code
├── docs/                  # Documentation
├── config/                # Configuration files
├── scripts/               # Deployment scripts
├── requirements.txt       # Dependencies
├── setup.py              # Package setup
└── README.md             # Project documentation
```

### **2. Module Organization**

```python
# Good: Clear module structure
from src.core.interfaces import IDataLoader, IModelTrainer
from src.core.data_loader import EthnicityDataLoader
from src.core.model_trainers import RandomForestTrainer
from src.utils.logging import TrainingLogger

# Bad: Unclear imports
from core import *
from utils import logger, config, data_loader
```

### **3. Class Organization**

```python
# Good: Logical method ordering
class DataProcessor:
    # Class variables
    DEFAULT_CONFIG = {}
    
    def __init__(self, config: dict):
        # Constructor
        pass
    
    # Public methods
    def process(self, data):
        pass
    
    # Private methods
    def _validate_data(self, data):
        pass
    
    # Properties
    @property
    def is_configured(self):
        pass
```

---

## 🚨 Error Handling

### **1. Exception Hierarchy**

```python
# Define custom exceptions
class TrainingError(Exception):
    """Base exception for training errors"""
    pass

class DataLoadError(TrainingError):
    """Data loading specific errors"""
    pass

class ModelTrainingError(TrainingError):
    """Model training specific errors"""
    pass

class ValidationError(TrainingError):
    """Validation specific errors"""
    pass
```

### **2. Error Handling Patterns**

```python
# Good: Specific exception handling
def load_data(self, data_path: str):
    try:
        if not os.path.exists(data_path):
            raise DataLoadError(f"Dataset directory '{data_path}' not found")
        
        # Load data logic
        return data
        
    except FileNotFoundError as e:
        self.logger.error(f"File not found: {e}")
        raise DataLoadError(f"Required file missing: {e}") from e
    except PermissionError as e:
        self.logger.error(f"Permission denied: {e}")
        raise DataLoadError(f"Insufficient permissions: {e}") from e
    except Exception as e:
        self.logger.error(f"Unexpected error: {e}")
        raise DataLoadError(f"Data loading failed: {e}") from e

# Bad: Generic exception handling
def load_data(self, data_path: str):
    try:
        # Load data logic
        return data
    except:
        print("Error occurred")
        return None
```

### **3. Validation Patterns**

```python
# Good: Comprehensive validation
def validate_data(self, data: np.ndarray, labels: np.ndarray) -> bool:
    """Validate loaded data with detailed error messages"""
    
    if data is None:
        raise ValidationError("Data cannot be None")
    
    if labels is None:
        raise ValidationError("Labels cannot be None")
    
    if len(data) != len(labels):
        raise ValidationError(f"Data length ({len(data)}) != labels length ({len(labels)})")
    
    if len(data) == 0:
        raise ValidationError("No data loaded")
    
    if not isinstance(data, np.ndarray):
        raise ValidationError(f"Data must be numpy array, got {type(data)}")
    
    if data.dtype != np.uint8:
        raise ValidationError(f"Data must be uint8, got {data.dtype}")
    
    return True
```

---

## 📊 Logging Best Practices

### **1. Structured Logging**

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Structured logging with JSON format"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info with structured data"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'INFO',
            'message': message,
            **kwargs
        }
        self.logger.info(json.dumps(log_entry))
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error with structured data"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'ERROR',
            'message': message,
            'error_type': type(error).__name__ if error else None,
            'error_message': str(error) if error else None,
            **kwargs
        }
        self.logger.error(json.dumps(log_entry))
```

### **2. Context Managers for Logging**

```python
from contextlib import contextmanager

@contextmanager
def log_execution_time(logger, operation_name: str):
    """Context manager for logging execution time"""
    start_time = time.time()
    logger.info(f"Starting {operation_name}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Completed {operation_name} in {duration:.2f} seconds")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed {operation_name} after {duration:.2f} seconds: {e}")
        raise

# Usage
with log_execution_time(logger, "data loading"):
    data = loader.load_data("dataset_path")
```

### **3. Logging Configuration**

```python
def setup_logging(config: dict):
    """Setup logging configuration"""
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.get('console_level', 'INFO')))
    console_handler.setFormatter(simple_formatter)
    
    # File handler
    if config.get('log_file'):
        file_handler = logging.FileHandler(config['log_file'])
        file_handler.setLevel(getattr(logging, config.get('file_level', 'DEBUG')))
        file_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    
    if config.get('log_file'):
        root_logger.addHandler(file_handler)
```

---

## 🧪 Testing Strategies

### **1. Test Structure**

```python
import unittest
from unittest.mock import Mock, patch
import pytest

class TestDataLoader(unittest.TestCase):
    """Test suite for DataLoader"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.mock_logger = Mock(spec=ILogger)
        self.data_loader = EthnicityDataLoader(self.mock_logger)
        self.sample_data = np.random.rand(10, 100, 100, 3).astype(np.uint8)
        self.sample_labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    
    def tearDown(self):
        """Cleanup after tests"""
        # Cleanup any test files or resources
        pass
    
    def test_load_data_valid_path(self):
        """Test data loading with valid path"""
        # Test implementation
        pass
    
    def test_load_data_invalid_path(self):
        """Test data loading with invalid path"""
        with self.assertRaises(FileNotFoundError):
            self.data_loader.load_data("invalid_path")
    
    def test_validate_data_valid(self):
        """Test data validation with valid data"""
        result = self.data_loader.validate_data(self.sample_data, self.sample_labels)
        self.assertTrue(result)
    
    @patch('os.path.exists')
    def test_load_data_with_mock(self, mock_exists):
        """Test with mocked file system"""
        mock_exists.return_value = True
        # Test implementation
        pass
```

### **2. Fixture Management**

```python
import pytest

@pytest.fixture
def sample_images():
    """Fixture for sample images"""
    return np.random.rand(10, 100, 100, 3).astype(np.uint8)

@pytest.fixture
def sample_labels():
    """Fixture for sample labels"""
    return np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

@pytest.fixture
def mock_logger():
    """Fixture for mock logger"""
    return Mock(spec=ILogger)

@pytest.fixture
def data_loader(mock_logger):
    """Fixture for data loader"""
    return EthnicityDataLoader(mock_logger)

def test_data_loading(data_loader, sample_images, sample_labels):
    """Test using fixtures"""
    # Test implementation
    pass
```

### **3. Integration Testing**

```python
class TestTrainingPipeline(unittest.TestCase):
    """Integration tests for complete training pipeline"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.test_config = TrainingConfig({
            'dataset_path': 'test_dataset',
            'model_output_path': 'test_model.pkl',
            'model_type': 'random_forest'
        })
        
        # Create test dataset
        self._create_test_dataset()
    
    def tearDown(self):
        """Cleanup test environment"""
        self._cleanup_test_files()
    
    def test_complete_training_pipeline(self):
        """Test complete training pipeline"""
        logger = TrainingLogger('test_training')
        pipeline = PipelineFactory.create_pipeline(self.test_config, logger)
        
        results = pipeline.run_pipeline(
            self.test_config.get('dataset_path'),
            self.test_config.get('model_output_path')
        )
        
        self.assertTrue(results['model_saved'])
        self.assertIn('cross_validation', results)
        self.assertIn('model_info', results)
    
    def _create_test_dataset(self):
        """Create test dataset for integration tests"""
        # Implementation to create test dataset
        pass
    
    def _cleanup_test_files(self):
        """Cleanup test files"""
        # Implementation to cleanup test files
        pass
```

---

## ⚙️ Configuration Management

### **1. Configuration Classes**

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml

@dataclass
class ModelConfig:
    """Model-specific configuration"""
    type: str = 'random_forest'
    n_estimators: int = 200
    random_state: int = 220
    cv_folds: int = 6

@dataclass
class DataConfig:
    """Data-specific configuration"""
    dataset_path: str = 'dataset_periorbital'
    image_size: tuple = (400, 200)
    validation_split: float = 0.2

@dataclass
class LoggingConfig:
    """Logging-specific configuration"""
    level: str = 'INFO'
    file: Optional[str] = None
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

@dataclass
class TrainingConfig:
    """Complete training configuration"""
    model: ModelConfig
    data: DataConfig
    logging: LoggingConfig
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_data.get('model', {})),
            data=DataConfig(**config_data.get('data', {})),
            logging=LoggingConfig(**config_data.get('logging', {}))
        )
    
    def to_file(self, config_path: str):
        """Save configuration to YAML file"""
        config_data = {
            'model': {
                'type': self.model.type,
                'n_estimators': self.model.n_estimators,
                'random_state': self.model.random_state,
                'cv_folds': self.model.cv_folds
            },
            'data': {
                'dataset_path': self.data.dataset_path,
                'image_size': self.data.image_size,
                'validation_split': self.data.validation_split
            },
            'logging': {
                'level': self.logging.level,
                'file': self.logging.file,
                'format': self.logging.format
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
```

### **2. Environment-Specific Configuration**

```python
import os
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigManager:
    """Manages environment-specific configuration"""
    
    def __init__(self, environment: Environment = None):
        self.environment = environment or self._detect_environment()
        self.config = self._load_config()
    
    def _detect_environment(self) -> Environment:
        """Detect environment from environment variables"""
        env_str = os.getenv('ENVIRONMENT', 'development').lower()
        return Environment(env_str)
    
    def _load_config(self) -> TrainingConfig:
        """Load configuration for current environment"""
        config_path = f"config/{self.environment.value}.yaml"
        
        if os.path.exists(config_path):
            return TrainingConfig.from_file(config_path)
        else:
            # Fallback to default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> TrainingConfig:
        """Get default configuration"""
        return TrainingConfig(
            model=ModelConfig(),
            data=DataConfig(),
            logging=LoggingConfig()
        )
    
    def get_config(self) -> TrainingConfig:
        """Get current configuration"""
        return self.config
```

---

## 📚 Documentation Standards

### **1. Docstring Standards**

```python
def extract_features(self, images: np.ndarray) -> np.ndarray:
    """
    Extract GLCM features from grayscale images.
    
    This method calculates Gray Level Co-occurrence Matrix features
    including contrast, homogeneity, correlation, ASM, and entropy
    for texture analysis of ethnicity detection.
    
    Args:
        images (np.ndarray): Grayscale images array of shape (N, H, W)
            where N is the number of images, H is height, and W is width.
            Images should be in uint8 format with values 0-255.
    
    Returns:
        np.ndarray: Feature matrix of shape (N, num_features) where
            num_features is the total number of extracted features
            (typically 20 for GLCM features).
    
    Raises:
        ValueError: If images array is empty or has invalid shape.
        TypeError: If images is not a numpy array.
    
    Example:
        >>> extractor = GLCFeatureExtractor(logger)
        >>> images = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)
        >>> features = extractor.extract_features(images)
        >>> print(features.shape)
        (10, 20)
    
    Note:
        This method automatically handles image resizing if images
        are larger than 256x256 pixels for efficiency.
    """
    pass
```

### **2. Type Hints**

```python
from typing import List, Dict, Optional, Union, Tuple, Callable
import numpy as np
from sklearn.base import BaseEstimator

def process_data(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    config: Dict[str, Union[str, int, float]] = None,
    callback: Optional[Callable[[str], None]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Process data with optional labels and configuration.
    
    Args:
        data: Input data array
        labels: Optional labels array
        config: Optional configuration dictionary
        callback: Optional callback function for progress updates
    
    Returns:
        Tuple of (processed_data, processed_labels)
    """
    pass
```

### **3. API Documentation**

```python
class EthnicityDataLoader(IDataLoader):
    """
    Data loader for Indonesian ethnicity detection dataset.
    
    This class handles loading and validation of ethnicity detection
    data from directory structure. It supports various image formats
    and provides comprehensive data validation.
    
    Attributes:
        logger (ILogger): Logger instance for logging operations.
        label_map (Dict[int, str]): Mapping of label indices to ethnicity names.
        supported_formats (List[str]): List of supported image file formats.
    
    Example:
        >>> logger = TrainingLogger('data_loading')
        >>> loader = EthnicityDataLoader(logger)
        >>> images, labels, metadata = loader.load_data('dataset_periorbital')
        >>> print(f"Loaded {len(images)} images from {len(np.unique(labels))} classes")
    
    Note:
        The dataset should be organized in the following structure:
        dataset_periorbital/
        ├── Bugis/
        ├── Sunda/
        ├── Malay/
        ├── Jawa/
        └── Banjar/
    """
    
    def __init__(self, logger: ILogger, supported_formats: List[str] = None):
        """
        Initialize data loader.
        
        Args:
            logger: Logger instance for logging operations.
            supported_formats: List of supported image formats.
                Defaults to ['.jpg', '.jpeg', '.png'].
        """
        pass
```

---

## ⚡ Performance Optimization

### **1. Memory Management**

```python
import gc
from contextlib import contextmanager

@contextmanager
def memory_efficient_processing():
    """Context manager for memory-efficient processing"""
    try:
        yield
    finally:
        # Force garbage collection
        gc.collect()

class MemoryEfficientProcessor:
    """Memory-efficient data processor"""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    def process_large_dataset(self, data: np.ndarray):
        """Process large dataset in batches"""
        results = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            
            with memory_efficient_processing():
                processed_batch = self._process_batch(batch)
                results.append(processed_batch)
            
            # Clear batch from memory
            del batch
        
        return np.concatenate(results, axis=0)
```

### **2. Caching Strategies**

```python
import functools
import hashlib
import pickle
import os

def cached_method(cache_dir: str = "cache"):
    """Decorator for caching method results"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key
            cache_key = hashlib.md5(
                f"{func.__name__}_{str(args)}_{str(kwargs)}".encode()
            ).hexdigest()
            
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Check cache
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # Execute function
            result = func(self, *args, **kwargs)
            
            # Save to cache
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        return wrapper
    return decorator

class CachedFeatureExtractor:
    """Feature extractor with caching"""
    
    @cached_method("feature_cache")
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extract features with automatic caching"""
        # Feature extraction logic
        pass
```

### **3. Parallel Processing**

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

class ParallelProcessor:
    """Parallel processing utility"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or cpu_count()
    
    def process_parallel(self, data: List[Any], func: Callable, 
                        use_threads: bool = True) -> List[Any]:
        """Process data in parallel"""
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            futures = [executor.submit(func, item) for item in data]
            results = [future.result() for future in futures]
        
        return results
    
    def process_images_parallel(self, images: List[np.ndarray], 
                              processor_func: Callable) -> List[np.ndarray]:
        """Process images in parallel"""
        return self.process_parallel(images, processor_func, use_threads=True)
```

---

## 🔒 Security Considerations

### **1. Input Validation**

```python
import re
from pathlib import Path

class SecureDataLoader:
    """Secure data loader with input validation"""
    
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    def validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security"""
        path = Path(file_path)
        
        # Check for path traversal attacks
        if '..' in str(path):
            raise SecurityError("Path traversal detected")
        
        # Check file extension
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise SecurityError(f"Invalid file extension: {path.suffix}")
        
        # Check file size
        if path.exists() and path.stat().st_size > self.MAX_FILE_SIZE:
            raise SecurityError("File too large")
        
        return True
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for security"""
        # Remove dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        
        return sanitized
```

### **2. Configuration Security**

```python
import os
from cryptography.fernet import Fernet

class SecureConfigManager:
    """Secure configuration manager"""
    
    def __init__(self, encryption_key: bytes = None):
        self.encryption_key = encryption_key or self._generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _generate_key(self) -> bytes:
        """Generate encryption key"""
        return Fernet.generate_key()
    
    def encrypt_config(self, config: dict) -> bytes:
        """Encrypt configuration data"""
        config_str = json.dumps(config)
        return self.cipher.encrypt(config_str.encode())
    
    def decrypt_config(self, encrypted_config: bytes) -> dict:
        """Decrypt configuration data"""
        decrypted_str = self.cipher.decrypt(encrypted_config)
        return json.loads(decrypted_str.decode())
    
    def load_secure_config(self, config_path: str) -> dict:
        """Load encrypted configuration"""
        with open(config_path, 'rb') as f:
            encrypted_data = f.read()
        
        return self.decrypt_config(encrypted_data)
```

---

## 🎯 Conclusion

Following these best practices ensures:

1. **Maintainable Code**: Clear structure and documentation
2. **Reliable Systems**: Comprehensive error handling and testing
3. **Secure Applications**: Input validation and secure configuration
4. **Performant Software**: Optimized processing and caching
5. **Professional Quality**: Industry-standard practices

These practices transform research code into production-ready software that can be maintained, extended, and deployed with confidence.

---

## 📚 References

- [Python Best Practices](https://realpython.com/python-best-practices/)
- [PEP 8 - Style Guide](https://pep8.org/)
- [Python Logging Best Practices](https://docs.python.org/3/howto/logging.html)
- [Testing Best Practices](https://docs.python.org/3/library/unittest.html)
- [Security Best Practices](https://docs.python.org/3/library/security.html)
