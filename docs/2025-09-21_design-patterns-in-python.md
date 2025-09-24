# 🎨 Design Patterns in Python - Implementation Guide

## 📋 Overview

This document provides a comprehensive guide to implementing design patterns in Python, specifically demonstrated through the ethnicity detection training system. Each pattern is explained with practical examples and real-world applications.

## 🎯 Table of Contents

1. [Creational Patterns](#creational-patterns)
2. [Structural Patterns](#structural-patterns)
3. [Behavioral Patterns](#behavioral-patterns)
4. [Architectural Patterns](#architectural-patterns)
5. [Pattern Implementation Examples](#pattern-implementation-examples)
6. [Best Practices](#best-practices)
7. [Common Pitfalls](#common-pitfalls)
8. [Testing Design Patterns](#testing-design-patterns)

---

## 🏗️ Creational Patterns

### **1. Factory Pattern**

#### **Purpose**
Create objects without specifying their exact classes. The factory method pattern provides an interface for creating objects in a superclass, but allows subclasses to alter the type of objects that will be created.

#### **Implementation in Our System**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class IModelTrainer(ABC):
    """Abstract product"""
    @abstractmethod
    def train(self, features, labels):
        pass

class RandomForestTrainer(IModelTrainer):
    """Concrete product"""
    def train(self, features, labels):
        # Random Forest implementation
        pass

class SVMTrainer(IModelTrainer):
    """Concrete product"""
    def train(self, features, labels):
        # SVM implementation
        pass

class ModelFactory:
    """Factory class"""
    
    @staticmethod
    def create_trainer(trainer_type: str, logger, **kwargs) -> IModelTrainer:
        """Factory method"""
        trainer_type = trainer_type.lower()
        
        if trainer_type == 'random_forest':
            return RandomForestTrainer(logger, **kwargs)
        elif trainer_type == 'svm':
            return SVMTrainer(logger, **kwargs)
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")
    
    @staticmethod
    def get_available_trainers() -> list:
        """Get list of available trainer types"""
        return ['random_forest', 'svm']

# Usage
logger = TrainingLogger()
trainer = ModelFactory.create_trainer('random_forest', logger, n_estimators=200)
```

#### **Benefits**
- **Flexibility**: Easy to add new model types
- **Encapsulation**: Creation logic is centralized
- **Loose Coupling**: Client code doesn't depend on concrete classes

#### **When to Use**
- When you need to create objects of different types
- When the exact type of object isn't known at compile time
- When you want to centralize object creation logic

---

### **2. Abstract Factory Pattern**

#### **Purpose**
Provide an interface for creating families of related objects without specifying their concrete classes.

#### **Implementation Example**
```python
from abc import ABC, abstractmethod

class IPreprocessor(ABC):
    """Abstract product A"""
    @abstractmethod
    def preprocess(self, images):
        pass

class IFeatureExtractor(ABC):
    """Abstract product B"""
    @abstractmethod
    def extract_features(self, images):
        pass

class GLCMProcessor(IPreprocessor):
    """Concrete product A1"""
    def preprocess(self, images):
        return cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)

class ColorProcessor(IPreprocessor):
    """Concrete product A2"""
    def preprocess(self, images):
        return cv2.cvtColor(images, cv2.COLOR_RGB2HSV)

class GLCMExtractor(IFeatureExtractor):
    """Concrete product B1"""
    def extract_features(self, images):
        # GLCM feature extraction
        pass

class ColorExtractor(IFeatureExtractor):
    """Concrete product B2"""
    def extract_features(self, images):
        # Color feature extraction
        pass

class MLFactory(ABC):
    """Abstract factory"""
    
    @abstractmethod
    def create_preprocessor(self) -> IPreprocessor:
        pass
    
    @abstractmethod
    def create_feature_extractor(self) -> IFeatureExtractor:
        pass

class GLCMFactory(MLFactory):
    """Concrete factory 1"""
    
    def create_preprocessor(self) -> IPreprocessor:
        return GLCMProcessor()
    
    def create_feature_extractor(self) -> IFeatureExtractor:
        return GLCMExtractor()

class ColorFactory(MLFactory):
    """Concrete factory 2"""
    
    def create_preprocessor(self) -> IPreprocessor:
        return ColorProcessor()
    
    def create_feature_extractor(self) -> IFeatureExtractor:
        return ColorExtractor()

# Usage
factory = GLCMFactory()
preprocessor = factory.create_preprocessor()
extractor = factory.create_feature_extractor()
```

---

### **3. Singleton Pattern**

#### **Purpose**
Ensure a class has only one instance and provide a global point of access to it.

#### **Implementation Example**
```python
class SingletonMeta(type):
    """Metaclass for singleton pattern"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class ConfigManager(metaclass=SingletonMeta):
    """Singleton configuration manager"""
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.config = {}
    
    def set_config(self, key: str, value):
        self.config[key] = value
    
    def get_config(self, key: str, default=None):
        return self.config.get(key, default)

# Usage
config1 = ConfigManager()
config2 = ConfigManager()
print(config1 is config2)  # True - same instance
```

#### **Thread-Safe Singleton**
```python
import threading

class ThreadSafeSingleton:
    """Thread-safe singleton implementation"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

---

### **4. Builder Pattern**

#### **Purpose**
Construct complex objects step by step. The builder pattern allows you to produce different types and representations of an object using the same construction code.

#### **Implementation Example**
```python
from abc import ABC, abstractmethod

class TrainingConfig:
    """Product to be built"""
    
    def __init__(self):
        self.dataset_path = None
        self.model_type = None
        self.n_estimators = None
        self.cv_folds = None
        self.random_state = None
    
    def __str__(self):
        return f"TrainingConfig(dataset={self.dataset_path}, model={self.model_type})"

class IConfigBuilder(ABC):
    """Abstract builder"""
    
    @abstractmethod
    def set_dataset_path(self, path: str):
        pass
    
    @abstractmethod
    def set_model_type(self, model_type: str):
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs):
        pass
    
    @abstractmethod
    def build(self) -> TrainingConfig:
        pass

class RandomForestConfigBuilder(IConfigBuilder):
    """Concrete builder for Random Forest"""
    
    def __init__(self):
        self.config = TrainingConfig()
    
    def set_dataset_path(self, path: str):
        self.config.dataset_path = path
        return self
    
    def set_model_type(self, model_type: str):
        self.config.model_type = model_type
        return self
    
    def set_parameters(self, **kwargs):
        self.config.n_estimators = kwargs.get('n_estimators', 200)
        self.config.cv_folds = kwargs.get('cv_folds', 6)
        self.config.random_state = kwargs.get('random_state', 220)
        return self
    
    def build(self) -> TrainingConfig:
        return self.config

class SVMConfigBuilder(IConfigBuilder):
    """Concrete builder for SVM"""
    
    def __init__(self):
        self.config = TrainingConfig()
    
    def set_dataset_path(self, path: str):
        self.config.dataset_path = path
        return self
    
    def set_model_type(self, model_type: str):
        self.config.model_type = model_type
        return self
    
    def set_parameters(self, **kwargs):
        self.config.C = kwargs.get('C', 1.0)
        self.config.gamma = kwargs.get('gamma', 'scale')
        self.config.cv_folds = kwargs.get('cv_folds', 6)
        return self
    
    def build(self) -> TrainingConfig:
        return self.config

class ConfigDirector:
    """Director class"""
    
    def __init__(self, builder: IConfigBuilder):
        self.builder = builder
    
    def build_random_forest_config(self, dataset_path: str) -> TrainingConfig:
        return (self.builder
                .set_dataset_path(dataset_path)
                .set_model_type('random_forest')
                .set_parameters(n_estimators=300, cv_folds=10)
                .build())
    
    def build_svm_config(self, dataset_path: str) -> TrainingConfig:
        return (self.builder
                .set_dataset_path(dataset_path)
                .set_model_type('svm')
                .set_parameters(C=10.0, gamma='auto')
                .build())

# Usage
rf_builder = RandomForestConfigBuilder()
director = ConfigDirector(rf_builder)
config = director.build_random_forest_config('dataset_periorbital')
print(config)
```

---

## 🏗️ Structural Patterns

### **1. Adapter Pattern**

#### **Purpose**
Allow objects with incompatible interfaces to work together by wrapping the incompatible object with an adapter.

#### **Implementation Example**
```python
# Existing third-party library interface
class ThirdPartyImageProcessor:
    """Existing class with incompatible interface"""
    
    def process_image_file(self, file_path: str) -> str:
        """Expects file path, returns processed file path"""
        # Process image and save to new file
        return f"processed_{file_path}"
    
    def get_image_info(self, file_path: str) -> dict:
        """Expects file path, returns info dict"""
        return {"size": "1024x768", "format": "JPEG"}

# Our desired interface
class IImageProcessor(ABC):
    """Our desired interface"""
    
    @abstractmethod
    def process_images(self, images: np.ndarray) -> np.ndarray:
        """Expects numpy array, returns numpy array"""
        pass
    
    @abstractmethod
    def get_processing_info(self) -> dict:
        """Returns processing information"""
        pass

class ImageProcessorAdapter(IImageProcessor):
    """Adapter to make third-party class compatible"""
    
    def __init__(self, third_party_processor: ThirdPartyImageProcessor):
        self.third_party = third_party_processor
        self.temp_files = []
    
    def process_images(self, images: np.ndarray) -> np.ndarray:
        """Adapt numpy arrays to file-based processing"""
        processed_images = []
        
        for i, image in enumerate(images):
            # Save image to temporary file
            temp_file = f"temp_image_{i}.jpg"
            cv2.imwrite(temp_file, image)
            self.temp_files.append(temp_file)
            
            # Process using third-party library
            processed_file = self.third_party.process_image_file(temp_file)
            
            # Load processed image back to numpy array
            processed_image = cv2.imread(processed_file)
            processed_images.append(processed_image)
            
            # Clean up
            os.remove(temp_file)
            os.remove(processed_file)
        
        return np.array(processed_images)
    
    def get_processing_info(self) -> dict:
        """Adapt third-party info to our interface"""
        if self.temp_files:
            info = self.third_party.get_image_info(self.temp_files[0])
            return {
                "processor": "ThirdPartyAdapter",
                "image_size": info["size"],
                "format": info["format"]
            }
        return {"processor": "ThirdPartyAdapter"}

# Usage
third_party_processor = ThirdPartyImageProcessor()
adapter = ImageProcessorAdapter(third_party_processor)
processed_images = adapter.process_images(image_array)
```

---

### **2. Decorator Pattern**

#### **Purpose**
Attach additional behaviors to objects dynamically. The decorator pattern allows you to extend functionality without altering the original class.

#### **Implementation Example**
```python
from abc import ABC, abstractmethod

class IDataProcessor(ABC):
    """Base component"""
    
    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        pass

class BasicProcessor(IDataProcessor):
    """Concrete component"""
    
    def process(self, data: np.ndarray) -> np.ndarray:
        return data * 2

class DataProcessorDecorator(IDataProcessor):
    """Base decorator"""
    
    def __init__(self, processor: IDataProcessor):
        self.processor = processor
    
    def process(self, data: np.ndarray) -> np.ndarray:
        return self.processor.process(data)

class LoggingDecorator(DataProcessorDecorator):
    """Concrete decorator - adds logging"""
    
    def __init__(self, processor: IDataProcessor, logger):
        super().__init__(processor)
        self.logger = logger
    
    def process(self, data: np.ndarray) -> np.ndarray:
        self.logger.info(f"Processing data of shape {data.shape}")
        result = super().process(data)
        self.logger.info(f"Processed data of shape {result.shape}")
        return result

class ValidationDecorator(DataProcessorDecorator):
    """Concrete decorator - adds validation"""
    
    def process(self, data: np.ndarray) -> np.ndarray:
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be None or empty")
        
        result = super().process(data)
        
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            raise ValueError("Result contains NaN or infinite values")
        
        return result

class TimingDecorator(DataProcessorDecorator):
    """Concrete decorator - adds timing"""
    
    def __init__(self, processor: IDataProcessor):
        super().__init__(processor)
        self.processing_time = None
    
    def process(self, data: np.ndarray) -> np.ndarray:
        import time
        start_time = time.time()
        result = super().process(data)
        self.processing_time = time.time() - start_time
        return result

# Usage - compose decorators dynamically
basic_processor = BasicProcessor()

# Add multiple decorators
processor = TimingDecorator(
    ValidationDecorator(
        LoggingDecorator(basic_processor, logger)
    )
)

data = np.array([1, 2, 3, 4, 5])
result = processor.process(data)
print(f"Processing time: {processor.processing_time:.4f} seconds")
```

---

### **3. Facade Pattern**

#### **Purpose**
Provide a simplified interface to a complex subsystem. The facade pattern hides the complexity of the underlying system.

#### **Implementation Example**
```python
class DataLoader:
    """Subsystem component"""
    
    def load_data(self, path: str):
        print(f"Loading data from {path}")
        return np.random.rand(100, 10)
    
    def validate_data(self, data):
        print("Validating data")
        return True

class Preprocessor:
    """Subsystem component"""
    
    def preprocess_glcm(self, data):
        print("Preprocessing for GLCM")
        return data
    
    def preprocess_color(self, data):
        print("Preprocessing for color")
        return data

class FeatureExtractor:
    """Subsystem component"""
    
    def extract_glcm_features(self, data):
        print("Extracting GLCM features")
        return np.random.rand(100, 20)
    
    def extract_color_features(self, data):
        print("Extracting color features")
        return np.random.rand(100, 32)

class ModelTrainer:
    """Subsystem component"""
    
    def train_model(self, features, labels):
        print("Training model")
        return "trained_model"
    
    def validate_model(self, model):
        print("Validating model")
        return True

class MLTrainingFacade:
    """Facade - simplified interface to complex subsystem"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.feature_extractor = FeatureExtractor()
        self.model_trainer = ModelTrainer()
    
    def train_ethnicity_model(self, dataset_path: str) -> str:
        """Simplified interface for complex training process"""
        print("=== Starting ML Training Process ===")
        
        # Step 1: Load data
        data = self.data_loader.load_data(dataset_path)
        if not self.data_loader.validate_data(data):
            raise ValueError("Invalid data")
        
        # Step 2: Preprocessing
        glcm_data = self.preprocessor.preprocess_glcm(data)
        color_data = self.preprocessor.preprocess_color(data)
        
        # Step 3: Feature extraction
        glcm_features = self.feature_extractor.extract_glcm_features(glcm_data)
        color_features = self.feature_extractor.extract_color_features(color_data)
        
        # Step 4: Combine features
        combined_features = np.concatenate([glcm_features, color_features], axis=1)
        labels = np.random.randint(0, 5, 100)  # Dummy labels
        
        # Step 5: Train model
        model = self.model_trainer.train_model(combined_features, labels)
        if not self.model_trainer.validate_model(model):
            raise ValueError("Model validation failed")
        
        print("=== Training Complete ===")
        return model

# Usage - simple interface hides complexity
facade = MLTrainingFacade()
model = facade.train_ethnicity_model("dataset_periorbital")
```

---

## 🎭 Behavioral Patterns

### **1. Strategy Pattern**

#### **Purpose**
Define a family of algorithms, encapsulate each one, and make them interchangeable. The strategy pattern lets the algorithm vary independently from clients that use it.

#### **Implementation Example**
```python
from abc import ABC, abstractmethod

class IFeatureExtractionStrategy(ABC):
    """Strategy interface"""
    
    @abstractmethod
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        pass

class GLCMStrategy(IFeatureExtractionStrategy):
    """Concrete strategy 1"""
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        print("Using GLCM feature extraction strategy")
        # GLCM implementation
        return np.random.rand(len(images), 20)
    
    def get_strategy_name(self) -> str:
        return "GLCM"

class ColorHistogramStrategy(IFeatureExtractionStrategy):
    """Concrete strategy 2"""
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        print("Using Color Histogram feature extraction strategy")
        # Color histogram implementation
        return np.random.rand(len(images), 32)
    
    def get_strategy_name(self) -> str:
        return "ColorHistogram"

class LBPStrategy(IFeatureExtractionStrategy):
    """Concrete strategy 3"""
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        print("Using Local Binary Pattern feature extraction strategy")
        # LBP implementation
        return np.random.rand(len(images), 59)
    
    def get_strategy_name(self) -> str:
        return "LBP"

class FeatureExtractor:
    """Context class"""
    
    def __init__(self, strategy: IFeatureExtractionStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: IFeatureExtractionStrategy):
        """Allow strategy to be changed at runtime"""
        self.strategy = strategy
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Delegates to the current strategy"""
        return self.strategy.extract_features(images)
    
    def get_current_strategy(self) -> str:
        return self.strategy.get_strategy_name()

# Usage
images = np.random.rand(10, 100, 100, 3)

# Use different strategies
glcm_strategy = GLCMStrategy()
extractor = FeatureExtractor(glcm_strategy)
glcm_features = extractor.extract_features(images)

# Change strategy at runtime
color_strategy = ColorHistogramStrategy()
extractor.set_strategy(color_strategy)
color_features = extractor.extract_features(images)

# Use another strategy
lbp_strategy = LBPStrategy()
extractor.set_strategy(lbp_strategy)
lbp_features = extractor.extract_features(images)
```

---

### **2. Observer Pattern**

#### **Purpose**
Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

#### **Implementation Example**
```python
from abc import ABC, abstractmethod
from typing import List

class ITrainingObserver(ABC):
    """Observer interface"""
    
    @abstractmethod
    def on_training_started(self, message: str):
        pass
    
    @abstractmethod
    def on_epoch_completed(self, epoch: int, metrics: dict):
        pass
    
    @abstractmethod
    def on_training_completed(self, final_metrics: dict):
        pass

class ProgressObserver(ITrainingObserver):
    """Concrete observer 1"""
    
    def on_training_started(self, message: str):
        print(f"📊 Progress: {message}")
    
    def on_epoch_completed(self, epoch: int, metrics: dict):
        accuracy = metrics.get('accuracy', 0)
        print(f"📊 Epoch {epoch}: Accuracy = {accuracy:.2f}%")
    
    def on_training_completed(self, final_metrics: dict):
        final_accuracy = final_metrics.get('accuracy', 0)
        print(f"📊 Training completed! Final accuracy: {final_accuracy:.2f}%")

class LoggingObserver(ITrainingObserver):
    """Concrete observer 2"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
    
    def on_training_started(self, message: str):
        with open(self.log_file, 'a') as f:
            f.write(f"Training started: {message}\n")
    
    def on_epoch_completed(self, epoch: int, metrics: dict):
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch}: {metrics}\n")
    
    def on_training_completed(self, final_metrics: dict):
        with open(self.log_file, 'a') as f:
            f.write(f"Training completed: {final_metrics}\n")

class MetricsObserver(ITrainingObserver):
    """Concrete observer 3"""
    
    def __init__(self):
        self.epoch_metrics = []
    
    def on_training_started(self, message: str):
        self.epoch_metrics = []
    
    def on_epoch_completed(self, epoch: int, metrics: dict):
        self.epoch_metrics.append((epoch, metrics))
    
    def on_training_completed(self, final_metrics: dict):
        print(f"📈 Collected metrics for {len(self.epoch_metrics)} epochs")
        # Could save to database, generate plots, etc.

class TrainingSubject:
    """Subject - notifies observers of changes"""
    
    def __init__(self):
        self.observers: List[ITrainingObserver] = []
    
    def add_observer(self, observer: ITrainingObserver):
        self.observers.append(observer)
    
    def remove_observer(self, observer: ITrainingObserver):
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_training_started(self, message: str):
        for observer in self.observers:
            observer.on_training_started(message)
    
    def notify_epoch_completed(self, epoch: int, metrics: dict):
        for observer in self.observers:
            observer.on_epoch_completed(epoch, metrics)
    
    def notify_training_completed(self, final_metrics: dict):
        for observer in self.observers:
            observer.on_training_completed(final_metrics)
    
    def train_model(self, data, labels):
        """Simulate training process with notifications"""
        self.notify_training_started("Starting Random Forest training")
        
        # Simulate training epochs
        for epoch in range(1, 6):
            # Simulate training step
            accuracy = 0.7 + (epoch * 0.05) + np.random.normal(0, 0.02)
            metrics = {
                'accuracy': accuracy,
                'loss': 1 - accuracy,
                'epoch': epoch
            }
            self.notify_epoch_completed(epoch, metrics)
        
        final_metrics = {
            'accuracy': 0.95,
            'loss': 0.05,
            'total_epochs': 5
        }
        self.notify_training_completed(final_metrics)

# Usage
subject = TrainingSubject()

# Add observers
progress_observer = ProgressObserver()
logging_observer = LoggingObserver("training.log")
metrics_observer = MetricsObserver()

subject.add_observer(progress_observer)
subject.add_observer(logging_observer)
subject.add_observer(metrics_observer)

# Start training - all observers will be notified
data = np.random.rand(100, 10)
labels = np.random.randint(0, 5, 100)
subject.train_model(data, labels)
```

---

### **3. Command Pattern**

#### **Purpose**
Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.

#### **Implementation Example**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class ICommand(ABC):
    """Command interface"""
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def undo(self) -> Dict[str, Any]:
        pass

class LoadDataCommand(ICommand):
    """Concrete command 1"""
    
    def __init__(self, data_loader, data_path: str):
        self.data_loader = data_loader
        self.data_path = data_path
        self.loaded_data = None
    
    def execute(self) -> Dict[str, Any]:
        print(f"Loading data from {self.data_path}")
        self.loaded_data = self.data_loader.load_data(self.data_path)
        return {"status": "success", "data_shape": self.loaded_data.shape}
    
    def undo(self) -> Dict[str, Any]:
        if self.loaded_data is not None:
            print("Unloading data")
            self.loaded_data = None
        return {"status": "data_unloaded"}

class PreprocessDataCommand(ICommand):
    """Concrete command 2"""
    
    def __init__(self, preprocessor, data):
        self.preprocessor = preprocessor
        self.original_data = data
        self.processed_data = None
    
    def execute(self) -> Dict[str, Any]:
        print("Preprocessing data")
        self.processed_data = self.preprocessor.preprocess(self.original_data)
        return {"status": "success", "processed_shape": self.processed_data.shape}
    
    def undo(self) -> Dict[str, Any]:
        if self.processed_data is not None:
            print("Undoing preprocessing")
            self.processed_data = None
        return {"status": "preprocessing_undone"}

class ExtractFeaturesCommand(ICommand):
    """Concrete command 3"""
    
    def __init__(self, feature_extractor, data):
        self.feature_extractor = feature_extractor
        self.input_data = data
        self.extracted_features = None
    
    def execute(self) -> Dict[str, Any]:
        print("Extracting features")
        self.extracted_features = self.feature_extractor.extract_features(self.input_data)
        return {"status": "success", "feature_shape": self.extracted_features.shape}
    
    def undo(self) -> Dict[str, Any]:
        if self.extracted_features is not None:
            print("Undoing feature extraction")
            self.extracted_features = None
        return {"status": "feature_extraction_undone"}

class TrainingInvoker:
    """Invoker - executes commands"""
    
    def __init__(self):
        self.command_history = []
        self.current_command = None
    
    def execute_command(self, command: ICommand) -> Dict[str, Any]:
        """Execute a command and add to history"""
        self.current_command = command
        result = command.execute()
        self.command_history.append(command)
        return result
    
    def undo_last_command(self) -> Dict[str, Any]:
        """Undo the last executed command"""
        if self.command_history:
            last_command = self.command_history.pop()
            return last_command.undo()
        return {"status": "no_commands_to_undo"}
    
    def execute_pipeline(self, commands: list) -> Dict[str, Any]:
        """Execute a series of commands"""
        results = []
        for command in commands:
            result = self.execute_command(command)
            results.append(result)
        return {"status": "pipeline_completed", "results": results}

# Usage
invoker = TrainingInvoker()

# Create commands
load_cmd = LoadDataCommand(data_loader, "dataset_periorbital")
preprocess_cmd = PreprocessDataCommand(preprocessor, loaded_data)
extract_cmd = ExtractFeaturesCommand(feature_extractor, processed_data)

# Execute commands
pipeline_commands = [load_cmd, preprocess_cmd, extract_cmd]
result = invoker.execute_pipeline(pipeline_commands)

# Undo last command
undo_result = invoker.undo_last_command()
```

---

## 🏗️ Architectural Patterns

### **1. Pipeline Pattern**

#### **Purpose**
Process data through a series of stages where the output of one stage becomes the input of the next stage.

#### **Implementation Example**
```python
from abc import ABC, abstractmethod
from typing import Any, Optional

class IPipelineStage(ABC):
    """Pipeline stage interface"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass
    
    @abstractmethod
    def get_stage_name(self) -> str:
        pass

class DataValidationStage(IPipelineStage):
    """Stage 1: Data validation"""
    
    def process(self, data: np.ndarray) -> np.ndarray:
        print("🔍 Validating data...")
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be None or empty")
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Data contains NaN or infinite values")
        
        print(f"✅ Data validation passed: {data.shape}")
        return data
    
    def get_stage_name(self) -> str:
        return "DataValidation"

class PreprocessingStage(IPipelineStage):
    """Stage 2: Preprocessing"""
    
    def process(self, data: np.ndarray) -> np.ndarray:
        print("🔄 Preprocessing data...")
        # Normalize data
        normalized_data = (data - data.mean()) / data.std()
        print(f"✅ Preprocessing completed: {normalized_data.shape}")
        return normalized_data
    
    def get_stage_name(self) -> str:
        return "Preprocessing"

class FeatureExtractionStage(IPipelineStage):
    """Stage 3: Feature extraction"""
    
    def process(self, data: np.ndarray) -> np.ndarray:
        print("🧮 Extracting features...")
        # Simulate feature extraction
        features = np.random.rand(data.shape[0], 50)
        print(f"✅ Feature extraction completed: {features.shape}")
        return features
    
    def get_stage_name(self) -> str:
        return "FeatureExtraction"

class ModelTrainingStage(IPipelineStage):
    """Stage 4: Model training"""
    
    def process(self, data: np.ndarray) -> str:
        print("🤖 Training model...")
        # Simulate model training
        model = "trained_model"
        print(f"✅ Model training completed")
        return model
    
    def get_stage_name(self) -> str:
        return "ModelTraining"

class DataPipeline:
    """Pipeline orchestrator"""
    
    def __init__(self):
        self.stages: list[IPipelineStage] = []
        self.results = {}
    
    def add_stage(self, stage: IPipelineStage) -> 'DataPipeline':
        """Add a stage to the pipeline"""
        self.stages.append(stage)
        return self
    
    def process(self, initial_data: Any) -> Any:
        """Process data through all stages"""
        print("🚀 Starting pipeline processing...")
        
        current_data = initial_data
        stage_results = {}
        
        for i, stage in enumerate(self.stages):
            print(f"\n--- Stage {i+1}: {stage.get_stage_name()} ---")
            
            try:
                current_data = stage.process(current_data)
                stage_results[stage.get_stage_name()] = {
                    "status": "success",
                    "output_type": type(current_data).__name__
                }
            except Exception as e:
                print(f"❌ Stage {stage.get_stage_name()} failed: {e}")
                stage_results[stage.get_stage_name()] = {
                    "status": "failed",
                    "error": str(e)
                }
                raise
        
        self.results = stage_results
        print("\n🎉 Pipeline processing completed successfully!")
        return current_data
    
    def get_pipeline_info(self) -> dict:
        """Get information about the pipeline"""
        return {
            "num_stages": len(self.stages),
            "stage_names": [stage.get_stage_name() for stage in self.stages],
            "results": self.results
        }

# Usage
pipeline = (DataPipeline()
           .add_stage(DataValidationStage())
           .add_stage(PreprocessingStage())
           .add_stage(FeatureExtractionStage())
           .add_stage(ModelTrainingStage()))

# Process data through pipeline
data = np.random.rand(100, 10)
result = pipeline.process(data)

print(f"\nPipeline Info: {pipeline.get_pipeline_info()}")
```

---

## 💡 Best Practices

### **1. Choose the Right Pattern**
```python
# Use Factory when you need to create objects of different types
class ModelFactory:
    @staticmethod
    def create_trainer(type: str) -> IModelTrainer:
        pass

# Use Strategy when you need to switch algorithms at runtime
class FeatureExtractor:
    def __init__(self, strategy: IFeatureExtractionStrategy):
        self.strategy = strategy

# Use Observer when you need to notify multiple objects of changes
class TrainingSubject:
    def __init__(self):
        self.observers = []
```

### **2. Keep Patterns Simple**
```python
# Good: Simple and focused
class SimpleFactory:
    @staticmethod
    def create_logger(type: str) -> ILogger:
        if type == "console":
            return ConsoleLogger()
        elif type == "file":
            return FileLogger("app.log")

# Bad: Over-engineered
class ComplexFactoryWithRegistryAndBuilder:
    def __init__(self):
        self.registry = {}
        self.builder = None
        # Too complex for simple use case
```

### **3. Use Composition Over Inheritance**
```python
# Good: Composition
class DataProcessor:
    def __init__(self, validator: IValidator, logger: ILogger):
        self.validator = validator
        self.logger = logger

# Bad: Deep inheritance
class AdvancedDataProcessor(DataProcessor):
    pass

class SuperAdvancedDataProcessor(AdvancedDataProcessor):
    pass
```

### **4. Make Patterns Testable**
```python
# Good: Easy to test with mocks
def test_factory():
    trainer = ModelFactory.create_trainer('random_forest')
    assert isinstance(trainer, RandomForestTrainer)

def test_strategy():
    mock_strategy = Mock(spec=IFeatureExtractionStrategy)
    extractor = FeatureExtractor(mock_strategy)
    extractor.extract_features(data)
    mock_strategy.extract_features.assert_called_once_with(data)
```

---

## ⚠️ Common Pitfalls

### **1. Over-Engineering**
```python
# Bad: Using complex patterns for simple problems
class OverEngineeredSolution:
    def __init__(self):
        self.factory = Factory()
        self.builder = Builder()
        self.decorator = Decorator()
        # Too complex for simple data processing

# Good: Simple solution for simple problem
def process_data(data):
    return data * 2
```

### **2. Tight Coupling in Patterns**
```python
# Bad: Factory tightly coupled to concrete classes
class BadFactory:
    @staticmethod
    def create_processor():
        return ConcreteProcessor(Logger(), Validator())  # Hard-coded dependencies

# Good: Factory uses dependency injection
class GoodFactory:
    @staticmethod
    def create_processor(logger: ILogger, validator: IValidator):
        return ConcreteProcessor(logger, validator)
```

### **3. Ignoring Interface Segregation**
```python
# Bad: Bloated interface
class IMLProcessor(ABC):
    @abstractmethod
    def load_data(self): pass
    @abstractmethod
    def preprocess(self): pass
    @abstractmethod
    def train(self): pass
    @abstractmethod
    def save(self): pass
    @abstractmethod
    def validate(self): pass
    @abstractmethod
    def log(self): pass

# Good: Focused interfaces
class IDataLoader(ABC):
    @abstractmethod
    def load_data(self): pass

class IPreprocessor(ABC):
    @abstractmethod
    def preprocess(self): pass
```

---

## 🧪 Testing Design Patterns

### **1. Testing Factories**
```python
def test_model_factory():
    # Test all supported types
    for model_type in ['random_forest', 'svm']:
        trainer = ModelFactory.create_trainer(model_type)
        assert isinstance(trainer, IModelTrainer)
    
    # Test invalid type
    with pytest.raises(ValueError):
        ModelFactory.create_trainer('invalid_type')
```

### **2. Testing Strategies**
```python
def test_strategy_pattern():
    mock_strategy = Mock(spec=IFeatureExtractionStrategy)
    mock_strategy.extract_features.return_value = np.array([[1, 2, 3]])
    
    extractor = FeatureExtractor(mock_strategy)
    result = extractor.extract_features(np.array([[1, 2, 3, 4]]))
    
    mock_strategy.extract_features.assert_called_once()
    assert result.shape == (1, 3)
```

### **3. Testing Observers**
```python
def test_observer_pattern():
    subject = TrainingSubject()
    observer = Mock(spec=ITrainingObserver)
    
    subject.add_observer(observer)
    subject.notify_training_started("test")
    
    observer.on_training_started.assert_called_once_with("test")
```

---

## 🎯 Conclusion

Design patterns provide proven solutions to common software design problems. When implemented correctly, they:

1. **Improve Code Quality**: Make code more maintainable and extensible
2. **Enhance Communication**: Provide a common vocabulary for developers
3. **Reduce Complexity**: Break down complex problems into manageable pieces
4. **Enable Reusability**: Create components that can be reused in different contexts
5. **Facilitate Testing**: Make code easier to test and mock

The key is to choose the right pattern for the right problem and implement it correctly without over-engineering. The ethnicity detection training system demonstrates how multiple patterns can work together to create a robust, maintainable architecture.

---

## 📚 References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612)
- [Python Design Patterns](https://python-patterns.guide/)
- [Refactoring.Guru - Design Patterns](https://refactoring.guru/design-patterns)
- [Head First Design Patterns](https://www.amazon.com/Head-First-Design-Patterns-Brain-Friendly/dp/0596007124)
