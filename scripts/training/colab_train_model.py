"""
Plant Disease Detection Training for Google Colab - PlantVillage Dataset
=======================================================================

This script adapts the PlantVillage training pipeline for Google Colab environment
with GPU acceleration and memory optimization.

PlantVillage Dataset Features:
- 87,867 high-quality images
- 38 classes (plant species + diseases)  
- Well-balanced distribution (1.23:1 ratio)
- Professional annotations
- Expected accuracy: 60-80% (vs 21% with old dataset)

Optimizations for Colab:
- GPU acceleration with mixed precision
- Memory management functions
- Enhanced CNN architecture for 38 classes
- Optimized hyperparameters for PlantVillage
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import required packages
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Memory management functions for Colab
def clear_memory():
    """Clear system memory and reset GPU state for fresh training."""
    print("🧹 Clearing system memory and GPU state...")
    
    # Clear Python variables and garbage collection
    import gc
    gc.collect()
    
    # Clear TensorFlow/Keras session and models
    try:
        tf.keras.backend.clear_session()
        print("   ✅ Keras session cleared")
    except:
        pass
    
    # Reset GPU memory if available
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.reset_memory_stats(gpu)
            print("   ✅ GPU memory stats reset")
    except:
        pass
    
    # Force garbage collection again
    gc.collect()
    
    # Check memory usage
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        print(f"   📊 System RAM: {memory_info.percent:.1f}% used ({memory_info.available/1024**3:.1f}GB available)")
    except ImportError:
        print("   📊 Memory info not available (psutil not installed)")
    
    print("   ✅ Memory cleanup completed\n")

def smart_memory_clear():
    """Advanced memory clearing for repeated training runs."""
    print("🧠 SMART MEMORY MANAGEMENT")
    print("=" * 40)
    
    # Clear global variables safely
    import gc
    import sys
    
    # List of variables to keep
    keep_vars = {
        '__name__', '__doc__', '__package__', '__loader__', '__spec__', 
        '__annotations__', '__builtins__', 'tf', 'np', 'plt', 'os', 'sys',
        'clear_memory', 'smart_memory_clear', 'check_memory_usage', 
        'main_colab_training', 'ColabPlantDiseaseTrainer', 'configure_gpu', 
        'enable_mixed_precision', 'get_colab_config'
    }
    
    # Get current globals
    current_globals = list(globals().keys())
    
    # Clear trainer instances and large objects
    cleared_count = 0
    for var_name in current_globals:
        if var_name not in keep_vars and not var_name.startswith('_'):
            try:
                var_obj = globals()[var_name]
                # Check if it's a large object or trainer instance
                if (hasattr(var_obj, '__dict__') or 
                    isinstance(var_obj, (list, dict, tuple)) or
                    var_name in ['trainer', 'results', 'model', 'history']):
                    del globals()[var_name]
                    cleared_count += 1
            except:
                pass
    
    print(f"   ✅ Cleared {cleared_count} global variables")
    
    # Clear TensorFlow/Keras
    try:
        tf.keras.backend.clear_session()
        print("   ✅ TensorFlow session cleared")
    except:
        pass
    
    # Force garbage collection multiple times
    for i in range(3):
        gc.collect()
    
    # GPU memory reset
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.reset_memory_stats(gpu)
            print("   ✅ GPU memory reset")
    except:
        pass
    
    # Check final memory state
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   📊 RAM after cleanup: {memory.percent:.1f}% used")
        
        if memory.percent > 80:
            print("   ⚠️  WARNING: High memory usage detected!")
            print("   💡 Consider restarting runtime if training fails")
        else:
            print("   ✅ Memory state looks good for training")
            
    except ImportError:
        print("   📊 Memory monitoring not available")
    
    print("   🎉 Smart memory cleanup completed!\n")

def check_memory_usage():
    """Check current memory usage before training."""
    print("🔍 Checking memory usage...")
    
    try:
        import psutil
        
        # System memory
        memory = psutil.virtual_memory()
        print(f"   💾 System RAM: {memory.percent:.1f}% used ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
        
        # GPU memory if available
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                gpu_details = tf.config.experimental.get_memory_info('GPU:0')
                gpu_used = gpu_details['current'] / 1024**3
                gpu_peak = gpu_details['peak'] / 1024**3
                print(f"   🎮 GPU RAM: {gpu_used:.1f}GB used (peak: {gpu_peak:.1f}GB)")
        except:
            print("   🎮 GPU memory info not available")
            
    except ImportError:
        print("   ⚠️ psutil not available - install with: !pip install psutil")
    
    print()

# Helper function to check file location
def check_data_file():
    """Check if processed_data.npz exists and show location info."""
    print("🔍 Checking for processed_data.npz file...")
    
    possible_paths = [
        './processed_data.npz',
        '/content/processed_data.npz',
        './data/plant_diseases/processed_data.npz',
        '/content/data/plant_diseases/processed_data.npz',
        'processed_data.npz'
    ]
    
    found_files = []
    for path in possible_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            found_files.append((path, size))
    
    if found_files:
        print("✅ Found processed_data.npz file(s):")
        for path, size in found_files:
            print(f"   📁 {path} ({size:.1f} MB)")
        return found_files[0][0]  # Return first found file
    else:
        print("❌ No processed_data.npz found!")
        print("📂 Current directory contents:")
        try:
            files = [f for f in os.listdir('.') if f.endswith('.npz')]
            if files:
                print("   NPZ files found:")
                for f in files:
                    print(f"   - {f}")
            else:
                print("   No .npz files in current directory")
                all_files = os.listdir('.')[:10]  # Show first 10 files
                print(f"   First 10 files: {all_files}")
        except Exception as e:
            print(f"   Error listing directory: {e}")
        return None

# GPU Configuration for Colab
def configure_gpu():
    """Configure GPU settings for Colab environment."""
    # First, clear any existing GPU memory
    try:
        tf.keras.backend.clear_session()
    except:
        pass
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPU (prevents taking all memory at once)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Reset GPU memory stats for fresh start
            try:
                for gpu in gpus:
                    tf.config.experimental.reset_memory_stats(gpu)
            except:
                pass
            
            print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
            return True
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            return False
    else:
        print("⚠️ No GPU detected - will use CPU")
        return False

# Mixed Precision for better GPU utilization
def enable_mixed_precision():
    """Enable mixed precision training for better performance."""
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision enabled (float16)")
    except Exception as e:
        print(f"⚠️ Mixed precision not available: {e}")

class ColabPlantDiseaseTrainer:
    """
    Colab-optimized trainer class for plant disease detection model.
    """
    
    def __init__(self, config):
        """Initialize the trainer with configuration."""
        self.config = config
        self.results = {}
        
        # Set random seeds for reproducibility
        np.random.seed(config.get('random_seed', 42))
        tf.random.set_seed(config.get('random_seed', 42))
        
        print("🔧 Trainer initialized for Colab environment")
    
    def load_colab_data(self):
        """Load data optimized for Colab environment with memory management."""
        print("📁 Loading dataset for Colab...")
        
        # Check if data is already loaded to prevent reloading
        if hasattr(self, 'X_train') and self.X_train is not None:
            print("✅ Data already loaded - reusing existing data to save memory")
            print(f"   📊 Using cached: {len(self.X_train)} train, {len(self.X_val)} val, {len(self.X_test)} test")
            return
        
        # Check multiple possible locations for the data file
        possible_paths = [
            './processed_data.npz',
            '/content/processed_data.npz',
            './data/plant_diseases/processed_data.npz',
            '/content/data/plant_diseases/processed_data.npz',
            'processed_data.npz'
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                print(f"📦 Found data file at: {path}")
                break
        
        if data_path is None:
            print("❌ No processed data found!")
            print("🔍 Searched in these locations:")
            for path in possible_paths:
                print(f"   - {path}")
            print("\n📂 Current directory contents:")
            try:
                files_in_dir = os.listdir('.')
                for f in files_in_dir:
                    print(f"   - {f}")
            except:
                print("   Could not list directory")
            
            print("\n💡 Solutions:")
            print("1. Upload processed_data.npz to Colab using:")
            print("   from google.colab import files")
            print("   uploaded = files.upload()")
            print("2. Or specify the correct path in the script")
            raise FileNotFoundError("processed_data.npz not found in any expected location")
        
        try:
            print("📦 Loading cached processed data...")
            
            # Memory-conscious loading with progress
            import psutil
            initial_memory = psutil.virtual_memory().percent
            print(f"   📊 Initial RAM usage: {initial_memory:.1f}%")
            
            data = np.load(data_path, allow_pickle=True)
            
            # Load data arrays with memory monitoring
            print("   📥 Loading training data...")
            self.X_train = data['X_train']
            print("   📥 Loading validation data...")  
            self.X_val = data['X_val'] 
            print("   📥 Loading test data...")
            self.X_test = data['X_test']
            print("   📥 Loading labels...")
            self.y_train = data['y_train']
            self.y_val = data['y_val']
            self.y_test = data['y_test']
            self.class_names = data['class_names']
            
            # Check memory after loading
            final_memory = psutil.virtual_memory().percent
            memory_increase = final_memory - initial_memory
            print(f"   📊 Final RAM usage: {final_memory:.1f}% (+{memory_increase:.1f}%)")
            
            print(f"✅ Data loaded: {len(self.X_train)} train, {len(self.X_val)} val, {len(self.X_test)} test")
            print(f"📊 Classes: {len(self.class_names)}")
            
            # Update config
            self.config['num_classes'] = len(self.class_names)
            
            # Force garbage collection after loading
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error loading data from {data_path}: {e}")
            raise
    
    def create_memory_efficient_model(self):
        """Create a hybrid model that combines custom CNN with light transfer learning."""
        print("🧠 Building hybrid CNN model with pre-trained features...")
        
        # Option 1: Custom CNN (like your working train_model.py)
        if self.config.get('use_custom_cnn', True):
            return self._create_custom_cnn()
        else:
            return self._create_transfer_learning_model()
    
    def _create_custom_cnn(self):
        """Create enhanced CNN optimized for PlantVillage dataset (38 classes)."""
        print("🏗️ Building PlantVillage-optimized CNN (targeting 60-80% accuracy)...")
        
        inputs = tf.keras.layers.Input(shape=(*self.config['img_size'], 3))
        
        # Normalize inputs (fix TensorFlow compatibility)
        x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)
        
        # Conv Block 1 - Foundation features
        x1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x1)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Conv Block 2 - Enhanced pattern detection
        x2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x2)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        
        # Squeeze-and-Excitation block for channel attention
        se = tf.keras.layers.GlobalAveragePooling2D()(x2)
        se = tf.keras.layers.Dense(64//4, activation='relu')(se)
        se = tf.keras.layers.Dense(64, activation='sigmoid')(se)
        se = tf.keras.layers.Reshape((1, 1, 64))(se)
        x2_attended = tf.keras.layers.Multiply()([x2, se])
        
        x = tf.keras.layers.MaxPooling2D((2, 2))(x2_attended)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        # Conv Block 3 - Complex feature extraction
        x3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x3 = tf.keras.layers.BatchNormalization()(x3)
        x3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x3)
        x3 = tf.keras.layers.BatchNormalization()(x3)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x3)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Conv Block 4 - High-level features  
        x4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x4 = tf.keras.layers.BatchNormalization()(x4)
        x4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x4)
        x4 = tf.keras.layers.BatchNormalization()(x4)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x4)
        x = tf.keras.layers.Dropout(0.35)(x)
        
        # Conv Block 5 - Abstract feature extraction
        x5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x5 = tf.keras.layers.BatchNormalization()(x5)
        x5 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same')(x5)  # Bottleneck
        x5 = tf.keras.layers.BatchNormalization()(x5)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x5)
        
        # Classification head optimized for 38 classes
        x = tf.keras.layers.Dense(1024, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.Dense(512, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(256, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Output layer for 38 classes
        outputs = tf.keras.layers.Dense(
            self.config['num_classes'], 
            activation='softmax',
            dtype='float32'
        )(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Optimized optimizer for PlantVillage dataset
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print(f"✅ PlantVillage CNN created: {model.count_params():,} parameters")
        print(f"   🎯 Optimized for 38 classes, targeting 60-80% accuracy")
        
        self.model = model
        return model
    
    def _create_transfer_learning_model(self):
        """Create transfer learning model as fallback."""
        print("🔄 Building transfer learning model...")
        
        # Use MobileNetV2 as base model (pre-trained on ImageNet)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.config['img_size'], 3),
            alpha=1.0,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = tf.keras.layers.Input(shape=(*self.config['img_size'], 3))
        
        # Preprocessing layer
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom classification layers
        x = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
        x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)
        
        x = tf.keras.layers.Dense(256, activation='relu', name='dense_2')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
        x = tf.keras.layers.Dropout(0.3, name='dropout_2')(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            self.config['num_classes'], 
            activation='softmax', 
            dtype='float32',
            name='predictions'
        )(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Compile with appropriate learning rate for transfer learning
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print(f"✅ Transfer learning model created!")
        print(f"   📊 Total parameters: {model.count_params():,}")
        print(f"   🔒 Frozen parameters: {base_model.count_params():,}")
        print(f"   🧠 Base model: MobileNetV2 (ImageNet pre-trained)")
        
        self.model = model
        return model
    
    def create_data_generators(self):
        """Create robust data generators optimized for PlantVillage dataset."""
        print("🔄 Creating robust data generators for PlantVillage dataset...")
        
        # Optimized data augmentation for PlantVillage (high-quality dataset)
        if self.config.get('use_data_augmentation', True):
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,          # Slightly reduced for stability
                width_shift_range=0.15,     # Reduced for better crop preservation
                height_shift_range=0.15,    # Reduced for better crop preservation
                shear_range=0.1,
                zoom_range=0.15,            # Reduced to prevent excessive distortion
                horizontal_flip=True,
                brightness_range=[0.95, 1.05],  # More conservative brightness
                fill_mode='nearest',
                preprocessing_function=None  # Normalization handled in model
            )
            print("   ✅ Conservative data augmentation enabled for stability")
        else:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        
        # Validation generator (no augmentation, normalization handled in model)
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        
        # Create generators with improved settings
        self.train_generator = train_datagen.flow(
            self.X_train, self.y_train,
            batch_size=self.config['batch_size'],
            shuffle=True,
            seed=42  # Fixed seed for reproducible augmentation
        )
        
        self.val_generator = val_datagen.flow(
            self.X_val, self.y_val,
            batch_size=self.config['batch_size'],
            shuffle=False,  # Never shuffle validation data
            seed=42
        )
        
        # Validate generators
        print("✅ Data generators created and validated")
        print(f"   📊 Training batches per epoch: {len(self.train_generator)}")
        print(f"   📊 Validation batches per epoch: {len(self.val_generator)}")
        print(f"   🧮 Training samples: {self.X_train.shape[0]}")
        print(f"   🧮 Validation samples: {self.X_val.shape[0]}")
        
        # Test generator functionality
        try:
            # Test training generator
            train_batch = next(iter(self.train_generator))
            print(f"   ✅ Training batch test: {train_batch[0].shape}, {train_batch[1].shape}")
            
            # Test validation generator
            val_batch = next(iter(self.val_generator))
            print(f"   ✅ Validation batch test: {val_batch[0].shape}, {val_batch[1].shape}")
            
        except Exception as e:
            print(f"   ⚠️ Generator test warning: {e}")
            print("   🔄 Generators may still work during training")
    
    def calculate_class_weights(self):
        """Calculate class weights to handle imbalanced dataset."""
        print("⚖️ Calculating class weights for imbalanced dataset...")
        
        # Get class distribution
        y_integers = np.argmax(self.y_train, axis=1)
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        
        class_weight_dict = dict(enumerate(class_weights))
        
        # Print class distribution info
        unique, counts = np.unique(y_integers, return_counts=True)
        print(f"   📊 Most common class: {counts.max()} samples")
        print(f"   📊 Least common class: {counts.min()} samples")
        print(f"   📊 Imbalance ratio: {counts.max()/counts.min():.2f}:1")
        print("   ✅ Class weights calculated to balance training")
        
        return class_weight_dict
    
    def train_model(self):
        """Train the model with Colab optimizations and comprehensive epoch monitoring."""
        print("🚀 Starting model training...")
        
        # Calculate class weights for imbalanced dataset
        class_weights = self.calculate_class_weights()
        
        # Create comprehensive progress callback that shows status for EVERY epoch
        class ComprehensiveProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.start_time = None
                self.epoch_times = []
                
            def on_train_begin(self, logs=None):
                self.start_time = tf.timestamp()
                print("🔥 Training Started - Monitoring ALL epochs!")
                print("=" * 60)
                
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start = tf.timestamp()
                print(f"\n📅 EPOCH {epoch + 1}/{self.params['epochs']} - Starting...")
                
            def on_epoch_end(self, epoch, logs=None):
                epoch_end = tf.timestamp()
                epoch_time = epoch_end - self.epoch_start
                self.epoch_times.append(epoch_time.numpy())
                
                # Get metrics
                train_acc = logs.get('accuracy', 0)
                train_loss = logs.get('loss', 0)
                val_acc = logs.get('val_accuracy', 0)
                val_loss = logs.get('val_loss', 0)
                lr = logs.get('lr', self.model.optimizer.learning_rate.numpy())
                
                # Status indicator
                if val_acc > 0.9:
                    status = "🔥 EXCELLENT"
                elif val_acc > 0.8:
                    status = "🚀 GREAT"
                elif val_acc > 0.6:
                    status = "✅ GOOD"
                elif val_acc > 0.4:
                    status = "🟡 OKAY"
                else:
                    status = "🔴 POOR"
                
                print(f"✅ EPOCH {epoch + 1}/{self.params['epochs']} COMPLETED - {status}")
                print(f"   📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} ({train_acc*100:.2f}%)")
                print(f"   📈 Valid: Loss={val_loss:.4f}, Acc={val_acc:.4f} ({val_acc*100:.2f}%)")
                print(f"   ⏱️  Time: {epoch_time:.1f}s, LR: {lr:.2e}")
                
                # Show improvement/decline
                if epoch > 0:
                    prev_val_acc = self.model.history.history['val_accuracy'][epoch-1] if hasattr(self.model, 'history') else 0
                    if val_acc > prev_val_acc:
                        print(f"   📈 Improvement: +{(val_acc-prev_val_acc)*100:.2f}%")
                    elif val_acc < prev_val_acc:
                        print(f"   📉 Decline: {(val_acc-prev_val_acc)*100:.2f}%")
                    else:
                        print(f"   ➡️  No change")
                
                # Estimated time remaining
                if epoch > 0:
                    avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
                    remaining_epochs = self.params['epochs'] - (epoch + 1)
                    eta = avg_epoch_time * remaining_epochs
                    eta_mins = int(eta // 60)
                    eta_secs = int(eta % 60)
                    print(f"   ⏳ ETA: {eta_mins}m {eta_secs}s ({remaining_epochs} epochs remaining)")
                
                print("-" * 60)
            
            def on_train_end(self, logs=None):
                total_time = tf.timestamp() - self.start_time
                total_mins = int(total_time // 60)
                total_secs = int(total_time % 60)
                print(f"\n🏁 TRAINING COMPLETED in {total_mins}m {total_secs}s")
        
        # Enhanced callbacks with better monitoring
        callbacks = [
            # Comprehensive progress tracking for every epoch
            ComprehensiveProgressCallback(),
            
            # Model checkpoint with detailed logging (modern format - fixes warning)
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_colab.keras',  # Modern format instead of .h5
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            ),
            
            # Backup H5 checkpoint for compatibility
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_colab.h5',
                monitor='val_accuracy', 
                save_best_only=True,
                mode='max',
                verbose=0,  # Less verbose for backup
                save_weights_only=False
            ),
            
            # Learning rate reduction with more conservative settings
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Less aggressive reduction
                patience=5,   # More patience
                min_lr=1e-8,
                verbose=1,
                cooldown=3
            ),
            
            # Early stopping with reasonable patience
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # Monitor accuracy instead of loss
                patience=12,  # Allow time for convergence
                restore_best_weights=True,
                verbose=1,
                min_delta=0.005,  # Require meaningful improvement
                mode='max'
            ),
            
            # Cosine annealing scheduler
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: self.config['learning_rate'] * 0.5 * (1 + np.cos(np.pi * epoch / 60)),  # HARDCODED 60
                verbose=0
            )
        ]
        
        # Calculate steps more carefully to ensure all data is used
        steps_per_epoch = max(1, len(self.X_train) // self.config['batch_size'])
        validation_steps = max(1, len(self.X_val) // self.config['batch_size'])
        
        # Ensure we don't skip any samples
        if len(self.X_train) % self.config['batch_size'] != 0:
            steps_per_epoch += 1
        if len(self.X_val) % self.config['batch_size'] != 0:
            validation_steps += 1
        
        print(f"📊 Training Configuration:")
        print(f"   🎯 Target Epochs: 60 (HARDCODED)")
        print(f"   📦 Batch Size: {self.config['batch_size']}")
        print(f"   📈 Steps per Epoch: {steps_per_epoch}")
        print(f"   📊 Validation Steps: {validation_steps}")
        print(f"   🧮 Total Training Samples: {len(self.X_train)}")
        print(f"   🧮 Total Validation Samples: {len(self.X_val)}")
        print(f"   📚 Classes: {self.config['num_classes']}")
        print(f"   🎓 Learning Rate: {self.config['learning_rate']}")
        
        print("\n🎯 PlantVillage CNN Training Session:")
        print("   🏗️  Enhanced Architecture for 38-class classification")
        print("   ⚙️  Optimized hyperparameters for high-quality PlantVillage data")
        print("   📊 87,867 professionally curated plant disease images")
        print("   🚀 GPU acceleration with mixed precision training")
        print("   💾 Automatic best model saving on accuracy improvement")
        print("   🔄 Adaptive learning rate with cosine annealing")
        print("   🌟 Target: 60-80% validation accuracy")
        print("   📈 Every epoch will be monitored and reported!\n")
        
        # Start training with comprehensive monitoring
        try:
            history = self.model.fit(
                self.train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=60,  # HARDCODED: Always 60 epochs
                validation_data=self.val_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=0,  # Set to 0 since our callback handles all output
                use_multiprocessing=False,  # Ensure stable training
                workers=1,  # Single worker to prevent data loading issues
                max_queue_size=10
            )
        except Exception as e:
            print(f"❌ Training error: {e}")
            print("🔄 Attempting recovery with reduced complexity...")
            
            # Fallback training with simpler setup - STILL USE 60 EPOCHS
            history = self.model.fit(
                self.train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=60,  # HARDCODED: Always 60 epochs, even in recovery
                validation_data=self.val_generator,
                validation_steps=validation_steps,
                callbacks=callbacks[:2],  # Only use checkpoint and progress callbacks
                verbose=1
            )
        
        self.history = history
        
        # Print training summary
        print("\n" + "="*50)
        print("🏁 TRAINING COMPLETED!")
        print("="*50)
        
        actual_epochs = len(history.history['accuracy'])
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        
        print(f"📊 Training Summary:")
        print(f"   Total epochs trained: {actual_epochs}/60 (HARDCODED)")
        print(f"   Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"   Best model saved at epoch: {best_epoch}")
        
        if actual_epochs < 60:
            print(f"   ⚡ Early stopping triggered - training was efficient!")
        else:
            print(f"   🔥 Full training completed!")
        
        # Save CPU-compatible model versions
        self.save_cpu_compatible_models(best_epoch)
        
        return history
    
    def plot_training_history(self):
        """Plot and display training history."""
        print("📈 Plotting training history...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', color='blue')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='red')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', color='blue')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print final metrics
        final_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        print(f"🎯 Final Training Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
        print(f"🎯 Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        print("📊 Evaluating model on test set...")
        
        # Load best model
        self.model = tf.keras.models.load_model('best_model_colab.h5')
        
        # Test data (normalization is handled inside the model now)
        X_test_eval = self.X_test.astype('float32')
        
        # Evaluate
        test_loss, test_accuracy, test_top3_accuracy = self.model.evaluate(
            X_test_eval, self.y_test, verbose=0
        )
        
        print(f"🏆 TEST RESULTS:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   Test Top-3 Accuracy: {test_top3_accuracy:.4f} ({test_top3_accuracy*100:.2f}%)")
        
        # Store results
        self.results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top3_accuracy': test_top3_accuracy
        }
        
        return self.results
    
    def save_cpu_compatible_models(self, best_epoch):
        """Save model in CPU-compatible formats after training."""
        print("\n💾 Saving CPU-compatible model versions...")
        
        try:
            from datetime import datetime
            import json
            
            # Load the best model
            best_model = tf.keras.models.load_model('best_model_colab.keras')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"plantvillage_cpu_model_{timestamp}"
            
            # 1. Save in multiple formats
            formats_saved = []
            
            # Keras format (already have best_model_colab.keras)
            keras_path = f"{base_name}.keras"
            best_model.save(keras_path)
            formats_saved.append(('Modern Keras', keras_path))
            
            # H5 format for compatibility (already have best_model_colab.h5)  
            h5_path = f"{base_name}.h5"
            best_model.save(h5_path, save_format='h5')
            formats_saved.append(('Legacy H5', h5_path))
            
            # Architecture + weights (maximum compatibility)
            arch_path = f"{base_name}_architecture.json"
            weights_path = f"{base_name}_weights.h5"
            
            with open(arch_path, 'w') as f:
                f.write(best_model.to_json())
            best_model.save_weights(weights_path)
            formats_saved.append(('Architecture JSON', arch_path))
            formats_saved.append(('Weights H5', weights_path))
            
            # 2. Save comprehensive class mapping
            class_info = {
                'class_names': self.class_names.tolist() if hasattr(self.class_names, 'tolist') else list(self.class_names),
                'num_classes': len(self.class_names),
                'model_timestamp': timestamp,
                'tensorflow_version': tf.__version__,
                'training_epoch': best_epoch,
                'validation_accuracy': max(self.history.history['val_accuracy']),
                'model_files': {
                    'recommended': keras_path,
                    'compatible': h5_path,
                    'architecture': arch_path,
                    'weights': weights_path
                },
                'cpu_compatibility': {
                    'tensorflow_2x': True,
                    'cpu_optimized': True,
                    'loading_order': [keras_path, h5_path, arch_path + '+' + weights_path]
                }
            }
            
            mapping_path = f"{base_name}_class_mapping.json"
            with open(mapping_path, 'w') as f:
                json.dump(class_info, f, indent=2)
            formats_saved.append(('Class Mapping', mapping_path))
            
            # 3. Create simple CPU loader
            loader_code = f'''
import tensorflow as tf
import numpy as np
import json

# Load class mapping
with open('{mapping_path}', 'r') as f:
    class_info = json.load(f)

class_names = class_info['class_names']

# Load model (try multiple formats)
try:
    model = tf.keras.models.load_model('{keras_path}')
    print("✅ Loaded modern Keras format")
except:
    try:
        model = tf.keras.models.load_model('{h5_path}')
        print("✅ Loaded H5 format")
    except:
        # Load from architecture + weights
        with open('{arch_path}', 'r') as f:
            model = tf.keras.models.model_from_json(f.read())
        model.load_weights('{weights_path}')
        print("✅ Loaded from architecture + weights")

print(f"🎯 PlantVillage model loaded with {{len(class_names)}} classes")
print(f"📊 Input shape: {{model.input_shape}}")

# Prediction function
def predict_plant_disease(image_array):
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    predictions = model.predict(image_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = float(predictions[0][predicted_idx])
    
    return predicted_class, confidence, predictions[0]

print("🚀 Use: predict_plant_disease(image_array)")
'''
            
            loader_path = f"{base_name}_cpu_loader.py"
            with open(loader_path, 'w') as f:
                f.write(loader_code)
            formats_saved.append(('CPU Loader', loader_path))
            
            # Summary
            print(f"\n🎯 CPU-COMPATIBLE MODELS SAVED:")
            for format_name, file_path in formats_saved:
                print(f"   ✅ {format_name}: {file_path}")
            
            print(f"\n📦 DOWNLOAD THESE FILES FOR CPU USE:")
            download_files = [keras_path, h5_path, mapping_path, loader_path]
            for file_path in download_files:
                print(f"   📁 files.download('{file_path}')")
            
            return {
                'saved_files': formats_saved,
                'download_files': download_files,
                'recommended_model': keras_path,
                'class_mapping': mapping_path
            }
            
        except Exception as e:
            print(f"   ❌ Error saving CPU-compatible models: {e}")
            return None
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline for Colab."""
        print("🚀 Starting Plant Disease Detection Training Pipeline for Colab")
        print("=" * 70)
        
        try:
            # Clear memory first for fresh start
            clear_memory()
            
            # Check memory usage
            check_memory_usage()
            
            # Configure environment
            gpu_available = configure_gpu()
            enable_mixed_precision()
            
            # Load data
            self.load_colab_data()
            
            # Create model
            self.create_memory_efficient_model()
            
            # Create data generators
            self.create_data_generators()
            
            # Train model
            self.train_model()
            
            # Plot results
            self.plot_training_history()
            
            # Evaluate model
            self.evaluate_model()
            
            # Final memory check
            print("\n📊 Final memory usage:")
            check_memory_usage()
            
            print("🎉 Training pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error in training pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

# Configuration for Colab
def get_colab_config():
    """Configuration optimized for PlantVillage dataset in Colab."""
    return {
        # Data settings
        'img_size': (224, 224),
        'batch_size': 24,  # Optimized for Colab GPU memory
        
        # Model settings
        'num_classes': 38,  # PlantVillage has 38 classes
        'use_custom_cnn': True,  # Use PlantVillage-optimized CNN
        
        # Training settings - optimized for PlantVillage
        'epochs': 60,  # HARDCODED: Always 60 epochs for PlantVillage
        'learning_rate': 0.0005,  # Lower LR for stable convergence
        'use_data_augmentation': True,
        
        # Reproducibility
        'random_seed': 42
    }

def main_colab_training():
    """Main function for PlantVillage Colab training."""
    print("🌱 PlantVillage Disease Detection Training - Colab Version")
    print("=" * 60)
    
    # Get configuration
    config = get_colab_config()
    
    print("⚙️ PlantVillage Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Initialize trainer
    trainer = ColabPlantDiseaseTrainer(config)
    
    # Run training pipeline
    success = trainer.run_complete_pipeline()
    
    if success:
        print("✅ PlantVillage training completed successfully!")
        print("🎯 Expected: 60-80% accuracy (major improvement from 21%)")
        return trainer.results
    else:
        print("❌ Training failed!")
        return None

# Example usage for Colab cells:
"""
# CELL 1: Memory Management (Run this first, especially between training runs)
clear_memory()
check_memory_usage()

# CELL 2: Check data file location:
check_data_file()

# CELL 3: If file not found, upload it:
from google.colab import files
uploaded = files.upload()  # Upload processed_data.npz

# CELL 4: Run the training (memory is automatically cleared at start):
results = main_colab_training()

# CELL 5: Download the trained model:
files.download('best_model_colab.h5')

# CELL 6: If you want to run again, clear memory first:
clear_memory()
results = main_colab_training()  # Fresh training run
"""

# Standalone memory management function for direct use
def reset_colab_environment():
    """Complete reset of Colab environment - use between training runs."""
    print("🔄 COMPLETE COLAB RESET")
    print("=" * 40)
    
    # Clear all variables from global namespace (be careful with this)
    try:
        # Get current globals
        current_globals = list(globals().keys())
        
        # Keep essential modules and functions
        keep_vars = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', 
                    '__annotations__', '__builtins__', 'tf', 'np', 'plt', 'os', 'sys',
                    'clear_memory', 'check_memory_usage', 'main_colab_training', 
                    'ColabPlantDiseaseTrainer', 'configure_gpu', 'enable_mixed_precision']
        
        # Clear other variables
        for var_name in current_globals:
            if not var_name.startswith('_') and var_name not in keep_vars:
                try:
                    del globals()[var_name]
                except:
                    pass
        
        print("   ✅ Global variables cleared")
    except Exception as e:
        print(f"   ⚠️ Could not clear all variables: {e}")
    
    # Clear memory
    clear_memory()
    
    # Check final state
    check_memory_usage()
    
    print("🎉 Colab environment reset complete!")
    print("   Ready for fresh training run\n")

if __name__ == "__main__":
    # For running directly
    results = main_colab_training()
    if results:
        print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")