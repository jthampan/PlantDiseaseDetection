"""
Process PlantVillage Dataset for Plant Disease Detection
======================================================

This script processes the New Plant Diseases Dataset (PlantVillage) which contains
38 classes of plant diseases and healthy plants. This is a much higher quality
dataset compared to the previous one and should achieve 60-80% accuracy.

The PlantVillage dataset is professionally curated with:
- High-resolution images
- Consistent lighting and backgrounds
- Better class balance
- Expert annotations

Expected Performance: 60-80% accuracy (vs 21% with old dataset)
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from collections import Counter
import json

class PlantVillageProcessor:
    """
    Process PlantVillage dataset for optimal plant disease detection.
    """
    
    def __init__(self, config):
        """Initialize processor with configuration."""
        self.config = config
        self.images = []
        self.labels = []
        self.class_names = []
        self.label_encoder = LabelEncoder()
        
        print("🌱 PlantVillage Dataset Processor Initialized")
        print("=" * 50)
        
    def load_plantvillage_data(self):
        """Load and process PlantVillage dataset."""
        print("📁 Loading PlantVillage dataset...")
        
        data_dir = self.config['data_dir']
        img_size = self.config['img_size']
        
        # Process train and validation splits to get class names, skip test for now
        splits = ['train', 'validation']
        
        all_images = []
        all_labels = []
        
        for split in splits:
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(split_dir):
                print(f"⚠️ {split} directory not found: {split_dir}")
                continue
                
            print(f"📂 Processing {split} data...")
            
            class_folders = sorted([f for f in os.listdir(split_dir) 
                                  if os.path.isdir(os.path.join(split_dir, f))])
            
            if not self.class_names:  # First time, set class names
                self.class_names = class_folders
                print(f"   ✅ Found {len(self.class_names)} classes")
                
            for class_idx, class_name in enumerate(class_folders):
                if class_name not in self.class_names:
                    continue  # Skip if not in our class list
                    
                class_path = os.path.join(split_dir, class_name)
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                print(f"   📊 {class_name}: {len(image_files)} images")
                
                for img_file in image_files:
                    img_path = os.path.join(class_path, img_file)
                    try:
                        # Load and preprocess image
                        image = cv2.imread(img_path)
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = cv2.resize(image, img_size)
                            
                            all_images.append(image)
                            all_labels.append(class_name)
                    except Exception as e:
                        print(f"   ⚠️ Error loading {img_path}: {e}")
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   📊 Total images: {len(all_images)}")
        print(f"   📊 Total classes: {len(self.class_names)}")
        
        # Convert to numpy arrays
        self.images = np.array(all_images, dtype=np.uint8)
        
        # Encode labels
        self.labels = self.label_encoder.fit_transform(all_labels)
        self.labels = to_categorical(self.labels, num_classes=len(self.class_names))
        
        print(f"   📊 Image shape: {self.images.shape}")
        print(f"   📊 Labels shape: {self.labels.shape}")
        
        # Update config with actual number of classes
        self.config['num_classes'] = len(self.class_names)
        
        return self.images, self.labels, self.class_names
    
    def analyze_dataset(self):
        """Analyze the dataset distribution and quality."""
        print("\n🔍 Dataset Analysis")
        print("=" * 30)
        
        # Get class distribution
        class_counts = Counter([self.label_encoder.inverse_transform([np.argmax(label)])[0] 
                               for label in self.labels])
        
        print("📊 Class Distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"   {class_name}: {count} images")
        
        # Calculate balance metrics
        counts = list(class_counts.values())
        min_count = min(counts)
        max_count = max(counts)
        mean_count = np.mean(counts)
        
        print(f"\n📈 Balance Analysis:")
        print(f"   Minimum samples per class: {min_count}")
        print(f"   Maximum samples per class: {max_count}")
        print(f"   Average samples per class: {mean_count:.1f}")
        print(f"   Imbalance ratio: {max_count/min_count:.2f}:1")
        
        if max_count/min_count < 5:
            print("   ✅ Well-balanced dataset!")
        else:
            print("   ⚠️ Some class imbalance detected")
        
        # Image quality analysis
        sample_images = self.images[:100]  # Analyze first 100 images
        mean_brightness = np.mean(sample_images)
        std_brightness = np.std(sample_images)
        
        print(f"\n🖼️ Image Quality:")
        print(f"   Mean brightness: {mean_brightness:.1f}")
        print(f"   Brightness std: {std_brightness:.1f}")
        print(f"   Image size: {self.images.shape[1:3]}")
        
        return class_counts
    
    def create_optimized_splits(self):
        """Create optimized train/validation/test splits."""
        print("\n📊 Creating optimized data splits...")
        
        # Convert one-hot back to class indices for stratification
        y_indices = np.argmax(self.labels, axis=1)
        
        # First split: separate test set (15%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, self.labels,
            test_size=self.config['test_size'],
            stratify=y_indices,
            random_state=self.config['random_seed']
        )
        
        # Second split: train/validation from remaining data
        y_temp_indices = np.argmax(y_temp, axis=1)
        val_ratio = self.config['val_size'] / (1 - self.config['test_size'])
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp_indices,
            random_state=self.config['random_seed']
        )
        
        print(f"✅ Data splits created:")
        print(f"   Training: {len(X_train)} samples ({len(X_train)/len(self.images)*100:.1f}%)")
        print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(self.images)*100:.1f}%)")
        print(f"   Test: {len(X_test)} samples ({len(X_test)/len(self.images)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save processed data for future use."""
        print("\n💾 Saving processed data...")
        
        save_path = os.path.join(self.config['data_dir'], 'processed_data.npz')
        
        np.savez_compressed(
            save_path,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            class_names=np.array(self.class_names)
        )
        
        file_size = os.path.getsize(save_path) / (1024*1024)  # MB
        print(f"✅ Data saved to: {save_path}")
        print(f"   File size: {file_size:.1f} MB")
        
        # Save class mapping for reference
        class_mapping = {i: name for i, name in enumerate(self.class_names)}
        mapping_path = os.path.join(self.config['data_dir'], 'class_mapping.json')
        
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print(f"✅ Class mapping saved to: {mapping_path}")
        
        return save_path
    
    def visualize_samples(self, X_train, y_train, num_samples=20):
        """Visualize sample images from each class."""
        print(f"\n🖼️ Visualizing {num_samples} sample images...")
        
        # Get one sample from each class
        unique_classes = np.unique(np.argmax(y_train, axis=1))
        samples_per_row = 5
        num_rows = (len(unique_classes) + samples_per_row - 1) // samples_per_row
        
        fig, axes = plt.subplots(num_rows, samples_per_row, 
                                figsize=(15, 3 * num_rows))
        axes = axes.flatten() if num_rows > 1 else [axes]
        
        for i, class_idx in enumerate(unique_classes):
            if i >= len(axes):
                break
                
            # Find first image of this class
            class_indices = np.where(np.argmax(y_train, axis=1) == class_idx)[0]
            sample_idx = class_indices[0]
            
            axes[i].imshow(X_train[sample_idx])
            axes[i].set_title(self.class_names[class_idx], fontsize=8)
            axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(len(unique_classes), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['data_dir'], 'sample_images.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Sample visualization saved")
    
    def run_complete_processing(self):
        """Run the complete data processing pipeline."""
        print("🚀 Starting PlantVillage Dataset Processing")
        print("=" * 60)
        
        try:
            # Load data
            self.load_plantvillage_data()
            
            # Analyze dataset
            self.analyze_dataset()
            
            # Create splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_optimized_splits()
            
            # Visualize samples
            self.visualize_samples(X_train, y_train)
            
            # Save processed data
            save_path = self.save_processed_data(X_train, X_val, X_test, 
                                               y_train, y_val, y_test)
            
            print("\n🎉 Processing completed successfully!")
            print(f"✅ Ready for training with {len(self.class_names)} classes")
            print(f"✅ Expected accuracy: 60-80% (much better than previous 21%)")
            
            return {
                'num_classes': len(self.class_names),
                'class_names': self.class_names,
                'data_path': save_path,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            print(f"❌ Error in processing: {e}")
            import traceback
            traceback.print_exc()
            return None

def get_plantvillage_config():
    """Configuration optimized for PlantVillage dataset."""
    return {
        # Data paths
        'data_dir': './data/plant_diseases',
        
        # Image settings
        'img_size': (224, 224),  # Standard size for good balance of detail and speed
        
        # Split ratios
        'test_size': 0.15,   # 15% for test
        'val_size': 0.15,    # 15% for validation (from remaining 85%)
        
        # Reproducibility
        'random_seed': 42
    }

def main():
    """Main function to process PlantVillage dataset."""
    print("🌱 PlantVillage Dataset Processing")
    print("=" * 40)
    
    config = get_plantvillage_config()
    
    print("⚙️ Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Check if data directory exists
    if not os.path.exists(config['data_dir']):
        print(f"❌ Data directory not found: {config['data_dir']}")
        print("Please ensure the dataset is in the correct location")
        return None
    
    # Initialize processor
    processor = PlantVillageProcessor(config)
    
    # Run processing
    results = processor.run_complete_processing()
    
    if results:
        print("\n📊 Processing Summary:")
        print(f"   Classes: {results['num_classes']}")
        print(f"   Training samples: {results['train_samples']}")
        print(f"   Validation samples: {results['val_samples']}")
        print(f"   Test samples: {results['test_samples']}")
        print(f"   Data saved to: {results['data_path']}")
        
        return results
    else:
        print("❌ Processing failed!")
        return None

if __name__ == "__main__":
    results = main()