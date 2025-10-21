"""
Google Colab Package Creator for Plant Disease Detection
=======================================================

This script creates a zip file containing all necessary components
for training the plant disease detection model in Google Colab.

Features:
- Intelligent file selection based on what's available
- Size optimization options
- Progress tracking during zip creation
- Verification of included components
- Colab-ready structure

Usage:
    python create_colab_package.py [options]
"""

import os
import zipfile
import shutil
from pathlib import Path
import argparse
import json
from datetime import datetime

class ColabPackageCreator:
    def __init__(self, output_name="plant_disease_colab.zip"):
        """Initialize the package creator."""
        self.project_root = Path.cwd()
        self.output_name = output_name
        self.total_files = 0
        self.total_size = 0
        self.included_files = []
        
    def get_file_size_mb(self, file_path):
        """Get file size in MB."""
        try:
            size_bytes = Path(file_path).stat().st_size
            return size_bytes / (1024 * 1024)
        except:
            return 0
    
    def format_size(self, size_mb):
        """Format size for display."""
        if size_mb < 1:
            return f"{size_mb * 1024:.1f} KB"
        elif size_mb < 1024:
            return f"{size_mb:.1f} MB"
        else:
            return f"{size_mb / 1024:.1f} GB"
    
    def scan_directory(self, directory, exclude_patterns=None):
        """Scan directory and return files with sizes."""
        if exclude_patterns is None:
            exclude_patterns = ['.git', '__pycache__', '.pyc', '.DS_Store']
        
        files_info = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return files_info
        
        for root, dirs, files in os.walk(dir_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                if not any(pattern in file for pattern in exclude_patterns):
                    file_path = Path(root) / file
                    try:
                        relative_path = file_path.relative_to(self.project_root)
                    except ValueError:
                        # Handle case where file is not in project root subpath
                        relative_path = file_path.relative_to(dir_path)
                    size_mb = self.get_file_size_mb(file_path)
                    files_info.append({
                        'path': file_path,
                        'relative_path': relative_path,
                        'size_mb': size_mb
                    })
        
        return files_info
    
    def create_package(self, include_data=True, include_processed=True, include_models=False):
        """Create the Colab package zip file."""
        
        print("📦 Creating Google Colab Package for Plant Disease Detection")
        print("=" * 65)
        print(f"📁 Project directory: {self.project_root}")
        print(f"📄 Output file: {self.output_name}")
        print()
        
        # Define what to include
        essential_files = [
            'scripts/training/colab_train_model.py',
            'scripts/data_processing/process_plantvillage_dataset.py',
            'requirements.txt',
            'COLAB_SETUP_GUIDE.md'
        ]
        
        optional_files = [
            'QUICK_START.md',
            'DATASET_SETUP.md',
            'README.md'
        ]
        
        # Check essential files
        print("🔍 Checking essential files:")
        missing_essential = []
        for file_path in essential_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                size = self.get_file_size_mb(full_path)
                print(f"   ✅ {file_path} ({self.format_size(size)})")
            else:
                print(f"   ❌ {file_path} - MISSING!")
                missing_essential.append(file_path)
        
        if missing_essential:
            print(f"\n❌ Missing essential files: {missing_essential}")
            print("Cannot create package without essential files.")
            return False
        
        print()
        
        # Start creating zip file
        print("📋 Creating zip file contents:")
        
        with zipfile.ZipFile(self.output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # Add essential files
            print("   📄 Adding essential files...")
            for file_path in essential_files:
                full_path = self.project_root / file_path
                zipf.write(full_path, file_path)
                self.included_files.append(file_path)
                self.total_files += 1
                self.total_size += self.get_file_size_mb(full_path)
            
            # Add optional documentation
            print("   📚 Adding documentation...")
            for file_path in optional_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    zipf.write(full_path, file_path)
                    self.included_files.append(file_path)
                    self.total_files += 1
                    self.total_size += self.get_file_size_mb(full_path)
            
            # Add dataset if requested
            if include_data:
                data_dir = self.project_root / 'data' / 'plant_diseases'
                if data_dir.exists():
                    print("   🌱 Adding dataset (this may take a while)...")
                    data_files = self.scan_directory(data_dir)
                    
                    for i, file_info in enumerate(data_files):
                        if i % 1000 == 0:  # Progress update every 1000 files
                            print(f"      Progress: {i}/{len(data_files)} files...")
                        
                        zipf.write(file_info['path'], f"data/plant_diseases/{file_info['relative_path'].relative_to('data/plant_diseases')}")
                        self.total_files += 1
                        self.total_size += file_info['size_mb']
                    
                    print(f"      ✅ Added {len(data_files)} dataset files")
                else:
                    print("   ⚠️  Dataset directory not found: data/plant_diseases")
            
            # Add processed data if available and requested
            if include_processed:
                processed_file = self.project_root / 'models' / 'processed_data.npz'
                if processed_file.exists():
                    print("   ⚡ Adding processed data...")
                    zipf.write(processed_file, 'models/processed_data.npz')
                    self.included_files.append('models/processed_data.npz')
                    self.total_files += 1
                    self.total_size += self.get_file_size_mb(processed_file)
                    print(f"      ✅ Added processed_data.npz ({self.format_size(self.get_file_size_mb(processed_file))})")
                else:
                    print("   ⚠️  Processed data not found: models/processed_data.npz")
            
            # Add existing models if requested
            if include_models:
                models_dir = self.project_root / 'models'
                if models_dir.exists():
                    print("   🤖 Adding existing models...")
                    model_files = [
                        'best_model_colab.keras',
                        'class_mapping.json'
                    ]
                    
                    for model_file in model_files:
                        model_path = models_dir / model_file
                        if model_path.exists():
                            zipf.write(model_path, f'models/{model_file}')
                            self.included_files.append(f'models/{model_file}')
                            self.total_files += 1
                            self.total_size += self.get_file_size_mb(model_path)
                            print(f"      ✅ Added {model_file}")
            
            # Create a manifest file
            print("   📋 Creating package manifest...")
            manifest = {
                'created_at': datetime.now().isoformat(),
                'total_files': self.total_files,
                'total_size_mb': round(self.total_size, 2),
                'includes_data': include_data,
                'includes_processed': include_processed,
                'includes_models': include_models,
                'files': self.included_files
            }
            
            manifest_str = json.dumps(manifest, indent=2)
            zipf.writestr('package_manifest.json', manifest_str)
            
            # Create Colab setup instructions
            colab_instructions = self._create_colab_instructions()
            zipf.writestr('COLAB_INSTRUCTIONS.md', colab_instructions)
        
        print()
        print("🎉 Package created successfully!")
        print(f"📄 File: {self.output_name}")
        print(f"📊 Total files: {self.total_files}")
        print(f"📦 Total size: {self.format_size(self.total_size)}")
        
        # File size warnings
        if self.total_size > 5000:  # 5GB
            print("⚠️  WARNING: File is very large (>5GB) - may be slow to upload")
        elif self.total_size > 2000:  # 2GB
            print("⚠️  Large file (>2GB) - upload may take some time")
        
        return True
    
    def _create_colab_instructions(self):
        """Create Colab setup instructions."""
        return """# Quick Colab Setup Instructions

## 📤 Upload and Extract

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Extract package (update path to your zip file)
import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/plant_disease_colab.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

# 3. Change to project directory
import os
os.chdir('/content/')
```

## ⚙️ Install Dependencies

```python
!pip install -r requirements.txt
```

## 🚀 Start Training

```python
from scripts.training.colab_train_model import main_colab_training
main_colab_training()
```

## 📊 Monitor Progress

```python
# Check GPU usage
!nvidia-smi

# View training files
!ls models/
```

## 💾 Save Results

```python
# Copy trained model to Google Drive
!cp models/best_model_colab.keras /content/drive/MyDrive/
!cp models/class_mapping.json /content/drive/MyDrive/
```
"""

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Create Google Colab package for plant disease detection')
    parser.add_argument('--output', '-o', default='plant_disease_colab.zip', 
                       help='Output zip filename (default: plant_disease_colab.zip)')
    parser.add_argument('--no-data', action='store_true', 
                       help='Exclude dataset (smaller package, process data in Colab)')
    parser.add_argument('--no-processed', action='store_true',
                       help='Exclude processed data file')
    parser.add_argument('--include-models', action='store_true',
                       help='Include existing trained models')
    parser.add_argument('--size-only', action='store_true',
                       help='Only show size estimates, don\'t create package')
    
    args = parser.parse_args()
    
    creator = ColabPackageCreator(args.output)
    
    if args.size_only:
        # Just show size estimates
        print("📊 Package Size Estimates:")
        print("=" * 30)
        
        essential_size = 0
        for file_path in ['scripts/training/colab_train_model.py', 'requirements.txt']:
            full_path = Path(file_path)
            if full_path.exists():
                essential_size += creator.get_file_size_mb(full_path)
        
        print(f"Essential files: {creator.format_size(essential_size)}")
        
        if not args.no_data:
            data_dir = Path('data/plant_diseases')
            if data_dir.exists():
                data_files = creator.scan_directory(data_dir)
                data_size = sum(f['size_mb'] for f in data_files)
                print(f"Dataset: {creator.format_size(data_size)} ({len(data_files)} files)")
        
        if not args.no_processed:
            processed_file = Path('models/processed_data.npz')
            if processed_file.exists():
                processed_size = creator.get_file_size_mb(processed_file)
                print(f"Processed data: {creator.format_size(processed_size)}")
        
        return
    
    # Create the package
    success = creator.create_package(
        include_data=not args.no_data,
        include_processed=not args.no_processed,
        include_models=args.include_models
    )
    
    if success:
        print("\n🚀 Ready for Google Colab!")
        print("Next steps:")
        print("1. Upload the zip file to Google Drive")
        print("2. Open Google Colab with GPU runtime")
        print("3. Follow instructions in COLAB_INSTRUCTIONS.md")
        print("4. Start training!")
    else:
        print("\n❌ Package creation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())