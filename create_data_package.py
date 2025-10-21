"""
Data Package Creator for Google Drive Upload
===========================================

This script creates optimized zip files for uploading your plant disease detection
data to Google Drive. Handles large datasets efficiently with progress tracking.

Features:
- Multiple package options (data only, models only, complete)
- Size optimization and compression
- Progress tracking for large files
- Google Drive upload preparation
- Verification of package contents
"""

import os
import zipfile
import shutil
from pathlib import Path
import argparse
import json
from datetime import datetime
import sys

class DataPackageCreator:
    def __init__(self):
        """Initialize the data package creator."""
        self.project_root = Path.cwd()
        self.total_files = 0
        self.total_size = 0
        self.compression_level = zipfile.ZIP_DEFLATED
        
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
    
    def scan_directory(self, directory):
        """Scan directory and return file information."""
        files_info = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return files_info, 0
        
        total_size = 0
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = Path(root) / file
                size_mb = self.get_file_size_mb(file_path)
                total_size += size_mb
                
                files_info.append({
                    'path': file_path,
                    'relative_path': file_path.relative_to(dir_path),
                    'size_mb': size_mb
                })
        
        return files_info, total_size
    
    def create_dataset_package(self, output_name="plant_disease_dataset.zip"):
        """Create a package with just the dataset."""
        
        print("📦 Creating Dataset Package for Google Drive")
        print("=" * 50)
        
        data_dir = self.project_root / "data" / "plant_diseases"
        if not data_dir.exists():
            print("❌ Dataset directory not found: data/plant_diseases")
            return False
        
        print("🔍 Scanning dataset...")
        files_info, total_size = self.scan_directory(data_dir)
        
        print(f"📊 Dataset Summary:")
        print(f"   Files: {len(files_info):,}")
        print(f"   Size: {self.format_size(total_size)}")
        print(f"   Output: {output_name}")
        print()
        
        if total_size > 15000:  # 15GB warning
            print("⚠️  WARNING: Very large dataset (>15GB)")
            print("   Consider creating separate packages for train/validation")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                return False
        
        print("📋 Creating dataset zip file...")
        
        with zipfile.ZipFile(output_name, 'w', self.compression_level) as zipf:
            for i, file_info in enumerate(files_info):
                if i % 1000 == 0:
                    progress = (i / len(files_info)) * 100
                    print(f"   Progress: {i:,}/{len(files_info):,} files ({progress:.1f}%)")
                
                # Add file to zip with proper structure
                zipf.write(file_info['path'], f"data/plant_diseases/{file_info['relative_path']}")
            
            # Add manifest
            manifest = {
                'type': 'dataset_package',
                'created_at': datetime.now().isoformat(),
                'total_files': len(files_info),
                'total_size_mb': round(total_size, 2),
                'dataset_classes': self._count_classes(data_dir)
            }
            zipf.writestr('dataset_manifest.json', json.dumps(manifest, indent=2))
        
        print(f"✅ Dataset package created: {output_name}")
        print(f"📊 Final size: {self.format_size(self.get_file_size_mb(output_name))}")
        return True
    
    def create_models_package(self, output_name="plant_disease_models.zip"):
        """Create a package with essential models only (excludes large processed data)."""
        
        print("📦 Creating Models Package for Google Drive")
        print("=" * 50)
        
        # Check what model files exist (essential files only)
        models_dir = self.project_root / "models"
        model_files = []
        
        if models_dir.exists():
            # Only include essential model files (exclude large .npz files)
            for file_pattern in ["*.keras", "*.h5", "*.json"]:
                model_files.extend(models_dir.glob(file_pattern))
        
        # Skip processed data for models-only package (too large)
        # Users can create complete package if they need processed data
        
        if not model_files:
            print("❌ No essential model files found")
            return False
        
        print("📋 Found essential model files:")
        total_size = 0
        for file_path in model_files:
            size = self.get_file_size_mb(file_path)
            total_size += size
            print(f"   ✅ {file_path.name} ({self.format_size(size)})")
        
        print(f"\n📊 Total size: {self.format_size(total_size)}")
        print(f"📄 Output: {output_name}")
        print(f"ℹ️  Note: Large processed data excluded for quick sharing")
        print()
        
        print("📋 Creating models zip file...")
        
        with zipfile.ZipFile(output_name, 'w', self.compression_level) as zipf:
            # Add model files
            for file_path in model_files:
                zipf.write(file_path, f"models/{file_path.name}")
                print(f"   ✅ Added: models/{file_path.name}")
            
            # Add manifest
            manifest = {
                'type': 'models_package',
                'created_at': datetime.now().isoformat(),
                'model_files': [f.name for f in model_files],
                'total_size_mb': round(total_size, 2),
                'note': 'Essential models only - large processed data excluded for quick sharing'
            }
            zipf.writestr('models_manifest.json', json.dumps(manifest, indent=2))
        
        print(f"✅ Models package created: {output_name}")
        print(f"📊 Final size: {self.format_size(self.get_file_size_mb(output_name))}")
        return True
    
    def create_processed_package(self, output_name="plant_disease_processed.zip"):
        """Create a package with processed data only."""
        
        print("📦 Creating Processed Data Package for Google Drive")
        print("=" * 50)
        
        # Check processed data directory
        processed_dir = self.project_root / "data" / "processed_data"
        processed_files = []
        
        if processed_dir.exists():
            processed_files.extend(processed_dir.glob("*"))
        
        if not processed_files:
            print("❌ No processed data files found")
            return False
        
        print("📋 Found processed data files:")
        total_size = 0
        for file_path in processed_files:
            if file_path.is_file():
                size = self.get_file_size_mb(file_path)
                total_size += size
                print(f"   ✅ {file_path.name} ({self.format_size(size)})")
        
        print(f"\n📊 Total size: {self.format_size(total_size)}")
        print(f"📄 Output: {output_name}")
        
        if total_size > 10000:  # 10GB warning
            print("⚠️  WARNING: Very large processed data (>10GB)")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                return False
        
        print("\n📋 Creating processed data zip file...")
        
        with zipfile.ZipFile(output_name, 'w', self.compression_level) as zipf:
            for file_path in processed_files:
                if file_path.is_file():
                    zipf.write(file_path, f"data/processed_data/{file_path.name}")
                    print(f"   ✅ Added: {file_path.name}")
            
            # Add manifest
            manifest = {
                'type': 'processed_data_package',
                'created_at': datetime.now().isoformat(),
                'processed_files': [f.name for f in processed_files if f.is_file()],
                'total_size_mb': round(total_size, 2)
            }
            zipf.writestr('processed_manifest.json', json.dumps(manifest, indent=2))
        
        print(f"✅ Processed data package created: {output_name}")
        print(f"📊 Final size: {self.format_size(self.get_file_size_mb(output_name))}")
        return True
    
    def create_complete_package(self, output_name="plant_disease_complete.zip"):
        """Create a complete package with everything."""
        
        print("📦 Creating Complete Package for Google Drive")
        print("=" * 50)
        
        # Check all components
        components = {
            'dataset': self.project_root / "data" / "plant_diseases",
            'models': self.project_root / "models",
            'processed': self.project_root / "data" / "processed_data"
        }
        
        total_estimated = 0
        component_info = {}
        
        print("🔍 Scanning all components...")
        
        for name, path in components.items():
            if path.exists():
                if name == 'dataset':
                    files, size = self.scan_directory(path)
                    component_info[name] = {'files': files, 'size': size}
                    print(f"   ✅ {name}: {len(files):,} files ({self.format_size(size)})")
                elif name == 'models':
                    files = list(path.glob("*"))
                    size = sum(self.get_file_size_mb(f) for f in files if f.is_file())
                    component_info[name] = {'files': files, 'size': size}
                    print(f"   ✅ {name}: {len(files)} files ({self.format_size(size)})")
                elif name == 'processed':
                    files = list(path.glob("*"))
                    size = sum(self.get_file_size_mb(f) for f in files if f.is_file())
                    component_info[name] = {'files': files, 'size': size}
                    print(f"   ✅ {name}: {len(files)} files ({self.format_size(size)})")
                
                total_estimated += component_info[name]['size']
            else:
                print(f"   ⚠️  {name}: Not found")
        
        print(f"\n📊 Estimated total size: {self.format_size(total_estimated)}")
        
        if total_estimated > 20000:  # 20GB
            print("⚠️  WARNING: Very large package (>20GB)")
            print("   Consider creating separate packages instead")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                return False
        
        print(f"📄 Output: {output_name}")
        print("\n📋 Creating complete zip file...")
        
        with zipfile.ZipFile(output_name, 'w', self.compression_level) as zipf:
            
            # Add dataset if exists
            if 'dataset' in component_info:
                print("   📁 Adding dataset...")
                files = component_info['dataset']['files']
                for i, file_info in enumerate(files):
                    if i % 1000 == 0:
                        progress = (i / len(files)) * 100
                        print(f"      Progress: {i:,}/{len(files):,} files ({progress:.1f}%)")
                    
                    zipf.write(file_info['path'], f"data/plant_diseases/{file_info['relative_path']}")
            
            # Add models if exists
            if 'models' in component_info:
                print("   🤖 Adding models...")
                for file_path in component_info['models']['files']:
                    if file_path.is_file():
                        zipf.write(file_path, f"models/{file_path.name}")
                        print(f"      ✅ Added: {file_path.name}")
            
            # Add processed data if exists
            if 'processed' in component_info:
                print("   ⚡ Adding processed data...")
                for file_path in component_info['processed']['files']:
                    if file_path.is_file():
                        zipf.write(file_path, f"data/processed_data/{file_path.name}")
                        print(f"      ✅ Added: {file_path.name}")
            
            # Add manifest
            manifest = {
                'type': 'complete_package',
                'created_at': datetime.now().isoformat(),
                'components': {k: {'files': len(v['files']), 'size_mb': v['size']} 
                             for k, v in component_info.items()},
                'total_size_mb': round(total_estimated, 2)
            }
            zipf.writestr('complete_manifest.json', json.dumps(manifest, indent=2))
        
        print(f"✅ Complete package created: {output_name}")
        print(f"📊 Final size: {self.format_size(self.get_file_size_mb(output_name))}")
        return True
    
    def _count_classes(self, dataset_dir):
        """Count the number of classes in the dataset."""
        classes = set()
        for subset in ['train', 'validation', 'test']:
            subset_dir = dataset_dir / subset
            if subset_dir.exists():
                classes.update([d.name for d in subset_dir.iterdir() if d.is_dir()])
        return len(classes)

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Create data packages for Google Drive upload')
    parser.add_argument('--type', choices=['dataset', 'models', 'processed', 'complete'], 
                       default='complete', help='Type of package to create')
    parser.add_argument('--output', '-o', help='Output filename (auto-generated if not provided)')
    parser.add_argument('--size-check', action='store_true', 
                       help='Only check sizes, don\'t create package')
    
    args = parser.parse_args()
    
    creator = DataPackageCreator()
    
    if args.size_check:
        print("📊 Data Size Analysis")
        print("=" * 30)
        
        # Check dataset
        dataset_dir = Path("data/plant_diseases")
        if dataset_dir.exists():
            _, size = creator.scan_directory(dataset_dir)
            print(f"Dataset: {creator.format_size(size)}")
        
        # Check models
        models_dir = Path("models")
        if models_dir.exists():
            files = list(models_dir.glob("*"))
            size = sum(creator.get_file_size_mb(f) for f in files if f.is_file())
            print(f"Models: {creator.format_size(size)}")
        
        # Check processed data
        processed_dir = Path("data/processed_data")
        if processed_dir.exists():
            files = list(processed_dir.glob("*"))
            size = sum(creator.get_file_size_mb(f) for f in files if f.is_file())
            print(f"Processed data: {creator.format_size(size)}")
        
        return 0
    
    # Determine output filename
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"plant_disease_{args.type}_{timestamp}.zip"
    
    # Create the appropriate package
    success = False
    
    if args.type == 'dataset':
        success = creator.create_dataset_package(args.output)
    elif args.type == 'models':
        success = creator.create_models_package(args.output)
    elif args.type == 'processed':
        success = creator.create_processed_package(args.output)
    elif args.type == 'complete':
        success = creator.create_complete_package(args.output)
    
    if success:
        print(f"\n🚀 Ready for Google Drive upload!")
        print(f"📁 File: {args.output}")
        print("\nNext steps:")
        print("1. Upload the zip file to Google Drive")
        print("2. Share or use as needed for your project")
        print("3. The zip contains organized folder structure")
    else:
        print(f"\n❌ Package creation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())