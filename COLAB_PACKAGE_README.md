# 📦 Google Colab Package Creation

## Quick Start

Create a zip file ready for Google Colab training in 3 simple steps:

### 1. Basic Package (Scripts + Dataset)
```bash
python create_colab_package.py
```
Creates: `plant_disease_colab.zip` (~1.3GB)

### 2. Lightweight Package (Scripts Only)
```bash
python create_colab_package.py --no-data
```
Creates: `plant_disease_colab.zip` (~100KB)
*Download dataset directly in Colab*

### 3. Full Package (Everything)
```bash
python create_colab_package.py --include-models
```
Creates: Complete package with pre-trained models

## 📋 Package Options

| Option | Description | Size Impact |
|--------|-------------|-------------|
| `--no-data` | Exclude dataset | -1.3GB |
| `--no-processed` | Exclude processed data | -900MB |
| `--include-models` | Include trained models | +100MB |
| `--output filename.zip` | Custom output name | - |

## 📊 Size Estimates

```bash
# Check sizes before creating
python create_colab_package.py --size-only
```

Sample output:
```
Essential files: 48.1 KB
Dataset: 1.3 GB (87900 files)  
Processed data: 900.0 MB
```

## 🚀 Colab Usage

### Upload to Google Drive
1. Upload your zip file to Google Drive
2. Open Google Colab (with GPU runtime)
3. Run the setup cells:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Extract package
import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/plant_disease_colab.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

# Install dependencies  
!pip install -r requirements.txt

# Start training
from scripts.training.colab_train_model import main_colab_training
main_colab_training()
```

## 💡 Recommendations

### For First-Time Users:
```bash
# Create lightweight package first
python create_colab_package.py --no-data --output my_first_test.zip
```
- Upload faster
- Test setup quickly  
- Download dataset in Colab

### For Regular Training:
```bash
# Include processed data for faster training
python create_colab_package.py
```
- Longer upload, faster training start
- Pre-processed data included
- Ready to train immediately

### For Advanced Users:
```bash
# Full package with everything
python create_colab_package.py --include-models --output complete_package.zip
```
- Transfer learning ready
- All components included
- Maximum flexibility

## 🔧 Troubleshooting

### Large File Upload Issues:
- Use `--no-data` option
- Download dataset directly in Colab:
```python
!wget -O dataset.zip "https://your-dataset-link"
!unzip dataset.zip
```

### Colab Timeout Issues:
- Save checkpoints frequently
- Use Google Drive sync
- Resume training from checkpoints

### Memory Issues:
- Reduce batch size in colab_train_model.py
- Clear memory between runs
- Monitor GPU usage with `!nvidia-smi`

## 📁 Package Contents

Your zip file will include:
- ✅ Training script (`colab_train_model.py`)
- ✅ Data processing script  
- ✅ Requirements and dependencies
- ✅ Setup documentation
- ✅ Colab instructions
- ✅ Package manifest
- 📁 Dataset (if included)
- 📁 Processed data (if available)
- 📁 Models (if requested)

Ready to train in Google Colab! 🚀