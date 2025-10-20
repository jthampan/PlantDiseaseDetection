# Dataset Setup Guide

## Overview
The PlantVillage dataset is not included in this repository due to size constraints (2.27 GB). Follow these steps to set up the dataset locally.

## Dataset Download

### Option 1: Kaggle (Recommended)
1. Install Kaggle CLI: `pip install kaggle`
2. Set up Kaggle credentials (place `kaggle.json` in `~/.kaggle/`)
3. Download dataset:
   ```bash
   kaggle datasets download -d abdallahalidev/plantvillage-dataset
   unzip plantvillage-dataset.zip -d data/
   ```

### Option 2: Manual Download
1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download the dataset manually
3. Extract to: `data/plant_diseases/`

## Expected Directory Structure
```
data/
├── plant_diseases/
│   ├── train/
│   │   ├── Apple___Apple_scab/
│   │   ├── Apple___Black_rot/
│   │   ├── Apple___Cedar_apple_rust/
│   │   ├── Apple___healthy/
│   │   └── ... (38 classes total)
│   ├── validation/
│   │   └── ... (same 38 classes)
│   └── test/
│       └── test/ (66 test images)
└── processed_data/
    └── processed_data.npz (will be generated)
```

## Processed Data Generation

The `processed_data.npz` file (917 MB) is not included in the repository due to GitHub's 100 MB file size limit. This file will be automatically generated when you first run the data processing script:

```bash
python scripts/data_processing/process_plantvillage_dataset.py
```

This file contains:
- Pre-processed training data (X_train, y_train)
- Pre-processed validation data (X_val, y_val) 
- Class mappings and labels
- Normalized image arrays ready for training

## Model Files

### Pre-trained Model
The trained model (`best_model_colab.keras`) is not included due to size. You have two options:

1. **Train your own model**: Run the training script
   ```bash
   python scripts/training/colab_train_model.py
   ```

2. **Download pre-trained model**: Contact the project maintainer or use cloud storage

## Verification

After setup, verify the installation:
```bash
python main.py
# Choose option 8: Test accuracy comprehensive
```

## File Sizes
- Complete dataset: ~2.27 GB
- Training images: ~2.1 GB (70,295 images)
- Validation images: ~150 MB (17,572 images)
- Test images: ~8 MB (66 images)
- Processed data file: ~917 MB (generated automatically)
- Trained model: ~80 MB

## Alternative: Cloud Storage
For easier sharing, consider:
1. Google Drive
2. OneDrive
3. AWS S3
4. Dropbox

Upload the dataset and model files, then share download links with team members.