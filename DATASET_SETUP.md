# Dataset Setup Guide

## 🚨 Important Update: Google Drive Integration Available!

**NEW**: The complete dataset is now available via Google Drive! Choose your preferred setup method below.

---

## 🚀 Option 1: Google Drive Download (Recommended)

### Quick Setup with Pre-packaged Dataset

**✅ Complete Package Available:**
- **File**: `plant_disease_complete_20251021_084958.zip`
- **Size**: 10.3 GB (includes dataset + models + processed data)
- **Location**: Google Drive
- **Status**: Ready for immediate use

**📥 Download Steps:**
1. **Access the Google Drive package** (contact project maintainer for link)
2. **Download** `plant_disease_complete_20251021_084958.zip`
3. **Extract** to your project directory:
   ```bash
   # Windows
   Expand-Archive -Path "plant_disease_complete_20251021_084958.zip" -DestinationPath "."
   
   # Linux/Mac
   unzip plant_disease_complete_20251021_084958.zip
   ```
4. **Verify** the structure matches the expected layout below

**⚡ Advantages:**
- ✅ No manual dataset download required
- ✅ Includes pre-processed data (saves 30+ minutes)
- ✅ Includes trained models (93.9% accuracy)
- ✅ Ready for immediate use

---

## 📦 Option 2: Manual Dataset Setup (Alternative)

### Prerequisites

Before setting up the dataset manually, ensure you have:
- Python 3.8 or higher installed
- All required dependencies installed: `pip install -r requirements.txt`
- At least 4GB of free disk space
- Stable internet connection for dataset download

### Dataset Download

#### Kaggle Download (Manual Method)
1. Install Kaggle CLI: `pip install kaggle`
2. Set up Kaggle credentials (place `kaggle.json` in `~/.kaggle/`)
3. Download dataset:
   ```bash
   kaggle datasets download -d abdallahalidev/plantvillage-dataset
   unzip plantvillage-dataset.zip -d data/
   ```

#### Manual Download from Kaggle
1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download the dataset manually
3. Extract to: `data/plant_diseases/`

**⚠️ Note**: Manual setup requires additional processing time and does not include pre-trained models.

---

## 📁 Expected Directory Structure

After setup (either Google Drive or manual), your project should have:

```
Final_plantDisease_Project/
├── data/
│   ├── plant_diseases/
│   │   ├── train/
│   │   │   ├── Apple___Apple_scab/
│   │   │   ├── Apple___Black_rot/
│   │   │   ├── Apple___Cedar_apple_rust/
│   │   │   ├── Apple___healthy/
│   │   │   └── ... (38 classes total)
│   │   ├── validation/
│   │   │   └── ... (same 38 classes)
│   │   └── test/
│   │       └── test/ (66 test images)
│   └── processed_data/
│       └── processed_data.npz (8.9 GB - included in Google Drive package)
├── models/
│   ├── best_model_colab.keras (65.6 MB - included in Google Drive package)
│   └── class_mapping.json
└── ... (other project files)
```

## 🔄 Data Processing

### Google Drive Package (Recommended)
✅ **No processing needed!** The Google Drive package includes:
- Complete dataset (1.3 GB)
- Pre-processed data (8.9 GB) 
- Trained models (65.6 MB)

### Manual Setup Only
If you downloaded manually, generate processed data:

```bash
python scripts/data_processing/process_plantvillage_dataset.py
```

**⚠️ Processing Requirements:**
- **Time**: 30-45 minutes
- **Memory**: 8+ GB RAM recommended
- **Storage**: Additional 8.9 GB for processed data

The `processed_data.npz` file contains:
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