# Google Colab Training Setup Guide

## 🎉 Quick Start: Google Drive Integration Available!

**✅ Complete Package Ready!** The dataset is now available via Google Drive for immediate Colab use.

**Current Package**: `plant_disease_complete_20251021_084958.zip` (10.3 GB)
- ✅ Complete dataset (87,900 images)
- ✅ Pre-trained models (93.9% accuracy)
- ✅ Pre-processed data (saves 30+ minutes)
- ✅ All training scripts included

---

## 🚀 Option 1: Use Google Drive Package (Recommended)

### Step 1: Access the Complete Package
The complete package is already uploaded to Google Drive:
- **File**: `plant_disease_complete_20251021_084958.zip`
- **Size**: 10.3 GB
- **Contents**: Everything needed for Colab training

### Step 2: Google Colab Setup
```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Navigate to your package
import os
os.chdir('/content/drive/MyDrive')  # Adjust path as needed

# 3. Extract the complete package
import zipfile
with zipfile.ZipFile('plant_disease_complete_20251021_084958.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

# 4. Verify extraction
print("📁 Contents extracted to /content/:")
for item in ['data', 'models', 'scripts']:
    if os.path.exists(f'/content/{item}'):
        print(f"✅ {item}/ - Ready")
    else:
        print(f"❌ {item}/ - Missing")
```

### Step 3: Start Training
```python
# Navigate to training directory
os.chdir('/content')

# Install dependencies
!pip install tensorflow opencv-python scikit-learn matplotlib pillow

# Run training
!python scripts/training/colab_train_model.py
```

**⚡ Advantages:**
- ✅ **No file preparation needed** - everything included
- ✅ **Pre-processed data included** - saves 30+ minutes
- ✅ **Pre-trained models available** - for comparison or fine-tuning
- ✅ **Immediate start** - no manual dataset downloads

---

## 📦 Option 2: Custom Package Creation (Advanced)

### If You Need a Custom Package

### Create Custom Packages (If Needed)
```bash
# Use the built-in package creator for custom needs

# Models only (65 MB - quick upload)
python create_data_package.py --type models

# Dataset only (1.3 GB - for retraining)
python create_data_package.py --type dataset

# Complete package (10.3 GB - full backup)
python create_data_package.py --type complete
```

### Required Components for Training:
1. **Training Script**: `colab_train_model.py`
2. **Data Processing Script**: `process_plantvillage_dataset.py` 
3. **Dataset**: PlantVillage dataset (train/validation folders)
4. **Processed Data**: Pre-processed data files (saves 30+ minutes)
5. **Dependencies**: Requirements list

### Package Structure (Auto-generated)
```
plant_disease_package.zip
├── 📁 scripts/
│   ├── 📁 training/
│   │   └── colab_train_model.py      # Main training script
│   └── 📁 data_processing/
│       └── process_plantvillage_dataset.py  # Data preprocessing
├── 📁 data/
│   ├── 📁 plant_diseases/
│   │   ├── 📁 train/                 # Training images (87,900 files)
│   │   │   ├── Apple___Apple_scab/
│   │   │   ├── Apple___Black_rot/
│   │   │   └── ... (all 38 classes)
│   │   └── 📁 validation/            # Validation images
│   │       ├── Apple___Apple_scab/
│   │       └── ... (all 38 classes)
│   └── 📁 processed_data/            # Pre-processed features (8.9 GB)
│       └── processed_data.npz
├── 📁 models/
│   ├── best_model_colab.keras        # Pre-trained model (93.9% accuracy)
│   └── class_mapping.json            # Class labels
├── 📄 requirements.txt               # Python dependencies
└── 📄 manifest.json                 # Package information
---

## ⚙️ File Size Considerations

### Current Google Drive Package:
- **Complete Package**: 10.3 GB (everything included)
- **Models Only**: 65.6 MB (for quick sharing)
- **Dataset Only**: 1.3 GB (for retraining)
- **Processed Data Only**: 8.9 GB (advanced preprocessing)

### Upload Strategy:
1. **Google Drive Package** ✅ **Recommended** - Complete setup, no preparation needed
2. **Custom packages** - Use `create_data_package.py` for specific needs
3. **Models only** - For quick model testing and deployment

---

## 🚀 Detailed Colab Setup Process

### Method 1: Google Drive Package (Recommended)

#### Step 1: Mount Google Drive
```python
# In Colab cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

#### Step 2: Extract Complete Package
```python
# In Colab cell 2: Extract the complete package
import zipfile
import os

# Extract the complete package (adjust path as needed)
zip_path = '/content/drive/MyDrive/plant_disease_complete_20251021_084958.zip'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/')

# Verify extraction
print("✅ Package extracted successfully!")
print("\n📁 Available directories:")
for item in ['data', 'models', 'scripts']:
    if os.path.exists(f'/content/{item}'):
        print(f"✅ {item}/ - Ready")
    else:
        print(f"❌ {item}/ - Missing")

# Change to project directory
os.chdir('/content/')
```

#### Step 3: Install Dependencies & Start Training
```python
# In Colab cell 3: Install required packages
!pip install tensorflow opencv-python scikit-learn matplotlib pillow seaborn

# Verify GPU is available
import tensorflow as tf
print(f"🚀 GPU Available: {tf.test.is_gpu_available()}")
print(f"📊 GPU Device: {tf.config.list_physical_devices('GPU')}")

# Start training
!python scripts/training/colab_train_model.py
```

### Method 2: Custom Package Upload

#### If Using Custom Package
```python
# In Colab cell 2: Extract custom zip file (adjust filename)
zip_path = '/content/drive/MyDrive/your_custom_package.zip'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/')

# Check contents and proceed with training setup
```

---

## 📋 Pre-Training Checklist

### ✅ Google Drive Package (Recommended):
- [x] **Complete package uploaded** (`plant_disease_complete_20251021_084958.zip`)
- [x] **All components included** (dataset + models + processed data)
- [x] **Ready for immediate use** - no preparation needed

### ⚙️ Custom Package Checklist:
- [ ] **Package created** using `create_data_package.py`
- [ ] **Uploaded to Google Drive** 
- [ ] **File paths verified** for Colab compatibility
- [ ] **Package size** appropriate for your needs

### 🖥️ Colab Environment Setup:
- [ ] **GPU runtime selected** (Runtime → Change runtime type → GPU)
- [ ] **High-RAM option** enabled if available
- [ ] **Google Drive mounted** successfully
- [ ] **Package extracted** without errors

---

## 🔧 Troubleshooting Common Issues

### Package Extraction Issues:
```python
# If extraction fails, check file path and permissions
import os
zip_path = '/content/drive/MyDrive/plant_disease_complete_20251021_084958.zip'
if os.path.exists(zip_path):
    print(f"✅ File found: {zip_path}")
    print(f"📊 File size: {os.path.getsize(zip_path) / (1024**3):.1f} GB")
else:
    print(f"❌ File not found: {zip_path}")
    print("📁 Available files:")
    !ls /content/drive/MyDrive/
```

### Memory Management:
```python
# Monitor GPU memory in Colab
!nvidia-smi

# Clear memory if needed
import gc
gc.collect()
```

# Clear memory if needed
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

### Path Issues:
```python
# Check current directory and files
import os
print("Current directory:", os.getcwd())
print("Files:", os.listdir('.'))
```

## 📊 Expected Training Process

### Training Timeline:
- **Setup**: 5-10 minutes (upload + extract)
- **Data Processing**: 10-15 minutes (if not pre-processed)
- **Training**: 2-4 hours (60 epochs on GPU)
- **Total**: 3-5 hours

### Expected Outputs:
- **Trained Model**: `best_model_colab.keras`
- **Training Logs**: Loss and accuracy curves
- **Class Mapping**: `class_mapping.json`
- **Performance Metrics**: Accuracy reports

## 💡 Pro Tips

### For Faster Setup:
1. **Pre-process data locally** - Include `processed_data.npz` in zip
2. **Use Google Drive sync** - Keep files accessible across sessions
3. **Save checkpoints** - Resume training if disconnected

### For Better Results:
1. **Verify dataset quality** - Check for corrupted images
2. **Monitor training progress** - Use TensorBoard or manual logging
3. **Save intermediate models** - Don't lose progress

### For Colab Limitations:
1. **Session timeouts** - Colab disconnects after ~12 hours
2. **Storage limits** - Clean up large files when done
3. **GPU quotas** - Use GPU efficiently, free tier has limits

## 🎯 Next Steps

After creating your zip file:
1. **Upload to Google Drive**
2. **Follow Colab setup steps**
3. **Monitor training progress**
4. **Download trained model**
5. **Test model performance**

Use the provided `create_colab_package.py` script to automatically generate the zip file with all necessary components!