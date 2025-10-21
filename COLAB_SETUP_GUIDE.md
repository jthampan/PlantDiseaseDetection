# Google Colab Training Setup Guide

## 📦 Preparing Files for Colab Training

To train the plant disease detection model in Google Colab, you need to upload a zip file containing all necessary components. This guide explains what to include and how to set it up.

## 🎯 What You Need for Colab Training

### Required Components:
1. **Training Script**: `colab_train_model.py`
2. **Data Processing Script**: `process_plantvillage_dataset.py` 
3. **Dataset**: PlantVillage dataset (train/validation folders)
4. **Processed Data**: Pre-processed data files (optional, saves time)
5. **Dependencies**: Requirements list

### Optional Components:
- **Previous Models**: For transfer learning or fine-tuning
- **Class Mappings**: Pre-defined class labels
- **Documentation**: Setup and usage guides

## 📁 Recommended Zip File Structure

```
plant_disease_colab.zip
├── 📁 scripts/
│   ├── 📁 training/
│   │   └── colab_train_model.py      # Main training script
│   └── 📁 data_processing/
│       └── process_plantvillage_dataset.py  # Data preprocessing
├── 📁 data/
│   └── 📁 plant_diseases/
│       ├── 📁 train/                 # Training images
│       │   ├── Apple___Apple_scab/
│       │   ├── Apple___Black_rot/
│       │   └── ... (all 38 classes)
│       └── 📁 validation/            # Validation images
│           ├── Apple___Apple_scab/
│           └── ... (all 38 classes)
├── 📁 models/ (optional)
│   ├── processed_data.npz           # Pre-processed data (saves time)
│   └── class_mapping.json           # Class labels
├── 📄 requirements.txt              # Python dependencies
├── 📄 COLAB_SETUP.md               # Colab setup instructions
└── 📄 README_COLAB.md              # Quick start guide
```

## ⚙️ File Size Considerations

### Large Files to Consider:
- **Dataset**: ~2-3GB (87,867 images)
- **Processed Data**: ~900MB (if pre-processed)
- **Total Zip**: ~3-4GB

### Optimization Options:
1. **Include processed data** - Faster training start (larger zip)
2. **Dataset only** - Process in Colab (smaller zip, longer setup)
3. **Subset for testing** - Small sample for quick experiments

## 🚀 Colab Upload Process

### Step 1: Upload to Google Drive
```python
# In Colab cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Extract Files
```python
# In Colab cell 2: Extract your zip file
import zipfile
import os

# Extract the uploaded zip file
with zipfile.ZipFile('/content/drive/MyDrive/plant_disease_colab.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

# Change to project directory
os.chdir('/content/')
```

### Step 3: Install Dependencies
```python
# In Colab cell 3: Install required packages
!pip install -r requirements.txt
```

### Step 4: Run Training
```python
# In Colab cell 4: Start training
from scripts.training.colab_train_model import main_colab_training
main_colab_training()
```

## 📋 Pre-Training Checklist

### Before Creating Zip:
- [ ] **Dataset downloaded** and organized in train/validation folders
- [ ] **Training script tested** locally (optional)
- [ ] **Dependencies verified** in requirements.txt
- [ ] **File paths checked** for Colab compatibility
- [ ] **Zip file size** under Google Drive limits

### Colab Environment Check:
- [ ] **GPU runtime** selected (Runtime → Change runtime type → GPU)
- [ ] **Sufficient storage** (~15GB recommended)
- [ ] **Google Drive space** for zip file and outputs
- [ ] **Stable internet** for upload and training

## 🔧 Troubleshooting Common Issues

### Large File Issues:
```python
# If zip file too large, split dataset processing:
# 1. Upload scripts only first
# 2. Download dataset directly in Colab:
!wget -O plant_data.zip "https://your-dataset-url"
!unzip plant_data.zip
```

### Memory Issues:
```python
# Monitor GPU memory in Colab
!nvidia-smi

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