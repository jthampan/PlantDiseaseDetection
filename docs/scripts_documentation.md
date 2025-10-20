# Plant Disease Detection Scripts Documentation

## Overview
This project contains a complete pipeline for plant disease detection using deep learning with the PlantVillage dataset. The system can classify 38 different plant diseases across multiple plant species with up to 93.9% accuracy.

## Scripts Structure

### 1. Data Processing (`scripts/data_processing/`)
- **`process_plantvillage_dataset.py`** - Comprehensive data preprocessing pipeline

### 2. Training (`scripts/training/`)
- **`colab_train_model.py`** - Training script optimized for Google Colab

### 3. Inference (`scripts/inference/`)
- **`predict_disease.py`** - Model inference and prediction script

## Quick Start Guide

### Prerequisites
- Python 3.11
- TensorFlow 2.20
- Required packages from `requirements.txt`

### Installation
1. Navigate to project directory:
   ```bash
   cd "c:\Kaplan\BDMLA\Final_plantDisease_Project"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Scripts

#### 1. Data Processing
```bash
python scripts/data_processing/process_plantvillage_dataset.py
```
This will:
- Load and analyze the PlantVillage dataset
- Create optimized train/validation/test splits
- Save processed data as `processed_data.npz`
- Generate class mapping and visualization

#### 2. Model Training (if needed)
```bash
python scripts/training/colab_train_model.py
```
Note: You already have a trained model (`models/best_model_colab.keras`)

#### 3. Making Predictions
```bash
python scripts/inference/predict_disease.py
```

## Performance Metrics
- **Training Accuracy**: >95%
- **Validation Accuracy**: >90%
- **Test Accuracy**: 93.9%
- **Classes**: 38 plant diseases
- **Input Size**: 224x224x3 RGB images

## Dataset Information
The PlantVillage dataset contains:
- **87,867 high-quality images**
- **38 classes** covering multiple plant species and diseases
- **Well-balanced distribution** (1.23:1 ratio)
- **Professional annotations**

### Supported Plants
- Apple (4 conditions)
- Corn/Maize (4 conditions) 
- Grape (4 conditions)
- Tomato (10 conditions)
- Pepper, Bell (2 conditions)
- Potato (3 conditions)
- And many more...

## Model Architecture
- **Base**: Enhanced CNN optimized for 38 classes
- **Input Shape**: (224, 224, 3)
- **Output Shape**: (38,) - Softmax classification
- **Optimization**: Mixed precision training
- **Framework**: TensorFlow 2.20/Keras

## File Structure
```
Final_plantDisease_Project/
├── data/plant_diseases/          # Dataset location
├── models/                       # Trained models
│   ├── best_model_colab.keras   # Main trained model
│   └── class_mapping.json       # Class index mapping
├── scripts/
│   ├── data_processing/
│   ├── training/
│   └── inference/
└── outputs/                      # Generated outputs
```

## Troubleshooting

### Common Issues
1. **Model loading errors**: Ensure paths are correct in prediction script
2. **Memory issues**: Use memory management functions in training script
3. **Dataset not found**: Verify data is in `data/plant_diseases/`

### GPU Support
The scripts support GPU acceleration if available:
- CUDA-compatible GPU recommended for training
- Automatic fallback to CPU if GPU unavailable

## Next Steps
1. Use the prediction script to test on new plant images
2. Fine-tune the model with additional data if needed
3. Deploy the model for production use
4. Extend to additional plant species