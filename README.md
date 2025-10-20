# Plant Disease Classification Project

## Overview
This project implements a state-of-the-art deep learning solution for plant disease classification using convolutional neural networks (CNNs). The system can identify various plant diseases from leaf images across multiple plant species with **93.9% accuracy**.

## 🚀 Quick Start

### Prerequisites
- **Python 3.11**
- **TensorFlow 2.20** 
- Windows/Linux/macOS

### Installation
1. Navigate to the project directory:
   ```bash
   cd "c:\Kaplan\BDMLA\Final_plantDisease_Project"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

#### Option 1: Interactive Mode (Recommended)
```bash
python main.py
```
This opens an interactive menu with all available options.

#### Option 2: Command Line Usage
```bash
# Check environment
python main.py --mode check

# Make a prediction
python main.py --mode predict --image "path/to/your/plant_image.jpg"

# Test model accuracy
python main.py --mode test

# Run complete pipeline
python main.py --mode pipeline
```

#### Option 3: Direct Script Usage
```bash
# Data processing
python scripts/data_processing/process_plantvillage_dataset.py

# Make predictions
python scripts/inference/predict_disease.py

# Training (optional - model already trained)
python scripts/training/colab_train_model.py
```

## 📊 Performance Metrics
- **Training Accuracy**: >95%
- **Validation Accuracy**: >90%
- **Test Accuracy**: **93.9%**
- **Classes**: 38 plant diseases and healthy conditions
- **Dataset**: PlantVillage (87,867 high-quality images)

## 🌱 Supported Plants and Diseases

### Apple (4 conditions)
- Apple scab
- Black rot  
- Cedar apple rust
- Healthy

### Corn/Maize (4 conditions)
- Cercospora leaf spot
- Common rust
- Northern Leaf Blight
- Healthy

### Grape (4 conditions)
- Black rot
- Esca (Black Measles)
- Leaf blight (Isariopsis Leaf Spot)
- Healthy

### Tomato (10 conditions)
- Bacterial spot
- Early blight
- Late blight
- Leaf Mold
- Septoria leaf spot
- Spider mites (Two-spotted spider mite)
- Target Spot
- Tomato mosaic virus
- Tomato Yellow Leaf Curl Virus
- Healthy

### Other Plants
- **Blueberry**: Healthy
- **Cherry**: Powdery mildew, Healthy
- **Orange**: Huanglongbing (Citrus greening)
- **Peach**: Bacterial spot, Healthy
- **Pepper (Bell)**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery mildew
- **Strawberry**: Leaf scorch, Healthy

## 📁 Project Structure
```
Final_plantDisease_Project/
├── data/
│   └── plant_diseases/          # Dataset (train/validation/test)
├── scripts/
│   ├── data_processing/         # Data preprocessing scripts
│   ├── training/               # Model training scripts
│   └── inference/              # Prediction scripts
├── models/
│   ├── best_model_colab.keras  # Trained model (93.9% accuracy)
│   └── class_mapping.json     # Class label mappings
├── docs/                      # Documentation
├── outputs/                   # Generated outputs
├── main.py                   # Main runner script
└── requirements.txt          # Dependencies
```

## 🔧 Main Features

### 1. Data Processing (`scripts/data_processing/`)
- **Automatic dataset loading** from PlantVillage format
- **Intelligent data splitting** with stratification
- **Quality analysis** and visualization
- **Memory-efficient processing** for large datasets

### 2. Model Training (`scripts/training/`)
- **GPU-accelerated training** with mixed precision
- **Advanced CNN architecture** optimized for 38 classes
- **Memory management** for Google Colab compatibility
- **Comprehensive monitoring** and logging

### 3. Inference (`scripts/inference/`)
- **Real-time prediction** on new images
- **Confidence scoring** and top-N results
- **Batch processing** capabilities
- **Easy integration** with web applications

### 4. Main Runner (`main.py`)
- **Interactive menu** system
- **Command-line interface** 
- **Complete pipeline** automation
- **Environment validation**

## 📈 Model Architecture
- **Base**: Enhanced CNN optimized for plant disease classification
- **Input**: 224×224×3 RGB images
- **Output**: 38-class softmax classification
- **Training**: Mixed precision with GPU acceleration
- **Framework**: TensorFlow 2.20/Keras

## 🚀 Usage Examples

### Making a Single Prediction
```python
from scripts.inference.predict_disease import PlantVillageAccuracyTest

# Initialize predictor
predictor = PlantVillageAccuracyTest()

# Make prediction
disease, confidence = predictor.predict_image("path/to/leaf_image.jpg")
print(f"Disease: {disease}")
print(f"Confidence: {confidence:.1%}")
```

### Processing New Dataset
```python
from scripts.data_processing.process_plantvillage_dataset import PlantVillageProcessor

# Initialize processor
config = {'data_dir': './data/plant_diseases', 'img_size': (224, 224)}
processor = PlantVillageProcessor(config)

# Run complete processing
results = processor.run_complete_processing()
```

## 📋 Requirements
See `requirements.txt` for complete dependency list. Key requirements:
- `tensorflow>=2.8.0`
- `numpy>=1.21.0`
- `opencv-python>=4.5.0`
- `scikit-learn>=1.0.0`
- `matplotlib>=3.4.0`
- `pillow>=8.3.0`

## 🔧 Troubleshooting

### Common Issues
1. **Model loading errors**: Ensure `models/best_model_colab.keras` exists
2. **Memory issues**: Close other applications, use CPU fallback
3. **Import errors**: Verify Python path and dependencies
4. **Dataset not found**: Check `data/plant_diseases/` structure

### GPU Support
- **CUDA**: Automatically detected if available
- **Memory management**: Built-in optimization for limited VRAM
- **CPU fallback**: Automatic fallback for systems without GPU

## 📖 Documentation
Detailed documentation available in `docs/`:
- `scripts_documentation.md` - Overview of all scripts
- `data_processing_guide.md` - Data processing details
- `prediction_guide.md` - Prediction and inference guide

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Add tests if applicable
5. Submit a pull request

## 📜 License
This project is licensed under the MIT License.

## 🏆 Achievements
- **93.9% accuracy** on PlantVillage test set
- **38 plant disease classes** supported
- **Production-ready** inference pipeline
- **Comprehensive documentation** and examples
- **Cross-platform compatibility**

## 📞 Support
For questions and support:
1. Check the documentation in `docs/` directory
2. Review troubleshooting section above
3. Open an issue with detailed error information

---
**Note**: This project uses a pre-trained model achieving 93.9% accuracy. The model was trained using Google Colab with GPU acceleration on the complete PlantVillage dataset.