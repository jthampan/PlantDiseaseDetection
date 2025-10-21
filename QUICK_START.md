# Quick Usage Guide

## � Prerequisites & Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)
- At least 4GB of free disk space for the dataset

### Installation Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/jthampan/PlantDiseaseDetection.git
cd PlantDiseaseDetection
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv plant_disease_env

# Activate virtual environment
# On Windows:
plant_disease_env\Scripts\activate
# On macOS/Linux:
source plant_disease_env/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Download Dataset
Follow the instructions in `DATASET_SETUP.md` to download and set up the PlantVillage dataset.

## �🚀 Getting Started in 3 Steps

### Step 1: Check Everything is Working
```bash
python main.py --mode check
```
This will verify:
- ✅ Python 3.11 and TensorFlow 2.20 are installed
- ✅ Required directories exist
- ✅ Trained model is available (93.9% accuracy)

### Step 2: Test a Prediction
```bash
python main.py --mode predict --image "path/to/your/plant_image.jpg"
```
Example output:
```
🎯 Prediction: Tomato: Early blight
📊 Confidence: 94.2%
🏆 Top 3:
   1. Tomato: Early blight: 94.2%
   2. Tomato: Late blight: 3.8%
   3. Tomato: Septoria leaf spot: 1.2%
```

### Step 3: Use Interactive Mode
```bash
python main.py
```
This opens a menu with all options:
```
🌱 Plant Disease Detection System
========================================
Choose an option:
1. 🔍 Check Environment
2. 📊 Process Dataset
3. 🏋️ Train Model (Optional)
4. 🔮 Predict Single Image
5. 📈 Test Model Accuracy
6. � Comprehensive Test with Plots
7. �🚀 Run Complete Pipeline
0. ❌ Exit
```

## 📸 Supported Image Types
- **Formats**: JPG, JPEG, PNG
- **Size**: Any size (automatically resized to 224x224)
- **Color**: RGB (automatically converted if needed)

## 🎯 What the System Can Detect
The model can identify **38 different conditions** across **14 plant types**:

### 🍎 Apple (4 conditions)
- Apple scab, Black rot, Cedar apple rust, Healthy

### 🌽 Corn/Maize (4 conditions)  
- Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy

### 🍇 Grape (4 conditions)
- Black rot, Esca, Leaf blight, Healthy

### 🍅 Tomato (10 conditions)
- Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Tomato mosaic virus, Yellow Leaf Curl Virus, Healthy

### 🌶️ Other Plants
- Bell Pepper, Potato, Peach, Orange, Cherry, Blueberry, Raspberry, Soybean, Squash, Strawberry

## 📊 Comprehensive Testing with Plots

For detailed analysis of your model's performance, use the comprehensive testing feature:

```bash
python main.py
# Then select option 6: "📋 Comprehensive Test with Plots"
```

This will:
- Test the model on all available test images
- Generate detailed accuracy statistics
- Create comprehensive visualization plots:
  - **Accuracy Summary**: Overall correct vs wrong predictions
  - **Confusion Matrix**: Detailed class-by-class performance
  - **Per-Class Accuracy**: Individual accuracy for each disease
  - **Confidence Distribution**: Confidence scores for correct vs wrong predictions
  - **Sample Predictions**: Visual examples of correct and wrong predictions
  - **Confusion Analysis**: Most commonly confused disease pairs

All plots are automatically saved to `outputs/plots/` directory.

### Sample Output:
```
📊 Test Results Summary:
   Total images tested: 1,847
   Correct predictions: 1,735
   Wrong predictions: 112
   Overall accuracy: 93.9%

📈 Generating comprehensive plots...
   ✅ Accuracy summary plot saved
   ✅ Confusion matrix plot saved
   ✅ Per-class accuracy plot saved
   ✅ Confidence distribution plot saved
   ✅ Correct predictions samples saved
   ✅ Wrong predictions samples saved
   ✅ Confusion analysis plot saved
✅ All plots saved to: outputs/plots/
```

## ⚡ Performance
- **Accuracy**: 93.9% on test dataset
- **Speed**: ~100ms per prediction (CPU)
- **Memory**: ~2GB RAM required
- **Confidence**: Always shows prediction confidence percentage

## 🔧 Troubleshooting

### "Model not found" error
```bash
# Check if model exists
ls models/best_model_colab.keras

# If missing, you may need to retrain or download the model
```

### "TensorFlow not found" error
```bash
# Install TensorFlow
pip install tensorflow>=2.8.0
```

### Import errors
```bash
# Make sure you're in the project directory
cd "c:\Kaplan\BDMLA\Final_plantDisease_Project"

# Check Python version
python --version  # Should be 3.11.x
```

## 💡 Pro Tips

1. **Best image quality**: Use clear, well-lit photos of plant leaves
2. **Single leaf focus**: Avoid images with multiple plants or objects
3. **Disease visibility**: Ensure disease symptoms are clearly visible
4. **File formats**: JPG and PNG work best
5. **Confidence threshold**: Predictions above 80% confidence are most reliable

## 🎯 Example Workflow

```bash
# 1. Check system
python main.py --mode check

# 2. Test with a sample image
python main.py --mode predict --image "data/plant_diseases/test/Apple___Apple_scab/image_001.jpg"

# 3. Run comprehensive accuracy test with plots
python main.py  # Select option 6: Comprehensive Test with Plots

# 4. Use interactive mode for multiple predictions
python main.py
```

## 📊 Expected Results
With the pre-trained model, you should see:
- **High accuracy** predictions (typically 85-95% confidence)
- **Fast processing** (under 1 second per image)
- **Detailed results** with top 3 predictions
- **Clean disease names** (formatted for readability)

Ready to start? Run `python main.py` and select option 4 to predict your first plant disease! 🌱