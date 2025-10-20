# Prediction Script Documentation

## Script: `predict_disease.py`

### Purpose
Loads the trained PlantVillage model and performs plant disease prediction on new images. Achieves 93.9% accuracy on the test dataset.

### Features
- **High Accuracy**: 93.9% accuracy on PlantVillage test set
- **Real-time Prediction**: Fast inference on single images
- **Confidence Scores**: Provides prediction confidence percentages
- **Top-N Results**: Shows top 3 most likely diseases
- **Batch Testing**: Can test accuracy on entire test sets
- **Model Compatibility**: Works with TensorFlow 2.20 and Python 3.11

### Model Information
- **Input Shape**: (224, 224, 3) RGB images
- **Output Shape**: (38,) classification probabilities
- **Classes**: 38 plant diseases and healthy conditions
- **Framework**: TensorFlow/Keras

### Usage

#### Basic Prediction
```python
from scripts.inference.predict_disease import PlantVillageAccuracyTest

# Initialize predictor
predictor = PlantVillageAccuracyTest()

# Predict single image
predicted_class, confidence = predictor.predict_image("path/to/plant_image.jpg")
print(f"Disease: {predicted_class}, Confidence: {confidence:.1%}")
```

#### Command Line Usage
```bash
python scripts/inference/predict_disease.py
```

### Key Methods

#### `__init__()`
- Loads the trained model from `models/best_model_colab.keras`
- Loads class mapping from `models/class_mapping.json`
- Validates model architecture
- Reports system information

#### `predict_image(image_path)`
- Loads and preprocesses a single image
- Performs model inference
- Returns prediction and confidence
- Shows top 3 predictions

**Parameters:**
- `image_path` (str): Path to image file (JPG, PNG, JPEG)

**Returns:**
- `predicted_class` (str): Predicted disease name
- `confidence` (float): Prediction confidence (0-1)

#### `test_accuracy()`
- Tests model accuracy on all test images
- Calculates overall accuracy metrics
- Shows per-class performance
- Generates confusion matrix

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- RGB format (automatically converted)

### Image Preprocessing
1. **Load**: Using PIL Image
2. **Convert**: To RGB format
3. **Resize**: To 224x224 pixels
4. **Normalize**: Convert to float32
5. **Batch**: Add batch dimension

### Class Mapping
The model predicts one of 38 classes:
```json
{
  "0": "Apple___Apple_scab",
  "1": "Apple___Black_rot",
  "2": "Apple___Cedar_apple_rust",
  "3": "Apple___healthy",
  "4": "Blueberry___healthy",
  ...
  "37": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
}
```

### Performance Metrics
- **Overall Accuracy**: 93.9%
- **Inference Time**: ~100ms per image (CPU)
- **Memory Usage**: ~2GB for model loading
- **Batch Size**: Optimized for single image prediction

### Error Handling
- **File Not Found**: Clear error messages for missing files
- **Invalid Images**: Handles corrupted or invalid image files
- **Model Loading**: Detailed error reporting for model issues
- **Memory Management**: Graceful handling of memory constraints

### Example Output
```
🌱 PlantVillage Model Test - Python 3.11 + TensorFlow 2.20
============================================================
📊 Python version: 3.11.9
📊 TensorFlow version: 2.20.0

🔄 Loading Colab model...
✅ Model loaded successfully!
📊 Input shape: (None, 224, 224, 3)
📊 Output shape: (None, 38)
✅ Loaded 38 classes

🔍 Analyzing: tomato_leaf.jpg
🎯 Prediction: Tomato: Early blight
📊 Confidence: 94.2%
🏆 Top 3:
   1. Tomato: Early blight: 94.2%
   2. Tomato: Late blight: 3.8%
   3. Tomato: Septoria leaf spot: 1.2%
```

### Dependencies
```python
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image
import sys
```

### Configuration Requirements
- **TensorFlow**: Version 2.20+
- **Python**: Version 3.11
- **Model File**: `models/best_model_colab.keras`
- **Class Mapping**: `models/class_mapping.json`

### Troubleshooting

#### Model Loading Issues
```python
# Enable unsafe deserialization for Colab models
tf.keras.config.enable_unsafe_deserialization()
```

#### Path Issues
- Ensure model path is correct: `models/best_model_colab.keras`
- Verify class mapping exists: `models/class_mapping.json`
- Use absolute paths if relative paths fail

#### Memory Issues
- Close other applications to free RAM
- Reduce batch size for multiple predictions
- Use CPU inference if GPU memory is limited

### Integration Examples

#### Web Application Integration
```python
from flask import Flask, request, jsonify
from scripts.inference.predict_disease import PlantVillageAccuracyTest

app = Flask(__name__)
predictor = PlantVillageAccuracyTest()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file.save('temp_image.jpg')
    
    disease, confidence = predictor.predict_image('temp_image.jpg')
    
    return jsonify({
        'disease': disease,
        'confidence': float(confidence),
        'status': 'success'
    })
```

#### Batch Processing
```python
import os
from pathlib import Path

predictor = PlantVillageAccuracyTest()
image_dir = "path/to/images"

for image_file in Path(image_dir).glob("*.jpg"):
    disease, confidence = predictor.predict_image(str(image_file))
    print(f"{image_file.name}: {disease} ({confidence:.1%})")
```