"""
Plant Disease Detection - Main Runner Script
===========================================

This script provides a unified interface to run all plant disease detection tasks:
- Data processing
- Model training (optional)
- Model prediction
- Accuracy testing

Compatible with Python 3.11 and TensorFlow 2.20
Achieves 93.9% accuracy on PlantVillage dataset
"""

import os
import sys
import argparse
from pathlib import Path
import json

# Add script directories to path
sys.path.append('scripts/data_processing')
sys.path.append('scripts/training') 
sys.path.append('scripts/inference')

def check_environment():
    """Check if the environment is properly set up."""
    print("🔍 Environment Check")
    print("=" * 30)
    
    # Check Python version
    import sys
    print(f"📊 Python: {sys.version}")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"📊 TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not found! Install with: pip install tensorflow")
        return False
    
    # Check required directories
    required_dirs = ['data/plant_diseases', 'models', 'outputs']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"⚠️ Directory missing: {dir_path}")
        else:
            print(f"✅ Directory found: {dir_path}")
    
    # Check for trained model
    model_path = "models/best_model_colab.keras"
    if os.path.exists(model_path):
        print(f"✅ Trained model found: {model_path}")
    else:
        print(f"⚠️ Trained model not found: {model_path}")
        
    print("✅ Environment check completed\n")
    return True

def process_data():
    """Run data processing pipeline."""
    print("🚀 Starting Data Processing")
    print("=" * 40)
    
    try:
        from process_plantvillage_dataset import main as process_main
        results = process_main()
        
        if results:
            print("✅ Data processing completed successfully!")
            return True
        else:
            print("❌ Data processing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return False

def train_model():
    """Run model training (optional)."""
    print("🚀 Starting Model Training")
    print("=" * 40)
    
    print("⚠️ Note: You already have a trained model with 93.9% accuracy.")
    print("Training a new model will take several hours and requires significant computational resources.")
    
    response = input("Do you want to proceed with training? (y/N): ").lower()
    
    if response == 'y':
        try:
            from colab_train_model import main_colab_training
            results = main_colab_training()
            
            if results:
                print("✅ Model training completed successfully!")
                return True
            else:
                print("❌ Model training failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error in model training: {e}")
            return False
    else:
        print("⏭️ Skipping model training")
        return True

def predict_single_image(image_path):
    """Make prediction on a single image."""
    print(f"🔍 Predicting Disease for: {Path(image_path).name}")
    print("=" * 50)
    
    try:
        from predict_disease import PlantVillageAccuracyTest
        
        predictor = PlantVillageAccuracyTest()
        disease, confidence = predictor.predict_image(image_path)
        
        if disease:
            print(f"\n🎉 Prediction Results:")
            print(f"🎯 Disease: {disease.replace('___', ': ').replace('_', ' ')}")
            print(f"📊 Confidence: {confidence:.1%}")
            return True
        else:
            print("❌ Prediction failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return False

def test_model_accuracy():
    """Test model accuracy on test dataset."""
    print("📊 Testing Model Accuracy")
    print("=" * 30)
    
    try:
        from predict_disease import PlantVillageAccuracyTest
        
        predictor = PlantVillageAccuracyTest()
        
        # Ask user which type of test to run
        print("Choose test type:")
        print("1. Quick accuracy test")
        print("2. Comprehensive test with plots")
        
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "2":
            print("\n🚀 Running comprehensive test with plots...")
            results = predictor.test_accuracy_comprehensive()
            
            if results:
                print(f"\n🎉 Comprehensive Test Results:")
                print(f"📊 Overall Accuracy: {results['accuracy']:.1%}")
                print(f"📊 Total Images Tested: {results['total_tested']}")
                print(f"✅ Correct Predictions: {results['correct']}")
                print(f"❌ Wrong Predictions: {results['wrong']}")
                print(f"📈 Plots saved to: outputs/plots/")
                
                # List generated plots
                plot_dir = Path("outputs/plots")
                if plot_dir.exists():
                    plot_files = list(plot_dir.glob("*.png"))
                    if plot_files:
                        print(f"\n📁 Generated Plots:")
                        for plot_file in plot_files:
                            print(f"   📄 {plot_file.name}")
        else:
            print("\n🚀 Running quick accuracy test...")
            predictor.test_accuracy()
        
        print("✅ Accuracy testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in accuracy testing: {e}")
        return False

def interactive_menu():
    """Display interactive menu for user selection."""
    while True:
        print("\n🌱 Plant Disease Detection System")
        print("=" * 40)
        print("Choose an option:")
        print("1. 🔍 Check Environment")
        print("2. 📊 Process Dataset")
        print("3. 🏋️ Train Model (Optional)")
        print("4. 🔮 Predict Single Image")
        print("5. 📈 Test Model Accuracy")
        print("6. � Comprehensive Test with Plots")
        print("7. �🚀 Run Complete Pipeline")
        print("0. ❌ Exit")
        
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            check_environment()
        elif choice == '2':
            process_data()
        elif choice == '3':
            train_model()
        elif choice == '4':
            image_path = input("Enter path to image file: ").strip()
            if os.path.exists(image_path):
                predict_single_image(image_path)
            else:
                print(f"❌ File not found: {image_path}")
        elif choice == '5':
            test_model_accuracy()
        elif choice == '6':
            # Comprehensive test with plots
            print("🚀 Running Comprehensive Test with Plots")
            print("=" * 50)
            try:
                from predict_disease import PlantVillageAccuracyTest
                predictor = PlantVillageAccuracyTest()
                results = predictor.test_accuracy_comprehensive()
                
                if results:
                    print(f"\n🎉 Comprehensive Test Completed!")
                    print(f"📊 Overall Accuracy: {results['accuracy']:.1%}")
                    print(f"📊 Total Images: {results['total_tested']}")
                    print(f"✅ Correct: {results['correct']}")
                    print(f"❌ Wrong: {results['wrong']}")
                    print(f"📈 All plots saved to: outputs/plots/")
                    
                    # List generated plots
                    plot_dir = Path("outputs/plots")
                    if plot_dir.exists():
                        plot_files = list(plot_dir.glob("*.png"))
                        if plot_files:
                            print(f"\n📁 Generated {len(plot_files)} plot files:")
                            for plot_file in plot_files:
                                print(f"   📄 {plot_file.name}")
                else:
                    print(f"\n❌ Comprehensive test failed!")
                    print(f"Please check if test images are available in:")
                    print(f"   - data/plant_diseases/test/ (organized in class folders)")
                    print(f"   - data/plant_diseases/test/test/ (flat structure with filenames)")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
        elif choice == '7':
            run_complete_pipeline()
        else:
            print("❌ Invalid choice! Please enter 0-7.")

def run_complete_pipeline():
    """Run the complete pipeline from start to finish."""
    print("🚀 Running Complete Pipeline")
    print("=" * 40)
    
    # Step 1: Environment check
    if not check_environment():
        print("❌ Environment check failed! Please fix issues before continuing.")
        return False
    
    # Step 2: Data processing (optional - only if not already done)
    processed_data_path = "data/plant_diseases/processed_data.npz"
    if not os.path.exists(processed_data_path):
        print("📊 Processed data not found. Running data processing...")
        if not process_data():
            print("❌ Pipeline failed at data processing step!")
            return False
    else:
        print("✅ Processed data already exists, skipping data processing")
    
    # Step 3: Check for trained model
    model_path = "models/best_model_colab.keras"
    if not os.path.exists(model_path):
        print("🏋️ Trained model not found. Training is required...")
        if not train_model():
            print("❌ Pipeline failed at training step!")
            return False
    else:
        print("✅ Trained model found, skipping training")
    
    # Step 4: Test model accuracy
    print("📈 Testing model accuracy...")
    if not test_model_accuracy():
        print("❌ Pipeline failed at accuracy testing step!")
        return False
    
    print("\n🎉 Complete pipeline executed successfully!")
    print("✅ System is ready for predictions!")
    
    # Offer to make a prediction
    response = input("\nWould you like to test with an image? (y/N): ").lower()
    if response == 'y':
        image_path = input("Enter path to image file: ").strip()
        if os.path.exists(image_path):
            predict_single_image(image_path)
    
    return True

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Plant Disease Detection System')
    parser.add_argument('--mode', choices=['check', 'process', 'train', 'predict', 'test', 'pipeline', 'interactive'],
                       default='interactive', help='Operation mode')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    
    args = parser.parse_args()
    
    print("🌱 Plant Disease Detection System")
    print(f"📊 Python 3.11 + TensorFlow 2.20")
    print(f"🎯 Expected Accuracy: 93.9%")
    print("=" * 60)
    
    if args.mode == 'check':
        check_environment()
    elif args.mode == 'process':
        process_data()
    elif args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        if args.image:
            if os.path.exists(args.image):
                predict_single_image(args.image)
            else:
                print(f"❌ Image file not found: {args.image}")
        else:
            print("❌ Please provide --image argument for prediction mode")
    elif args.mode == 'test':
        test_model_accuracy()
    elif args.mode == 'pipeline':
        run_complete_pipeline()
    elif args.mode == 'interactive':
        interactive_menu()
    else:
        print(f"❌ Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()