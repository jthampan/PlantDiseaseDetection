"""
Test script for comprehensive plant disease accuracy testing
"""

import sys
import os
from pathlib import Path

# Add the scripts path
sys.path.append('scripts/inference')

def test_comprehensive_analysis():
    """Test the comprehensive analysis functionality."""
    
    print("🧪 Testing Comprehensive Analysis Functionality")
    print("=" * 60)
    
    try:
        # Import the prediction class
        from predict_disease import PlantVillageAccuracyTest
        
        # Initialize predictor
        print("🔄 Initializing predictor...")
        predictor = PlantVillageAccuracyTest()
        print("✅ Predictor initialized successfully!")
        
        # Check if test directory exists
        test_dir = Path("data/plant_diseases/test")
        if not test_dir.exists():
            print(f"❌ Test directory not found: {test_dir}")
            print("Please ensure test images are available in the correct directory structure")
            return False
        
        # Count test images
        test_classes = [d for d in test_dir.iterdir() if d.is_dir()]
        total_images = 0
        for class_dir in test_classes:
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            total_images += len(images)
            print(f"   📂 {class_dir.name}: {len(images)} images")
        
        print(f"📊 Total test images found: {total_images}")
        
        if total_images == 0:
            print("❌ No test images found!")
            return False
        
        # Ask user if they want to proceed
        proceed = input(f"\nProceed with comprehensive testing on {total_images} images? (y/N): ").lower()
        
        if proceed != 'y':
            print("⏭️ Test cancelled by user")
            return True
        
        # Run comprehensive test
        print("\n🚀 Running comprehensive accuracy test...")
        results = predictor.test_accuracy_comprehensive()
        
        if results:
            print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
            print(f"📊 Overall Accuracy: {results['accuracy']:.1%}")
            print(f"📊 Total Images Tested: {results['total_tested']}")
            print(f"✅ Correct Predictions: {results['correct']}")
            print(f"❌ Wrong Predictions: {results['wrong']}")
            
            # Check if plots were created
            plot_dir = Path("outputs/plots")
            if plot_dir.exists():
                plot_files = list(plot_dir.glob("*.png"))
                print(f"\n📈 Generated {len(plot_files)} plot files:")
                for plot_file in plot_files:
                    print(f"   📄 {plot_file.name}")
                    
                print(f"\n✅ All plots saved to: {plot_dir}")
            else:
                print("⚠️ Plot directory not found")
            
            return True
        else:
            print("❌ Test failed!")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the predict_disease.py script is in scripts/inference/")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Run test
    success = test_comprehensive_analysis()
    
    if success:
        print("\n🎊 COMPREHENSIVE TESTING FUNCTIONALITY VERIFIED!")
    else:
        print("\n❌ TESTING FAILED!")