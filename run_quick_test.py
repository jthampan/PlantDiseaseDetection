"""
Quick comprehensive test script - runs automatically without user interaction
"""

import sys
import os
from pathlib import Path

# Add the scripts path
sys.path.append('scripts/inference')

def run_comprehensive_test():
    """Run comprehensive test automatically."""
    
    print("🚀 AUTOMATIC COMPREHENSIVE TEST")
    print("=" * 60)
    
    try:
        # Import the prediction class
        from predict_disease import PlantVillageAccuracyTest
        
        # Initialize predictor
        print("🔄 Initializing predictor...")
        predictor = PlantVillageAccuracyTest()
        print("✅ Predictor initialized successfully!")
        
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
    success = run_comprehensive_test()
    
    if success:
        print("\n🎊 COMPREHENSIVE TESTING COMPLETED SUCCESSFULLY!")
        print("\n📋 WHAT TO DO NEXT:")
        print("1. 📈 Check the plots in outputs/plots/ directory")
        print("2. 📊 Review the accuracy results above")
        print("3. 🔍 Analyze which diseases perform best/worst")
        print("4. 🖼️ Look at sample correct/wrong predictions")
    else:
        print("\n❌ TESTING FAILED!")