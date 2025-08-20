#!/usr/bin/env python3
"""
Test script for YOLO11m integration
"""

import os
import sys
from PIL import Image
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_yolo11m_loading():
    """Test if YOLO11m model can be loaded"""
    print("Testing YOLO11m model loading...")
    
    try:
        from utils.models import load_models, get_model_status
        
        # Load models
        models = load_models()
        
        # Check model status
        status = get_model_status(models)
        
        print("Model Status:")
        for model_name, is_available in status.items():
            print(f"  {model_name}: {'✅ Available' if is_available else '❌ Not Available'}")
        
        # Check if YOLO11m is available
        if models.get('yolo_model'):
            print("✅ YOLO11m model loaded successfully!")
            return True
        else:
            print("❌ YOLO11m model failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def test_yolo11m_detection():
    """Test YOLO11m detection on a sample image"""
    print("\nTesting YOLO11m detection...")
    
    try:
        from utils.models import load_models
        from utils.expert_food_recognition import YOLO11mFoodRecognitionSystem
        
        # Load models
        models = load_models()
        
        if not models.get('yolo_model'):
            print("❌ YOLO11m model not available")
            return False
        
        # Create a simple test image (you can replace this with an actual image)
        # For now, we'll create a blank image
        test_image = Image.new('RGB', (640, 480), color='white')
        
        # Initialize the recognition system
        yolo_system = YOLO11mFoodRecognitionSystem(models)
        
        # Run detection
        results = yolo_system.recognize_food(test_image)
        
        print(f"Detection Results:")
        print(f"  Success: {results.get('success', False)}")
        print(f"  Total Detections: {results.get('total_detections', 0)}")
        print(f"  Method: {results.get('method', 'Unknown')}")
        
        if results.get('success'):
            print("✅ YOLO11m detection test passed!")
            return True
        else:
            print(f"❌ YOLO11m detection failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during detection test: {e}")
        return False

def test_model_paths():
    """Test if yolo11m.pt file exists in expected locations"""
    print("\nTesting yolo11m.pt file locations...")
    
    possible_paths = [
        "yolo11m.pt",
        "../yolo11m.pt", 
        "../../yolo11m.pt",
        os.path.join(os.path.dirname(__file__), "yolo11m.pt"),
        os.path.join(os.path.dirname(__file__), "..", "yolo11m.pt"),
        os.path.join(os.path.dirname(__file__), "..", "..", "yolo11m.pt")
    ]
    
    found_paths = []
    for path in possible_paths:
        if os.path.exists(path):
            found_paths.append(path)
            print(f"✅ Found yolo11m.pt at: {path}")
        else:
            print(f"❌ Not found: {path}")
    
    if found_paths:
        print(f"✅ yolo11m.pt found in {len(found_paths)} location(s)")
        return True
    else:
        print("❌ yolo11m.pt not found in any expected location")
        return False

def main():
    """Run all tests"""
    print("🧪 YOLO11m Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Check if yolo11m.pt exists
    path_test = test_model_paths()
    
    # Test 2: Test model loading
    loading_test = test_yolo11m_loading()
    
    # Test 3: Test detection (only if loading passed)
    detection_test = False
    if loading_test:
        detection_test = test_yolo11m_detection()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"  Model File Check: {'✅ PASS' if path_test else '❌ FAIL'}")
    print(f"  Model Loading: {'✅ PASS' if loading_test else '❌ FAIL'}")
    print(f"  Detection Test: {'✅ PASS' if detection_test else '❌ FAIL'}")
    
    if path_test and loading_test and detection_test:
        print("\n🎉 All tests passed! YOLO11m integration is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
