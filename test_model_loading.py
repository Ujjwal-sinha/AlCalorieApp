#!/usr/bin/env python3
"""
Test script to verify model loading and food detection
"""

import os
import sys
from PIL import Image
import io
import base64

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

def test_model_loading():
    """Test if models can be loaded properly"""
    print("🧪 Testing Model Loading")
    print("========================")
    
    # Test 1: Import the API bridge
    print("\n1. Testing imports...")
    try:
        from python_api_bridge import initialize_models, describe_image_enhanced_api, models
        print("✅ Successfully imported API bridge functions")
    except Exception as e:
        print(f"❌ Failed to import API bridge: {e}")
        return False
    
    # Test 2: Initialize models
    print("\n2. Testing model initialization...")
    try:
        models_ready = initialize_models()
        print(f"Models initialization result: {models_ready}")
        
        # Check individual models
        blip_available = models.get('blip_model') is not None and models.get('processor') is not None
        yolo_available = models.get('yolo_model') is not None
        llm_available = models.get('llm') is not None
        
        print(f"   BLIP: {'✅' if blip_available else '❌'}")
        print(f"   YOLO: {'✅' if yolo_available else '❌'}")
        print(f"   LLM:  {'✅' if llm_available else '❌'}")
        
        if not (blip_available or yolo_available):
            print("⚠️  No vision models available - detection will not work")
            return False
            
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False
    
    # Test 3: Create a simple test image
    print("\n3. Testing image detection...")
    try:
        # Create a simple test image (red square)
        from PIL import Image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Test the detection function
        result = describe_image_enhanced_api(test_image)
        print(f"Detection result: {result}")
        
        if result and result != "Detection failed. Please try again.":
            print("✅ Detection function is working")
            return True
        else:
            print("❌ Detection function returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def test_api_endpoint():
    """Test the actual API endpoint"""
    print("\n4. Testing API endpoint...")
    try:
        import requests
        import json
        
        # Create a simple test image as base64
        test_image = Image.new('RGB', (224, 224), color='red')
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        img_data = buffer.getvalue()
        base64_img = base64.b64encode(img_data).decode('utf-8')
        
        # Test the API endpoint
        response = requests.post(
            "http://localhost:8000/api/describe-image-enhanced",
            json={
                "image": base64_img,
                "format": "image/jpeg"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API endpoint working: {result.get('method', 'unknown')}")
            print(f"   Description: {result.get('description', 'none')}")
            print(f"   Items found: {result.get('items_found', 0)}")
            return True
        else:
            print(f"❌ API endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        print("   Make sure the API is running: python python_api_bridge.py")
        return False

if __name__ == "__main__":
    print("🍱 AI Calorie App - Model Loading Test")
    print("======================================")
    
    # Test model loading
    models_ok = test_model_loading()
    
    if models_ok:
        print("\n🎉 Model loading test passed!")
        print("\nNext steps:")
        print("1. Start the API: python python_api_bridge.py")
        print("2. Test the API: python test_model_loading.py (run this again)")
        print("3. Start the frontend: npm run dev")
    else:
        print("\n❌ Model loading test failed!")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install ML packages: pip install torch torchvision transformers ultralytics")
        print("3. Set GROQ_API_KEY in .env file")
        print("4. Check internet connection for model downloads")
    
    # If models are working, test the API endpoint
    if models_ok:
        test_api_endpoint()