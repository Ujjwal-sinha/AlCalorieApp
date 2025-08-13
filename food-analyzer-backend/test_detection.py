#!/usr/bin/env python3
"""
Test script to verify food detection is working properly
"""

import json
import base64
from PIL import Image
import numpy as np
import sys
import os

# Add the python_models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'python_models'))

def create_test_image():
    """Create a simple test image with food-like colors"""
    # Create a 512x512 test image with food-like colors
    img_array = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Add some food-like colored regions
    # Orange/brown (like bread/toast)
    img_array[100:200, 100:300] = [139, 69, 19]
    
    # Green (like vegetables)
    img_array[250:350, 150:250] = [34, 139, 34]
    
    # Red (like tomato/apple)
    img_array[400:450, 200:300] = [255, 0, 0]
    
    # Yellow (like cheese/banana)
    img_array[50:100, 350:450] = [255, 255, 0]
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    return img

def test_detection():
    """Test the detection system"""
    print("🧪 Testing Food Detection System...")
    print("=" * 50)
    
    # Create test image
    test_img = create_test_image()
    
    # Convert to base64
    import io
    buffer = io.BytesIO()
    test_img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Test input data
    test_input = {
        "model_type": "yolo",
        "image_data": img_base64,
        "width": 512,
        "height": 512
    }
    
    print("📸 Created test image with food-like colors")
    print("🔍 Testing YOLO detection...")
    
    try:
        # Import and test the detection
        from python_models.detect_food import detect_with_yolo, decode_image
        
        # Decode the image
        decoded_img = decode_image(img_base64)
        if decoded_img:
            print("✅ Image decoded successfully")
            
            # Test YOLO detection
            result = detect_with_yolo(decoded_img)
            print(f"🎯 YOLO Result: {result}")
            
            if result.get('success'):
                foods = result.get('detected_foods', [])
                confidences = result.get('confidence_scores', {})
                
                if foods:
                    print(f"✅ Detected {len(foods)} food items:")
                    for food in foods:
                        conf = confidences.get(food, 0)
                        print(f"   - {food}: {conf:.3f}")
                else:
                    print("⚠️  No food items detected (this might be expected for a simple test image)")
            else:
                print(f"❌ Detection failed: {result.get('error', 'Unknown error')}")
        else:
            print("❌ Failed to decode test image")
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🧪 Testing other models...")
    
    # Test other models
    models_to_test = ['vit', 'swin', 'blip', 'clip']
    
    for model_type in models_to_test:
        try:
            print(f"\n🔍 Testing {model_type.upper()} detection...")
            
            if model_type == 'vit':
                from python_models.detect_food import detect_with_vit
                result = detect_with_vit(decoded_img)
            elif model_type == 'swin':
                from python_models.detect_food import detect_with_swin
                result = detect_with_swin(decoded_img)
            elif model_type == 'blip':
                from python_models.detect_food import detect_with_blip
                result = detect_with_blip(decoded_img)
            elif model_type == 'clip':
                from python_models.detect_food import detect_with_clip
                result = detect_with_clip(decoded_img)
            
            print(f"🎯 {model_type.upper()} Result: {result}")
            
            if result.get('success'):
                foods = result.get('detected_foods', [])
                if foods:
                    print(f"✅ {model_type.upper()} detected {len(foods)} items")
                else:
                    print(f"⚠️  {model_type.upper()} detected no items")
            else:
                print(f"❌ {model_type.upper()} failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ {model_type.upper()} test failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("✅ Detection system test completed!")

if __name__ == "__main__":
    test_detection()
