#!/usr/bin/env python3
"""
Test script to verify food detection is working properly
"""

import requests
import base64
import json
from PIL import Image
import io
import os

def test_food_detection():
    """Test the food detection API"""
    
    # Create a simple test image (or use an existing one)
    test_image_path = "test_food_image.jpg"
    
    if not os.path.exists(test_image_path):
        # Create a simple colored image for testing
        img = Image.new('RGB', (300, 300), color='red')
        img.save(test_image_path)
        print(f"Created test image: {test_image_path}")
    
    # Read and encode the image
    with open(test_image_path, 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Test the describe-image-enhanced endpoint
    print("🔍 Testing food detection endpoint...")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/describe-image-enhanced",
            json={
                "image": base64_image,
                "format": "image/jpeg"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Food detection successful!")
            print(f"Description: {result.get('description', 'No description')}")
            print(f"Method: {result.get('method', 'Unknown')}")
            print(f"Items found: {result.get('items_found', 0)}")
            
            if result.get('success') and result.get('description'):
                print("🎉 Food items detected successfully!")
                return True
            else:
                print("⚠️  No food items detected")
                return False
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_full_analysis():
    """Test the full analysis endpoint"""
    
    test_image_path = "test_food_image.jpg"
    
    if not os.path.exists(test_image_path):
        img = Image.new('RGB', (300, 300), color='green')
        img.save(test_image_path)
    
    with open(test_image_path, 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
    
    print("\n🔍 Testing full analysis endpoint...")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/analyze",
            json={
                "image": base64_image,
                "context": "lunch meal",
                "format": "image/jpeg"
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Full analysis successful!")
            
            if result.get('success'):
                analysis = result.get('analysis', '')
                food_items = result.get('food_items', [])
                nutritional_data = result.get('nutritional_data', {})
                
                print(f"Food items found: {len(food_items)}")
                print(f"Total calories: {nutritional_data.get('total_calories', 0)}")
                
                if food_items:
                    print("🎉 Food analysis completed successfully!")
                    for item in food_items[:3]:  # Show first 3 items
                        print(f"  - {item.get('item', 'Unknown')}: {item.get('calories', 0)} cal")
                    return True
                else:
                    print("⚠️  No food items found in analysis")
                    return False
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Food Detection Fix")
    print("=" * 40)
    
    # Test 1: Basic food detection
    detection_success = test_food_detection()
    
    # Test 2: Full analysis
    analysis_success = test_full_analysis()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"Food Detection: {'✅ PASS' if detection_success else '❌ FAIL'}")
    print(f"Full Analysis: {'✅ PASS' if analysis_success else '❌ FAIL'}")
    
    if detection_success and analysis_success:
        print("🎉 All tests passed! Food detection is working.")
    else:
        print("⚠️  Some tests failed. Check the API server and model loading.")