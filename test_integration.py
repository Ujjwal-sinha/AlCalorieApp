#!/usr/bin/env python3
"""
Test script to verify the AI Calorie App integration
"""

import requests
import base64
import json
import time
from PIL import Image
import io

def test_api_health():
    """Test if the API is running and healthy"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Health Check: PASSED")
            print(f"   Status: {data.get('status')}")
            print(f"   Models Available: {data.get('models_available')}")
            return True
        else:
            print(f"❌ API Health Check: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API Health Check: FAILED (Connection Error: {e})")
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a simple colored image for testing
    img = Image.new('RGB', (224, 224), color='red')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_data = buffer.getvalue()
    base64_img = base64.b64encode(img_data).decode('utf-8')
    
    return base64_img

def test_food_analysis():
    """Test the food analysis endpoint"""
    try:
        # Create test image
        test_image = create_test_image()
        
        # Test data
        test_data = {
            "image": test_image,
            "context": "This is a test image",
            "format": "image/jpeg"
        }
        
        print("🧪 Testing food analysis...")
        response = requests.post(
            "http://localhost:8000/api/analyze",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Food Analysis: PASSED")
            print(f"   Success: {data.get('success')}")
            print(f"   Food Items Found: {len(data.get('food_items', []))}")
            print(f"   Total Calories: {data.get('nutritional_data', {}).get('total_calories', 0)}")
            return True
        else:
            print(f"❌ Food Analysis: FAILED (Status: {response.status_code})")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Food Analysis: FAILED (Connection Error: {e})")
        return False

def test_nextjs_api():
    """Test the Next.js API route"""
    try:
        # This would test the Next.js API route at localhost:3000
        # For now, we'll just check if the port is accessible
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Next.js Frontend: ACCESSIBLE")
            return True
        else:
            print(f"⚠️  Next.js Frontend: Status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Next.js Frontend: Not accessible ({e})")
        print("   This is normal if you haven't started the frontend yet")
        return False

def main():
    """Run all integration tests"""
    print("🧪 AI Calorie App - Integration Test")
    print("====================================")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: API Health
    if test_api_health():
        tests_passed += 1
    
    print()
    
    # Test 2: Food Analysis
    if test_food_analysis():
        tests_passed += 1
    
    print()
    
    # Test 3: Frontend
    if test_nextjs_api():
        tests_passed += 1
    
    print()
    print("====================================")
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your integration is working perfectly.")
    elif tests_passed >= 1:
        print("⚠️  Some tests passed. Check the failures above.")
    else:
        print("❌ All tests failed. Please check your setup.")
        print("\n💡 Troubleshooting:")
        print("1. Make sure the Python API is running: python start_api.py")
        print("2. Check your .env file has a valid GROQ_API_KEY")
        print("3. Ensure all dependencies are installed: pip install -r requirements.txt")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()