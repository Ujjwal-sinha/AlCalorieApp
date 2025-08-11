#!/usr/bin/env python3
"""
Test script to check YOLO model loading on Streamlit Cloud
"""

import os
import sys
import time

def test_yolo_loading():
    """Test YOLO model loading with detailed logging"""
    print("🔍 Testing YOLO model loading...")
    
    # Test 1: Check if ultralytics is available
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics library imported successfully")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        return False
    
    # Test 2: Check if local file exists
    yolo_path = "yolov8n.pt"
    if os.path.exists(yolo_path):
        print(f"✅ Local YOLO file found: {yolo_path}")
        file_size = os.path.getsize(yolo_path) / (1024 * 1024)  # MB
        print(f"📁 File size: {file_size:.2f} MB")
    else:
        print(f"⚠️ Local YOLO file not found: {yolo_path}")
    
    # Test 3: Try loading YOLO model
    try:
        print("🔄 Attempting to load YOLO model...")
        start_time = time.time()
        
        # Try loading from local file first
        if os.path.exists(yolo_path):
            try:
                model = YOLO(yolo_path)
                load_time = time.time() - start_time
                print(f"✅ YOLO loaded from local file in {load_time:.2f} seconds")
                return True
            except Exception as e:
                print(f"❌ Local file loading failed: {e}")
        
        # Try downloading
        try:
            model = YOLO('yolov8n.pt')
            load_time = time.time() - start_time
            print(f"✅ YOLO downloaded and loaded in {load_time:.2f} seconds")
            return True
        except Exception as e:
            print(f"❌ Download loading failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ YOLO loading failed: {e}")
        return False

def test_environment():
    """Test the deployment environment"""
    print("\n🌍 Environment Information:")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Check available memory (if possible)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Available memory: {memory.available / (1024**3):.2f} GB")
    except ImportError:
        print("psutil not available for memory check")

if __name__ == "__main__":
    print("🚀 YOLO Loading Test for Streamlit Cloud")
    print("=" * 50)
    
    test_environment()
    print("\n" + "=" * 50)
    
    success = test_yolo_loading()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 YOLO loading test PASSED!")
    else:
        print("❌ YOLO loading test FAILED!")
        print("💡 The app will still work with BLIP-only detection")
    
    print("=" * 50)
