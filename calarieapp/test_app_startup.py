#!/usr/bin/env python3
"""
Test script to verify app startup without TensorFlow errors
"""

import sys
import os

def test_imports():
    """Test all imports to identify any issues"""
    print("🔍 Testing imports...")
    
    # Test basic imports
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ python-dotenv import failed: {e}")
        return False
    
    # Test AI/ML imports with error handling
    try:
        from langchain_groq import ChatGroq
        print("✅ LangChain Groq imported successfully")
    except ImportError as e:
        print(f"⚠️  LangChain Groq import failed: {e}")
    
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"⚠️  Transformers import failed: {e}")
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"⚠️  PyTorch import failed: {e}")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"⚠️  NumPy import failed: {e}")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"⚠️  OpenCV import failed: {e}")
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"⚠️  Matplotlib import failed: {e}")
    
    try:
        from captum.attr import GradientShap
        print("✅ Captum imported successfully")
    except ImportError as e:
        print(f"⚠️  Captum import failed: {e}")
    
    try:
        from lime.lime_image import LimeImageExplainer
        print("✅ LIME imported successfully")
    except ImportError as e:
        print(f"⚠️  LIME import failed: {e}")
    
    try:
        from ultralytics import YOLO
        print("✅ YOLO imported successfully")
    except ImportError as e:
        print(f"⚠️  YOLO import failed: {e}")
    
    # Test UI imports
    try:
        from modern_ui import load_css
        print("✅ Modern UI imported successfully")
    except ImportError as e:
        print(f"⚠️  Modern UI import failed: {e}")
    
    try:
        from visualizations import display_visualization_dashboard
        print("✅ Visualizations imported successfully")
    except ImportError as e:
        print(f"⚠️  Visualizations import failed: {e}")
    
    return True

def test_app_structure():
    """Test if app.py can be imported without errors"""
    print("\n🔍 Testing app.py import...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Try to import app module
        import app
        print("✅ app.py imported successfully")
        return True
    except Exception as e:
        print(f"❌ app.py import failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting AI Calorie Tracker App Tests")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test app structure
    app_ok = test_app_structure()
    
    print("\n" + "=" * 50)
    if imports_ok and app_ok:
        print("🎉 All tests passed! App should start successfully.")
        return True
    else:
        print("⚠️  Some tests failed. Check the warnings above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
