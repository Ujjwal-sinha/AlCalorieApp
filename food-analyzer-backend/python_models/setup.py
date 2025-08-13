#!/usr/bin/env python3
"""
Setup script for Python AI model integration

This script helps set up the Python environment for the food detection AI models
that integrate with the TypeScript backend.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install Python requirements"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_imports():
    """Test if all required libraries can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not available")
        return False
    
    try:
        import ultralytics
        print(f"✅ Ultralytics {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not available")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow available")
    except ImportError:
        print("❌ Pillow not available")
        return False
    
    return True

def test_detection_script():
    """Test the detection script"""
    print("🧪 Testing detection script...")
    
    script_path = Path(__file__).parent / "detect_food.py"
    if not script_path.exists():
        print("❌ detect_food.py not found")
        return False
    
    # Test with a simple JSON input
    test_input = {
        "model": "yolo",
        "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",  # 1x1 pixel
        "width": 1,
        "height": 1,
        "format": "png"
    }
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path), "yolo"
        ], input=json.dumps(test_input), text=True, capture_output=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Detection script test passed")
            return True
        else:
            print(f"❌ Detection script test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Detection script test timed out")
        return False
    except Exception as e:
        print(f"❌ Detection script test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Python AI model integration...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n💡 Try installing manually:")
        print("   pip install -r python_models/requirements.txt")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n💡 Some libraries failed to import. Check your installation.")
        sys.exit(1)
    
    # Test detection script
    if not test_detection_script():
        print("\n💡 Detection script test failed. Check the script and dependencies.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Start the TypeScript backend: npm run dev")
    print("   2. The backend will automatically use Python models when available")
    print("   3. Check /health endpoint for model status")

if __name__ == "__main__":
    import json
    main()
