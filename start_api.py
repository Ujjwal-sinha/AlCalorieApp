#!/usr/bin/env python3
"""
Startup script for the AI Calorie App API
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_requirements():
    """Install required packages"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        return False
    return True

def check_env_file():
    """Check if .env file exists with required variables"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Creating template...")
        with open(".env", "w") as f:
            f.write("# AI Calorie App Environment Variables\n")
            f.write("GROQ_API_KEY=your_groq_api_key_here\n")
            f.write("NEXT_PUBLIC_API_URL=http://localhost:8000\n")
        print("📝 Created .env template. Please add your GROQ_API_KEY")
        return False
    
    # Check if GROQ_API_KEY is set
    with open(".env", "r") as f:
        content = f.read()
        if "GROQ_API_KEY=your_groq_api_key_here" in content or "GROQ_API_KEY=" not in content:
            print("⚠️  Please set your GROQ_API_KEY in the .env file")
            return False
    
    print("✅ Environment configuration found")
    return True

def start_api():
    """Start the FastAPI server"""
    print("🚀 Starting AI Calorie App API...")
    print("📡 API will be available at: http://localhost:8000")
    print("📖 API documentation: http://localhost:8000/docs")
    print("🔗 Make sure your Next.js app is running on: http://localhost:3000")
    print("\n" + "="*50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "python_api_bridge:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 API server stopped")
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")

def main():
    """Main startup function"""
    print("🍱 AI Calorie App - Python API Setup")
    print("="*40)
    
    check_python_version()
    
    # Check if this is first run
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        sys.exit(1)
    
    # Install dependencies
    if not install_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_env_file():
        print("\n💡 Next steps:")
        print("1. Add your GROQ_API_KEY to the .env file")
        print("2. Run this script again: python start_api.py")
        sys.exit(1)
    
    # Start the API
    start_api()

if __name__ == "__main__":
    main()