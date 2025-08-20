#!/usr/bin/env python3
"""
Check deployment files for Streamlit Cloud
Verifies that all required files are present and properly configured
"""

import os
import sys
from pathlib import Path

def check_deployment_files():
    """Check if all required files for deployment are present"""
    print("🔍 Checking deployment files for Streamlit Cloud...")
    print("=" * 60)
    
    # Required files for deployment
    required_files = [
        "app.py",
        "requirements.txt", 
        "yolo11m.pt",
        ".streamlit/config.toml"
    ]
    
    # Required directories
    required_dirs = [
        "utils"
    ]
    
    # Check files
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            size = os.path.getsize(file_path)
            if file_path == "yolo11m.pt":
                print(f"✅ {file_path} ({size / (1024*1024):.1f} MB)")
            else:
                print(f"✅ {file_path} ({size} bytes)")
    
    # Check directories
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path}/ (directory)")
    
    # Check utils directory contents
    if os.path.isdir("utils"):
        utils_files = os.listdir("utils")
        print(f"   📁 utils/ contains: {', '.join(utils_files)}")
    
    # Check .gitignore
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r") as f:
            content = f.read()
            if "!yolo11m.pt" in content:
                print("✅ .gitignore properly configured (yolo11m.pt will be included)")
            else:
                print("⚠️  .gitignore may exclude yolo11m.pt - check configuration")
    else:
        print("⚠️  .gitignore not found")
    
    # Report results
    print("\n" + "=" * 60)
    
    if missing_files or missing_dirs:
        print("❌ Missing required files/directories:")
        for item in missing_files + missing_dirs:
            print(f"   - {item}")
        return False
    else:
        print("✅ All required files are present!")
        print("\n🚀 Ready for deployment to Streamlit Cloud!")
        print("\nNext steps:")
        print("1. Run: ./deploy_to_streamlit.sh")
        print("2. Create GitHub repository")
        print("3. Deploy to Streamlit Cloud")
        return True

if __name__ == "__main__":
    success = check_deployment_files()
    sys.exit(0 if success else 1)
