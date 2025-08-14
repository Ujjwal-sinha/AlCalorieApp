#!/usr/bin/env python3
"""
Test script to verify YOLO model path is working correctly
"""

import os
import sys

def test_model_paths():
    """Test different model paths"""
    print("Testing YOLO model paths...")
    
    # Current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Test different paths
    paths_to_test = [
        'yolo11m.pt',
        '../yolo11m.pt',
        '../../yolo11m.pt',
        'food-analyzer-backend/yolo11m.pt',
        'python_models/yolo11m.pt'
    ]
    
    for path in paths_to_test:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"   Size: {file_size:.1f} MB")
        else:
            print(f"❌ Not found: {path}")
    
    # List files in current directory
    print(f"\nFiles in current directory:")
    for file in os.listdir('.'):
        if file.endswith('.pt'):
            print(f"  - {file}")

if __name__ == "__main__":
    test_model_paths()
