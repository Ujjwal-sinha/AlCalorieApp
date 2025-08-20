#!/usr/bin/env python3
"""
Launcher script for AI-Powered Nutrition Analysis App
Run this script to start the Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    print("🚀 Starting AI-Powered Nutrition Analysis App...")
    print("=" * 60)
    print("🧠 Landing Page: Beautiful UI with AI-powered nutrition analysis")
    print("🔍 Analysis Page: YOLO11m food detection and calorie tracking")
    print("📊 Features: History tracking, analytics, and progress monitoring")
    print("=" * 60)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Run the Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
