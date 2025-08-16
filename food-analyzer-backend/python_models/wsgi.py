#!/usr/bin/env python3
"""
WSGI Configuration for Production Deployment
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Flask app
from app import app

if __name__ == "__main__":
    app.run()
