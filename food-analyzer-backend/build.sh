#!/bin/bash

echo "🚀 Building Food Analyzer Backend..."

# Exit on any error
set -e

echo "📦 Installing Node.js dependencies..."
npm install

echo "🔨 Building TypeScript..."
npm run build

echo "🐍 Installing Python dependencies..."
cd python_models
pip install -r requirements.txt
cd ..

echo "🔍 Testing Python setup..."
echo "Python version:"
python3 --version

echo "Testing Python packages:"
python3 -c "import torch; print('PyTorch:', torch.__version__)" || echo "PyTorch not available"
python3 -c "import transformers; print('Transformers available')" || echo "Transformers not available"
python3 -c "from ultralytics import YOLO; print('YOLO available')" || echo "YOLO not available"

echo "Testing Python script:"
python3 python_models/detect_food.py test || echo "Python script test failed"

echo "✅ Hybrid Node.js + Python backend setup complete"
echo "📁 Build output: dist/"
echo "🐍 Python models: python_models/"
