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

echo "✅ Hybrid Node.js + Python backend setup complete"
echo "📁 Build output: dist/"
echo "🐍 Python models: python_models/"
