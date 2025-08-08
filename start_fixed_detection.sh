#!/bin/bash

echo "🚀 Starting Fixed Food Detection System"
echo "======================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please create .env file with GROQ_API_KEY"
    exit 1
fi

# Check if GROQ_API_KEY is set
if ! grep -q "GROQ_API_KEY" .env; then
    echo "❌ GROQ_API_KEY not found in .env file!"
    echo "Please add GROQ_API_KEY to your .env file"
    exit 1
fi

echo "✅ Environment configuration found"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -q fastapi uvicorn python-multipart pillow torch torchvision transformers ultralytics python-dotenv langchain-groq requests

# Check if models directory exists, create if not
if [ ! -d "models" ]; then
    mkdir models
    echo "📁 Created models directory"
fi

# Start Python API server in background
echo "🐍 Starting Python API server..."
python python_api_bridge.py &
PYTHON_PID=$!

# Wait for Python server to start
echo "⏳ Waiting for Python server to initialize..."
sleep 10

# Test if Python server is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Python API server is running"
else
    echo "❌ Python API server failed to start"
    kill $PYTHON_PID 2>/dev/null
    exit 1
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install --silent

# Start Next.js development server
echo "🌐 Starting Next.js development server..."
npm run dev &
NEXTJS_PID=$!

# Wait for Next.js server to start
echo "⏳ Waiting for Next.js server to start..."
sleep 5

# Test if Next.js server is running
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Next.js server is running"
else
    echo "❌ Next.js server failed to start"
    kill $PYTHON_PID $NEXTJS_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 Food Detection System Started Successfully!"
echo "============================================="
echo "🐍 Python API: http://localhost:8000"
echo "🌐 Next.js App: http://localhost:3000"
echo "📊 API Health: http://localhost:8000/health"
echo ""
echo "🧪 To test food detection:"
echo "python test_food_detection_fix.py"
echo ""
echo "Press Ctrl+C to stop all servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $PYTHON_PID $NEXTJS_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Keep script running
wait