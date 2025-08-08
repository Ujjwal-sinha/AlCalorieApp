#!/bin/bash

# AI Calorie App - Complete Setup and Run Script

echo "🍱 AI Calorie App - Complete Setup"
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    echo "Please install Node.js 18 or higher and try again."
    exit 1
fi

echo "✅ Python $(python3 --version) detected"
echo "✅ Node.js $(node --version) detected"

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install Python dependencies
echo "📦 Installing Python dependencies..."
python3 -m pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOL
# AI Calorie App Environment Variables
GROQ_API_KEY=your_groq_api_key_here
NEXT_PUBLIC_API_URL=http://localhost:8000
EOL
    echo "⚠️  Please edit .env file and add your GROQ_API_KEY"
    echo "💡 You can get a GROQ API key from: https://console.groq.com/"
    exit 1
fi

# Check if GROQ_API_KEY is set
if grep -q "your_groq_api_key_here" .env; then
    echo "⚠️  Please set your GROQ_API_KEY in the .env file"
    echo "💡 Edit .env and replace 'your_groq_api_key_here' with your actual API key"
    exit 1
fi

echo "✅ Environment configuration found"

# Build the Next.js app
echo "🔨 Building Next.js application..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Please check the errors above."
    exit 1
fi

echo "✅ Build completed successfully"

# Start both frontend and backend
echo "🚀 Starting AI Calorie App..."
echo "📡 Frontend: http://localhost:3000"
echo "📡 Backend API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "=================================="

# Run both servers concurrently
npm run dev:full