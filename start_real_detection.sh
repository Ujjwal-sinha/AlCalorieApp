#!/bin/bash

# AI Calorie App - Real Detection Startup Script
echo "🍱 Starting AI Calorie App with Real Python Detection"
echo "===================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check Node.js
if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    echo -e "${YELLOW}   Please install Node.js from https://nodejs.org/${NC}"
    exit 1
fi

# Check Python
if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo -e "${YELLOW}   Please install Python 3 from https://python.org/${NC}"
    exit 1
fi

# Check pip
if ! command_exists pip3; then
    echo -e "${RED}❌ pip3 is not installed${NC}"
    echo -e "${YELLOW}   Please install pip3${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites found${NC}"

# Check .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating template...${NC}"
    cat > .env << EOF
# AI Calorie App Environment Variables
GROQ_API_KEY="your_groq_api_key_here"
NEXT_PUBLIC_GROQ_API_KEY="your_groq_api_key_here"
NEXT_PUBLIC_API_URL="http://localhost:8000"
NEXT_PUBLIC_PYTHON_API_URL="http://localhost:8000"
EOF
    echo -e "${YELLOW}📝 Please edit .env file with your GROQ_API_KEY${NC}"
    echo -e "${YELLOW}   You can get a free API key from https://console.groq.com/${NC}"
fi

# Check GROQ API key
if grep -q "your_groq_api_key_here" .env 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Please set your GROQ_API_KEY in the .env file${NC}"
    echo -e "${YELLOW}   The app will run in mock mode without it${NC}"
fi

# Install Node.js dependencies
echo -e "${BLUE}📦 Installing Node.js dependencies...${NC}"
if npm install; then
    echo -e "${GREEN}✅ Node.js dependencies installed${NC}"
else
    echo -e "${RED}❌ Failed to install Node.js dependencies${NC}"
    exit 1
fi

# Install Python dependencies
echo -e "${BLUE}🐍 Installing Python dependencies for real detection...${NC}"
echo -e "${BLUE}   This may take a while for first-time setup...${NC}"

# Install core requirements
if python3 -m pip install -r requirements.txt; then
    echo -e "${GREEN}✅ Core Python dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  Some Python dependencies failed to install${NC}"
    echo -e "${YELLOW}    The app will run in mock mode${NC}"
fi

# Try to install additional ML dependencies
echo -e "${BLUE}🤖 Installing ML model dependencies...${NC}"
python3 -m pip install torch torchvision transformers ultralytics pillow opencv-python-headless 2>/dev/null || {
    echo -e "${YELLOW}⚠️  ML dependencies installation had issues${NC}"
    echo -e "${YELLOW}    Some models may not be available${NC}"
}

# Check ports
echo -e "${BLUE}🔍 Checking ports...${NC}"
if port_in_use 3000; then
    echo -e "${YELLOW}⚠️  Port 3000 is already in use${NC}"
fi

if port_in_use 8000; then
    echo -e "${YELLOW}⚠️  Port 8000 is already in use${NC}"
fi

# Create logs directory
mkdir -p logs

echo -e "${GREEN}🚀 Starting AI Calorie App with Real Detection...${NC}"
echo -e "${BLUE}📡 Frontend: http://localhost:3000${NC}"
echo -e "${BLUE}🐍 Python API: http://localhost:8000${NC}"
echo -e "${BLUE}📖 API Docs: http://localhost:8000/docs${NC}"
echo ""
echo -e "${GREEN}🧠 Real Detection Features:${NC}"
echo -e "${GREEN}   • BLIP image captioning model${NC}"
echo -e "${GREEN}   • YOLO object detection${NC}"
echo -e "${GREEN}   • Groq LLM for analysis${NC}"
echo -e "${GREEN}   • Multi-strategy food detection${NC}"
echo -e "${GREEN}   • Advanced nutritional parsing${NC}"
echo ""
echo -e "${YELLOW}💡 Detection Modes:${NC}"
echo -e "${YELLOW}   1. Full AI Mode - All models loaded (best accuracy)${NC}"
echo -e "${YELLOW}   2. Partial Mode - Some models available${NC}"
echo -e "${YELLOW}   3. Mock Mode - No models (for development)${NC}"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop all services${NC}"
echo "===================================================="

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}🛑 Stopping all services...${NC}"
    kill $(jobs -p) 2>/dev/null
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start Python API in background
echo -e "${BLUE}🐍 Starting Python API with real detection...${NC}"
python3 python_api_bridge.py > logs/python_api.log 2>&1 &
PYTHON_PID=$!

# Wait for Python API to start
echo -e "${BLUE}⏳ Waiting for Python API to initialize...${NC}"
sleep 5

# Check if Python API started successfully
if kill -0 $PYTHON_PID 2>/dev/null; then
    echo -e "${GREEN}✅ Python API started successfully (PID: $PYTHON_PID)${NC}"
    
    # Test the API
    echo -e "${BLUE}🧪 Testing Python API...${NC}"
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✅ Python API is responding${NC}"
    else
        echo -e "${YELLOW}⚠️  Python API may not be fully ready${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Python API failed to start - check logs/python_api.log${NC}"
fi

# Start Next.js development server
echo -e "${BLUE}⚛️  Starting Next.js frontend...${NC}"
npm run dev

# This line will only be reached if npm run dev exits
cleanup