#!/bin/bash

# AI Calorie App - Full Stack Startup Script
echo "🍱 Starting AI Calorie App - Full Stack"
echo "========================================"

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

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites found${NC}"

# Check if .env file exists
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
echo -e "${BLUE}🐍 Installing Python dependencies...${NC}"
if python3 -m pip install -r requirements.txt; then
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  Some Python dependencies may have failed to install${NC}"
    echo -e "${YELLOW}    The app will run in TypeScript-only mode${NC}"
fi

# Check ports
echo -e "${BLUE}🔍 Checking ports...${NC}"

if port_in_use 3000; then
    echo -e "${YELLOW}⚠️  Port 3000 is already in use${NC}"
    echo -e "${YELLOW}    Please stop the process using port 3000 or the app may not start correctly${NC}"
fi

if port_in_use 8000; then
    echo -e "${YELLOW}⚠️  Port 8000 is already in use${NC}"
    echo -e "${YELLOW}    Python API may not start correctly${NC}"
fi

# Create log directory
mkdir -p logs

echo -e "${GREEN}🚀 Starting AI Calorie App...${NC}"
echo -e "${BLUE}📡 Frontend will be available at: http://localhost:3000${NC}"
echo -e "${BLUE}🔗 Python API will be available at: http://localhost:8000${NC}"
echo -e "${BLUE}📖 API documentation: http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}💡 The app will work in multiple modes:${NC}"
echo -e "${YELLOW}   1. Full mode (with Python API) - if Python dependencies are available${NC}"
echo -e "${YELLOW}   2. TypeScript mode (Next.js only) - if Python API is not available${NC}"
echo -e "${YELLOW}   3. Mock mode - for development and testing${NC}"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop all services${NC}"
echo "========================================"

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
echo -e "${BLUE}🐍 Starting Python API...${NC}"
python3 start_api.py > logs/python_api.log 2>&1 &
PYTHON_PID=$!

# Wait a moment for Python API to start
sleep 3

# Check if Python API started successfully
if kill -0 $PYTHON_PID 2>/dev/null; then
    echo -e "${GREEN}✅ Python API started successfully (PID: $PYTHON_PID)${NC}"
else
    echo -e "${YELLOW}⚠️  Python API failed to start - running in TypeScript-only mode${NC}"
fi

# Start Next.js development server
echo -e "${BLUE}⚛️  Starting Next.js development server...${NC}"
npm run dev

# This line will only be reached if npm run dev exits
cleanup