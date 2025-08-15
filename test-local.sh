#!/bin/bash

echo "🧪 Testing Local Development Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test backend health
test_backend() {
    print_status "Testing Backend Health..."
    
    # Check if backend is running
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "Backend is running and healthy"
        
        # Test API endpoint
        if curl -s http://localhost:8000/api > /dev/null; then
            print_success "API endpoint is accessible"
        else
            print_warning "API endpoint might not be working"
        fi
    else
        print_error "Backend is not running or not accessible"
        print_status "Start backend with: ./start-backend.sh"
        return 1
    fi
}

# Test frontend
test_frontend() {
    print_status "Testing Frontend..."
    
    # Check if frontend is running
    if curl -s http://localhost:5173 > /dev/null; then
        print_success "Frontend is running"
    else
        print_error "Frontend is not running or not accessible"
        print_status "Start frontend with: ./start-frontend.sh"
        return 1
    fi
}

# Test environment variables
test_env() {
    print_status "Testing Environment Variables..."
    
    # Check backend env
    if [ -f "food-analyzer-backend/.env" ]; then
        print_success "Backend .env file exists"
        
        # Check for GROQ API key
        if grep -q "GROQ_API_KEY" food-analyzer-backend/.env; then
            print_success "GROQ API key is configured"
        else
            print_warning "GROQ API key not found in .env"
        fi
    else
        print_warning "Backend .env file not found"
    fi
    
    # Check frontend env
    if [ -f "food-analyzer-frontend/.env" ]; then
        print_success "Frontend .env file exists"
        
        # Check API base URL
        if grep -q "VITE_API_BASE_URL" food-analyzer-frontend/.env; then
            print_success "API base URL is configured"
        else
            print_warning "API base URL not found in .env"
        fi
    else
        print_warning "Frontend .env file not found"
    fi
}

# Test dependencies
test_dependencies() {
    print_status "Testing Dependencies..."
    
    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node -v)
        print_success "Node.js: $NODE_VERSION"
    else
        print_error "Node.js not found"
        return 1
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python: $PYTHON_VERSION"
    else
        print_error "Python3 not found"
        return 1
    fi
    
    # Check if node_modules exist
    if [ -d "food-analyzer-backend/node_modules" ]; then
        print_success "Backend node_modules installed"
    else
        print_warning "Backend node_modules not found"
    fi
    
    if [ -d "food-analyzer-frontend/node_modules" ]; then
        print_success "Frontend node_modules installed"
    else
        print_warning "Frontend node_modules not found"
    fi
}

# Test model files
test_models() {
    print_status "Testing AI Models..."
    
    # Check YOLO model
    if [ -f "food-analyzer-backend/yolo11m.pt" ]; then
        print_success "YOLO model file exists"
    else
        print_warning "YOLO model file not found"
    fi
    
    # Check Python models directory
    if [ -d "food-analyzer-backend/python_models" ]; then
        print_success "Python models directory exists"
    else
        print_warning "Python models directory not found"
    fi
}

# Test ports
test_ports() {
    print_status "Testing Ports..."
    
    # Check if ports are in use
    if lsof -i :8000 > /dev/null 2>&1; then
        print_success "Port 8000 (Backend) is in use"
    else
        print_warning "Port 8000 (Backend) is not in use"
    fi
    
    if lsof -i :5173 > /dev/null 2>&1; then
        print_success "Port 5173 (Frontend) is in use"
    else
        print_warning "Port 5173 (Frontend) is not in use"
    fi
}

# Main test function
main() {
    echo "🔍 Running comprehensive local environment tests..."
    echo ""
    
    test_dependencies
    echo ""
    
    test_env
    echo ""
    
    test_models
    echo ""
    
    test_ports
    echo ""
    
    test_backend
    echo ""
    
    test_frontend
    echo ""
    
    echo "📋 Test Summary:"
    echo "=================="
    echo "✅ Dependencies: Node.js, Python, npm packages"
    echo "✅ Environment: .env files and configuration"
    echo "✅ Models: AI model files and directories"
    echo "✅ Ports: Backend (8000) and Frontend (5173)"
    echo "✅ Backend: Health check and API endpoints"
    echo "✅ Frontend: Web server accessibility"
    echo ""
    echo "🎉 If all tests passed, your development environment is ready!"
    echo ""
    echo "🚀 Next steps:"
    echo "1. Get your GROQ API key from https://console.groq.com/"
    echo "2. Update food-analyzer-backend/.env with your GROQ_API_KEY"
    echo "3. Run './start-dev.sh' to start both servers"
    echo "4. Open http://localhost:5173 in your browser"
    echo "5. Test the application by uploading a food image"
}

# Run main function
main
