#!/bin/bash

echo "🚀 Setting up Food Analyzer Development Environment..."

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

# Check if Node.js is installed
check_node() {
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18 or higher."
        exit 1
    fi
    
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        print_error "Node.js version 18 or higher is required. Current version: $(node -v)"
        exit 1
    fi
    
    print_success "Node.js version: $(node -v)"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_success "Python version: $PYTHON_VERSION"
}

# Check if pip is installed
check_pip() {
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed. Please install pip3."
        exit 1
    fi
    
    print_success "pip3 is available"
}

# Setup Backend
setup_backend() {
    print_status "Setting up Backend..."
    
    cd food-analyzer-backend
    
    # Install Node.js dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install Node.js dependencies"
        exit 1
    fi
    
    # Install Python dependencies
    print_status "Installing Python dependencies..."
    cd python_models
    pip3 install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install Python dependencies"
        exit 1
    fi
    
    cd ..
    
    # Build TypeScript
    print_status "Building TypeScript..."
    npm run build
    
    if [ $? -ne 0 ]; then
        print_error "Failed to build TypeScript"
        exit 1
    fi
    
    # Copy environment file
    if [ -f "env.development" ]; then
        cp env.development .env
        print_success "Environment file configured"
    else
        print_warning "No environment file found. Please create .env manually"
    fi
    
    cd ..
    print_success "Backend setup completed"
}

# Setup Frontend
setup_frontend() {
    print_status "Setting up Frontend..."
    
    cd food-analyzer-frontend
    
    # Install dependencies
    print_status "Installing dependencies..."
    npm install
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install frontend dependencies"
        exit 1
    fi
    
    # Copy environment file
    if [ -f "env.development" ]; then
        cp env.development .env
        print_success "Environment file configured"
    else
        print_warning "No environment file found. Please create .env manually"
    fi
    
    cd ..
    print_success "Frontend setup completed"
}

# Create development scripts
create_dev_scripts() {
    print_status "Creating development scripts..."
    
    # Backend dev script
    cat > start-backend.sh << 'EOF'
#!/bin/bash
cd food-analyzer-backend
echo "🚀 Starting Backend Development Server..."
echo "📡 API will be available at: http://localhost:8000"
echo "🏥 Health check at: http://localhost:8000/health"
echo "🔧 Press Ctrl+C to stop"
npm run dev
EOF
    
    # Frontend dev script
    cat > start-frontend.sh << 'EOF'
#!/bin/bash
cd food-analyzer-frontend
echo "🚀 Starting Frontend Development Server..."
echo "🌐 App will be available at: http://localhost:5173"
echo "🔧 Press Ctrl+C to stop"
npm run dev
EOF
    
    # Combined dev script
    cat > start-dev.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Food Analyzer Development Environment..."

# Function to cleanup background processes
cleanup() {
    echo "🛑 Stopping development servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Start backend
echo "📡 Starting backend..."
cd food-analyzer-backend
npm run dev &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🌐 Starting frontend..."
cd food-analyzer-frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Development environment started!"
echo "📡 Backend: http://localhost:8000"
echo "🌐 Frontend: http://localhost:5173"
echo "🏥 Health check: http://localhost:8000/health"
echo "🔧 Press Ctrl+C to stop all servers"

# Wait for both processes
wait
EOF
    
    # Make scripts executable
    chmod +x start-backend.sh start-frontend.sh start-dev.sh
    
    print_success "Development scripts created"
}

# Main execution
main() {
    print_status "Checking prerequisites..."
    check_node
    check_python
    check_pip
    
    print_status "Setting up development environment..."
    setup_backend
    setup_frontend
    create_dev_scripts
    
    print_success "🎉 Development environment setup completed!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Get your GROQ API key from https://console.groq.com/"
    echo "2. Update food-analyzer-backend/.env with your GROQ_API_KEY"
    echo "3. Run './start-dev.sh' to start both servers"
    echo "4. Open http://localhost:5173 in your browser"
    echo ""
    echo "🔧 Available scripts:"
    echo "  ./start-backend.sh  - Start backend only"
    echo "  ./start-frontend.sh - Start frontend only"
    echo "  ./start-dev.sh      - Start both servers"
}

# Run main function
main
