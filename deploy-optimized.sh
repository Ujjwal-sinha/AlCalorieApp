#!/bin/bash

# Optimized Python Models Deployment Script
echo "🚀 Deploying Optimized Python Models Service"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if we're in the right directory
if [ ! -f "python_models/app.py" ]; then
    print_error "Please run this script from the food-analyzer-backend directory"
    exit 1
fi

print_status "Checking files..."
if [ ! -f "python_models/models_optimized.py" ]; then
    print_error "models_optimized.py not found"
    exit 1
fi

if [ ! -f "python_models/requirements_optimized.txt" ]; then
    print_error "requirements_optimized.txt not found"
    exit 1
fi

print_success "All required files found"

print_status "Configuration Summary:"
echo "  - App: app.py (optimized with all models)"
echo "  - Models: models_optimized.py (with fallbacks)"
echo "  - Requirements: requirements_optimized.txt (minimal)"
echo "  - Render Config: render.yaml (updated)"

print_status "Next Steps:"
echo ""
echo "1. Go to your Render dashboard: https://dashboard.render.com/"
echo "2. Navigate to your food-detection-models service"
echo "3. Update these settings:"
echo "   - Build Command: pip install -r requirements_optimized.txt"
echo "   - Start Command: python app.py"
echo "4. Add environment variables:"
echo "   - YOLO_CONFIG_DIR=/tmp"
echo "   - PYTHONUNBUFFERED=1"
echo "5. Click 'Manual Deploy' → 'Deploy latest commit'"
echo ""
echo "6. Test the deployment:"
echo "   ./test-deployment.sh"
echo ""

print_success "Deployment script ready!"
print_warning "Note: This deployment will take 5-10 minutes due to ML model downloads"
