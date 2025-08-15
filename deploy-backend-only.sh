#!/bin/bash

echo "🚀 Deploying Food Analyzer Backend to Render..."

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
if [ ! -d "food-analyzer-backend" ]; then
    print_error "food-analyzer-backend directory not found!"
    print_status "Please run this script from the root of your AlCalorieApp repository"
    exit 1
fi

# Create a temporary deployment directory
TEMP_DIR="temp-backend-deploy"
print_status "Creating temporary deployment directory..."

# Clean up any existing temp directory
rm -rf $TEMP_DIR

# Create temp directory and copy backend files
mkdir $TEMP_DIR
cp -r food-analyzer-backend/* $TEMP_DIR/

# Copy necessary files from root
cp .gitignore $TEMP_DIR/ 2>/dev/null || true
cp README.md $TEMP_DIR/ 2>/dev/null || true

print_success "Backend files prepared for deployment"

# Instructions for manual deployment
echo ""
echo "📋 Manual Deployment Instructions:"
echo "=================================="
echo ""
echo "1. Go to https://render.com"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub repository: https://github.com/Ujjwal-sinha/AlCalorieApp"
echo "4. Configure the service:"
echo "   - Name: food-analyzer-backend"
echo "   - Environment: Node"
echo "   - Root Directory: food-analyzer-backend"
echo "   - Build Command: npm install && npm run build && cd python_models && pip install -r requirements.txt"
echo "   - Start Command: npm start"
echo "   - Health Check Path: /health"
echo ""
echo "5. Add Environment Variables:"
echo "   - NODE_ENV=production"
echo "   - PORT=10000"
echo "   - CORS_ORIGIN=https://al-calorie-app.vercel.app"
echo "   - GROQ_API_KEY=your_actual_groq_api_key"
echo ""
echo "6. Advanced Settings:"
echo "   - Add Disk: models, /opt/render/project/src/python_models, 10GB"
echo "   - Enable Auto-Deploy"
echo ""
echo "7. Click 'Create Web Service'"
echo ""
echo "✅ Your backend will be deployed from the food-analyzer-backend directory!"
echo ""

# Clean up
rm -rf $TEMP_DIR
print_success "Temporary files cleaned up"
