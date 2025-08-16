#!/bin/bash

# Python Models Deployment Script for Render
# This script helps deploy the Python models service to Render

set -e

echo "🚀 Python Models Deployment Script for Render"
echo "=============================================="

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

# Check if we're in the right directory
if [ ! -f "python_models/app.py" ]; then
    print_error "Please run this script from the food-analyzer-backend directory"
    exit 1
fi

print_status "Checking prerequisites..."

# Check if git is available
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install git first."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "This directory is not a git repository. Please initialize git first."
    exit 1
fi

# Check if we have the required files
required_files=(
    "python_models/app.py"
    "python_models/requirements.txt"
    "python_models/Dockerfile"
    "python_models/render.yaml"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "Required file $file is missing"
        exit 1
    fi
done

print_success "All required files found"

# Check git status
print_status "Checking git status..."
if [ -n "$(git status --porcelain)" ]; then
    print_warning "You have uncommitted changes. Please commit them before deploying."
    echo "Current changes:"
    git status --short
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Deployment cancelled"
        exit 0
    fi
fi

# Get current branch
current_branch=$(git branch --show-current)
print_status "Current branch: $current_branch"

# Check if we have a remote repository
if ! git remote get-url origin &> /dev/null; then
    print_error "No remote repository found. Please add a remote origin first."
    exit 1
fi

remote_url=$(git remote get-url origin)
print_status "Remote repository: $remote_url"

# Push changes to remote
print_status "Pushing changes to remote repository..."
git add .
git commit -m "Deploy Python models service to Render" || {
    print_warning "No changes to commit"
}

git push origin "$current_branch"
print_success "Changes pushed to remote repository"

# Display deployment instructions
echo
echo "📋 Deployment Instructions"
echo "=========================="
echo
echo "1. Go to your Render dashboard: https://dashboard.render.com/"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub repository"
echo "4. Configure the service:"
echo "   - Name: food-detection-models"
echo "   - Environment: Python 3"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: python app.py"
echo "   - Plan: Starter (free tier)"
echo
echo "5. Set Environment Variables:"
echo "   - PYTHON_VERSION=3.11.0"
echo "   - PORT=5000"
echo "   - FLASK_ENV=production"
echo "   - PYTHONUNBUFFERED=1"
echo
echo "6. Click 'Create Web Service'"
echo "7. Wait for the build to complete (10-15 minutes)"
echo
echo "🔗 After deployment, update your backend environment variables:"
echo "   - PYTHON_MODELS_URL=https://your-service-name.onrender.com"
echo "   - NODE_ENV=production"
echo
echo "🧪 Test the deployment:"
echo "   curl https://your-service-name.onrender.com/health"
echo
print_success "Deployment script completed successfully!"
echo
print_warning "Remember to update your backend environment variables after deployment"
