#!/bin/bash

# Deployment Verification Script
echo "🔍 Verifying Python Models Deployment"
echo "====================================="

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

# Get service URL
read -p "Enter your Render service URL (e.g., https://food-detection-models.onrender.com): " SERVICE_URL

echo ""
print_status "Testing service at: $SERVICE_URL"
echo ""

# Test 1: Root endpoint
print_status "1. Testing root endpoint..."
ROOT_RESPONSE=$(curl -s "$SERVICE_URL/")
if [[ $ROOT_RESPONSE == *"Food Detection API"* ]]; then
    print_success "✅ Root endpoint working"
else
    print_error "❌ Root endpoint failed"
    echo "Response: $ROOT_RESPONSE"
fi
echo ""

# Test 2: Health check
print_status "2. Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s "$SERVICE_URL/health")
if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    print_success "✅ Health endpoint working"
else
    print_error "❌ Health endpoint failed"
    echo "Response: $HEALTH_RESPONSE"
fi
echo ""

# Test 3: Models endpoint
print_status "3. Testing models endpoint..."
MODELS_RESPONSE=$(curl -s "$SERVICE_URL/models")
if [[ $MODELS_RESPONSE == *"available_models"* ]]; then
    print_success "✅ Models endpoint working"
else
    print_error "❌ Models endpoint failed"
    echo "Response: $MODELS_RESPONSE"
fi
echo ""

# Test 4: Test endpoint
print_status "4. Testing test endpoint..."
TEST_RESPONSE=$(curl -s "$SERVICE_URL/test")
if [[ $TEST_RESPONSE == *"working"* ]]; then
    print_success "✅ Test endpoint working"
else
    print_error "❌ Test endpoint failed"
    echo "Response: $TEST_RESPONSE"
fi
echo ""

# Test 5: Detection endpoint (mock)
print_status "5. Testing detection endpoint..."
DETECT_RESPONSE=$(curl -s -X POST "$SERVICE_URL/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "yolo",
    "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
  }')

if [[ $DETECT_RESPONSE == *"success"* ]] || [[ $DETECT_RESPONSE == *"detected_foods"* ]]; then
    print_success "✅ Detection endpoint working"
else
    print_error "❌ Detection endpoint failed"
    echo "Response: $DETECT_RESPONSE"
fi
echo ""

print_status "Deployment Summary:"
echo "====================="
echo "Service URL: $SERVICE_URL"
echo "Status: $(if [[ $ROOT_RESPONSE == *"Food Detection API"* ]]; then echo "✅ DEPLOYED SUCCESSFULLY"; else echo "❌ DEPLOYMENT ISSUES"; fi)"
echo ""

if [[ $ROOT_RESPONSE == *"Food Detection API"* ]]; then
    print_success "🎉 Your Python Models Service is deployed and working!"
    print_status "You can now update your backend configuration to use this service."
else
    print_error "⚠️  There are deployment issues. Check your Render logs."
    print_status "Common issues:"
    echo "  - Build command not updated"
    echo "  - Start command not updated"
    echo "  - Missing environment variables"
    echo "  - Dependencies not installed"
fi
