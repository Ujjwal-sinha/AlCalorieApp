#!/bin/bash

# Test script for Python Models Service Deployment
echo "🧪 Testing Python Models Service Deployment"
echo "============================================"

# Get the service URL from user input
read -p "Enter your Render service URL (e.g., https://your-service.onrender.com): " SERVICE_URL

echo ""
echo "Testing service at: $SERVICE_URL"
echo ""

# Test 1: Health check
echo "1. Testing health endpoint..."
curl -s "$SERVICE_URL/health" | jq '.' 2>/dev/null || curl -s "$SERVICE_URL/health"
echo ""

# Test 2: Test endpoint
echo "2. Testing test endpoint..."
curl -s "$SERVICE_URL/test" | jq '.' 2>/dev/null || curl -s "$SERVICE_URL/test"
echo ""

# Test 3: Models endpoint
echo "3. Testing models endpoint..."
curl -s "$SERVICE_URL/models" | jq '.' 2>/dev/null || curl -s "$SERVICE_URL/models"
echo ""

# Test 4: Detection endpoint (mock)
echo "4. Testing detection endpoint..."
curl -s -X POST "$SERVICE_URL/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "yolo",
    "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
  }' | jq '.' 2>/dev/null || curl -s -X POST "$SERVICE_URL/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "yolo",
    "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
  }'
echo ""

echo "✅ Testing complete!"
echo ""
echo "If all tests pass, your service is working correctly."
echo "You can now update your backend configuration to use this service."
