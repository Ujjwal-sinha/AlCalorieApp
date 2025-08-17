#!/bin/bash

echo "🧪 Testing Food Detection API Deployment"
echo "========================================"

SERVICE_URL="https://food-detection-models.onrender.com"

echo ""
echo "Testing service at: $SERVICE_URL"
echo ""

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s "$SERVICE_URL/health" | python3 -m json.tool 2>/dev/null || echo "❌ Health check failed"

echo ""
echo "2. Testing models endpoint..."
curl -s "$SERVICE_URL/models" | python3 -m json.tool 2>/dev/null || echo "❌ Models endpoint failed"

echo ""
echo "3. Testing test endpoint..."
curl -s "$SERVICE_URL/test" | python3 -m json.tool 2>/dev/null || echo "❌ Test endpoint failed"

echo ""
echo "✅ Testing complete!"
echo ""
echo "If all tests fail, the service is not deployed or not running."
echo "Follow the deployment guide in deploy-minimal.sh"
