#!/bin/bash

echo "🧪 Testing Services After Fix"
echo "============================="

echo ""
echo "Testing Python Models Service..."
echo "URL: https://food-detection-models.onrender.com"
echo ""

# Test Python service
PYTHON_URL="https://food-detection-models.onrender.com"
echo "1. Health Check:"
curl -s "$PYTHON_URL/health" | python3 -c "import sys, json; data=json.load(sys.stdin); print('Status:', data.get('status', 'unknown')); print('Version:', data.get('version', 'unknown')); print('Models:', data.get('models_enabled', []))" 2>/dev/null || echo "❌ Python service not responding"

echo ""
echo "2. Models Check:"
curl -s "$PYTHON_URL/models" | python3 -c "import sys, json; data=json.load(sys.stdin); print('Available models:', data.get('available_models', [])); print('Memory usage:', data.get('memory_usage', 'unknown'))" 2>/dev/null || echo "❌ Models endpoint not responding"

echo ""
echo "Testing Node.js Backend Service..."
echo "URL: https://food-analyzer-backend.onrender.com"
echo ""

# Test Node.js service
NODE_URL="https://food-analyzer-backend.onrender.com"
echo "3. Backend Health Check:"
curl -s "$NODE_URL/health" | python3 -c "import sys, json; data=json.load(sys.stdin); print('Status:', data.get('status', 'unknown'))" 2>/dev/null || echo "❌ Node.js service not responding"

echo ""
echo "✅ Testing complete!"
echo ""
echo "If services are still showing 502 errors:"
echo "1. Check Render dashboard for service status"
echo "2. Update start command to: python app_yolo_only.py"
echo "3. Redeploy the services"
