#!/bin/bash

echo "🚨 Fixing 502 Bad Gateway Errors"
echo "================================"

echo ""
echo "❌ Current Issues:"
echo "1. food-detection-models.onrender.com - 502 Bad Gateway"
echo "2. food-analyzer-backend.onrender.com - 502 Bad Gateway"
echo ""

echo "✅ Solution: Deploy YOLO-Only Version"
echo ""

echo "📋 Steps to Fix:"
echo ""

echo "1. Go to Render Dashboard:"
echo "   https://render.com"
echo ""

echo "2. For Python Models Service:"
echo "   - Find 'food-detection-models' service"
echo "   - Go to Settings"
echo "   - Update Start Command to: python app_yolo_only.py"
echo "   - Redeploy the service"
echo ""

echo "3. For Node.js Backend Service:"
echo "   - Find 'food-analyzer-backend' service"
echo "   - Go to Environment variables"
echo "   - Update PYTHON_MODELS_URL to: https://food-detection-models.onrender.com"
echo "   - Redeploy the service"
echo ""

echo "4. Alternative: Create New Services"
echo "   - Create new service with name: food-detection-yolo-only"
echo "   - Use Start Command: python app_yolo_only.py"
echo "   - This will give you a fresh deployment"
echo ""

echo "🧪 Test After Fix:"
echo "   curl https://food-detection-models.onrender.com/health"
echo "   curl https://food-analyzer-backend.onrender.com/health"
echo ""

echo "💡 Why 502 Error Occurs:"
echo "   - Service crashed due to memory issues"
echo "   - Wrong start command"
echo "   - Service not responding"
echo "   - YOLO-only version will fix this"
