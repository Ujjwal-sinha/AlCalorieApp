#!/bin/bash

echo "🚀 Render Deployment Script"
echo "=========================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Please run this from the python_models directory."
    exit 1
fi

echo "✅ Found app.py"
echo "✅ render.yaml created"

echo ""
echo "📋 To deploy to Render:"
echo "1. Go to https://render.com"
echo "2. Sign up/Login with GitHub"
echo "3. Click 'New +' → 'Web Service'"
echo "4. Connect your GitHub repository"
echo "5. Select this directory (food-analyzer-backend/python_models)"
echo "6. Configure:"
echo "   - Name: python-models-api"
echo "   - Environment: Python"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: python app.py"
echo "   - Plan: Free"
echo ""
echo "7. Add Environment Variables:"
echo "   - FLASK_ENV=production"
echo "   - PYTHONUNBUFFERED=1"
echo "   - PORT=5000"
echo ""
echo "8. Click 'Create Web Service'"
echo ""
echo "�� Your service will be available at:"
echo "   https://your-service-name.onrender.com"
echo ""
echo "🔗 Update your backend:"
echo "   PYTHON_MODELS_URL=https://your-service-name.onrender.com"
