#!/bin/bash

echo "🚀 Deploying Python Models to Railway..."
echo "========================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Navigate to python_models directory
cd food-analyzer-backend/python_models

# Initialize Railway project (if not already done)
if [ ! -f ".railway" ]; then
    echo "📦 Initializing Railway project..."
    railway init
fi

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

# Get the deployment URL
echo "🔗 Getting deployment URL..."
DEPLOYMENT_URL=$(railway status --json | jq -r '.deployment.url')

echo "✅ Deployment complete!"
echo "🌐 Python Models API URL: $DEPLOYMENT_URL"
echo ""
echo "📝 Next steps:"
echo "1. Copy the URL above"
echo "2. Add it to your backend environment variables as PYTHON_SERVICE_URL"
echo "3. Redeploy your backend to Vercel"
echo ""
echo "🔧 Environment variable to add:"
echo "PYTHON_SERVICE_URL=$DEPLOYMENT_URL"
