#!/bin/bash

# 🚀 Food Analyzer - Single Vercel Deployment Script
# Built by Ujjwal Sinha

echo "🚀 Starting Food Analyzer Deployment..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI is not installed. Installing..."
    npm install -g vercel
fi

# Check if user is logged in to Vercel
if ! vercel whoami &> /dev/null; then
    echo "🔐 Please login to Vercel..."
    vercel login
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm run install:all

# Build the project
echo "🔨 Building project..."
npm run build

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment completed!"
echo "🌐 Your app will be available at the URL shown above"
echo "📝 Don't forget to configure environment variables in Vercel Dashboard"
echo ""
echo "📋 Next steps:"
echo "1. Go to Vercel Dashboard → Your project → Settings → Environment Variables"
echo "2. Add the required environment variables (see DEPLOYMENT_GUIDE.md)"
echo "3. Test your application"
echo ""
echo "🔗 Built by Ujjwal Sinha • GitHub: https://github.com/Ujjwal-sinha"
