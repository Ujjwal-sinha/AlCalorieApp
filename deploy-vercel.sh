#!/bin/bash

echo "🚀 Starting Vercel deployment preparation..."

# Build backend
echo "📦 Building backend..."
cd food-analyzer-backend
npm install
npm run build
cd ..

# Build frontend
echo "📦 Building frontend..."
cd food-analyzer-frontend
npm install
npm run build
cd ..

echo "✅ Build complete! Ready for Vercel deployment."
echo "🌐 Deploy with: vercel --prod"
