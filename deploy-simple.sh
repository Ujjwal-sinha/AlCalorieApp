#!/bin/bash

# 🚀 Simple Two-Folder Deployment Script
# Built by Ujjwal Sinha

echo "🚀 Starting Simple Two-Folder Deployment..."

echo "📋 What will be deployed:"
echo "✅ food-analyzer-backend"
echo "✅ food-analyzer-frontend"
echo "❌ Other folders (NOT included)"

echo ""
echo "📦 Installing dependencies..."
npm run install:all

echo ""
echo "🔨 Building both applications..."
npm run build

echo ""
echo "✅ Build completed!"
echo ""
echo "🌐 Next steps:"
echo "1. Go to https://vercel.com"
echo "2. Click 'New Project'"
echo "3. Import: Ujjwal-sinha/AlCalorieApp"
echo "4. Configure:"
echo "   - Project Name: alcalorieapp"
echo "   - Framework: Other"
echo "   - Build Command: npm run build"
echo "   - Output Directory: food-analyzer-frontend/dist"
echo "   - Install Command: npm run install:all"
echo "5. Click 'Deploy'"
echo ""
echo "📝 After deployment, add environment variables (see SIMPLE_DEPLOYMENT_GUIDE.md)"
echo ""
echo "🔗 Built by Ujjwal Sinha • GitHub: https://github.com/Ujjwal-sinha"
