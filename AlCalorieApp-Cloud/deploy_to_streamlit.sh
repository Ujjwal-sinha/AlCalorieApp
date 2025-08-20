#!/bin/bash

# 🚀 Streamlit Cloud Deployment Script
# This script helps you prepare and deploy your app to Streamlit Cloud

echo "🚀 Starting Streamlit Cloud Deployment Process..."
echo "================================================"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Please run this script from the AlCalorieApp-Cloud directory."
    exit 1
fi

# Check if required files exist
echo "📋 Checking required files..."
required_files=("app.py" "requirements.txt" "yolo11m.pt" ".streamlit/config.toml")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

echo "✅ All required files found!"

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files
echo "📝 Adding files to Git..."
git add .

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "💾 Committing changes..."
    git commit -m "Deploy to Streamlit Cloud: AI-Powered Nutrition Analysis App"
    echo "✅ Changes committed"
fi

# Check if remote origin exists
if ! git remote get-url origin &> /dev/null; then
    echo ""
    echo "🌐 GitHub Repository Setup Required:"
    echo "====================================="
    echo "1. Create a new repository on GitHub:"
    echo "   - Go to https://github.com/new"
    echo "   - Name it something like 'alcalorie-app'"
    echo "   - Make it PUBLIC (required for Streamlit Cloud)"
    echo "   - Don't initialize with README, .gitignore, or license"
    echo ""
    echo "2. After creating the repository, run these commands:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo "3. Then deploy to Streamlit Cloud:"
    echo "   - Go to https://share.streamlit.io"
    echo "   - Sign in with GitHub"
    echo "   - Click 'New app'"
    echo "   - Select your repository"
    echo "   - Set main file path to: app.py"
    echo "   - Click 'Deploy!'"
    echo ""
    echo "📖 For detailed instructions, see: STREAMLIT_CLOUD_DEPLOYMENT.md"
else
    echo "🌐 Remote origin already configured"
    echo "📤 Pushing to GitHub..."
    git push origin main
    echo "✅ Code pushed to GitHub!"
    echo ""
    echo "🎉 Next Steps:"
    echo "1. Go to https://share.streamlit.io"
    echo "2. Sign in with GitHub"
    echo "3. Click 'New app'"
    echo "4. Select your repository"
    echo "5. Set main file path to: app.py"
    echo "6. Click 'Deploy!'"
fi

echo ""
echo "✅ Deployment preparation complete!"
echo "📖 See STREAMLIT_CLOUD_DEPLOYMENT.md for detailed instructions"
