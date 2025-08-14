# 🚀 **Simple Vercel Deployment Guide - Two Folders Only**

This guide will help you deploy **only** the `food-analyzer-backend` and `food-analyzer-frontend` folders in a single Vercel deployment.

## 📋 **What We're Deploying**

- ✅ `food-analyzer-backend` - Node.js/TypeScript backend with AI models
- ✅ `food-analyzer-frontend` - React frontend application
- ❌ Other folders (AlCalorieApp-Cloud, calarieapp, etc.) - NOT included

## 🏗️ **Project Structure**

```
AlCalorieApp/
├── vercel.json                    # Vercel configuration
├── package.json                   # Root package.json
├── food-analyzer-backend/         # ✅ Backend (Node.js/TypeScript)
│   ├── src/
│   ├── python_models/
│   ├── package.json
│   └── dist/
├── food-analyzer-frontend/        # ✅ Frontend (React)
│   ├── src/
│   ├── package.json
│   └── dist/
└── SIMPLE_DEPLOYMENT_GUIDE.md     # This guide
```

## 🚀 **Step-by-Step Deployment**

### **Step 1: Prepare Your Repository**

1. **Commit all changes**:
```bash
git add .
git commit -m "Configure for two-folder Vercel deployment"
git push origin main
```

### **Step 2: Deploy to Vercel**

1. **Go to [vercel.com](https://vercel.com)** and sign in
2. **Click "New Project"**
3. **Import your GitHub repository**: `Ujjwal-sinha/AlCalorieApp`
4. **Configure the project**:
   - **Project Name**: `alcalorieapp` (or any name you prefer)
   - **Framework Preset**: Select "Other"
   - **Root Directory**: Leave as `/` (root)
   - **Build Command**: `npm run build`
   - **Output Directory**: `food-analyzer-frontend/dist`
   - **Install Command**: `npm run install:all`

5. **Click "Deploy"**

### **Step 3: Configure Environment Variables**

After deployment, go to **Project Settings → Environment Variables** and add:

```env
# Backend Configuration
NODE_ENV=production
CORS_ORIGIN=https://your-vercel-domain.vercel.app
API_TIMEOUT=45000
MAX_FILE_SIZE=10485760
YOLO_MODEL_PATH=yolo11m.pt
YOLO_ENABLED=true
PYTHON_PATH=python3
PYTHON_SCRIPT_PATH=python_models/detect_food.py
JWT_SECRET=your-secure-jwt-secret-here
RATE_LIMIT_WINDOW=900000
RATE_LIMIT_MAX=100

# Frontend Configuration
VITE_API_BASE_URL=/api
```

### **Step 4: Handle Python Models**

Since Vercel doesn't support Python, you need to handle the Python models:

**Option A: External Python Service (Recommended)**
1. Deploy Python models to [Railway](https://railway.app):
   - Create new project
   - Upload `food-analyzer-backend/python_models/` folder
   - Upload `food-analyzer-backend/yolo11m.pt` (39MB model file)
   - Deploy as Python service
2. Add environment variable: `PYTHON_SERVICE_URL=https://your-python-service.railway.app`

**Option B: Vercel Functions**
Create `food-analyzer-backend/api/detect.py`:
```python
from http.server import BaseHTTPRequestHandler
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_models'))
from detect_food import main

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            result = main(request_data)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
```

## 🔧 **How It Works**

### **Build Process**:
1. **Backend**: TypeScript compiles to JavaScript in `food-analyzer-backend/dist/`
2. **Frontend**: React builds to static files in `food-analyzer-frontend/dist/`

### **Python Models**:
- **Fixed**: YOLO model path issue resolved (looks for `yolo11m.pt` in multiple locations)
- **Location**: `food-analyzer-backend/yolo11m.pt` (39MB model file)
- **Python Scripts**: `food-analyzer-backend/python_models/detect_food.py`

### **Routing**:
- `/api/*` → Backend serverless functions
- `/*` → Frontend static files

### **Single URL**:
Everything will be available at: `https://your-project-name.vercel.app`

## ✅ **Testing Your Deployment**

1. **Visit your Vercel URL**
2. **Test Frontend**: Navigate through pages
3. **Test Backend**: Try uploading an image
4. **Check API**: Visit `/api/health`

## 🔍 **Troubleshooting**

### **Build Failures**:
- Check Vercel build logs
- Ensure all dependencies are installed
- Verify TypeScript compilation

### **API Errors**:
- Check environment variables
- Verify CORS configuration
- Test Python service connectivity

### **Python Model Issues**:
- Use external Python service (Railway/Render)
- Check model file paths
- Verify Python dependencies

## 📊 **Expected URLs**

After successful deployment:
- **Main App**: `https://alcalorieapp.vercel.app`
- **API Health**: `https://alcalorieapp.vercel.app/api/health`
- **Analysis**: `https://alcalorieapp.vercel.app/analysis`

## 🎉 **Success!**

Your Food Analyzer app with both frontend and backend will be deployed as a single application!

**Built by Ujjwal Sinha** • [GitHub](https://github.com/Ujjwal-sinha) • [LinkedIn](https://www.linkedin.com/in/sinhaujjwal01/)
