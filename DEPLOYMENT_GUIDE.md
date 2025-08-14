# 🚀 **Single Vercel Deployment Guide for Food Analyzer**

This guide will help you deploy both the frontend and backend of your Food Analyzer application in a single Vercel deployment.

## 📋 **Prerequisites**

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally with `npm install -g vercel`
3. **Git Repository**: Your code should be in a Git repository

## 🏗️ **Project Structure**

Your project is now configured as a monorepo with the following structure:

```
AlCalorieApp/
├── vercel.json                 # Vercel configuration
├── package.json                # Root package.json for monorepo
├── food-analyzer-frontend/     # React frontend
├── food-analyzer-backend/      # Node.js backend
└── DEPLOYMENT_GUIDE.md         # This guide
```

## 🔧 **Configuration Files Created**

### 1. **Root `vercel.json`**
- Configures both frontend and backend builds
- Routes API calls to backend and static files to frontend
- Sets function timeout to 60 seconds

### 2. **Root `package.json`**
- Monorepo configuration with workspaces
- Build scripts for both applications
- Development scripts for local testing

### 3. **Updated Frontend Config**
- API base URL automatically detects production vs development
- Uses relative `/api` path in production

## 🚀 **Deployment Steps**

### **Step 1: Prepare Your Repository**

1. **Commit all changes**:
```bash
git add .
git commit -m "Configure monorepo for Vercel deployment"
git push origin main
```

### **Step 2: Deploy to Vercel**

1. **Navigate to your project root**:
```bash
cd /Users/ujjwalsinha/AlCalorieApp
```

2. **Login to Vercel** (if not already logged in):
```bash
vercel login
```

3. **Deploy the project**:
```bash
vercel --prod
```

4. **Follow the prompts**:
   - Set up and deploy: `Y`
   - Which scope: Select your account
   - Link to existing project: `N`
   - Project name: `alcalorieapp-monorepo`
   - Directory: `./` (current directory)
   - Override settings: `N`

### **Step 3: Configure Environment Variables**

1. **Go to Vercel Dashboard** → Your project → Settings → Environment Variables

2. **Add these environment variables**:
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

### **Step 4: Handle Python Dependencies**

Since Vercel doesn't support Python by default, you have two options:

#### **Option A: Use External Python Service (Recommended)**

1. **Deploy Python models to Railway/Render**:
   - Create a new project on [Railway](https://railway.app) or [Render](https://render.com)
   - Upload your `food-analyzer-backend/python_models/` directory
   - Deploy as a Python service

2. **Update backend environment variables**:
```env
PYTHON_SERVICE_URL=https://your-python-service.railway.app
```

#### **Option B: Use Vercel Functions with Python Runtime**

Create `food-analyzer-backend/api/detect.py`:
```python
from http.server import BaseHTTPRequestHandler
import json
import sys
import os

# Add the python_models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_models'))

from detect_food import main

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse the request
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Call the detection function
            result = main(request_data)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
```

### **Step 5: Test Your Deployment**

1. **Visit your Vercel URL** (e.g., `https://food-analyzer-monorepo-abc123.vercel.app`)
2. **Test the frontend**: Navigate through different pages
3. **Test the backend**: Try uploading an image for analysis
4. **Check API endpoints**: Visit `/api/health` to verify backend is working

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **Build Failures**:
   - Check Vercel build logs
   - Ensure all dependencies are installed
   - Verify TypeScript compilation

2. **API Errors**:
   - Check environment variables
   - Verify CORS configuration
   - Test Python service connectivity

3. **Python Model Issues**:
   - Use external Python service (Railway/Render)
   - Check model file paths
   - Verify Python dependencies

4. **Frontend API Calls**:
   - Ensure `VITE_API_BASE_URL` is set correctly
   - Check network tab for API errors
   - Verify routing configuration

### **Performance Optimization:**

1. **Enable Vercel Edge Functions** for faster response times
2. **Use CDN** for static assets
3. **Optimize images** before upload
4. **Implement caching** for repeated requests

## 📊 **Monitoring**

1. **Vercel Analytics**: Monitor performance and usage
2. **Function Logs**: Check backend execution logs
3. **Error Tracking**: Set up error monitoring
4. **Performance Monitoring**: Track response times

## 🔄 **Updates and Redeployment**

To update your application:

1. **Make your changes** in the code
2. **Commit and push** to your repository
3. **Vercel will automatically redeploy** (if connected to Git)
4. **Or manually redeploy**: `vercel --prod`

## 🌐 **Custom Domain (Optional)**

1. **Go to Vercel Dashboard** → Your project → Settings → Domains
2. **Add your custom domain** (e.g., `food-analyzer.yourdomain.com`)
3. **Update DNS records** as instructed by Vercel

## ✅ **Deployment Checklist**

- [ ] Repository pushed to Git
- [ ] Vercel project created
- [ ] Environment variables configured
- [ ] Python service deployed (if using external service)
- [ ] Frontend accessible
- [ ] Backend API working
- [ ] Image upload and analysis functional
- [ ] Custom domain configured (optional)
- [ ] SSL certificates active
- [ ] Performance monitoring set up

## 🎉 **Success!**

Your Food Analyzer application is now deployed as a single Vercel application with both frontend and backend running together!

**Your app will be available at**: `https://alcalorieapp-monorepo.vercel.app`

**API endpoints will be at**: `https://alcalorieapp-monorepo.vercel.app/api/*`

---

**Built by Ujjwal Sinha** • [GitHub](https://github.com/Ujjwal-sinha) • [LinkedIn](https://www.linkedin.com/in/sinhaujjwal01/)
