# 🚀 Backend-Only Deployment Guide

## Problem
You want to deploy only the `food-analyzer-backend` directory from your [AlCalorieApp repository](https://github.com/Ujjwal-sinha/AlCalorieApp), but Render is trying to deploy the entire repository.

## Solution: Use Root Directory Configuration

### Step 1: Deploy to Render with Root Directory

1. **Go to [render.com](https://render.com)**
2. **Click "New +" → "Web Service"**
3. **Connect Repository**: 
   - Select your GitHub account
   - Choose: `Ujjwal-sinha/AlCalorieApp`
   - Branch: `main`

### Step 2: Configure Service Settings

#### Basic Settings:
- **Name**: `food-analyzer-backend`
- **Environment**: `Node`
- **Region**: Choose closest to your users
- **Branch**: `main`

#### ⭐ **IMPORTANT: Root Directory Setting**
- **Root Directory**: `food-analyzer-backend`

This tells Render to only use the `food-analyzer-backend` directory for deployment.

#### Build & Deploy Settings:
- **Build Command**: 
  ```bash
  npm install && npm run build && cd python_models && pip install -r requirements.txt
  ```
- **Start Command**: `npm start`
- **Health Check Path**: `/health`

### Step 3: Environment Variables

Add these environment variables:

#### Required:
```
NODE_ENV=production
PORT=10000
CORS_ORIGIN=https://al-calorie-app.vercel.app
GROQ_API_KEY=your_actual_groq_api_key_here
```

#### Optional:
```
CACHE_ENABLED=true
VIT_ENABLED=true
SWIN_ENABLED=true
BLIP_ENABLED=true
CLIP_ENABLED=true
YOLO_ENABLED=true
LLM_ENABLED=true
```

### Step 4: Advanced Settings

#### Disk Configuration:
- **Add Disk**: Click "Add Disk"
- **Name**: `models`
- **Mount Path**: `/opt/render/project/src/python_models`
- **Size**: 10GB

#### Auto-Deploy:
- ✅ Enable auto-deploy for automatic updates

### Step 5: Deploy

1. **Click "Create Web Service"**
2. **Monitor Build Process**:
   - Render will only use files from `food-analyzer-backend/`
   - Build time: 10-20 minutes (first time)
   - AI models will be downloaded during build

## Alternative: Using render.yaml

If you prefer using the `render.yaml` file, it's already configured with:

```yaml
services:
  - type: web
    name: food-analyzer-backend
    rootDir: food-analyzer-backend  # This is the key setting
    # ... other configuration
```

## Expected File Structure in Render

Render will see this structure:
```
/opt/render/project/src/
├── src/                    # TypeScript source
├── python_models/          # Python AI models
├── yolo11m.pt             # YOLO model file
├── package.json           # Node.js dependencies
├── render.yaml            # Render configuration
├── Dockerfile             # Container configuration
└── .env                   # Environment variables
```

## Verification

### 1. Check Build Logs
Look for these in the build logs:
```
✓ Cloning repository
✓ Setting root directory to: food-analyzer-backend
✓ Installing Node.js dependencies
✓ Building TypeScript
✓ Installing Python dependencies
✓ Hybrid Node.js + Python backend setup complete
```

### 2. Test Deployment
```bash
# Test health endpoint
curl https://your-backend-name.onrender.com/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "uptime": 123.456,
  "environment": "production"
}
```

## Troubleshooting

### Issue: Still deploying entire repository
**Solution**: Make sure you set the **Root Directory** to `food-analyzer-backend` in the Render dashboard.

### Issue: Build fails because files not found
**Solution**: Verify that the `food-analyzer-backend` directory contains:
- `package.json`
- `src/` directory
- `python_models/` directory
- `yolo11m.pt` file

### Issue: Python dependencies not found
**Solution**: The build command should be:
```bash
npm install && npm run build && cd python_models && pip install -r requirements.txt
```

## Update Frontend Configuration

After successful deployment:

1. **Go to Vercel Dashboard**
2. **Navigate to your frontend project**
3. **Go to Settings → Environment Variables**
4. **Add/Update Variable**:
   - **Name**: `VITE_API_BASE_URL`
   - **Value**: `https://your-backend-name.onrender.com/api`
   - **Environment**: Production, Preview, Development

## Final Checklist

- [ ] Root Directory set to `food-analyzer-backend`
- [ ] Build command includes Python dependencies
- [ ] Environment variables configured
- [ ] Disk storage configured
- [ ] Build process completed successfully
- [ ] Health check endpoint working
- [ ] Frontend environment variable updated
- [ ] Complete image analysis flow tested

## Quick Deployment Script

Run this script for step-by-step guidance:
```bash
./deploy-backend-only.sh
```

---

**🎉 Your backend will now be deployed from the correct directory!**
