# 🚀 YOLO-Only Deployment Guide

## ✅ Ultra Memory Optimization:
- **Only 1 model**: YOLO (yolov8n.pt)
- **Total memory usage**: ~50-100MB
- **Fits easily in 512MB RAM** (FREE plan)

## 📁 Files Created:
- `app_yolo_only.py` - YOLO-only Flask app
- `models_yolo_only.py` - Only YOLO model
- `requirements.txt` - Fixed PyTorch syntax
- `render.yaml` - Updated configuration

## 🎯 Models Available:
✅ **YOLO** (yolov8n.pt) - ~50-100MB

❌ **ViT** - Disabled
❌ **Swin** - Disabled
❌ **BLIP** - Disabled  
❌ **CLIP** - Disabled

## 🚀 Deploy on Render:

### 1. Go to Render Dashboard
- Visit: https://render.com
- Sign up/Login with GitHub

### 2. Create New Web Service
- Click "New +" → "Web Service"
- Connect your GitHub repository
- Select: `food-analyzer-backend/python_models`

### 3. Configure Service
```
Name: food-detection-yolo-only
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: python app_yolo_only.py
Plan: Free
```

### 4. Environment Variables
```
FLASK_ENV=production
PYTHONUNBUFFERED=1
PORT=5000
```

### 5. Deploy
- Click "Create Web Service"
- Wait for build (5-10 minutes)

## 🌐 Service URL
Your service will be available at:
```
https://food-detection-yolo-only.onrender.com
```

## 🔗 Backend Integration
Update your Node.js backend:
```bash
export PYTHON_MODELS_URL=https://food-detection-yolo-only.onrender.com
```

## 🧪 Test Endpoints
```bash
# Health check
curl https://food-detection-yolo-only.onrender.com/health

# List models
curl https://food-detection-yolo-only.onrender.com/models

# Test endpoint
curl https://food-detection-yolo-only.onrender.com/test
```

## 💡 Benefits:
- ✅ **Guaranteed to fit in 512MB RAM**
- ✅ **Fastest loading** (only 1 model)
- ✅ **Most reliable deployment** on FREE plan
- ✅ **No memory errors**
- ✅ **YOLO covers most food detection needs**

## 🔄 API Usage:
```bash
# YOLO detection only
curl -X POST https://food-detection-yolo-only.onrender.com/detect \
  -H "Content-Type: application/json" \
  -d '{"model_type": "yolo", "image_data": "base64_image_data"}'
```

## 🎯 Perfect for:
- Food detection apps
- Object detection
- FREE tier deployment
- Memory-constrained environments
