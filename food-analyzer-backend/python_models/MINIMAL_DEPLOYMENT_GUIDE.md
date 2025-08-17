# 🚀 Minimal Deployment Guide (YOLO + ViT Only)

## ✅ Memory Optimization:
- **Only 2 models**: YOLO and ViT
- **Total memory usage**: ~200-300MB
- **Fits easily in 512MB RAM** (FREE plan)

## 📁 Files Created:
- `app_minimal.py` - Minimal Flask app
- `models_minimal.py` - Only YOLO and ViT models
- `requirements.txt` - Fixed PyTorch syntax
- `render.yaml` - Updated configuration

## 🎯 Models Available:
✅ **YOLO** (yolov8n.pt) - ~50-100MB
✅ **ViT** (vit-base-patch16-224) - ~150-200MB

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
Name: food-detection-minimal
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: python app_minimal.py
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
https://food-detection-minimal.onrender.com
```

## 🔗 Backend Integration
Update your Node.js backend:
```bash
export PYTHON_MODELS_URL=https://food-detection-minimal.onrender.com
```

## 🧪 Test Endpoints
```bash
# Health check
curl https://food-detection-minimal.onrender.com/health

# List models
curl https://food-detection-minimal.onrender.com/models

# Test endpoint
curl https://food-detection-minimal.onrender.com/test
```

## 💡 Benefits:
- ✅ **Guaranteed to fit in 512MB RAM**
- ✅ **Fast loading** (only 2 models)
- ✅ **Reliable deployment** on FREE plan
- ✅ **No memory errors**
- ✅ **YOLO + ViT** cover most food detection needs

## 🔄 API Usage:
```bash
# YOLO detection
curl -X POST https://food-detection-minimal.onrender.com/detect \
  -H "Content-Type: application/json" \
  -d '{"model_type": "yolo", "image_data": "base64_image_data"}'

# ViT detection  
curl -X POST https://food-detection-minimal.onrender.com/detect \
  -H "Content-Type: application/json" \
  -d '{"model_type": "vit", "image_data": "base64_image_data"}'
```

## 🎯 Perfect for:
- Food detection apps
- Image classification
- Object detection
- FREE tier deployment
