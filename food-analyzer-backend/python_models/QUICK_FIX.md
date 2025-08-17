# 🚨 Quick Fix for https://food-detection-models.onrender.com

## ❌ Current Issue:
The URL `https://food-detection-models.onrender.com` shows "Not Found" because:
1. The service is not deployed
2. The service is not running
3. Wrong configuration

## ✅ Solution:

### Option 1: Deploy New Service (Recommended)
1. Go to https://render.com
2. Create new web service
3. Use these settings:
   - **Name**: `food-detection-models`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app_minimal.py`
   - **Plan**: Free

### Option 2: Fix Existing Service
If service exists but not working:
1. Go to Render dashboard
2. Find your service
3. Update start command to: `python app_minimal.py`
4. Redeploy

## 📁 Required Files:
- ✅ `app_minimal.py` - Minimal Flask app
- ✅ `models_minimal.py` - YOLO + ViT only
- ✅ `requirements.txt` - Fixed PyTorch
- ✅ `render.yaml` - Configuration

## 🧪 Test After Deployment:
```bash
curl https://food-detection-models.onrender.com/health
curl https://food-detection-models.onrender.com/models
```

## 🔗 Backend Integration:
```bash
export PYTHON_MODELS_URL=https://food-detection-models.onrender.com
```

## 💡 Why This Will Work:
- Only 2 models (YOLO + ViT)
- ~200-300MB memory usage
- Fits in 512MB FREE plan
- No memory errors
