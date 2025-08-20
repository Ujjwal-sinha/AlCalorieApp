# 🚨 Emergency Fix for 502 Bad Gateway Errors

## ❌ Current Problems:
1. `food-detection-models.onrender.com` - 502 Bad Gateway
2. `food-analyzer-backend.onrender.com` - 502 Bad Gateway

## ✅ Immediate Solutions:

### Option 1: Fix Existing Services (Recommended)

#### For Python Models Service:
1. Go to https://render.com
2. Find service: `food-detection-models`
3. Go to **Settings** tab
4. Update **Start Command** to: `python app_yolo_only.py`
5. Click **Save Changes**
6. Go to **Events** tab
7. Click **Manual Deploy**

#### For Node.js Backend Service:
1. Go to https://render.com
2. Find service: `food-analyzer-backend`
3. Go to **Environment** tab
4. Verify `PYTHON_MODELS_URL` is set to: `https://food-detection-models.onrender.com`
5. Go to **Events** tab
6. Click **Manual Deploy**

### Option 2: Create New Services (Fresh Start)

#### Create New Python Service:
1. Go to https://render.com
2. Click **New +** → **Web Service**
3. Connect GitHub repository
4. Select directory: `food-analyzer-backend/python_models`
5. Configure:
   - **Name**: `food-detection-yolo-only`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app_yolo_only.py`
   - **Plan**: Free
6. Add Environment Variables:
   - `FLASK_ENV=production`
   - `PYTHONUNBUFFERED=1`
   - `PORT=5000`
7. Click **Create Web Service**

#### Update Node.js Backend:
1. Go to **Environment** tab
2. Update `PYTHON_MODELS_URL` to new service URL
3. Redeploy

## 🧪 Test After Fix:
```bash
# Test Python service
curl https://food-detection-models.onrender.com/health

# Test Node.js service
curl https://food-analyzer-backend.onrender.com/health
```

## 💡 Why 502 Error Occurs:
- **Memory issues**: Service crashed due to 512MB limit
- **Wrong start command**: Using wrong Python file
- **Service not responding**: Application failed to start
- **YOLO-only version**: Will fix all memory issues

## 🎯 Expected Results:
- ✅ Services respond with 200 OK
- ✅ Health checks pass
- ✅ YOLO model loads successfully
- ✅ No more memory errors
