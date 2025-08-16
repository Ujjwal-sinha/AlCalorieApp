# Python Models Deployment Guide for Render

This guide will help you deploy the Python models service on Render and connect it to your existing backend and frontend.

## Overview

The Python models service contains AI/ML models for food detection:
- YOLO (You Only Look Once) for object detection
- Vision Transformer (ViT) for image classification
- Swin Transformer for advanced image analysis
- BLIP for image captioning
- CLIP for image-text understanding

## Prerequisites

1. A Render account (free tier available)
2. Your existing backend deployed on Render: `https://food-analyzer-backend.onrender.com`
3. Your frontend deployed on Vercel: `https://al-calorie-app.vercel.app/`

## Step 1: Deploy Python Models Service on Render

### Option A: Using Render Dashboard (Recommended)

1. **Create a new Web Service on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure the service:**
   - **Name:** `food-detection-models`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Plan:** `Starter` (free tier)

3. **Set Environment Variables:**
   ```
   PYTHON_VERSION=3.11.0
   PORT=5000
   FLASK_ENV=production
   PYTHONUNBUFFERED=1
   ```

4. **Deploy:**
   - Click "Create Web Service"
   - Wait for the build to complete (may take 10-15 minutes)

### Option B: Using render.yaml (Alternative)

1. **Push the render.yaml file to your repository**
2. **Create a new Web Service from GitHub**
3. **Render will automatically detect the configuration**

## Step 2: Update Backend Configuration

### Update Environment Variables

1. **Go to your backend service on Render:**
   - Navigate to your `food-analyzer-backend` service
   - Go to "Environment" tab

2. **Add the Python Models URL:**
   ```
   PYTHON_MODELS_URL=https://your-python-models-service.onrender.com
   ```
   Replace `your-python-models-service` with your actual service name.

3. **Set NODE_ENV to production:**
   ```
   NODE_ENV=production
   ```

### Update Local Development

For local development, update your `env.development` file:

```bash
# Python Models Service Configuration
# Comment out for local development (will use local Python process)
# PYTHON_MODELS_URL=https://your-python-models-service.onrender.com
```

## Step 3: Test the Integration

### Test Python Models Service

1. **Health Check:**
   ```bash
   curl https://your-python-models-service.onrender.com/health
   ```

2. **Test Detection:**
   ```bash
   curl -X POST https://your-python-models-service.onrender.com/detect \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "yolo",
       "image_data": "base64_encoded_image_data",
       "width": 1024,
       "height": 1024
     }'
   ```

### Test Backend Integration

1. **Check backend health:**
   ```bash
   curl https://food-analyzer-backend.onrender.com/api/health
   ```

2. **Test food analysis:**
   - Upload an image through your frontend
   - Check the backend logs to see if it's using the remote Python service

## Step 4: Monitor and Debug

### Check Logs

1. **Python Models Service Logs:**
   - Go to your Python service on Render
   - Click "Logs" tab
   - Monitor for any errors during model loading or inference

2. **Backend Logs:**
   - Go to your backend service on Render
   - Click "Logs" tab
   - Look for messages like "Using remote Python service for yolo detection"

### Common Issues and Solutions

1. **Model Loading Timeout:**
   - The first request may take longer as models are downloaded
   - Subsequent requests will be faster
   - Consider using a paid plan for better performance

2. **Memory Issues:**
   - If you encounter memory errors, upgrade to a paid plan
   - The free tier has limited memory (512MB)

3. **Cold Start Delays:**
   - Render free tier services sleep after 15 minutes of inactivity
   - First request after sleep may take 30-60 seconds
   - Consider using a paid plan for consistent performance

## Step 5: Production Optimization

### Performance Optimization

1. **Model Caching:**
   - Models are cached in memory after first load
   - Subsequent requests are much faster

2. **Request Optimization:**
   - Images are automatically resized to 1024x1024
   - JPEG compression is applied for faster transmission

3. **Error Handling:**
   - The service includes comprehensive error handling
   - Failed detections fall back gracefully

### Monitoring

1. **Health Checks:**
   - Service includes automatic health checks
   - Monitor `/health` endpoint for service status

2. **Performance Metrics:**
   - Processing time is included in responses
   - Monitor for performance degradation

## Configuration Files

### render.yaml
```yaml
services:
  - type: web
    name: food-detection-models
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: PORT
        value: 5000
      - key: FLASK_ENV
        value: production
      - key: PYTHONUNBUFFERED
        value: 1
    healthCheckPath: /health
    autoDeploy: true
```

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgcc-s1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/models
EXPOSE 5000
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 FLASK_ENV=production
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1
CMD ["python", "app.py"]
```

## API Endpoints

### Health Check
```
GET /health
```

### Food Detection
```
POST /detect
Content-Type: application/json

{
  "model_type": "yolo|vit|swin|blip|clip",
  "image_data": "base64_encoded_image",
  "width": 1024,
  "height": 1024
}
```

### List Models
```
GET /models
```

## Environment Variables

### Required
- `PORT`: Service port (default: 5000)
- `FLASK_ENV`: Environment (production/development)

### Optional
- `PYTHON_VERSION`: Python version (default: 3.11.0)
- `PYTHONUNBUFFERED`: Python output buffering (default: 1)

## Troubleshooting

### Service Won't Start
1. Check build logs for dependency issues
2. Verify Python version compatibility
3. Check memory limits on free tier

### Models Not Loading
1. Check internet connectivity for model downloads
2. Verify sufficient disk space
3. Check memory allocation

### Slow Performance
1. Upgrade to paid plan for better resources
2. Monitor memory usage
3. Check for memory leaks

### Connection Issues
1. Verify CORS configuration
2. Check network connectivity
3. Verify service URLs

## Support

For issues specific to:
- **Render deployment:** Check Render documentation
- **Python models:** Check the model-specific documentation
- **Backend integration:** Check the backend service logs

## Next Steps

1. **Monitor performance** and adjust resources as needed
2. **Set up alerts** for service health
3. **Consider upgrading** to paid plans for better performance
4. **Implement caching** strategies for frequently used models
