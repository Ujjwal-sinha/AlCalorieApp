# 🚀 Render Deployment Guide for Python Models

## Quick Start

### 1. Go to Render
- Visit: https://render.com
- Sign up/Login with GitHub

### 2. Create New Web Service
- Click "New +" → "Web Service"
- Connect your GitHub repository
- Select the `food-analyzer-backend/python_models` directory

### 3. Configure Service
```
Name: python-models-api
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: python app.py
Plan: Free
```

### 4. Environment Variables
Add these in Render dashboard:
```
FLASK_ENV=production
PYTHONUNBUFFERED=1
PORT=5000
```

### 5. Deploy
- Click "Create Web Service"
- Wait for build to complete (5-10 minutes)

## Service URL
Your service will be available at:
```
https://your-service-name.onrender.com
```

## API Endpoints
- `GET /health` - Health check
- `GET /models` - List available models
- `GET /test` - Test endpoint
- `POST /detect` - Food detection API

## Backend Integration
Update your Node.js backend environment:
```bash
export PYTHON_MODELS_URL=https://your-service-name.onrender.com
```

## Testing
```bash
# Health check
curl https://your-service-name.onrender.com/health

# List models
curl https://your-service-name.onrender.com/models

# Test endpoint
curl https://your-service-name.onrender.com/test
```

## Troubleshooting
1. **Build fails**: Check requirements.txt
2. **Memory issues**: Models will load on first use
3. **Timeout**: First request may take 30-60 seconds
4. **Cold start**: Service may sleep after inactivity

## Cost
- **Free Tier**: 750 hours/month
- **Paid**: $7/month for always-on service
