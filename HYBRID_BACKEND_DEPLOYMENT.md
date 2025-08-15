# 🚀 Hybrid Backend Deployment Guide (Node.js + Python)

## Architecture Overview

Your backend uses a **hybrid architecture**:
- **Node.js/TypeScript**: Main API server (Express.js)
- **Python**: AI/ML models for food detection
- **Communication**: Node.js calls Python scripts via child_process
- **AI Models**: YOLO, Vision Transformers, Swin, CLIP, BLIP

## Prerequisites

1. **GitHub Repository**: Your code should be in a GitHub repository
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **GROQ API Key**: Get from [console.groq.com](https://console.groq.com/)
4. **Vercel Frontend URL**: [https://al-calorie-app.vercel.app/](https://al-calorie-app.vercel.app/)

## Step-by-Step Deployment

### Step 1: Verify Repository Structure

Ensure your repository has this structure:
```
food-analyzer-backend/
├── src/                    # TypeScript source
│   ├── services/
│   │   └── ModelManager.ts # Manages AI models
│   └── server.ts           # Main server
├── python_models/          # Python AI models
│   ├── detect_food.py      # Main detection script
│   ├── requirements.txt    # Python dependencies
│   └── color_analysis.py   # Color analysis
├── yolo11m.pt             # YOLO model file (39MB)
├── render.yaml            # Render configuration
├── Dockerfile             # Container configuration
└── package.json           # Node.js dependencies
```

### Step 2: Deploy to Render

#### 2.1 Create New Web Service
1. Go to [render.com](https://render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select the repository containing your backend code

#### 2.2 Configure Service Settings

**Basic Settings:**
- **Name**: `food-analyzer-backend`
- **Environment**: `Node`
- **Region**: Choose closest to your users
- **Branch**: `main`

**Build & Deploy Settings:**
- **Build Command**: 
  ```bash
  npm install && npm run build && cd python_models && pip install -r requirements.txt
  ```
- **Start Command**: `npm start`
- **Health Check Path**: `/health`

#### 2.3 Environment Variables

Add these environment variables:

**Required:**
```
NODE_ENV=production
PORT=10000
CORS_ORIGIN=https://al-calorie-app.vercel.app
GROQ_API_KEY=your_actual_groq_api_key_here
```

**Optional:**
```
CACHE_ENABLED=true
VIT_ENABLED=true
SWIN_ENABLED=true
BLIP_ENABLED=true
CLIP_ENABLED=true
YOLO_ENABLED=true
LLM_ENABLED=true
```

#### 2.4 Advanced Settings

**Disk Configuration:**
- **Add Disk**: Click "Add Disk"
- **Name**: `models`
- **Mount Path**: `/opt/render/project/src/python_models`
- **Size**: 10GB (for AI models)

**Auto-Deploy:**
- ✅ Enable auto-deploy for automatic updates

### Step 3: Deploy and Monitor

1. **Click "Create Web Service"**
2. **Monitor Build Process**:
   - **Phase 1**: Node.js dependencies installation (2-3 minutes)
   - **Phase 2**: TypeScript compilation (1-2 minutes)
   - **Phase 3**: Python dependencies installation (5-10 minutes)
   - **Phase 4**: AI model downloads (3-5 minutes)
   - **Total**: 10-20 minutes for first build

3. **Watch for these in build logs**:
   ```
   ✓ Installing Node.js dependencies
   ✓ Building TypeScript
   ✓ Installing Python dependencies
   ✓ Downloading AI models
   ✓ Hybrid Node.js + Python backend setup complete
   ```

## Expected Build Process

### Node.js Setup
```
npm install
├── Installing Express, TypeScript, etc.
└── Building TypeScript to JavaScript
```

### Python Setup
```
cd python_models && pip install -r requirements.txt
├── Installing PyTorch (large download)
├── Installing Transformers
├── Installing Ultralytics (YOLO)
├── Installing OpenCV, Pillow, etc.
└── Setting up Python environment
```

### AI Models Loading
```
Initializing ModelManager...
├── Loading YOLO model (yolo11m.pt)
├── Loading Vision Transformer models
├── Loading Swin Transformer
├── Loading CLIP model
├── Loading BLIP model
└── All models ready
```

## Post-Deployment Testing

### 1. Test Backend Health
```bash
curl https://your-backend-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "uptime": 123.456,
  "environment": "production"
}
```

### 2. Test Python Integration
```bash
curl -X POST https://your-backend-name.onrender.com/api/analyze \
  -F "image=@test-image.jpg"
```

### 3. Check Model Status
```bash
curl https://your-backend-name.onrender.com/api/models/status
```

## Troubleshooting Hybrid Backend

### Common Issues

#### 1. Python Dependencies Fail
**Symptoms**: Build fails during pip install
**Solutions**:
- Check `python_models/requirements.txt` syntax
- Ensure all packages are compatible
- Try installing packages individually

#### 2. AI Models Not Loading
**Symptoms**: Backend starts but models fail to load
**Solutions**:
- Check if YOLO model file is present
- Verify disk space allocation
- Check Python path configuration

#### 3. Child Process Communication Issues
**Symptoms**: Node.js can't call Python scripts
**Solutions**:
- Verify Python executable path
- Check file permissions
- Ensure Python scripts are executable

#### 4. Memory Issues
**Symptoms**: Backend crashes due to memory
**Solutions**:
- AI models are memory-intensive
- Consider upgrading to higher plan
- Implement model lazy loading

### Debug Commands

```bash
# Test Python availability
curl https://your-backend-name.onrender.com/api/debug/python

# Test model loading
curl https://your-backend-name.onrender.com/api/debug/models

# Test child process communication
curl https://your-backend-name.onrender.com/api/debug/process
```

## Performance Considerations

### 1. Cold Start Times
- **First Request**: 30-60 seconds (models loading)
- **Subsequent Requests**: 2-5 seconds
- **Model Caching**: Enable for better performance

### 2. Memory Usage
- **YOLO Model**: ~500MB
- **Vision Transformers**: ~200MB each
- **Total Memory**: ~2-3GB recommended

### 3. Response Times
- **Image Analysis**: 5-15 seconds
- **Model Loading**: 30-60 seconds (first time)
- **Cached Models**: 2-5 seconds

## Update Frontend Configuration

1. **Go to Vercel Dashboard**
2. **Navigate to your frontend project**
3. **Go to Settings → Environment Variables**
4. **Add/Update Variable**:
   - **Name**: `VITE_API_BASE_URL`
   - **Value**: `https://your-backend-name.onrender.com/api`
   - **Environment**: Production, Preview, Development

## Monitoring and Logs

### Render Dashboard
- Monitor service status
- Check build logs
- View resource usage
- Monitor memory consumption

### Application Logs
- Node.js logs in Render dashboard
- Python script outputs
- Model loading status
- Error tracking

## Cost Optimization

### 1. Plan Selection
- **Free Tier**: Good for testing (limited resources)
- **Starter Plan**: $7/month (recommended for production)
- **Standard Plan**: $25/month (for high traffic)

### 2. Resource Management
- Monitor memory usage
- Implement model caching
- Use lazy loading for models

## Security Considerations

### 1. API Keys
- Store GROQ API key in Render environment variables
- Never commit API keys to repository
- Rotate keys regularly

### 2. File Upload Security
- Validate file types and sizes
- Implement rate limiting
- Sanitize uploaded images

### 3. CORS Configuration
- Restrict to your frontend domain
- Don't use wildcard origins

## Final Checklist

- [ ] Repository contains all necessary files
- [ ] YOLO model file (yolo11m.pt) is present
- [ ] Python requirements.txt is correct
- [ ] Render service created and configured
- [ ] Environment variables set
- [ ] Disk storage configured
- [ ] Build process completed successfully
- [ ] Health check endpoint working
- [ ] Python integration tested
- [ ] Frontend environment variable updated
- [ ] Complete image analysis flow tested

## Support Resources

- **Render Documentation**: [docs.render.com](https://docs.render.com)
- **PyTorch Documentation**: [pytorch.org/docs](https://pytorch.org/docs)
- **Transformers Documentation**: [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- **GROQ API Documentation**: [console.groq.com/docs](https://console.groq.com/docs)

---

**🎉 Your hybrid Node.js + Python backend is now deployed and ready to serve your frontend!**
