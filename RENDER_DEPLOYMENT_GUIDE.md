# 🚀 Render Deployment Guide for Food Analyzer Backend

## Prerequisites

1. **GitHub Repository**: Your code should be in a GitHub repository
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **GROQ API Key**: Get from [console.groq.com](https://console.groq.com/)
4. **Vercel Frontend URL**: Your deployed frontend URL

## Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Commit and Push Your Changes**
   ```bash
   cd food-analyzer-backend
   git add .
   git commit -m "Add Render deployment configuration"
   git push origin main
   ```

2. **Verify Files Are Pushed**
   - `render.yaml`
   - `Dockerfile`
   - `.dockerignore`
   - `yolo11m.pt` (39MB model file)

### Step 2: Sign Up/Login to Render

1. Go to [render.com](https://render.com)
2. Sign up with your GitHub account
3. Authorize Render to access your repositories

### Step 3: Create New Web Service

1. **Click "New +" → "Web Service"**
2. **Connect Repository**:
   - Select your GitHub account
   - Choose the repository containing your backend code
   - Select the branch (usually `main`)

### Step 4: Configure Service Settings

#### Basic Settings:
- **Name**: `food-analyzer-backend`
- **Environment**: `Node`
- **Region**: Choose closest to your users
- **Branch**: `main`

#### Build & Deploy Settings:
- **Build Command**: 
  ```bash
  npm install && npm run build && cd python_models && pip install -r requirements.txt
  ```
- **Start Command**: `npm start`
- **Health Check Path**: `/health`

### Step 5: Environment Variables

Click "Environment" tab and add these variables:

#### Required Variables:
```
NODE_ENV=production
PORT=10000
CORS_ORIGIN=https://your-actual-frontend-app.vercel.app
GROQ_API_KEY=your_actual_groq_api_key_here
```

#### Optional Variables:
```
NUTRITION_API_KEY=your_nutrition_api_key
FOOD_DB_API_KEY=your_food_db_api_key
CACHE_ENABLED=true
VIT_ENABLED=true
YOLO_ENABLED=true
```

### Step 6: Advanced Settings

#### Disk Configuration:
- **Add Disk**: Click "Add Disk"
- **Name**: `models`
- **Mount Path**: `/opt/render/project/src/python_models`
- **Size**: 10GB

#### Auto-Deploy:
- ✅ Enable auto-deploy for automatic updates

### Step 7: Deploy

1. **Click "Create Web Service"**
2. **Monitor Build Process**:
   - Watch the build logs
   - This may take 5-10 minutes for the first build
   - AI models will be downloaded during build

3. **Wait for Deployment**:
   - Build status will show "Building..."
   - Then "Deploying..."
   - Finally "Live"

### Step 8: Get Your Backend URL

Once deployed, you'll get a URL like:
```
https://your-backend-name.onrender.com
```

## Post-Deployment Configuration

### Step 1: Test Your Backend

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

### Step 2: Update Frontend Configuration

1. **Go to Vercel Dashboard**
2. **Navigate to your frontend project**
3. **Go to Settings → Environment Variables**
4. **Add/Update Variable**:
   - **Name**: `VITE_API_BASE_URL`
   - **Value**: `https://your-backend-name.onrender.com/api`
   - **Environment**: Production, Preview, Development

### Step 3: Redeploy Frontend

```bash
# In your frontend directory
git add .
git commit -m "Update API base URL for production"
git push origin main
```

Vercel will automatically redeploy with the new environment variable.

## Testing the Complete Setup

### 1. Test Backend API
```bash
# Health check
curl https://your-backend-name.onrender.com/health

# Test analysis endpoint (replace with actual image)
curl -X POST https://your-backend-name.onrender.com/api/analyze \
  -F "image=@test-image.jpg"
```

### 2. Test Frontend Integration
1. Open your Vercel frontend URL
2. Upload a food image
3. Verify the analysis works
4. Check browser network tab for API calls

## Troubleshooting

### Common Issues

#### 1. Build Failures
**Symptoms**: Build fails in Render dashboard
**Solutions**:
- Check build logs for specific errors
- Ensure all dependencies are in `package.json`
- Verify Python requirements are correct
- Check if YOLO model file is present

#### 2. Model Loading Issues
**Symptoms**: Backend starts but models don't load
**Solutions**:
- Ensure YOLO model file is in repository
- Check disk space allocation
- Verify model paths in configuration

#### 3. CORS Issues
**Symptoms**: Frontend can't connect to backend
**Solutions**:
- Update `CORS_ORIGIN` in Render environment variables
- Ensure frontend URL is correct
- Check browser console for CORS errors

#### 4. API Key Issues
**Symptoms**: GROQ API calls fail
**Solutions**:
- Verify GROQ API key is set correctly
- Test API key manually
- Check API key permissions

#### 5. Memory Issues
**Symptoms**: Backend crashes or times out
**Solutions**:
- AI models are memory-intensive
- Consider upgrading to a higher plan
- Optimize model loading

### Debug Commands

```bash
# Test backend health
curl https://your-backend-name.onrender.com/health

# Test API endpoint
curl https://your-backend-name.onrender.com/api

# Check CORS headers
curl -H "Origin: https://your-frontend-app.vercel.app" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -X OPTIONS https://your-backend-name.onrender.com/api/analyze
```

## Performance Optimization

### 1. Model Caching
- Enable model caching in configuration
- Use persistent disk for model storage

### 2. Response Optimization
- Implement response compression
- Add request caching where appropriate

### 3. Resource Management
- Monitor memory usage in Render dashboard
- Implement proper cleanup in shutdown handlers

## Cost Optimization

### 1. Plan Selection
- Start with free tier for testing
- Upgrade based on usage patterns
- Monitor resource usage

### 2. Auto-Scaling
- Use Render's auto-scaling features
- Set appropriate scaling thresholds

## Monitoring

### 1. Render Dashboard
- Monitor service status
- Check build logs
- View resource usage

### 2. Application Logs
- Check logs in Render dashboard
- Monitor for errors
- Track performance metrics

## Security Considerations

### 1. API Keys
- Never commit API keys to repository
- Use Render's environment variables
- Rotate keys regularly

### 2. CORS
- Restrict CORS to specific domains
- Don't use wildcard origins in production

### 3. Rate Limiting
- Implement rate limiting for API endpoints
- Monitor usage patterns

## Final Checklist

- [ ] Backend deployed to Render
- [ ] Health check endpoint working
- [ ] API endpoints responding
- [ ] GROQ API key configured
- [ ] CORS origin set correctly
- [ ] Frontend updated with new API URL
- [ ] Image upload and analysis working
- [ ] Error handling implemented
- [ ] Monitoring set up
- [ ] Performance optimized

## Support Resources

- **Render Documentation**: [docs.render.com](https://docs.render.com)
- **Render Community**: [community.render.com](https://community.render.com)
- **GROQ API Documentation**: [console.groq.com/docs](https://console.groq.com/docs)
- **Vercel Documentation**: [vercel.com/docs](https://vercel.com/docs)

---

**🎉 Congratulations! Your backend is now deployed and ready to serve your frontend!**
