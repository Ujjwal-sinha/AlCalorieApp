# Frontend-Backend Integration Guide

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env.local` file in the frontend directory:

```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000/api

# Feature Flags
VITE_ENABLE_OFFLINE_MODE=false
VITE_ENABLE_CAMERA_CAPTURE=true
VITE_ENABLE_EXPORT=true

# Development Configuration
VITE_DEBUG_MODE=true
VITE_LOG_LEVEL=info
```

### 2. Start Both Services

**Terminal 1 - Backend:**
```bash
cd food-analyzer-backend
npm run dev
```

**Terminal 2 - Frontend:**
```bash
cd food-analyzer-frontend
npm run dev
```

### 3. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000/api
- **Backend Health**: http://localhost:8000/health

## 🔧 Configuration

### Backend Configuration

The backend is configured in `food-analyzer-backend/src/config/index.ts`:

```typescript
export const config = {
  port: 8000,
  apiPrefix: '/api',
  corsOrigin: 'http://localhost:5173',
  // ... other settings
};
```

### Frontend Configuration

The frontend configuration is in `food-analyzer-frontend/src/config/index.ts`:

```typescript
export const APP_CONFIG = {
  api: {
    baseUrl: 'http://localhost:8000/api',
    timeout: 30000,
    retries: 3,
  },
  // ... other settings
};
```

## 🔌 API Integration

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze/advanced` | POST | Advanced food analysis with ensemble detection |
| `/api/analyze/model/{model}` | POST | Analysis with specific AI model |
| `/api/analyze/batch` | POST | Batch analysis of multiple images |
| `/api/food/validate` | POST | Validate food items |
| `/api/food/search` | GET | Search for food items |
| `/api/nutrition/calculate` | POST | Calculate nutrition for food items |
| `/api/models/status` | GET | Get AI model status |
| `/health` | GET | Backend health check |

### Example API Usage

```typescript
import { AnalysisService } from './services/AnalysisService';

const analysisService = AnalysisService.getInstance();

// Analyze an image
const result = await analysisService.analyzeImage(imageFile);

// Get model status
const modelStatus = await analysisService.getModelStatus();

// Check backend health
const health = await analysisService.getServiceHealth();
```

## 🎯 Features

### 1. Real-time Status Monitoring

The frontend includes a **System Status** tab that shows:
- Backend health status
- AI model availability
- Service uptime
- Environment information

### 2. Advanced Error Handling

- Automatic retry logic for failed requests
- Graceful fallback to simulated detection
- User-friendly error messages
- Network status indicators

### 3. Model Selection

Users can choose specific AI models for analysis:
- YOLO (Object Detection)
- ViT (Vision Transformer)
- Swin (Swin Transformer)
- BLIP (Image Captioning)
- CLIP (Contrastive Learning)
- LLM (Language Model)

### 4. Ensemble Detection

The system automatically combines multiple AI models for better accuracy:
- Confidence-based weighting
- Fallback mechanisms
- Performance optimization

## 🔍 Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure backend CORS is configured for frontend origin
   - Check `corsOrigin` in backend config

2. **Connection Refused**
   - Verify backend is running on port 8000
   - Check firewall settings
   - Ensure no other service is using the port

3. **Model Loading Failures**
   - Check Python dependencies are installed
   - Verify model files are present
   - Review backend logs for errors

4. **Image Upload Issues**
   - Check file size limits (10MB default)
   - Verify supported image formats
   - Ensure proper MIME type handling

### Debug Mode

Enable debug mode in frontend:

```bash
VITE_DEBUG_MODE=true npm run dev
```

This will show detailed API requests and responses in the browser console.

## 📊 Performance Optimization

### Backend Optimizations

1. **Model Caching**: AI models are cached in memory
2. **Request Batching**: Multiple images processed efficiently
3. **Compression**: Responses are compressed for faster transfer
4. **Connection Pooling**: Efficient database connections

### Frontend Optimizations

1. **Request Retries**: Automatic retry with exponential backoff
2. **Image Compression**: Client-side image optimization
3. **Caching**: Local storage for history and settings
4. **Lazy Loading**: Components loaded on demand

## 🔒 Security

### Backend Security

- Helmet.js for security headers
- CORS configuration
- Input validation with Joi
- Rate limiting
- File upload restrictions

### Frontend Security

- Environment variable protection
- Input sanitization
- Secure API communication
- No sensitive data in client code

## 🚀 Deployment

### Production Setup

1. **Backend Deployment**:
   ```bash
   cd food-analyzer-backend
   npm run build
   npm start
   ```

2. **Frontend Deployment**:
   ```bash
   cd food-analyzer-frontend
   npm run build
   # Deploy dist/ folder to your hosting service
   ```

3. **Environment Variables**:
   - Set `NODE_ENV=production`
   - Configure production API URLs
   - Set up proper CORS origins

### Docker Deployment

Both services include Docker support for easy deployment:

```bash
# Backend
docker build -t food-analyzer-backend .
docker run -p 8000:8000 food-analyzer-backend

# Frontend
docker build -t food-analyzer-frontend .
docker run -p 80:80 food-analyzer-frontend
```

## 📈 Monitoring

### Health Checks

- Backend health endpoint: `/health`
- Model status endpoint: `/api/models/status`
- Frontend status monitoring in System Status tab

### Logging

- Backend logs with Morgan
- Frontend console logging
- Error tracking and reporting

## 🤝 Contributing

1. Follow the existing code structure
2. Add proper TypeScript types
3. Include error handling
4. Write tests for new features
5. Update documentation

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review backend and frontend logs
3. Verify configuration settings
4. Test with different images/formats
