# Food Analyzer Backend

A powerful TypeScript backend for AI-powered food analysis and nutrition tracking. This backend converts the Python-based food recognition system into a robust Node.js/TypeScript application with advanced AI model integration.

## 🚀 Features

- **Advanced AI Food Detection**: Multiple AI models including YOLO, Vision Transformers, Swin Transformers, BLIP, and CLIP
- **Comprehensive Nutrition Analysis**: Extensive nutrition database with external API integration
- **Real-time Image Processing**: High-performance image analysis with Sharp
- **Modular Architecture**: Clean separation of concerns with services and middleware
- **Health Monitoring**: Built-in health checks and model status monitoring
- **Error Handling**: Robust error handling and graceful degradation
- **TypeScript**: Full TypeScript support with strict type checking

## 🏗️ Architecture

```
src/
├── config/           # Configuration management
├── middleware/       # Express middleware (error handling, validation)
├── routes/          # API route definitions
├── services/        # Core business logic
│   ├── FoodDetectionService.ts    # AI-powered food detection
│   ├── ModelManager.ts            # AI model management
│   └── NutritionService.ts        # Nutrition analysis
├── types/           # TypeScript type definitions
└── server.ts        # Main server entry point
```

## 🛠️ Installation

### Prerequisites

- Node.js 18+ 
- npm 8+
- Python 3.8+ (for AI model integration)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd food-analyzer-backend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   # Server Configuration
   PORT=3000
   NODE_ENV=development
   
   # AI Model Configuration
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_API_ENDPOINT=https://api.groq.com
   
   # Nutrition API (Optional)
   NUTRITIONIX_APP_ID=your_nutritionix_app_id
   NUTRITIONIX_APP_KEY=your_nutritionix_app_key
   
   # CORS Configuration
   CORS_ORIGIN=http://localhost:3001
   
   # Logging
   LOG_LEVEL=info
   ```

4. **Set up Python AI models (Optional but recommended)**
   ```bash
   # Navigate to python_models directory
   cd python_models
   
   # Run setup script
   python3 setup.py
   
   # Or install manually
   pip install -r requirements.txt
   ```

5. **Build the project**
   ```bash
   npm run build
   ```

6. **Start the server**
   ```bash
   # Development mode
   npm run dev
   
   # Production mode
   npm start
   ```

## 🔧 Development

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build TypeScript to JavaScript
- `npm start` - Start production server
- `npm test` - Run tests
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues

### Code Structure

The backend follows a clean architecture pattern:

- **Services**: Core business logic (FoodDetectionService, ModelManager, NutritionService)
- **Routes**: API endpoint definitions
- **Middleware**: Request/response processing
- **Types**: TypeScript type definitions
- **Config**: Application configuration

## 📡 API Endpoints

### Health Check
```
GET /health
```
Returns server health status and model availability.

### Food Analysis
```
POST /api/analysis
```
Analyze food images and get nutrition information.

**Request Body:**
```json
{
  "image": "base64_encoded_image",
  "context": "optional_context_description",
  "model_type": "ensemble",
  "confidence_threshold": 0.6
}
```

**Response:**
```json
{
  "success": true,
  "sessionId": "abc123def456",
  "detectedFoods": ["chicken", "rice", "broccoli"],
  "nutritionData": {
    "chicken": {
      "calories": 165,
      "protein": 31,
      "carbs": 0,
      "fat": 3.6,
      "fiber": 0
    }
  },
  "totalNutrition": {
    "calories": 300,
    "protein": 35,
    "carbs": 25,
    "fat": 8,
    "fiber": 3
  },
  "insights": [
    "This meal is high in protein, great for muscle building and satiety."
  ],
  "confidence": 0.85,
  "processingTime": 1250
}
```

### Model Status
```
GET /api/models/status
```
Get status of all AI models.

### Nutrition Lookup
```
GET /api/nutrition/:food
```
Look up nutrition information for a specific food item.

## 🤖 AI Models

The backend supports multiple AI models for food detection with a hybrid approach:

### Detection Models
- **YOLO**: Object detection for food items
- **Vision Transformer (ViT)**: Advanced image classification
- **Swin Transformer**: Hierarchical vision transformer
- **BLIP**: Vision-language model for image understanding
- **CLIP**: Contrastive language-image pre-training

### Hybrid Architecture
- **TypeScript Backend**: Main application logic, API endpoints, and business logic
- **Python AI Models**: Actual AI model inference via child_process integration
- **Graceful Fallback**: Simulated detection when Python models aren't available
- **Automatic Detection**: Backend automatically chooses between Python models and simulation

### Model Management
- Automatic model loading with retry logic
- Health monitoring and status reporting
- Graceful degradation when models fail
- Model reloading capabilities
- Python model availability checking

## 🍎 Nutrition Analysis

### Features
- **Comprehensive Database**: 100+ food items with accurate nutrition data
- **External API Integration**: Nutritionix API for additional data
- **Fuzzy Matching**: Intelligent food name matching
- **Estimation**: Fallback nutrition estimation for unknown foods
- **Health Insights**: Automated nutrition insights and recommendations

### Nutrition Data Structure
```typescript
interface NutritionData {
  calories: number;    // Calories per 100g
  protein: number;     // Protein per 100g
  carbs: number;       // Carbohydrates per 100g
  fat: number;         // Fat per 100g
  fiber: number;       // Fiber per 100g
}
```

## 🔒 Security

- **Helmet**: Security headers
- **CORS**: Configurable cross-origin resource sharing
- **Input Validation**: Request validation middleware
- **Rate Limiting**: Built-in rate limiting (configurable)
- **Error Handling**: Secure error responses

## 📊 Monitoring

### Health Checks
- Server health status
- Model availability
- Database connectivity
- External API status

### Logging
- Structured logging with different levels
- Request/response logging
- Error tracking
- Performance monitoring

## 🧪 Testing

### Running Tests
```bash
npm test
```

### Test Structure
- Unit tests for services
- Integration tests for API endpoints
- Mock AI model responses
- Nutrition calculation validation

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY dist ./dist
EXPOSE 3000
CMD ["node", "dist/server.js"]
```

### Environment Variables for Production
```env
NODE_ENV=production
PORT=3000
CORS_ORIGIN=https://your-frontend-domain.com
LOG_LEVEL=warn
```

## 🔧 Configuration

### Model Configuration
Models can be configured in the ModelManager:

```typescript
const modelConfig: ModelConfig = {
  name: 'yolo',
  type: 'detection',
  enabled: true,
  confidence_threshold: 0.5,
  model_path: 'yolov8n.pt'
};
```

### Performance Tuning
- Image processing limits
- Model loading timeouts
- API rate limits
- Memory usage optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Style
- Use TypeScript strict mode
- Follow ESLint configuration
- Write comprehensive tests
- Document new features

## 📝 License

MIT License - see LICENSE file for details

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API examples

## 🔄 Hybrid Architecture

This backend uses a hybrid approach combining TypeScript and Python for optimal performance:

### Architecture Benefits
- **TypeScript Backend**: Fast API responses, type safety, and modern development
- **Python AI Models**: Access to the full ML ecosystem and pre-trained models
- **Best of Both Worlds**: TypeScript for business logic, Python for AI inference
- **Graceful Degradation**: Works with or without Python models

### Setup Options
1. **Full Setup**: Install Python models for actual AI detection
2. **Simulation Only**: Use TypeScript backend with simulated detection
3. **Mixed Mode**: Some models in Python, others simulated

### Migration Notes
- Python AI models are integrated via child_process
- TypeScript handles all business logic and API endpoints
- Nutrition database is converted to TypeScript Map
- Error handling is improved with TypeScript types
- Automatic fallback to simulation when Python models fail