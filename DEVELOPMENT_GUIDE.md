# Food Analyzer Development Guide

## 🚀 Quick Start

### Prerequisites

- **Node.js 18+** - [Download here](https://nodejs.org/)
- **Python 3.8+** - [Download here](https://python.org/)
- **pip3** - Usually comes with Python
- **Git** - [Download here](https://git-scm.com/)

### One-Command Setup

```bash
# Make the setup script executable and run it
chmod +x setup-dev.sh
./setup-dev.sh
```

### Manual Setup

If you prefer to set up manually:

#### 1. Backend Setup

```bash
cd food-analyzer-backend

# Install Node.js dependencies
npm install

# Install Python dependencies
cd python_models
pip3 install -r requirements.txt
cd ..

# Build TypeScript
npm run build

# Copy environment file
cp env.development .env

# Edit .env with your GROQ API key
nano .env
```

#### 2. Frontend Setup

```bash
cd food-analyzer-frontend

# Install dependencies
npm install

# Copy environment file
cp env.development .env
```

## 🔧 Development Scripts

After running the setup script, you'll have these convenient scripts:

### Start Both Servers
```bash
./start-dev.sh
```
- Starts backend on http://localhost:8000
- Starts frontend on http://localhost:5173
- Press Ctrl+C to stop both

### Start Backend Only
```bash
./start-backend.sh
```

### Start Frontend Only
```bash
./start-frontend.sh
```

## 🌐 API Endpoints

### Health Check
- **GET** `/health` - Server health status

### Food Analysis
- **POST** `/api/analyze` - Analyze food image
- **POST** `/api/detect` - Detect food items
- **POST** `/api/nutrition` - Get nutrition info

### Diet Chat
- **POST** `/api/diet/chat` - Diet chat endpoint
- **POST** `/api/diet/plan` - Generate diet plan

### Testing Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Test analysis (replace with actual image)
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@test-image.jpg"
```

## 🔑 Environment Variables

### Backend (.env)
```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (for development)
NODE_ENV=development
PORT=8000
CORS_ORIGIN=http://localhost:5173
LOG_LEVEL=debug
```

### Frontend (.env)
```bash
VITE_API_BASE_URL=http://localhost:8000/api
VITE_APP_ENV=development
VITE_DEBUG_MODE=true
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
lsof -i :8000
lsof -i :5173

# Kill the process
kill -9 <PID>
```

#### 2. Python Dependencies Issues
```bash
# Upgrade pip
pip3 install --upgrade pip

# Install with verbose output
pip3 install -r requirements.txt -v

# Try installing individually
pip3 install torch torchvision
pip3 install transformers
pip3 install ultralytics
```

#### 3. Node.js Dependencies Issues
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### 4. YOLO Model Not Found
```bash
# Check if model file exists
ls -la food-analyzer-backend/yolo11m.pt

# If missing, download it (you'll need to provide the download URL)
# wget -O food-analyzer-backend/yolo11m.pt "your-model-url"
```

#### 5. GROQ API Key Issues
```bash
# Check if API key is set
echo $GROQ_API_KEY

# Test API key
curl -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models
```

### Debug Mode

#### Backend Debug
```bash
# Enable debug logging
export LOG_LEVEL=debug

# Start with debug output
cd food-analyzer-backend
DEBUG=* npm run dev
```

#### Frontend Debug
```bash
# Enable debug mode
export VITE_DEBUG_MODE=true

# Start with debug output
cd food-analyzer-frontend
npm run dev
```

## 📁 Project Structure

```
AlCalorieApp/
├── food-analyzer-backend/          # Backend API
│   ├── src/                        # TypeScript source
│   │   ├── config/                 # Configuration
│   │   ├── middleware/             # Express middleware
│   │   ├── routes/                 # API routes
│   │   ├── services/               # Business logic
│   │   └── types/                  # TypeScript types
│   ├── python_models/              # Python AI models
│   │   ├── detect_food.py          # Food detection logic
│   │   ├── color_analysis.py       # Color analysis
│   │   └── requirements.txt        # Python dependencies
│   ├── yolo11m.pt                  # YOLO model file
│   └── package.json                # Node.js dependencies
├── food-analyzer-frontend/         # React frontend
│   ├── src/                        # React source
│   │   ├── components/             # React components
│   │   ├── pages/                  # Page components
│   │   ├── services/               # API services
│   │   └── types/                  # TypeScript types
│   └── package.json                # Frontend dependencies
└── setup-dev.sh                    # Development setup script
```

## 🔍 Development Tools

### VS Code Extensions (Recommended)
- **TypeScript and JavaScript Language Features**
- **Python**
- **ESLint**
- **Prettier**
- **Thunder Client** (API testing)
- **GitLens**

### Browser Extensions
- **React Developer Tools**
- **Redux DevTools** (if using Redux)

### API Testing
```bash
# Using curl
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@test-image.jpg" \
  -H "Content-Type: multipart/form-data"

# Using Thunder Client (VS Code extension)
# Create a new request to http://localhost:8000/api/analyze
```

## 🧪 Testing

### Backend Tests
```bash
cd food-analyzer-backend

# Run all tests
npm test

# Run specific test file
npm test -- test_file.test.js

# Run with coverage
npm test -- --coverage
```

### Frontend Tests
```bash
cd food-analyzer-frontend

# Run tests (if configured)
npm test

# Run tests in watch mode
npm test -- --watch
```

### Manual Testing Checklist
- [ ] Backend health check returns 200
- [ ] Frontend loads without errors
- [ ] Image upload works
- [ ] Food detection returns results
- [ ] Nutrition analysis works
- [ ] Diet chat responds
- [ ] Error handling works
- [ ] CORS is configured correctly

## 📊 Performance Monitoring

### Backend Monitoring
```bash
# Monitor CPU and memory usage
top -p $(pgrep -f "node.*server")

# Monitor network connections
netstat -an | grep :8000

# Monitor logs
tail -f food-analyzer-backend/logs/app.log
```

### Frontend Monitoring
- Open browser DevTools
- Check Network tab for API calls
- Monitor Console for errors
- Check Performance tab for bottlenecks

## 🔄 Hot Reload

### Backend Hot Reload
The backend uses `ts-node-dev` which automatically restarts when files change.

### Frontend Hot Reload
The frontend uses Vite which provides fast hot module replacement.

## 🚀 Production Build

### Backend Production Build
```bash
cd food-analyzer-backend
npm run build
npm start
```

### Frontend Production Build
```bash
cd food-analyzer-frontend
npm run build
npm run preview
```

## 📝 Code Style

### TypeScript/JavaScript
- Use ESLint and Prettier
- Follow TypeScript best practices
- Use meaningful variable names
- Add JSDoc comments for functions

### Python
- Follow PEP 8 style guide
- Use type hints
- Add docstrings for functions
- Use meaningful variable names

## 🐛 Debugging Tips

### Backend Debugging
1. Use `console.log()` for quick debugging
2. Use VS Code debugger with breakpoints
3. Check logs in terminal output
4. Use Postman/Thunder Client for API testing

### Frontend Debugging
1. Use browser DevTools Console
2. Use React Developer Tools
3. Add `debugger;` statements
4. Use browser Network tab for API calls

### Common Debug Commands
```bash
# Check if servers are running
ps aux | grep node
ps aux | grep python

# Check port usage
netstat -tulpn | grep :8000
netstat -tulpn | grep :5173

# Check environment variables
env | grep NODE
env | grep VITE
```

## 📚 Additional Resources

- [Node.js Documentation](https://nodejs.org/docs/)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
- [Express.js Documentation](https://expressjs.com/)
- [Python Documentation](https://docs.python.org/)
- [GROQ API Documentation](https://console.groq.com/docs)

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request
5. Ensure all tests pass
6. Update documentation if needed

## 📞 Support

If you encounter issues:
1. Check this troubleshooting guide
2. Search existing issues
3. Create a new issue with detailed information
4. Include error messages and steps to reproduce
