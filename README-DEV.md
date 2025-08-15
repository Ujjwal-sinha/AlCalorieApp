# 🚀 Food Analyzer - Local Development

## Quick Start

### 1. One-Command Setup
```bash
./setup-dev.sh
```

### 2. Get GROQ API Key
- Go to [https://console.groq.com/](https://console.groq.com/)
- Sign up and get your API key
- Update `food-analyzer-backend/.env` with your key

### 3. Start Development Servers
```bash
./start-dev.sh
```

### 4. Open in Browser
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- Health Check: http://localhost:8000/health

## 🔧 Available Scripts

| Script | Description |
|--------|-------------|
| `./setup-dev.sh` | Complete development setup |
| `./start-dev.sh` | Start both servers |
| `./start-backend.sh` | Start backend only |
| `./start-frontend.sh` | Start frontend only |
| `./test-local.sh` | Test local environment |

## 📁 Project Structure

```
AlCalorieApp/
├── food-analyzer-backend/     # Node.js + Python API
├── food-analyzer-frontend/    # React + Vite App
├── setup-dev.sh              # Development setup
├── start-dev.sh              # Start both servers
├── test-local.sh             # Test environment
└── DEVELOPMENT_GUIDE.md      # Detailed guide
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   lsof -i :8000  # Check backend port
   lsof -i :5173  # Check frontend port
   kill -9 <PID>  # Kill process
   ```

2. **Missing Dependencies**
   ```bash
   ./setup-dev.sh  # Re-run setup
   ```

3. **GROQ API Key Issues**
   - Get key from https://console.groq.com/
   - Update `food-analyzer-backend/.env`

4. **Test Environment**
   ```bash
   ./test-local.sh  # Run comprehensive tests
   ```

## 📚 More Information

- **Detailed Guide**: [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)
- **Backend API**: http://localhost:8000/health
- **Frontend App**: http://localhost:5173

## 🎯 Next Steps

1. ✅ Run `./setup-dev.sh`
2. ✅ Get GROQ API key
3. ✅ Update `.env` file
4. ✅ Run `./start-dev.sh`
5. ✅ Test the application
6. ✅ Start developing!

---

**Happy Coding! 🎉**
