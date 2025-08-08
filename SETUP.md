# AI Calorie App - Setup Guide

This guide will help you set up and run the AI Calorie App with both the Next.js frontend and Python backend.

## Prerequisites

- **Node.js** (v18 or higher)
- **Python** (v3.8 or higher)
- **npm** or **yarn**
- **GROQ API Key** (for AI analysis)

## Quick Start

### 1. Clone and Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
# AI Calorie App Environment Variables
GROQ_API_KEY=your_groq_api_key_here
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Important**: Replace `your_groq_api_key_here` with your actual GROQ API key.

### 3. Run the Application

#### Option A: Run Both Frontend and Backend Together (Recommended)
```bash
npm run dev:full
```

#### Option B: Run Separately
```bash
# Terminal 1: Start Python API Backend
npm run api
# or
python start_api.py

# Terminal 2: Start Next.js Frontend
npm run dev
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Project Structure

```
ai-calorie-app/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   ├── analyze/           # Analysis page
│   ├── dashboard/         # Dashboard page
│   └── globals.css        # Global styles
├── lib/                   # Utility libraries
│   └── api.ts            # API client
├── types/                 # TypeScript type definitions
│   └── index.ts          # Main types
├── calarieapp/           # Python Streamlit app
│   └── app.py            # Main Python application
├── python_api_bridge.py  # FastAPI bridge
├── start_api.py          # API startup script
├── requirements.txt      # Python dependencies
└── package.json          # Node.js dependencies
```

## Features

### Frontend (Next.js)
- 🖼️ **Image Upload**: Drag & drop or click to upload food images
- 📊 **Nutrition Analysis**: Detailed calorie and macronutrient breakdown
- 📈 **Dashboard**: Track your daily nutrition intake
- 🎨 **Modern UI**: Clean, responsive design with Tailwind CSS

### Backend (Python)
- 🤖 **AI Models**: BLIP for image captioning, YOLO for object detection
- 🧠 **LLM Analysis**: GROQ-powered nutritional analysis
- 🔍 **Computer Vision**: Advanced food detection and recognition
- 📊 **Nutritional Database**: Comprehensive food nutrition data

## API Endpoints

### Health Check
```
GET /health
```

### Analyze Food (Base64)
```
POST /api/analyze
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "context": "optional_context_string"
}
```

### Analyze Food (File Upload)
```
POST /api/analyze-file
Content-Type: multipart/form-data

file: image_file
context: optional_context_string
```

## Development

### Frontend Development
```bash
npm run dev
```

### Backend Development
```bash
python start_api.py
```

### Build for Production
```bash
npm run build
```

## Troubleshooting

### Common Issues

1. **"Cannot find module '@/types'"**
   - Make sure TypeScript paths are configured correctly in `tsconfig.json`
   - Try restarting your development server

2. **Python API not starting**
   - Check if all Python dependencies are installed: `pip install -r requirements.txt`
   - Verify your GROQ_API_KEY is set in the `.env` file
   - Make sure port 8000 is not in use

3. **CORS errors**
   - The FastAPI backend is configured to allow requests from localhost:3000
   - If you're running on different ports, update the CORS settings in `python_api_bridge.py`

4. **Model loading errors**
   - The app will run in mock mode if AI models fail to load
   - Check your internet connection for downloading models
   - Ensure you have sufficient disk space and memory

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your GROQ API key for LLM analysis | Required |
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |

## Performance Tips

1. **First Run**: Initial model loading may take a few minutes
2. **Memory**: Ensure you have at least 4GB RAM available
3. **GPU**: CUDA-compatible GPU will significantly speed up analysis
4. **Network**: Stable internet connection required for model downloads

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

If you encounter any issues:
1. Check this setup guide
2. Review the troubleshooting section
3. Check the API documentation at http://localhost:8000/docs
4. Open an issue on GitHub

---

**Happy analyzing! 🍱**