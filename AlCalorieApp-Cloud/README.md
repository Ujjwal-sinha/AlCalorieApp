# 🍱 AI Calorie Tracker - Cloud Optimized

An AI-powered calorie tracking application built with Streamlit that analyzes food images and provides comprehensive nutritional information. This version is specifically optimized for Streamlit Cloud deployment.

## ✨ Features

- 📷 **AI Food Detection**: Upload food images for automatic identification using BLIP and YOLO models
- 📊 **Comprehensive Nutrition Analysis**: Get detailed calorie and macronutrient breakdown with LLM analysis
- 📈 **Progress Tracking**: Monitor daily calorie intake and goals with visual progress indicators
- 📋 **History Management**: View and manage your food analysis history with detailed insights
- 🎯 **Goal Setting**: Set and track daily calorie targets with progress visualization
- 🔬 **Advanced Analytics**: AI interpretability features (Grad-CAM, SHAP, LIME) for model transparency
- 📊 **Data Visualization**: Interactive charts and analytics dashboard

## 🚀 Deployment on Streamlit Cloud

### Prerequisites

1. **Groq API Key**: Get your API key from [Groq Console](https://console.groq.com/)
2. **GitHub Account**: For hosting the code
3. **Streamlit Cloud Account**: For deployment

### Quick Setup

1. **Fork/Clone this repository** to your GitHub account

2. **Add your Groq API Key**:
   - Go to your Streamlit Cloud dashboard
   - Navigate to your app settings
   - Add a new secret with key `GROQ_API_KEY` and your API key as the value

3. **Deploy on Streamlit Cloud**:
   - Connect your GitHub repository
   - Set the main file path to: `AlCalorieApp-Cloud/app.py`
   - Deploy!

### Environment Variables

The app requires the following environment variable:
- `GROQ_API_KEY`: Your Groq API key for LLM features

## 📁 Project Structure

```
AlCalorieApp-Cloud/
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies (optimized)
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── utils/
│   ├── models.py            # AI model loading and management
│   ├── analysis.py          # Food analysis functions
│   └── ui.py               # UI components and utilities
├── assets/                  # Static files (if any)
└── README.md               # This file
```

## 🛠️ Technical Architecture

### AI Models Used

- **BLIP (Salesforce)**: For image captioning and food detection
- **YOLO (Ultralytics)**: For object detection and food identification
- **Groq LLM**: For nutritional analysis and recommendations
- **CNN (DenseNet)**: For AI interpretability visualizations

### Key Optimizations

- **Modular Design**: Separated concerns into utils modules for better maintainability
- **Error Handling**: Comprehensive error handling with graceful fallbacks
- **Caching**: Model loading is cached using `@st.cache_resource`
- **Lazy Loading**: Models are loaded only when needed
- **Memory Management**: Optimized for Streamlit Cloud's memory constraints
- **Version Constraints**: All dependencies have proper version bounds

### Dependencies

#### Core Dependencies
- **Streamlit**: Web framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **LangChain**: LLM integration
- **Ultralytics**: YOLO model framework

#### Visualization Dependencies
- **Matplotlib**: Data visualization
- **Plotly**: Interactive charts
- **Seaborn**: Statistical visualizations

#### Optional Dependencies
- **Captum**: Model interpretability
- **LIME**: Local interpretability
- **OpenCV**: Computer vision

## 🤖 Enhanced Food Agent

The app includes an advanced **Enhanced Food Agent** that provides comprehensive food analysis with web search capabilities:

### Agent Features

1. **Image Analysis**: Advanced food detection using multiple AI models
2. **Web Search**: Real-time information from the web about food items
3. **Context Management**: Store analysis context for follow-up questions
4. **Comprehensive Information**: 
   - Nutritional facts and health benefits
   - Cultural and historical background
   - Recipe suggestions and cooking methods
   - Dietary considerations and allergen information

### How to Use the Enhanced Agent

1. **Navigate to the "🤖 Enhanced Agent" tab**
2. **Upload a food image** for analysis
3. **Click "Analyze with Enhanced Agent"** to start processing
4. **View comprehensive results** including web-sourced information
5. **Ask follow-up questions** about the food without re-uploading
6. **Explore detailed insights** about nutrition, culture, and preparation

### Agent Architecture

- **Modular Design**: Each component can be upgraded independently
- **Caching System**: Efficient storage of search results and context
- **Error Handling**: Graceful fallbacks when web search is unavailable
- **Session Management**: Unique session IDs for each analysis

## 📖 Usage Guide

### 1. Upload Food Image
- Use the file uploader to select a food image (PNG, JPG, JPEG)
- Ensure good lighting and clear visibility of all food items

### 2. Add Context (Optional)
- Provide additional description of the meal
- Include cooking methods, portion sizes, or special ingredients

### 3. Analyze
- Click "Analyze Food" to start the AI analysis
- The app will detect food items and provide nutritional breakdown

### 4. View Results
- See detailed nutritional information (calories, protein, carbs, fats)
- View comprehensive health analysis and recommendations
- Check AI interpretability visualizations (if available)

### 5. Track Progress
- Monitor daily calorie intake in the sidebar
- View progress towards your daily calorie target
- Access analysis history and trends

## 🔧 Troubleshooting

### Common Issues

1. **"LLM service unavailable"**
   - Check your Groq API key in Streamlit secrets
   - Ensure the API key is valid and has sufficient credits

2. **"Image analysis unavailable"**
   - Models may take time to load on first run
   - Check if all dependencies are properly installed

3. **"Models not available"**
   - Some models may fail to load due to memory constraints
   - The app will work with available models and provide fallbacks

4. **Slow performance**
   - First run may be slower due to model downloading
   - Subsequent runs will be faster due to caching

### Performance Tips

- Use clear, well-lit food images
- Add context descriptions for better analysis
- The app works best with common food items
- Close other browser tabs to free up memory

## 🎯 Features Breakdown

### Core Features
- ✅ Image upload and processing
- ✅ AI-powered food detection
- ✅ Nutritional analysis
- ✅ Progress tracking
- ✅ History management

### Advanced Features
- ✅ AI interpretability (Grad-CAM, SHAP, LIME)
- ✅ Multiple detection strategies (BLIP + YOLO)
- ✅ Comprehensive error handling
- ✅ Responsive UI design
- ✅ Data visualization

### Deployment Optimizations
- ✅ Modular code structure
- ✅ Proper caching strategies
- ✅ Memory optimization
- ✅ Error resilience
- ✅ Version constraints

## 🤝 Contributing

Feel free to submit issues and enhancement requests! This project is designed to be easily extensible.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web framework
- **Hugging Face**: For the transformers library
- **Groq**: For fast LLM inference
- **Ultralytics**: For YOLO models
- **Salesforce**: For BLIP models

---

**Note**: This version is specifically optimized for Streamlit Cloud deployment while maintaining all the advanced features of the original application. The modular design ensures better maintainability and easier debugging.
