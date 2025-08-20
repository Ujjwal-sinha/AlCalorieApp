# 🍱 YOLO11m Calorie Tracker - Cloud Optimized

An AI-powered calorie tracking application built with Streamlit that analyzes food images using YOLO11m object detection and provides comprehensive nutritional information. This version is specifically optimized for Streamlit Cloud deployment.

## ✨ Features

- 🔍 **YOLO11m Object Detection**: State-of-the-art food identification using YOLO11m model
- 📷 **Advanced Food Detection**: Upload food images for automatic identification using YOLO11m
- 📊 **Comprehensive Nutrition Analysis**: Get detailed calorie and macronutrient breakdown with LLM analysis
- 📈 **Progress Tracking**: Monitor daily calorie intake and goals with visual progress indicators
- 📋 **History Management**: View and manage your food analysis history with detailed insights
- 🎯 **Goal Setting**: Set and track daily calorie targets with progress visualization
- 📊 **Data Visualization**: Interactive charts and analytics dashboard
- 🚀 **Fast Detection**: Optimized YOLO11m model for quick and accurate food detection

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
├── requirements.txt          # Python dependencies (YOLO11m optimized)
├── yolo11m.pt               # YOLO11m model file
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── utils/
│   ├── models.py            # YOLO11m model loading and management
│   ├── expert_food_recognition.py  # YOLO11m food detection system
│   └── ui.py               # UI components and utilities
├── test_yolo11m_integration.py  # Test script for YOLO11m integration
└── README.md               # This file
```

## 🛠️ Technical Architecture

### AI Models Used

- **YOLO11m**: Advanced object detection model for food identification
- **Groq LLM**: For nutritional analysis and recommendations

### Key Optimizations

- **YOLO11m Focus**: Simplified architecture using only YOLO11m for detection
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
- **LangChain**: LLM integration
- **Ultralytics**: YOLO11m model framework

#### Visualization Dependencies
- **Matplotlib**: Data visualization
- **Plotly**: Interactive charts
- **Seaborn**: Statistical visualizations

#### Computer Vision
- **OpenCV**: Computer vision operations
- **Pillow**: Image processing

## 🔍 YOLO11m Features

### Object Detection Capabilities

YOLO11m is trained on the COCO dataset and can detect various food-related objects including:

- **Fruits**: Apple, Orange, Banana
- **Vegetables**: Carrot, Broccoli
- **Food Items**: Pizza, Hot Dog, Sandwich, Cake, Donut, Cookie
- **Utensils**: Cup, Bowl, Spoon, Fork, Knife, Wine Glass, Bottle
- **Furniture**: Chair, Couch, Bed, Dining Table
- **Appliances**: Microwave, Oven, Toaster, Refrigerator

### Detection Process

1. **Image Upload**: User uploads a food image
2. **YOLO11m Detection**: The model identifies objects in the image
3. **Food Classification**: Detected objects are classified as food items
4. **Nutrition Analysis**: LLM provides nutritional information
5. **Results Display**: Comprehensive analysis with visualizations

## 🧪 Testing

Run the integration test to verify YOLO11m functionality:

```bash
python test_yolo11m_integration.py
```

This test will:
- Check if yolo11m.pt file exists
- Test model loading
- Test detection on sample images

## 🚀 Performance

- **Fast Detection**: YOLO11m provides real-time object detection
- **High Accuracy**: State-of-the-art detection performance
- **Memory Efficient**: Optimized for cloud deployment
- **Scalable**: Can handle multiple concurrent users

## 🔧 Customization

### Adding New Food Categories

To add support for new food categories, modify the `food_classes` set in `utils/expert_food_recognition.py`.

### Adjusting Detection Thresholds

Modify the `confidence_threshold` parameter in the `YOLO11mFoodRecognitionSystem` class to adjust detection sensitivity.

## 📊 Usage

1. **Upload Image**: Click "Choose a food image" and select your food photo
2. **Run Analysis**: Click "Run YOLO11m Analysis" to start detection
3. **View Results**: See detected items, nutritional information, and recommendations
4. **Track Progress**: Monitor your daily calorie intake and goals

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the integration script
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For the YOLO11m model
- **Groq**: For LLM capabilities
- **Streamlit**: For the web framework
- **COCO Dataset**: For training data

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the test script for troubleshooting
- Review the model status in the app sidebar

---

**Built with ❤️ using Streamlit & YOLO11m**
