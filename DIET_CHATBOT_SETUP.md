# 🤖 Diet Chatbot Setup Guide

## Overview
Your diet chatbot is already well-built with both basic and enhanced versions! Here's how to get it running and test it.

## 🚀 Quick Start

### 1. Backend Setup
```bash
cd food-analyzer-backend
npm install
npm run dev
```

### 2. Frontend Setup
```bash
cd food-analyzer-frontend
npm install
npm run dev
```

### 3. Test the Chatbot
```bash
# Run the test script from the root directory
node test_diet_chatbot.js
```

## 📋 Prerequisites

### Required Environment Variables
Make sure your `food-analyzer-backend/.env` file has:
```env
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile
```

### API Key Setup
1. Get a GROQ API key from [console.groq.com](https://console.groq.com)
2. Add it to your `.env` file
3. Restart the backend server

## 🎯 Features Available

### Basic Diet Chat (`DietChat.tsx`)
- ✅ AI-powered nutrition advice using GROQ API
- ✅ Smart suggestions and related topics
- ✅ Confidence scoring for responses
- ✅ Connection status monitoring
- ✅ Sample questions for easy start
- ✅ Modern, responsive UI
- ✅ Message history tracking

### Enhanced Diet Chat (`EnhancedDietChat.tsx`)
- ✅ All basic features PLUS:
- 🆕 Food image upload and analysis
- 🆕 User profile management
- 🆕 Quick action buttons
- 🆕 Integration with food detection system
- 🆕 Personalized advice based on goals
- 🆕 Dietary restrictions support

## 🧪 Testing Your Chatbot

### 1. Health Check
```bash
curl http://localhost:8000/api/analysis/diet-chat/health
```

### 2. Sample Questions
```bash
curl http://localhost:8000/api/analysis/diet-chat/sample-questions
```

### 3. Chat Test
```bash
curl -X POST http://localhost:8000/api/analysis/diet-chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I eat to lose weight?"}'
```

### 4. Automated Testing
```bash
node test_diet_chatbot.js
```

## 🎨 Using the Components

### Basic Usage
```tsx
import DietChat from './components/DietChat';

function App() {
  return <DietChat />;
}
```

### Enhanced Usage
```tsx
import EnhancedDietChat from './components/EnhancedDietChat';

function App() {
  return <EnhancedDietChat />;
}
```

### Demo with Toggle
```tsx
import DietChatDemo from './components/DietChatDemo';

function App() {
  return <DietChatDemo />;
}
```

## 🔧 Customization Options

### 1. Modify Sample Questions
Edit the `getSampleQuestions()` method in `DietChatService.ts`

### 2. Adjust AI Model
Change the model in your `.env` file:
```env
LLM_MODEL=mixtral-8x7b-32768  # Faster, less detailed
LLM_MODEL=llama-3.3-70b-versatile  # Slower, more detailed
```

### 3. Customize UI Theme
Modify colors in `DietChat.css`:
```css
/* Change primary color */
:root {
  --primary-color: #4CAF50;  /* Green */
  --primary-hover: #45a049;
}
```

### 4. Add More Quick Actions
Edit the `quickActions` object in `EnhancedDietChat.tsx`

## 🐛 Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not configured"**
   - Add your API key to `.env` file
   - Restart the backend server

2. **"Connection failed"**
   - Check if backend is running on port 8000
   - Verify internet connection
   - Check GROQ API status

3. **"No response from AI"**
   - Check API key validity
   - Try a different model
   - Check rate limits

4. **Image upload not working**
   - Ensure backend food detection is set up
   - Check file size limits
   - Verify image format (jpg, png, webp)

### Debug Mode
Enable debug logging in your backend:
```env
LOG_LEVEL=debug
```

## 📈 Performance Tips

1. **Reduce Response Time**
   - Use faster model: `mixtral-8x7b-32768`
   - Implement response caching
   - Optimize image processing

2. **Handle Rate Limits**
   - Add request queuing
   - Implement exponential backoff
   - Cache common responses

3. **Improve User Experience**
   - Add typing indicators
   - Implement message streaming
   - Add voice input/output

## 🚀 Next Steps

1. **Test Current Setup**: Run the test script
2. **Choose Your Version**: Basic or Enhanced
3. **Customize**: Modify colors, questions, features
4. **Deploy**: Use Vercel, Netlify, or your preferred platform
5. **Monitor**: Add analytics and error tracking

## 📞 Support

If you encounter issues:
1. Check the console for error messages
2. Run the test script to identify problems
3. Verify all environment variables are set
4. Ensure both frontend and backend are running

Your diet chatbot is ready to help users with nutrition advice, meal planning, and healthy eating guidance! 🎉