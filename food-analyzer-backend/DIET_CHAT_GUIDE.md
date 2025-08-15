# 🍎 AI Diet Chat System

## Overview

The AI Diet Chat system provides an intelligent nutrition assistant powered by the `llama-3.3-70b-versatile` model via GROQ API. Users can ask questions about diet, nutrition, and healthy eating, receiving comprehensive, evidence-based responses with actionable suggestions.

## 🚀 Key Features

### 1. **Intelligent Query Processing**
- **Natural Language Understanding**: Processes questions in natural language
- **Context Awareness**: Considers user history and context for personalized responses
- **Rate Limiting**: Built-in rate limiting to handle API constraints efficiently
- **Error Handling**: Robust error handling with fallback responses

### 2. **Comprehensive Responses**
- **Detailed Answers**: In-depth, evidence-based nutrition advice
- **Actionable Suggestions**: Practical tips and recommendations
- **Related Topics**: Suggests related nutrition topics for further exploration
- **Confidence Scoring**: Shows AI confidence level for each response

### 3. **User Experience**
- **Sample Questions**: Pre-built questions to help users get started
- **Chat Interface**: Modern, responsive chat UI
- **Real-time Feedback**: Typing indicators and loading states
- **Mobile Optimized**: Works perfectly on all devices

## 🏗️ Architecture

### Backend Components

#### `DietChatService`
```typescript
class DietChatService {
  // Singleton pattern for efficient resource management
  private static instance: DietChatService;
  
  // Core methods
  async answerDietQuery(query: DietQuery): Promise<DietResponse>
  async healthCheck(): Promise<HealthStatus>
  getSampleQuestions(): string[]
  
  // Rate limiting and queue management
  private rateLimitedRequest<T>(requestFn: () => Promise<T>): Promise<T>
  private processQueue(): Promise<void>
}
```

#### Data Structures
```typescript
interface DietQuery {
  question: string;
  context?: string;
  userHistory?: string[];
}

interface DietResponse {
  answer: string;
  suggestions: string[];
  relatedTopics: string[];
  confidence: number;
}
```

### Frontend Components

#### `DietChat`
- **Real-time Chat Interface**: Modern chat UI with message bubbles
- **Sample Questions**: Clickable question buttons for easy interaction
- **Response Details**: Shows confidence, suggestions, and related topics
- **Loading States**: Typing indicators and loading animations

## 🔧 Setup Instructions

### 1. Environment Configuration
```bash
# Set GROQ API key
export GROQ_API_KEY=gsk_your_api_key_here

# Or add to .env file
GROQ_API_KEY=gsk_your_api_key_here
```

### 2. Backend Installation
```bash
cd food-analyzer-backend
npm install
npm run build
```

### 3. Frontend Installation
```bash
cd food-analyzer-frontend
npm install
npm run dev
```

### 4. Testing
```bash
# Test diet chat functionality
node test_diet_chat.js

# Test backend health
node check_backend_status.js
```

## 📊 API Endpoints

### POST `/api/analysis/diet-chat`
Submit a diet-related question to the AI assistant.

**Request Body:**
```json
{
  "question": "What should I eat to lose weight healthily?",
  "context": "I'm a 30-year-old looking to lose 10 pounds",
  "userHistory": ["How much protein do I need?"]
}
```

**Response:**
```json
{
  "success": true,
  "answer": "To lose weight healthily, focus on a balanced diet with...",
  "suggestions": [
    "Create a calorie deficit of 500-750 calories per day",
    "Include plenty of protein to preserve muscle mass",
    "Eat more vegetables and whole foods"
  ],
  "relatedTopics": [
    "Protein Requirements",
    "Calorie Counting",
    "Meal Planning"
  ],
  "confidence": 0.85
}
```

### GET `/api/analysis/diet-chat/health`
Check the health status of the diet chat service.

**Response:**
```json
{
  "service": "Diet Chat Service",
  "status": "healthy",
  "available": true
}
```

### GET `/api/analysis/diet-chat/sample-questions`
Get a list of sample questions for the chat interface.

**Response:**
```json
{
  "success": true,
  "questions": [
    "What should I eat to lose weight healthily?",
    "How can I build muscle through diet?",
    "What foods are good for heart health?",
    "How much protein do I need daily?",
    "What's the best diet for diabetes?"
  ]
}
```

## 🎯 Usage Examples

### 1. Basic Query
```typescript
const dietChatService = DietChatService.getInstance();
const result = await dietChatService.answerDietQuery({
  question: "What should I eat to lose weight healthily?"
});
```

### 2. Contextual Query
```typescript
const result = await dietChatService.answerDietQuery({
  question: "How much protein do I need?",
  context: "I'm a 25-year-old male who works out 3 times per week",
  userHistory: ["What should I eat to lose weight?"]
});
```

### 3. Frontend Integration
```typescript
const response = await fetch('/api/analysis/diet-chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: userQuestion,
    context: userContext,
    userHistory: previousQuestions
  })
});
```

## 🎨 User Interface Features

### Chat Interface
- **Message Bubbles**: User and AI messages with different styling
- **Timestamps**: Shows when each message was sent
- **Confidence Indicator**: Visual confidence score with color coding
- **Suggestions Section**: Actionable tips displayed below AI responses
- **Related Topics**: Clickable topic tags for further exploration

### Sample Questions
- **Grid Layout**: Responsive grid of sample questions
- **Hover Effects**: Visual feedback on hover
- **Click to Send**: One-click question submission
- **Loading States**: Disabled state during processing

### Input System
- **Real-time Validation**: Prevents empty submissions
- **Loading States**: Visual feedback during processing
- **Auto-scroll**: Automatically scrolls to new messages
- **Mobile Optimized**: Touch-friendly interface

## 🔍 Error Handling

### Common Issues
1. **API Key Missing**: Ensure `GROQ_API_KEY` is set
2. **Rate Limits**: System automatically handles rate limiting
3. **Network Issues**: Graceful fallback for connection problems
4. **Invalid Queries**: Validation and error messages

### Fallback Mechanisms
- **Default Responses**: Pre-defined responses for common errors
- **Retry Logic**: Automatic retry with exponential backoff
- **User Feedback**: Clear error messages and suggestions
- **Graceful Degradation**: System continues working with reduced functionality

## 📈 Performance Considerations

### Optimization Strategies
1. **Rate Limiting**: Prevents API overload
2. **Request Queuing**: Manages concurrent requests
3. **Caching**: Cache common responses (future enhancement)
4. **Response Streaming**: Stream long responses (future enhancement)

### Monitoring
- **Response Times**: Track query processing duration
- **Success Rates**: Monitor API reliability
- **Error Rates**: Track failure patterns
- **User Engagement**: Monitor feature usage

## 🚀 Future Enhancements

### Planned Features
1. **Conversation Memory**: Remember full conversation context
2. **Personalized Responses**: Learn from user preferences
3. **Voice Integration**: Voice-to-text and text-to-speech
4. **Image Analysis**: Analyze food images in chat
5. **Recipe Suggestions**: Provide recipe recommendations

### Technical Improvements
1. **Response Caching**: Cache common queries
2. **Streaming Responses**: Real-time response streaming
3. **Multi-language Support**: International language support
4. **Advanced Analytics**: Detailed usage analytics

## 📚 Sample Questions

The system includes 10 pre-built sample questions:

1. "What should I eat to lose weight healthily?"
2. "How can I build muscle through diet?"
3. "What foods are good for heart health?"
4. "How much protein do I need daily?"
5. "What's the best diet for diabetes?"
6. "How can I improve my gut health?"
7. "What should I eat before and after workouts?"
8. "How can I reduce inflammation through diet?"
9. "What are the best sources of omega-3?"
10. "How can I boost my immune system with food?"

## 🔧 Configuration

### Model Settings
```typescript
{
  modelName: 'llama-3.3-70b-versatile',
  temperature: 0.7,
  maxTokens: 1500,
  minRequestInterval: 1000 // 1 second between requests
}
```

### Rate Limiting
- **Request Queue**: Manages concurrent requests
- **Minimum Interval**: 1 second between API calls
- **Automatic Retry**: Retries failed requests
- **Error Recovery**: Graceful handling of rate limits

## 📖 Best Practices

### For Users
1. **Be Specific**: Ask detailed questions for better responses
2. **Provide Context**: Include relevant personal information
3. **Follow Suggestions**: Implement the actionable tips provided
4. **Explore Topics**: Click on related topics for more information

### For Developers
1. **Handle Errors**: Implement proper error handling
2. **Rate Limiting**: Respect API rate limits
3. **User Feedback**: Provide clear loading and error states
4. **Accessibility**: Ensure the interface is accessible

## 📞 Support

### Documentation
- **API Reference**: Complete endpoint documentation
- **Code Examples**: Working code samples
- **Troubleshooting**: Common issues and solutions

### Resources
- [GROQ API Documentation](https://console.groq.com/docs)
- [LangChain Documentation](https://js.langchain.com/docs/)
- [React Component Library](https://react.dev/)

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Model**: llama-3.3-70b-versatile  
**Maintainer**: Food Analyzer Team
