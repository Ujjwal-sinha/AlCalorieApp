# GROQ AI Integration Guide

This guide explains how to set up and use the GROQ AI integration for comprehensive food analysis in the Food Analyzer application.

## Overview

The GROQ integration provides advanced AI-powered nutrition analysis using the Llama3-8b-8192 model through LangChain. It generates detailed nutritional insights, health scores, and personalized recommendations.

## Features

- **Comprehensive Food Analysis**: Detailed breakdown of nutritional content
- **Health Scoring**: AI-generated nutritional quality score (1-10)
- **Personalized Recommendations**: Actionable health and dietary advice
- **Dietary Considerations**: Allergen information and dietary restrictions
- **Executive Summary**: Concise overview of meal nutritional profile

## Setup Instructions

### 1. Get GROQ API Key

1. Visit [GROQ Console](https://console.groq.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the API key (starts with `gsk_`)

### 2. Environment Configuration

#### Backend Setup

Add the GROQ API key to your environment variables:

```bash
# .env file
GROQ_API_KEY=gsk_your_api_key_here
```

#### Frontend Setup

The frontend automatically receives GROQ analysis through the backend API.

### 3. Install Dependencies

#### Backend Dependencies

The following packages are automatically installed:

```json
{
  "@langchain/groq": "^0.1.3",
  "@langchain/core": "^0.2.21"
}
```

#### Python Dependencies (AlCalorieApp-Cloud)

```bash
pip install langchain-groq langchain-core
```

## Usage

### Backend API Endpoints

#### 1. Integrated Analysis (Recommended)

The GROQ analysis is automatically included in the main analysis endpoint:

```bash
POST /api/analysis/advanced
Content-Type: multipart/form-data

{
  "image": <file>,
  "context": "optional meal context"
}
```

Response includes:
```json
{
  "success": true,
  "groq_analysis": {
    "summary": "Executive summary of the meal...",
    "detailedAnalysis": "Full comprehensive analysis...",
    "healthScore": 8,
    "recommendations": [
      "Consider adding more vegetables",
      "Monitor portion sizes"
    ],
    "dietaryConsiderations": [
      "Contains common allergens",
      "Suitable for high-protein diets"
    ]
  }
}
```

#### 2. Standalone GROQ Analysis

```bash
POST /api/analysis/groq
Content-Type: application/json

{
  "detectedFoods": ["chicken", "rice", "broccoli"],
  "nutritionalData": {
    "total_calories": 450,
    "total_protein": 35,
    "total_carbs": 45,
    "total_fats": 12
  },
  "foodItems": [...],
  "imageDescription": "Optional description",
  "mealContext": "Optional context"
}
```

#### 3. Health Check

```bash
GET /api/analysis/groq/health
```

### Frontend Integration

The GROQ analysis is automatically displayed in the AnalysisResults component when available:

```tsx
import GroqAnalysis from './components/GroqAnalysis';

// Automatically rendered when groq_analysis is present
{result.groq_analysis && (
  <GroqAnalysis analysis={result.groq_analysis} />
)}
```

### Python Integration (AlCalorieApp-Cloud)

```python
from utils.models import generate_comprehensive_food_analysis, load_models

# Load models
models = load_models()

# Generate analysis
analysis = generate_comprehensive_food_analysis(
    detected_foods=['chicken', 'rice', 'broccoli'],
    nutritional_data={
        'total_calories': 450,
        'total_protein': 35,
        'total_carbs': 45,
        'total_fats': 12
    },
    models=models
)

print(f"Health Score: {analysis['health_score']}")
print(f"Summary: {analysis['summary']}")
print(f"Recommendations: {analysis['recommendations']}")
```

## Analysis Output Format

The GROQ analysis provides structured output in the following format:

### Executive Summary
- 2-3 sentence overview of nutritional profile
- Key health implications

### Detailed Nutritional Analysis
- Food-by-food breakdown
- Portion size estimates
- Cooking method impact

### Meal Composition Assessment
- Meal type (Breakfast/Lunch/Dinner/Snack)
- Cuisine style identification
- Portion size classification
- Cooking methods used
- Main macronutrient focus

### Nutritional Quality Score (1-10)
- AI-generated score with justification
- Based on nutritional balance and variety

### Strengths
- What's nutritionally good about the meal
- 2-3 key positive points

### Areas for Improvement
- Specific suggestions for better nutrition
- 2-3 actionable improvements

### Health Recommendations
1. **Immediate Suggestions**: Specific tips for this meal
2. **Portion Adjustments**: If needed
3. **Complementary Foods**: What to add for better nutrition
4. **Timing Considerations**: Best time to eat this meal

### Dietary Considerations
- **Allergen Information**: Common allergens present
- **Dietary Restrictions**: Vegan/Vegetarian/Gluten-free compatibility
- **Blood Sugar Impact**: High/Medium/Low glycemic impact
- **Special Considerations**: Other important dietary notes

## Error Handling

### Common Issues

1. **API Key Not Configured**
   ```
   Error: GROQ_API_KEY not configured
   Solution: Set the GROQ_API_KEY environment variable
   ```

2. **Network Issues**
   ```
   Error: GROQ API error
   Solution: Check internet connection and GROQ service status
   ```

3. **Rate Limiting**
   ```
   Error: Rate limit exceeded
   Solution: Wait and retry, or upgrade GROQ plan
   ```

### Fallback Behavior

When GROQ analysis fails:
- Application continues to work with basic analysis
- User receives notification about GROQ unavailability
- Basic nutritional data is still provided

## Testing

### Backend Testing

```bash
# Build the project
npm run build

# Test GROQ integration
node test_groq_integration.js
```

### Frontend Testing

The GROQ analysis component can be tested by:
1. Uploading an image with food
2. Checking if the GROQ analysis section appears
3. Verifying the health score and recommendations

## Performance Considerations

- **Response Time**: GROQ analysis adds ~2-5 seconds to processing time
- **Token Usage**: Each analysis uses ~500-1000 tokens
- **Cost**: Approximately $0.001-0.005 per analysis (based on GROQ pricing)

## Security

- API keys are stored securely in environment variables
- No API keys are exposed to the frontend
- All GROQ requests go through the backend

## Troubleshooting

### GROQ Analysis Not Appearing

1. Check if GROQ_API_KEY is set
2. Verify GROQ service health: `GET /api/analysis/groq/health`
3. Check browser console for errors
4. Verify backend logs for GROQ-related errors

### Poor Analysis Quality

1. Ensure detected foods are accurate
2. Provide clear image descriptions
3. Include meal context when available
4. Check if nutritional data is complete

### Performance Issues

1. Monitor GROQ API response times
2. Check network connectivity
3. Verify GROQ service status
4. Consider implementing caching for repeated analyses

## Support

For issues with GROQ integration:
1. Check GROQ service status: https://status.groq.com/
2. Review GROQ documentation: https://console.groq.com/docs
3. Check application logs for detailed error messages
4. Verify API key permissions and quotas

## Future Enhancements

- Caching of similar analyses
- Batch processing for multiple images
- Custom analysis templates
- Integration with dietary preferences
- Historical analysis tracking
