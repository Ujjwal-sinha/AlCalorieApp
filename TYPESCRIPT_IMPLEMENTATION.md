# TypeScript Food Analysis Implementation

This document explains the TypeScript equivalents of the Python `agents.py` and `app.py` files, allowing you to run food analysis directly in your Next.js application without requiring the Python backend.

## 🚀 Quick Start

```typescript
import { apiClient } from '@/lib/api'
import FoodAnalyzer from '@/lib/food-analyzer'

// Option 1: Use the enhanced API client (recommended)
const result = await apiClient.analyzeFoodDirect(imageFile, 'lunch meal')

// Option 2: Use the food analyzer directly
const analyzer = new FoodAnalyzer({ enableMockMode: true })
const analysis = await analyzer.analyzeFoodWithEnhancedPrompt('grilled chicken with rice', 'dinner')
```

## 📁 File Structure

```
lib/
├── agents.ts              # TypeScript equivalent of agents.py
├── food-analyzer.ts       # TypeScript equivalent of app.py
├── api.ts                 # Enhanced API client with TypeScript methods
└── food-analysis-example.ts # Usage examples and demonstrations
```

## 🔧 Core Components

### 1. FoodDetectionAgent (`lib/agents.ts`)

Equivalent to Python's `FoodDetectionAgent` class:

```typescript
import { FoodDetectionAgent } from '@/lib/agents'

const agent = new FoodDetectionAgent({
  groqApiKey: 'your-api-key', // optional
  temperature: 0.1,
  maxTokens: 1000
})

// Detect food from image description
const result = await agent.detectFoodFromImageDescription(
  'grilled salmon with vegetables',
  'healthy dinner'
)

// Search for food information
const searchResult = await agent.searchFoodInformation('quinoa salad')
```

**Key Features:**
- ✅ Enhanced food item extraction
- ✅ Comprehensive nutritional analysis
- ✅ Food categorization (protein, vegetable, fruit, etc.)
- ✅ Search integration for unknown foods
- ✅ Mock mode for development

### 2. FoodAnalyzer (`lib/food-analyzer.ts`)

Equivalent to Python's main `app.py` functionality:

```typescript
import FoodAnalyzer from '@/lib/food-analyzer'

const analyzer = new FoodAnalyzer({
  groqApiKey: 'your-api-key',
  enableMockMode: false
})

// Analyze food with enhanced prompt
const analysis = await analyzer.analyzeFoodWithEnhancedPrompt(
  'chicken breast with broccoli and rice',
  'post-workout meal'
)

// Describe image (if you have image processing)
const description = await analyzer.describeImageEnhanced(imageFile)

// Extract food items from text
const foodItems = analyzer.extractFoodItemsFromText(
  'I had pizza, salad, and orange juice for lunch'
)
```

**Key Features:**
- ✅ Comprehensive food analysis
- ✅ Nutritional data extraction
- ✅ Image description processing
- ✅ Text parsing for food items
- ✅ Mock data generation
- ✅ File validation

### 3. Enhanced API Client (`lib/api.ts`)

Extended API client with TypeScript methods:

```typescript
import { apiClient } from '@/lib/api'

// Direct TypeScript analysis (no Python backend needed)
const result = await apiClient.analyzeFoodDirect(imageFile, 'breakfast')

// Analyze food description directly
const descResult = await apiClient.analyzeFoodDescription(
  'oatmeal with berries and nuts',
  'morning meal'
)

// Traditional Python backend analysis (if available)
const backendResult = await apiClient.analyzeFood(imageFile, 'lunch')
```

## 🎯 Usage Examples

### Example 1: Basic Food Analysis

```typescript
import { apiClient } from '@/lib/api'

async function analyzeMyMeal(imageFile: File) {
  try {
    // Use direct TypeScript analysis
    const result = await apiClient.analyzeFoodDirect(imageFile, 'dinner')
    
    if (result.success && result.data) {
      console.log('Food items:', result.data.food_items)
      console.log('Total calories:', result.data.nutritional_data.total_calories)
      console.log('Analysis:', result.data.analysis)
      return result.data
    }
  } catch (error) {
    console.error('Analysis failed:', error)
  }
}
```

### Example 2: Agent-Based Analysis

```typescript
import { initializeAgents } from '@/lib/agents'

async function detailedFoodAnalysis(description: string) {
  const { foodAgent, searchAgent } = initializeAgents({
    enableMockMode: true
  })
  
  if (foodAgent && searchAgent) {
    // Search for additional information
    const searchInfo = await searchAgent.searchFoodInformation(description)
    
    // Perform comprehensive detection
    const detection = await foodAgent.detectFoodFromImageDescription(
      description,
      'detailed analysis'
    )
    
    return { searchInfo, detection }
  }
}
```

### Example 3: Complete Workflow

```typescript
import { exampleCompleteWorkflow } from '@/lib/food-analysis-example'

async function handleImageUpload(imageFile: File) {
  try {
    // This will try multiple analysis methods automatically
    const result = await exampleCompleteWorkflow(imageFile)
    
    console.log(`Analysis completed using: ${result.method}`)
    console.log('Result:', result.result)
    
    return result.result
  } catch (error) {
    console.error('All analysis methods failed:', error)
  }
}
```

## 🔄 Migration from Python

### Python → TypeScript Equivalents

| Python Function | TypeScript Equivalent | Location |
|----------------|----------------------|----------|
| `FoodDetectionAgent` | `FoodDetectionAgent` | `lib/agents.ts` |
| `describe_image_enhanced()` | `describeImageEnhanced()` | `lib/food-analyzer.ts` |
| `analyze_food_with_enhanced_prompt()` | `analyzeFoodWithEnhancedPrompt()` | `lib/food-analyzer.ts` |
| `extract_food_items_from_text()` | `extractFoodItemsFromText()` | `lib/food-analyzer.ts` |
| `extract_items_and_nutrients()` | `extractItemsAndNutrients()` | `lib/food-analyzer.ts` |
| `query_langchain()` | `queryLLM()` (private) | `lib/agents.ts` |

### Key Differences

1. **Async/Await**: All methods are properly async in TypeScript
2. **Type Safety**: Full TypeScript type definitions
3. **Error Handling**: Consistent error handling with try/catch
4. **Mock Mode**: Built-in mock mode for development
5. **Browser Compatible**: Works directly in the browser

## 🛠️ Configuration

### Environment Variables

```env
# Optional - for LLM analysis
NEXT_PUBLIC_GROQ_API_KEY=your_groq_api_key

# API endpoint (defaults to localhost:8000)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Initialization Options

```typescript
// Food Analyzer Configuration
const analyzer = new FoodAnalyzer({
  groqApiKey: 'your-key',           // Optional
  apiEndpoint: 'http://localhost:8000', // Optional
  enableMockMode: false             // Set to true for development
})

// Agent Configuration
const { foodAgent, searchAgent } = initializeAgents({
  groqApiKey: 'your-key',
  temperature: 0.1,
  maxTokens: 1000
})
```

## 🎨 Integration with UI Components

### React Component Example

```typescript
import { useState } from 'react'
import { apiClient } from '@/lib/api'
import { AnalysisResult } from '@/types'

export function FoodAnalysisComponent() {
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  
  const handleFileUpload = async (file: File) => {
    setLoading(true)
    try {
      const response = await apiClient.analyzeFoodDirect(file)
      if (response.success && response.data) {
        setResult(response.data)
      }
    } catch (error) {
      console.error('Analysis failed:', error)
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div>
      <input 
        type="file" 
        accept="image/*"
        onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
      />
      
      {loading && <p>Analyzing...</p>}
      
      {result && (
        <div>
          <h3>Analysis Results</h3>
          <p>Total Calories: {result.nutritional_data.total_calories}</p>
          <ul>
            {result.food_items.map((item, index) => (
              <li key={index}>{item.description}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
```

## 🚀 Performance Benefits

### TypeScript Implementation Advantages:

1. **⚡ Faster**: No Python backend required
2. **🔧 Easier Setup**: No Python dependencies
3. **🌐 Browser Native**: Runs directly in the browser
4. **📱 Mobile Friendly**: Works on mobile devices
5. **🔄 Real-time**: Instant analysis without API calls
6. **💾 Offline Capable**: Can work without internet (in mock mode)

### When to Use Each Method:

| Method | Best For | Speed | Setup |
|--------|----------|-------|-------|
| `analyzeFoodDirect()` | Development, quick prototypes | ⚡⚡⚡ | Easy |
| `analyzeFoodDescription()` | Text-only analysis | ⚡⚡⚡ | Easy |
| `analyzeFood()` | Production with AI models | ⚡⚡ | Complex |
| `analyzeFoodBase64()` | Legacy compatibility | ⚡ | Complex |

## 🧪 Testing

Run the examples to test your implementation:

```typescript
import { foodAnalysisExamples } from '@/lib/food-analysis-example'

// Test direct analysis
const result1 = await foodAnalysisExamples.directAnalysis(imageFile)

// Test API client
const result2 = await foodAnalysisExamples.apiClientAnalysis(imageFile)

// Test complete workflow
const result3 = await foodAnalysisExamples.completeWorkflow(imageFile)

// Test utilities
const utils = foodAnalysisExamples.utilityFunctions()
```

## 🔧 Troubleshooting

### Common Issues:

1. **"Cannot find module '@/types'"**
   - Make sure your `tsconfig.json` paths are configured correctly
   - Check that `types/index.ts` exists

2. **Mock mode not working**
   - Set `enableMockMode: true` in the configuration
   - Check that you're not passing a GROQ API key

3. **Analysis returns empty results**
   - Check the input data format
   - Enable console logging to debug
   - Try the mock mode first

### Debug Mode:

```typescript
// Enable detailed logging
const analyzer = new FoodAnalyzer({
  enableMockMode: true // This will show more logs
})

// Check the browser console for detailed logs
```

## 🎉 Next Steps

1. **Try the examples** in `lib/food-analysis-example.ts`
2. **Integrate with your UI** components
3. **Add your GROQ API key** for real LLM analysis
4. **Customize the analysis prompts** for your use case
5. **Add more food categories** and nutritional data

The TypeScript implementation gives you all the power of the Python backend with the convenience of running directly in your Next.js application! 🚀