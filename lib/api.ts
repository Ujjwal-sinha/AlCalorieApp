// API utilities for AI Calorie App

import axios from 'axios'
import { AnalysisResult, ApiResponse } from '../types'

// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('❌ API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// API functions
export const apiClient = {
  // Health check
  async healthCheck(): Promise<ApiResponse> {
    try {
      const response = await api.get('/health')
      return {
        success: true,
        data: response.data
      }
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message
      }
    }
  },

  // Analyze food image
  async analyzeFood(imageFile: File, context: string = ''): Promise<ApiResponse<AnalysisResult>> {
    try {
      const formData = new FormData()
      formData.append('file', imageFile)
      formData.append('context', context)

      const response = await api.post('/api/analyze-file', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      return {
        success: true,
        data: response.data
      }
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message
      }
    }
  },

  // Analyze food from base64 image
  async analyzeFoodBase64(base64Image: string, context: string = ''): Promise<ApiResponse<AnalysisResult>> {
    try {
      const response = await api.post('/api/analyze', {
        image: base64Image,
        context: context,
        format: 'image/jpeg'
      })

      return {
        success: true,
        data: response.data
      }
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message
      }
    }
  }
}

// Utility functions
export const utils = {
  // Convert file to base64
  async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => {
        const result = reader.result as string
        // Remove data:image/jpeg;base64, prefix
        const base64 = result.split(',')[1]
        resolve(base64)
      }
      reader.onerror = (error) => reject(error)
    })
  },

  // Format calories
  formatCalories(calories: number): string {
    return new Intl.NumberFormat('en-US').format(Math.round(calories))
  },

  // Format macronutrients
  formatMacro(value: number, unit: string = 'g'): string {
    return `${value.toFixed(1)}${unit}`
  },

  // Calculate macro percentages
  calculateMacroPercentages(protein: number, carbs: number, fats: number) {
    const totalCalories = (protein * 4) + (carbs * 4) + (fats * 9)
    
    if (totalCalories === 0) {
      return { protein: 0, carbs: 0, fats: 0 }
    }

    return {
      protein: Math.round(((protein * 4) / totalCalories) * 100),
      carbs: Math.round(((carbs * 4) / totalCalories) * 100),
      fats: Math.round(((fats * 9) / totalCalories) * 100)
    }
  },

  // Validate image file
  validateImageFile(file: File): { valid: boolean; error?: string } {
    const maxSize = 10 * 1024 * 1024 // 10MB
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']

    if (!allowedTypes.includes(file.type)) {
      return {
        valid: false,
        error: 'Please upload a valid image file (JPEG, PNG, or WebP)'
      }
    }

    if (file.size > maxSize) {
      return {
        valid: false,
        error: 'Image file size must be less than 10MB'
      }
    }

    return { valid: true }
  },

  // Generate mock data for development
  generateMockAnalysis(): AnalysisResult {
    return {
      success: true,
      analysis: `## COMPREHENSIVE FOOD ANALYSIS

### IDENTIFIED FOOD ITEMS:
- Item: Grilled salmon fillet (150g), Calories: 280, Protein: 39g, Carbs: 0g, Fats: 12g
- Item: Quinoa salad (100g), Calories: 120, Protein: 4g, Carbs: 22g, Fats: 2g
- Item: Roasted vegetables (80g), Calories: 45, Protein: 2g, Carbs: 10g, Fats: 0.5g

### NUTRITIONAL TOTALS:
- Total Calories: 445 kcal
- Total Protein: 45g (40% of calories)
- Total Carbohydrates: 32g (29% of calories)
- Total Fats: 14.5g (31% of calories)

### MEAL COMPOSITION ANALYSIS:
- **Meal Type**: Dinner
- **Cuisine Style**: Mediterranean/Healthy
- **Portion Size**: Medium
- **Main Macronutrient**: Protein-rich

### NUTRITIONAL QUALITY ASSESSMENT:
- **Strengths**: Excellent protein source, omega-3 fatty acids, complex carbohydrates
- **Areas for Improvement**: Well-balanced meal
- **Missing Nutrients**: Could add more colorful vegetables

### HEALTH RECOMMENDATIONS:
1. **Excellent omega-3 source** - Salmon provides heart-healthy fats
2. **Complete protein** - Great for muscle maintenance
3. **Balanced macronutrients** - Ideal post-workout meal`,
      food_items: [
        { item: 'Grilled salmon fillet', description: 'Grilled salmon fillet - 280 calories', calories: 280 },
        { item: 'Quinoa salad', description: 'Quinoa salad - 120 calories', calories: 120 },
        { item: 'Roasted vegetables', description: 'Roasted vegetables - 45 calories', calories: 45 }
      ],
      nutritional_data: {
        total_calories: 445,
        total_protein: 45,
        total_carbs: 32,
        total_fats: 14.5,
        items: [
          {
              item: 'Grilled salmon fillet', calories: 280, protein: 39, carbs: 0, fats: 12,
              description: ''
          },
          {
              item: 'Quinoa salad', calories: 120, protein: 4, carbs: 22, fats: 2,
              description: ''
          },
          {
              item: 'Roasted vegetables', calories: 45, protein: 2, carbs: 10, fats: 0.5,
              description: ''
          }
        ]
      },
      improved_description: 'grilled salmon fillet, quinoa salad, roasted vegetables',
      detailed: true
    }
  }
}

export default api