// API client for AI Calorie App
// Handles communication with the backend API

import { config } from './config'
import { AnalysisResult, ComprehensiveAnalysisResult, ApiResponse } from '../types'

class ApiClient {
  analyzeFoodDescription(arg0: string, context: string | undefined) {
    throw new Error('Method not implemented.')
  }
  private baseUrl: string
  private timeout: number
  private retryAttempts: number

  constructor() {
    this.baseUrl = config.api.nextjsApiBase
    this.timeout = config.api.timeout
    this.retryAttempts = config.api.retryAttempts
  }

  /**
   * Analyze food image with comprehensive detection
   */
  async analyzeFoodDirect(
    file: File, 
    context: string = ''
  ): Promise<ApiResponse<ComprehensiveAnalysisResult>> {
    try {
      console.log('📡 Sending food analysis request...')
      
      const formData = new FormData()
      formData.append('file', file)
      formData.append('context', context)

      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), this.timeout)

      const response = await fetch(`${this.baseUrl}/analyze`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      
      if (!result.success) {
        throw new Error(result.error || 'Analysis failed')
      }

      console.log('✅ Food analysis completed successfully')
      return result

    } catch (error) {
      console.error('❌ Food analysis failed:', error)
      
      // Return fallback response
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Analysis failed',
        fallback: true
      }
    }
  }

  /**
   * Analyze food with retry logic
   */
  async analyzeFoodWithRetry(
    file: File, 
    context: string = ''
  ): Promise<ApiResponse<ComprehensiveAnalysisResult>> {
    let lastError: Error | null = null

    for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
      try {
        console.log(`🔄 Attempt ${attempt}/${this.retryAttempts}`)
        
        const result = await this.analyzeFoodDirect(file, context)
        
        if (result.success) {
          return result
        }
        
        lastError = new Error(result.error || 'Analysis failed')
        
      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Unknown error')
        console.warn(`⚠️ Attempt ${attempt} failed:`, lastError.message)
        
        // Wait before retrying (exponential backoff)
        if (attempt < this.retryAttempts) {
          const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000)
          await new Promise(resolve => setTimeout(resolve, delay))
        }
      }
    }

    console.error('❌ All retry attempts failed')
    return {
      success: false,
      error: lastError?.message || 'All retry attempts failed',
      fallback: true
    }
  }

  /**
   * Check API health
   */
  async checkHealth(): Promise<boolean> {
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), config.api.healthCheckTimeout)

      const response = await fetch(`${this.baseUrl}/analyze`, {
        method: 'GET',
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        return false
      }

      const result = await response.json()
      return result.status === 'active'

    } catch (error) {
      console.warn('⚠️ Health check failed:', error)
      return false
    }
  }

  /**
   * Get API status
   */
  async getStatus(): Promise<{
    status: string
    version: string
    features: string[]
  } | null> {
    try {
      const response = await fetch(`${this.baseUrl}/analyze`, {
        method: 'GET',
        signal: AbortSignal.timeout(config.api.healthCheckTimeout)
      })

      if (!response.ok) {
        return null
      }

      const result = await response.json()
      return {
        status: result.status || 'unknown',
        version: result.version || 'unknown',
        features: result.features || []
      }

    } catch (error) {
      console.warn('⚠️ Status check failed:', error)
      return null
    }
  }
}

// Utility functions for data processing
export const utils = {
  /**
   * Format calories for display
   */
  formatCalories(calories: number): string {
    if (calories >= 1000) {
      return `${(calories / 1000).toFixed(1)}k`
    }
    return calories.toString()
  },

  /**
   * Format macronutrients for display
   */
  formatMacro(macro: number): string {
    return macro.toFixed(1)
  },

  /**
   * Generate mock analysis for fallback
   */
  generateMockAnalysis(
    fileName: string, 
    context: string = ''
  ): ComprehensiveAnalysisResult {
    const description = context || fileName.replace(/\.[^/.]+$/, '')
    const estimatedCalories = 400
    
    return {
      success: true,
      analysis: `## BASIC FOOD ANALYSIS

### DETECTED ITEMS:
- Item: ${description}, Calories: ${estimatedCalories} (estimated)

### NUTRITIONAL ESTIMATE:
- Estimated Calories: ${estimatedCalories} kcal
- Estimated Protein: ${(estimatedCalories * 0.15 / 4).toFixed(1)}g
- Estimated Carbs: ${(estimatedCalories * 0.50 / 4).toFixed(1)}g
- Estimated Fats: ${(estimatedCalories * 0.35 / 9).toFixed(1)}g

### NOTE:
This is a basic estimate. For detailed analysis, try uploading a clearer image.`,
      food_items: [{
        item: description,
        description: `${description} - ${estimatedCalories} calories`,
        calories: estimatedCalories,
        protein: estimatedCalories * 0.15 / 4,
        carbs: estimatedCalories * 0.50 / 4,
        fats: estimatedCalories * 0.35 / 9,
        fiber: 3
      }],
      nutritional_data: {
        total_calories: estimatedCalories,
        total_protein: estimatedCalories * 0.15 / 4,
        total_carbs: estimatedCalories * 0.50 / 4,
        total_fats: estimatedCalories * 0.35 / 9,
        items: []
      },
      improved_description: description,
      detailed: false,
      error: '',
      blip_detection: {
        success: false,
        description: description,
        confidence: 0.3,
        detected_items: [description],
        processing_time: 0
      },
      visualizations: {
        gradcam: {
          success: false,
          error: 'Visualization not available',
          type: 'gradcam' as const,
          processingTime: 0
        },
        shap: {
          success: false,
          error: 'Visualization not available',
          type: 'shap' as const,
          processingTime: 0
        },
        lime: {
          success: false,
          error: 'Visualization not available',
          type: 'lime' as const,
          processingTime: 0
        },
        edge: {
          success: false,
          error: 'Visualization not available',
          type: 'edge' as const,
          processingTime: 0
        }
      },
      detection_metadata: {
        success: false,
        total_items: 1,
        confidence: 0.3,
        detection_methods: ['fallback'],
        enhanced_description: description,
        processing_time: 0
      },
      enhanced: false
    }
  },

  /**
   * Validate file for upload
   */
  validateFile(file: File): { valid: boolean; error?: string } {
    // Check file type
    if (!file.type.startsWith('image/')) {
      return { valid: false, error: 'File must be an image' }
    }

    // Check file size
    if (file.size > config.upload.maxFileSize) {
      return { 
        valid: false, 
        error: `File too large (max ${config.upload.maxFileSizeMB}MB)` 
      }
    }

    return { valid: true }
  },

  /**
   * Calculate macro percentages
   */
  calculateMacroPercentages(protein: number, carbs: number, fats: number): {
    protein: number
    carbs: number
    fats: number
  } {
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

  /**
   * Format processing time
   */
  formatProcessingTime(milliseconds: number): string {
    if (milliseconds < 1000) {
      return `${milliseconds}ms`
    }
    return `${(milliseconds / 1000).toFixed(1)}s`
  },

  /**
   * Get confidence level description
   */
  getConfidenceLevel(confidence: number): {
    level: 'high' | 'medium' | 'low'
    description: string
    color: string
  } {
    if (confidence >= 0.8) {
      return {
        level: 'high',
        description: 'High Confidence',
        color: 'text-green-600'
      }
    } else if (confidence >= 0.5) {
      return {
        level: 'medium',
        description: 'Medium Confidence',
        color: 'text-yellow-600'
      }
    } else {
      return {
        level: 'low',
        description: 'Low Confidence',
        color: 'text-red-600'
      }
    }
  }
}

// Create and export API client instance
export const apiClient = new ApiClient()

// Export the class for testing
export { ApiClient }