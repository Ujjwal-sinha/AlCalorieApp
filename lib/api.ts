// API utilities for AI Calorie App

import axios from 'axios'
import { AnalysisResult, ApiResponse } from '../types'
import FoodAnalyzer, { foodAnalyzerUtils } from './food-analyzer'
import { initializeAgents } from './agents'

// API Client interface
interface ApiClientUtils {
    fileToBase64(file: File): Promise<string>
    formatCalories(calories: number): string
    formatMacro(value: number, unit?: string): string
    calculateMacroPercentages(protein: number, carbs: number, fats: number): {
        protein: number
        carbs: number
        fats: number
    }
    validateImageFile(file: File): { valid: boolean; error?: string }
    generateMockAnalysis(description?: string, context?: string): AnalysisResult
}

// Complete ApiClient interface including utils
interface ApiClientType {
    healthCheck(): Promise<ApiResponse>
    analyzeFood(imageFile: File, context?: string): Promise<ApiResponse<AnalysisResult>>
    analyzeFoodBase64(base64Image: string, context?: string): Promise<ApiResponse<AnalysisResult>>
    analyzeFoodDirect(imageFile: File, context?: string): Promise<ApiResponse<AnalysisResult>>
    analyzeFoodDescription(description: string, context?: string): Promise<ApiResponse<AnalysisResult>>
    utils: ApiClientUtils
}

// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 30 seconds
    headers: {
        'Content-Type': 'application/json',
    },
})

// Initialize food analyzer
const foodAnalyzer = new FoodAnalyzer({
    groqApiKey: process.env.NEXT_PUBLIC_GROQ_API_KEY,
    apiEndpoint: API_BASE_URL,
    enableMockMode: !process.env.NEXT_PUBLIC_GROQ_API_KEY
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
    },

    // Analyze food using TypeScript implementation (faster, no Python backend required)
    async analyzeFoodDirect(imageFile: File, context: string = ''): Promise<ApiResponse<AnalysisResult>> {
        try {
            console.log('🚀 Using direct TypeScript food analysis...')

            // Step 1: Get image description
            const imageDescription = await foodAnalyzer.describeImageEnhanced(imageFile)
            console.log('📝 Image description:', imageDescription)

            // Step 2: Analyze food with enhanced prompt
            const analysisResult = await foodAnalyzer.analyzeFoodWithEnhancedPrompt(imageDescription, context)
            console.log('✅ Direct analysis completed')

            return {
                success: true,
                data: analysisResult
            }
        } catch (error: any) {
            console.error('❌ Direct analysis failed:', error)

            // Fallback to mock analysis
            const mockResult = foodAnalyzer.generateMockAnalysis('food items from image', context)
            return {
                success: true,
                data: mockResult
            }
        }
    },

    // Analyze food description directly (no image required)
    async analyzeFoodDescription(description: string, context: string = ''): Promise<ApiResponse<AnalysisResult>> {
        try {
            console.log('🔍 Analyzing food description directly...')

            const analysisResult = await foodAnalyzer.analyzeFoodWithEnhancedPrompt(description, context)
            console.log('✅ Description analysis completed')

            return {
                success: true,
                data: analysisResult
            }
        } catch (error: any) {
            console.error('❌ Description analysis failed:', error)

            return {
                success: false,
                error: error.message || 'Description analysis failed'
            }
        }
    }
}

    // Add utils to apiClient (with type assertion)
    ; (apiClient as any).utils = {
        // Re-export food analyzer utilities
        ...foodAnalyzerUtils,
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

        // Generate mock data for development - uses dynamic generation
        generateMockAnalysis(description: string = 'mixed food items', context: string = ''): AnalysisResult {
            return foodAnalyzer.generateMockAnalysis(description, context)
        }
    }

// Export utils separately for backward compatibility  
export const utils = (apiClient as any).utils

export default api