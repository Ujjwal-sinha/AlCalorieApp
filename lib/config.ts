// Configuration system for AI Calorie App
// Centralizes all configurable values to avoid hardcoding

export interface AppConfig {
    // API Configuration
    api: {
        timeout: number
        healthCheckTimeout: number
        baseUrl: string
        nextjsApiBase: string
        retryAttempts: number
    }

    // File Upload Configuration
    upload: {
        maxFileSize: number // in bytes
        allowedTypes: string[]
        maxFileSizeMB: number // for display purposes
    }

    // Analysis Configuration
    analysis: {
        progressUpdateInterval: number // milliseconds
        maxProgressIncrement: number
        defaultConfidence: number
        estimatedAnalysisTime: number // seconds
        modelCount: number
        blipBatchSize: number // number of prompts to process in parallel
        blipMaxTokens: number
        blipTemperature: number
    }

    // UI Configuration
    ui: {
        animationDuration: number
        transitionDuration: number
        progressBarHeight: string
        cardPadding: string
        borderRadius: string
    }

    // Nutrition Configuration
    nutrition: {
        defaultCalorieTarget: number
        macroCaloriesPerGram: {
            protein: number
            carbs: number
            fats: number
        }
        defaultPortionSizes: {
            small: number
            medium: number
            large: number
        }
    }

    // Detection Configuration (from Python implementation)
    detection: {
        blipPrompts: string[]
        yoloConfidence: number
        yoloIou: number
        contrastEnhancement: number
        maxNewTokens: number
        numBeams: number
        temperature: number
        topP: number
        repetitionPenalty: number
        essentialFoodKeywords: string[]
        nonFoodKeywords: string[]
        blipModel: string
        yoloModel: string
        llmModel: string
    }
    
    // Mock Data Configuration
    mockData: {
        analysisTime: string
        confidence: string
        defaultCalories: number
        sampleFoodItems: Array<{
            name: string
            calories: number
            protein: number
            carbs: number
            fats: number
        }>
    }
}

// Default configuration
const defaultConfig: AppConfig = {
    api: {
        timeout: 30000, // 30 seconds
        healthCheckTimeout: 5000, // 5 seconds
        baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
        nextjsApiBase: '/api',
        retryAttempts: 3
    },

    upload: {
        maxFileSize: 10 * 1024 * 1024, // 10MB
        allowedTypes: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
        maxFileSizeMB: 10
    },

    analysis: {
        progressUpdateInterval: 200, // milliseconds
        maxProgressIncrement: 30,
        defaultConfidence: 96.2,
        estimatedAnalysisTime: 12, // seconds
        modelCount: 4,
        blipBatchSize: 3, // process 3 prompts at a time
        blipMaxTokens: 200,
        blipTemperature: 0.3
    },

    ui: {
        animationDuration: 200, // milliseconds
        transitionDuration: 300, // milliseconds
        progressBarHeight: '8px',
        cardPadding: '24px',
        borderRadius: '12px'
    },

    nutrition: {
        defaultCalorieTarget: 2000,
        macroCaloriesPerGram: {
            protein: 4,
            carbs: 4,
            fats: 9
        },
        defaultPortionSizes: {
            small: 200,
            medium: 400,
            large: 600
        }
    },

    detection: {
        // Exact prompts from Python implementation
        blipPrompts: [
            "List every food item, ingredient, dish, sauce, and beverage visible in this image:",
            "What are all the foods, vegetables, fruits, meats, grains, and drinks you can see?",
            "Identify each food component including main dishes, sides, garnishes, and condiments:"
        ],
        // YOLO parameters from Python
        yoloConfidence: 0.1,
        yoloIou: 0.4,
        // Image enhancement from Python
        contrastEnhancement: 1.3,
        // BLIP generation parameters from Python
        maxNewTokens: 300,
        numBeams: 6,
        temperature: 0.3,
        topP: 0.9,
        repetitionPenalty: 1.1,
        // Food keywords from Python (essential_food_keywords)
        essentialFoodKeywords: [
            'apple', 'banana', 'orange', 'tomato', 'potato', 'carrot', 'onion', 'garlic',
            'chicken', 'beef', 'pork', 'fish', 'egg', 'cheese', 'milk', 'bread', 'rice',
            'pasta', 'pizza', 'burger', 'sandwich', 'salad', 'soup', 'cake', 'cookie',
            'coffee', 'tea', 'juice', 'water', 'sauce', 'oil', 'butter', 'salt', 'pepper',
            'lettuce', 'spinach', 'broccoli', 'corn', 'beans', 'meat', 'vegetable', 'fruit'
        ],
        // Non-food keywords from Python
        nonFoodKeywords: ['plate', 'bowl', 'cup', 'glass', 'table'],
        // Model names from Python
        blipModel: "Salesforce/blip-image-captioning-base",
        yoloModel: "yolov8n.pt",
        llmModel: "llama3-8b-8192"
    },

    mockData: {
        analysisTime: '~12 seconds',
        confidence: '96.2%',
        defaultCalories: 400,
        sampleFoodItems: [
            { name: 'Grilled chicken breast', calories: 250, protein: 46, carbs: 0, fats: 5 },
            { name: 'Mixed green salad', calories: 25, protein: 2, carbs: 5, fats: 0.3 },
            { name: 'Brown rice', calories: 180, protein: 4, carbs: 36, fats: 1.5 },
            { name: 'Olive oil dressing', calories: 90, protein: 0, carbs: 0, fats: 10 }
        ]
    }
}

// Environment-specific overrides
const getEnvironmentConfig = (): Partial<AppConfig> => {
    const env = process.env.NODE_ENV || 'development'

    switch (env) {
        case 'production':
            return {
                api: {
                    ...defaultConfig.api,
                    timeout: 45000, // Longer timeout for production
                    retryAttempts: 5
                },
                analysis: {
                    ...defaultConfig.analysis,
                    progressUpdateInterval: 150 // Faster updates in production
                }
            }

        case 'development':
            return {
                api: {
                    ...defaultConfig.api,
                    timeout: 60000 // Longer timeout for development/debugging
                },
                analysis: {
                    ...defaultConfig.analysis,
                    progressUpdateInterval: 300 // Slower updates for debugging
                }
            }

        case 'test':
            return {
                api: {
                    ...defaultConfig.api,
                    timeout: 10000, // Shorter timeout for tests
                    retryAttempts: 1
                },
                analysis: {
                    ...defaultConfig.analysis,
                    progressUpdateInterval: 50, // Fast updates for tests
                    estimatedAnalysisTime: 2 // Quick analysis for tests
                }
            }

        default:
            return {}
    }
}

// Merge default config with environment-specific overrides
const createConfig = (): AppConfig => {
    const envConfig = getEnvironmentConfig()

    return {
        api: { ...defaultConfig.api, ...envConfig.api },
        upload: { ...defaultConfig.upload, ...envConfig.upload },
        analysis: { ...defaultConfig.analysis, ...envConfig.analysis },
        ui: { ...defaultConfig.ui, ...envConfig.ui },
        nutrition: { ...defaultConfig.nutrition, ...envConfig.nutrition },
        detection: { ...defaultConfig.detection, ...envConfig.detection },
        mockData: { ...defaultConfig.mockData, ...envConfig.mockData }
    }
}

// Export the final configuration
export const config = createConfig()

// Utility functions for common config access
export const getApiConfig = () => config.api
export const getUploadConfig = () => config.upload
export const getAnalysisConfig = () => config.analysis
export const getUIConfig = () => config.ui
export const getNutritionConfig = () => config.nutrition
export const getDetectionConfig = () => config.detection
export const getMockDataConfig = () => config.mockData

// Validation functions
export const validateConfig = (): boolean => {
    try {
        // Validate required environment variables
        if (!config.api.baseUrl) {
            console.warn('API base URL not configured')
            return false
        }

        // Validate numeric values
        if (config.upload.maxFileSize <= 0) {
            console.error('Invalid max file size configuration')
            return false
        }

        if (config.analysis.progressUpdateInterval <= 0) {
            console.error('Invalid progress update interval configuration')
            return false
        }

        // Validate arrays
        if (!Array.isArray(config.upload.allowedTypes) || config.upload.allowedTypes.length === 0) {
            console.error('Invalid allowed file types configuration')
            return false
        }

        return true
    } catch (error) {
        console.error('Configuration validation failed:', error)
        return false
    }
}

// Dynamic configuration updates (for runtime changes)
export const updateConfig = (updates: Partial<AppConfig>): void => {
    Object.assign(config, {
        api: { ...config.api, ...updates.api },
        upload: { ...config.upload, ...updates.upload },
        analysis: { ...config.analysis, ...updates.analysis },
        ui: { ...config.ui, ...updates.ui },
        nutrition: { ...config.nutrition, ...updates.nutrition },
        mockData: { ...config.mockData, ...updates.mockData }
    })
}

// AppConfig is already exported above as an interface

// Initialize and validate configuration on module load
if (typeof window !== 'undefined') {
    // Client-side validation
    if (!validateConfig()) {
        console.warn('Configuration validation failed - some features may not work correctly')
    }
}

export default config