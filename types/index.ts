// Type definitions for AI Calorie App

export interface FoodItem {
  item: string
  description: string
  calories: number
  protein: number
  carbs: number
  fats: number
  fiber: number
}

export interface NutritionData {
  total_calories: number
  total_protein: number
  total_carbs: number
  total_fats: number
  items: FoodItem[]
}

export interface AnalysisResult {
  error: string
  success: boolean
  analysis: string
  food_items: FoodItem[]
  nutritional_data: NutritionData
  improved_description: string
  detailed: boolean
}

export interface AnalysisStep {
  label: string
  progress: number
  completed?: boolean
}

// Enhanced types for comprehensive analysis
export interface BLIPResult {
  success: boolean
  description: string
  confidence: number
  detected_items: string[]
  processing_time: number
}

export interface VisualizationResult {
  success: boolean
  imageUrl?: string
  error?: string
  type: 'gradcam' | 'shap' | 'lime' | 'edge'
  processingTime: number
}

export interface ComprehensiveAnalysisResult extends AnalysisResult {
  blip_detection: BLIPResult
  visualizations: {
    gradcam: VisualizationResult
    shap: VisualizationResult
    lime: VisualizationResult
    edge: VisualizationResult
  }
  detection_metadata: {
    success: boolean
    total_items: number
    confidence: number
    detection_methods: string[]
    enhanced_description: string
    processing_time: number
  }
  enhanced: boolean
}

// API Response types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  fallback?: boolean
}

// Configuration types
export interface AppConfig {
  api: {
    timeout: number
    healthCheckTimeout: number
    baseUrl: string
    nextjsApiBase: string
    retryAttempts: number
  }
  upload: {
    maxFileSize: number
    allowedTypes: string[]
    maxFileSizeMB: number
  }
  analysis: {
    progressUpdateInterval: number
    maxProgressIncrement: number
    defaultConfidence: number
    estimatedAnalysisTime: number
    modelCount: number
    blipBatchSize: number
    blipMaxTokens: number
    blipTemperature: number
  }
  ui: {
    animationDuration: number
    transitionDuration: number
    progressBarHeight: string
    cardPadding: string
    borderRadius: string
  }
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

// Analysis options
export interface AnalysisOptions {
  enableVisualizations: boolean
  enableDetailedAnalysis: boolean
  enableFallback: boolean
  maxProcessingTime: number
}

// Visualization configuration
export interface VisualizationConfig {
  imageSize: number
  overlayAlpha: number
  colormap: string
  dpi: number
  saveFormat: string
}

export interface MacroData {
  name: string
  value: number
  color: string
}

export interface WeeklyData {
  day: string
  calories: number
  protein: number
  carbs: number
  fats: number
}

export interface QuickStat {
  label: string
  value: string
  target?: string
  change?: string
  icon: any
  color: string
  bgColor: string
  progress?: number
}

export interface RecentAnalysis {
  id: number
  food: string
  calories: number
  time: string
  accuracy: number
}

export interface ModelStatus {
  name: string
  status: 'Active' | 'Inactive' | 'Error'
  color: 'green' | 'red' | 'yellow'
}

// Form types
export interface AnalyzeFormData {
  image: File | null
  context: string
}

// Chart data types
export interface ChartDataPoint {
  name: string
  value: number
  color?: string
}

// User preferences
export interface UserPreferences {
  calorieTarget: number
  dietaryRestrictions: string[]
  activityLevel: 'low' | 'moderate' | 'high'
  units: 'metric' | 'imperial'
}

// Visualization types
export interface VisualizationData {
  edgeDetection?: string
  gradCam?: string
  shap?: string
  lime?: string
}

export interface AnalysisHistory {
  id: string
  timestamp: Date
  image: string
  result: AnalysisResult
  context?: string
}