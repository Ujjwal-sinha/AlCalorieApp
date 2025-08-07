// Type definitions for AI Calorie App

export interface FoodItem {
  item: string
  description: string
  calories: number
  protein?: number
  carbs?: number
  fats?: number
  fiber?: number
}

export interface NutritionData {
  total_calories: number
  total_protein: number
  total_carbs: number
  total_fats: number
  items: FoodItem[]
}

export interface AnalysisResult {
  success: boolean
  analysis: string
  food_items: FoodItem[]
  nutritional_data: NutritionData
  improved_description: string
  detailed?: boolean
  error?: string
}

export interface AnalysisStep {
  label: string
  progress: number
  completed?: boolean
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

// API Response types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
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