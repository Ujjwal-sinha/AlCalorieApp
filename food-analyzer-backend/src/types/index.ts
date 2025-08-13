// Core types for food analysis backend

export interface FoodItem {
  name: string;
  calories: number;
  protein: number;
  carbs: number;
  fats: number;
  quantity?: string;
  confidence?: number;
}

export interface NutritionData {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
}

export interface NutritionalData {
  total_calories: number;
  total_protein: number;
  total_carbs: number;
  total_fats: number;
  items: FoodItem[];
}

export interface FoodAnalysisContext {
  sessionId?: string;
  timestamp: string;
  processingTime: number;
  confidence: number;
  detectionMethods: string[];
}

export interface AnalysisResult {
  success: boolean;
  error?: string;
  description?: string;
  analysis?: string;
  nutritional_data?: NutritionalData;
  confidence_scores?: Record<string, number>;
  food_details?: Record<string, FoodDetails>;
  detection_methods?: Record<string, string>;
  image_analysis?: ImageAnalysis;
  // New fields for enhanced analysis
  sessionId?: string;
  detectedFoods?: string[];
  nutritionData?: Map<string, NutritionData>;
  totalNutrition?: NutritionData;
  insights?: string[];
  detectionMethods?: string[];
  processingTime?: number;
  confidence?: number;
  timestamp?: string;
}

export interface FoodDetails {
  category: string;
  common_name: string;
  nutritional_category: string;
}

export interface ImageAnalysis {
  total_foods: number;
  complexity: 'simple' | 'medium' | 'high';
  detection_quality: string;
}

export interface ModelConfig {
  name: string;
  type: 'vision' | 'language' | 'detection';
  enabled: boolean;
  confidence_threshold: number;
  model_path?: string;
  api_endpoint?: string;
}

export interface DetectionResult {
  method: string;
  foods: string[];
  confidence_scores: Record<string, number>;
  weight: number;
}

export interface ModelStatus {
  [key: string]: boolean;
}

export interface FoodVocabulary {
  vocabulary: string[];
  categories: FoodCategories;
  visual_features: VisualFeatures;
}

export interface FoodCategories {
  proteins: string[];
  vegetables: string[];
  fruits: string[];
  grains: string[];
  dairy: string[];
  prepared: string[];
  snacks: string[];
  desserts: string[];
  beverages: string[];
}

export interface VisualFeatures {
  color_profiles: Record<string, ColorProfile>;
  texture_patterns: Record<string, string>;
  shape_characteristics: Record<string, string>;
}

export interface ColorProfile {
  red: [number, number];
  green: [number, number];
  blue: [number, number];
}

export interface AIModel {
  name: string;
  type: string;
  loaded: boolean;
  instance?: any;
  processor?: any;
  config?: any;
}

export interface ProcessedImage {
  buffer: Buffer;
  width: number;
  height: number;
  channels: number;
  format: string;
}

export interface AnalysisRequest {
  image: ProcessedImage;
  context?: string | undefined;
  model_type?: string | undefined;
  confidence_threshold?: number | undefined;
  ensemble_threshold?: number | undefined;
  use_advanced_detection?: boolean | undefined;
}