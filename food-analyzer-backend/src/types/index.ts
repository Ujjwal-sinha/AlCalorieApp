// Core types for food analysis application

export interface FoodItem {
  name: string;
  calories: number;
  protein: number;
  carbs: number;
  fats: number;
  quantity?: string;
  confidence?: number;
}

export interface NutritionalData {
  total_calories: number;
  total_protein: number;
  total_carbs: number;
  total_fats: number;
  items: FoodItem[];
}

export interface NutritionData {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
}

export interface AnalysisResult {
  success: boolean;
  error?: string;
  description: string;
  analysis: string;
  nutritional_data: NutritionalData;
  confidence_scores?: Record<string, number>;
  food_details?: Record<string, FoodDetails>;
  detection_methods?: Record<string, string>;
  image_analysis?: ImageAnalysis;
  detected_foods?: string[];
  confidence?: number;
  processing_time?: number;
  model_used?: string;
  sessionId?: string;
  insights?: string[];
}

export interface ExpertAnalysisResult {
  success: boolean;
  description: string;
  analysis: string;
  nutritional_data: NutritionalData;
  detected_foods: string[];
  confidence: number;
  processing_time: number;
  model_used: string;
  insights: string[];
  sessionId: string;
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

export interface ColorProfile {
  red: [number, number];
  green: [number, number];
  blue: [number, number];
}

export interface VisualFeatures {
  color_profiles: Record<string, ColorProfile>;
  texture_patterns: Record<string, string>;
  shape_characteristics: Record<string, string>;
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

export interface DetectionResult {
  method: string;
  foods: Set<string>;
  weight: number;
}

export interface ModelStatus {
  'BLIP (Image Analysis)': boolean;
  'ViT-B/16 (Vision Transformer)': boolean;
  'Swin Transformer': boolean;
  'CLIP (Similarity Scoring)': boolean;
  'LLM (Nutrition Analysis)': boolean;
  'YOLO (Object Detection)': boolean;
  'CNN (Visualizations)': boolean;
  [key: string]: boolean; // Index signature for dynamic access
}

export interface AIModel {
  name: string;
  type: string;
  loaded: boolean;
  instance?: any;
  processor?: any;
  config?: any;
}

export interface ModelConfig {
  name: string;
  type: 'vision' | 'language' | 'detection';
  enabled: boolean;
  confidence_threshold: number;
  model_path?: string;
  api_endpoint?: string;
}

export interface CulturalInfo {
  origin: string;
  history: string;
  cultural_significance: string;
  traditional_uses: string[];
}

export interface Recipe {
  title: string;
  description: string;
  source: string;
}

export interface NutritionalBalance {
  balance_score: number;
  categories: Record<string, string[]>;
  recommendations: string[];
  total_foods: number;
}

export interface HistoryEntry {
  id: string;
  timestamp: Date;
  image_url?: string;
  analysis_result: AnalysisResult;
  context?: string;
}

export interface TrendData {
  date: string;
  calories: number;
  protein: number;
  carbs: number;
  fats: number;
}

export interface ChartData {
  name: string;
  value: number;
  color?: string;
}

export interface ProcessedImage {
  buffer: Buffer;
  width: number;
  height: number;
  format: string;
}

export interface AnalysisRequest {
  image: ProcessedImage;
  context?: string;
  confidence_threshold?: number;
  ensemble_threshold?: number;
  use_advanced_detection?: boolean;
  model_type?: string;
}

export interface FoodAnalysisContext {
  meal_type?: string;
  cuisine_style?: string;
  dietary_restrictions?: string[];
  health_goals?: string[];
  user_preferences?: Record<string, any>;
}