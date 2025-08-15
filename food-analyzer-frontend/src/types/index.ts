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

export interface AnalysisResult {
  success: boolean;
  description?: string;
  analysis?: string;
  nutritional_data?: {
    total_calories: number;
    total_protein: number;
    total_carbs: number;
    total_fats: number;
    items: Array<{
      name: string;
      calories: number;
      protein: number;
      carbs: number;
      fats: number;
      confidence?: number;
    }>;
  };
  detected_foods?: string[];
  confidence?: number;
  confidence_scores?: { [key: string]: number };
  food_details?: any;
  image_analysis?: any;
  processing_time?: number;
  model_used?: string;
  sessionId?: string;
  insights?: string[];
  detection_methods?: string[];
  error?: string;
  groq_analysis?: {
    summary: string;
    detailedAnalysis: string;
    healthScore: number;
    recommendations: string[];
    dietaryConsiderations: string[];
    foodItemReports?: {
      [foodName: string]: {
        nutritionProfile: string;
        healthBenefits: string;
        nutritionalHistory: string;
        cookingMethods: string;
        servingSuggestions: string;
        potentialConcerns: string;
        alternatives: string;
      };
    };
    dailyMealPlan?: {
      breakfast: string[];
      lunch: string[];
      dinner: string[];
      snacks: string[];
      hydration: string[];
      totalCalories: number;
      notes: string;
    };
  };
  diet_chat_response?: {
    answer: string;
    suggestions: string[];
    relatedTopics: string[];
    confidence: number;
  };
  model_info?: {
    detection_count: number;
    total_confidence: number;
    model_performance: { 
      [key: string]: { 
        success: boolean; 
        detection_count: number; 
        error?: string 
      } 
    };
    detailed_detections: Array<{
      food: string;
      count: number;
      methods: string[];
      avg_confidence: number;
      model_details: any[];
    }>;
  };
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