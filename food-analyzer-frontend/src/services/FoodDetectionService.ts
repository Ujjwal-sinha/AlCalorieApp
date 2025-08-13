import type { FoodCategories, VisualFeatures, AnalysisResult } from '../types';

export class AdvancedFoodDetector {
  private confidenceThreshold: number = 0.3;
  private ensembleThreshold: number = 0.6;
  private models: Record<string, any> = {};
  private apiBaseUrl: string;

  constructor(apiBaseUrl: string = 'http://localhost:8000') {
    this.apiBaseUrl = apiBaseUrl;
  }

  async initializeModels(): Promise<void> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/models/status`);
      if (response.ok) {
        this.models = await response.json();
      }
    } catch (error) {
      console.warn('Failed to initialize models:', error);
    }
  }

  private async loadFoodVocabulary(): Promise<Set<string>> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/food/vocabulary`);
      if (response.ok) {
        const data = await response.json();
        return new Set(data.vocabulary || []);
      }
    } catch (error) {
      console.warn('Failed to load food vocabulary from API:', error);
    }
    return new Set();
  }

  private async loadFoodCategories(): Promise<FoodCategories> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/food/categories`);
      if (response.ok) {
        const data = await response.json();
        return data.categories || {};
      }
    } catch (error) {
      console.warn('Failed to load food categories from API:', error);
    }
    return {
      proteins: [], vegetables: [], fruits: [], grains: [],
      dairy: [], prepared: [], snacks: [], desserts: [], beverages: []
    };
  }

  private async loadVisualFeatures(): Promise<VisualFeatures> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/food/visual-features`);
      if (response.ok) {
        const data = await response.json();
        return data.features || {};
      }
    } catch (error) {
      console.warn('Failed to load visual features from API:', error);
    }
    return {
      color_profiles: {},
      texture_patterns: {},
      shape_characteristics: {}
    };
  }

  async detectFoodsAdvanced(imageFile: File): Promise<AnalysisResult> {
    try {
      // Initialize models if not already done
      if (Object.keys(this.models).length === 0) {
        await this.initializeModels();
      }

      // Create FormData for image upload
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('use_advanced_detection', 'true');
      formData.append('confidence_threshold', this.confidenceThreshold.toString());
      formData.append('ensemble_threshold', this.ensembleThreshold.toString());

      // Send to backend for processing
      const response = await fetch(`${this.apiBaseUrl}/api/analyze/advanced`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        return {
          success: true,
          description: result.description || 'Food analysis completed',
          analysis: result.analysis || 'Analysis completed successfully',
          nutritional_data: result.nutritional_data || {
            total_calories: 0,
            total_protein: 0,
            total_carbs: 0,
            total_fats: 0,
            items: []
          },
          confidence_scores: result.confidence_scores,
          food_details: result.food_details,
          detection_methods: result.detection_methods,
          image_analysis: result.image_analysis
        };
      } else {
        throw new Error(result.error || 'Analysis failed');
      }
    } catch (error) {
      console.error('Advanced food detection failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        description: 'Detection failed',
        analysis: 'Unable to analyze image. Please check your connection and try again.',
        nutritional_data: {
          total_calories: 0,
          total_protein: 0,
          total_carbs: 0,
          total_fats: 0,
          items: []
        }
      };
    }
  }

  // All detection methods now handled by backend API
  // This class serves as a client interface to the Python backend

  async getModelStatus(): Promise<Record<string, boolean>> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/models/status`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.warn('Failed to get model status:', error);
    }
    return {};
  }

  async detectWithSpecificModel(imageFile: File, modelType: string): Promise<AnalysisResult> {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('model_type', modelType);

      const response = await fetch(`${this.apiBaseUrl}/api/analyze/model/${modelType}`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Model ${modelType} analysis failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`${modelType} detection failed:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        description: `${modelType} detection failed`,
        analysis: 'Unable to analyze image with this model',
        nutritional_data: {
          total_calories: 0,
          total_protein: 0,
          total_carbs: 0,
          total_fats: 0,
          items: []
        }
      };
    }
  }

  async detectWithYOLO(imageFile: File): Promise<AnalysisResult> {
    return this.detectWithSpecificModel(imageFile, 'yolo');
  }

  async detectWithViT(imageFile: File): Promise<AnalysisResult> {
    return this.detectWithSpecificModel(imageFile, 'vit');
  }

  async detectWithBLIP(imageFile: File): Promise<AnalysisResult> {
    return this.detectWithSpecificModel(imageFile, 'blip');
  }

  async detectWithSwin(imageFile: File): Promise<AnalysisResult> {
    return this.detectWithSpecificModel(imageFile, 'swin');
  }

  async detectWithCLIP(imageFile: File): Promise<AnalysisResult> {
    return this.detectWithSpecificModel(imageFile, 'clip');
  }

  async detectLightweight(imageFile: File): Promise<AnalysisResult> {
    return this.detectWithSpecificModel(imageFile, 'lightweight');
  }

  async detectRobust(imageFile: File): Promise<AnalysisResult> {
    return this.detectWithSpecificModel(imageFile, 'robust');
  }

  async detectSimple(imageFile: File): Promise<AnalysisResult> {
    return this.detectWithSpecificModel(imageFile, 'simple');
  }
}