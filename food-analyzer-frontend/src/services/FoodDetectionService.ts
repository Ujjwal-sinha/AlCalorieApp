import type { AnalysisResult } from '../types';

export class FoodDetectionService {
  private static instance: FoodDetectionService;
  private apiBaseUrl: string;
  private models: Record<string, any> = {};
  private confidenceThreshold: number = 0.3;
  private ensembleThreshold: number = 0.6;

  private constructor() {
    this.apiBaseUrl = 'http://localhost:8000';
  }

  public static getInstance(): FoodDetectionService {
    if (!FoodDetectionService.instance) {
      FoodDetectionService.instance = new FoodDetectionService();
    }
    return FoodDetectionService.instance;
  }

  private async initializeModels(): Promise<void> {
    try {
      // This would typically load TensorFlow.js models
      // For now, we'll simulate model loading
      this.models = {
        'food-detection': 'loaded',
        'nutrition-analysis': 'loaded'
      };
    } catch (error) {
      console.warn('Failed to initialize models:', error);
    }
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
        error: error instanceof Error ? error.message : 'Detection failed',
        description: 'Failed to detect foods in image',
        analysis: 'Please try again with a clearer image',
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

  async detectFoodsWithModel(imageFile: File, modelType: string): Promise<AnalysisResult> {
    try {
      // Initialize models if not already done
      if (Object.keys(this.models).length === 0) {
        await this.initializeModels();
      }

      // Create FormData for image upload
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('confidence_threshold', this.confidenceThreshold.toString());

      // Send to backend for processing with specific model
      const response = await fetch(`${this.apiBaseUrl}/api/analyze/model/${modelType}`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`${modelType} analysis failed: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        return {
          success: true,
          description: `${modelType} analysis completed`,
          analysis: result.analysis || `${modelType} analysis completed successfully`,
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
        throw new Error(result.error || `${modelType} analysis failed`);
      }
    } catch (error) {
      console.error(`${modelType} detection failed:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : `${modelType} detection failed`,
        description: `Failed to detect foods with ${modelType}`,
        analysis: `Please try again with a clearer image or different model`,
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

  async getModelStatus(): Promise<Record<string, boolean>> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/models/status`);
      if (response.ok) {
        const data = await response.json();
        return data.summary || {};
      }
    } catch (error) {
      console.warn('Failed to get model status:', error);
    }
    return {};
  }

  async getServiceHealth(): Promise<any> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/health`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.warn('Failed to get service health:', error);
    }
    return { status: 'unknown' };
  }

  setConfidenceThreshold(threshold: number): void {
    this.confidenceThreshold = Math.max(0, Math.min(1, threshold));
  }

  setEnsembleThreshold(threshold: number): void {
    this.ensembleThreshold = Math.max(0, Math.min(1, threshold));
  }

  getConfidenceThreshold(): number {
    return this.confidenceThreshold;
  }

  getEnsembleThreshold(): number {
    return this.ensembleThreshold;
  }
}