import type { AnalysisResult, FoodItem, NutritionalData } from '../types';
import config, { APP_CONFIG } from '../config';

export class AnalysisService {
  private static instance: AnalysisService;
  private apiBaseUrl: string;

  private constructor() {
    this.apiBaseUrl = APP_CONFIG.api.baseUrl;
  }

  public static getInstance(): AnalysisService {
    if (!AnalysisService.instance) {
      AnalysisService.instance = new AnalysisService();
    }
    return AnalysisService.instance;
  }

  async analyzeImage(file: File, context?: string): Promise<AnalysisResult> {
    try {
      const formData = new FormData();
      formData.append('image', file);

      if (context) {
        formData.append('context', context);
      }

      formData.append('use_advanced_detection', 'true');
      formData.append('confidence_threshold', APP_CONFIG.analysis.confidenceThreshold.toString());
      formData.append('ensemble_threshold', APP_CONFIG.analysis.ensembleThreshold.toString());

      const response = await fetch(`${this.apiBaseUrl}/analyze/advanced`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Image analysis failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Analysis failed',
        description: 'Unable to analyze image',
        analysis: 'Please check your connection and try again with a clearer image',
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

  async analyzeWithSpecificModel(file: File, modelType: string, context?: string): Promise<AnalysisResult> {
    try {
      const formData = new FormData();
      formData.append('image', file);

      if (context) {
        formData.append('context', context);
      }

      formData.append('confidence_threshold', APP_CONFIG.analysis.confidenceThreshold.toString());

      const response = await fetch(`${this.apiBaseUrl}/analyze/model/${modelType}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `${modelType} analysis failed: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error(`${modelType} analysis failed:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : `${modelType} analysis failed`,
        description: `${modelType} analysis failed`,
        analysis: `Unable to analyze image with ${modelType} model`,
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

  async analyzeBatch(files: File[], context?: string): Promise<any> {
    try {
      const formData = new FormData();

      files.forEach(file => {
        formData.append('images', file);
      });

      if (context) {
        formData.append('context', context);
      }

      formData.append('use_advanced_detection', 'true');
      formData.append('confidence_threshold', APP_CONFIG.analysis.confidenceThreshold.toString());
      formData.append('ensemble_threshold', APP_CONFIG.analysis.ensembleThreshold.toString());

      const response = await fetch(`${this.apiBaseUrl}/analyze/batch`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Batch analysis failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Batch analysis failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Batch analysis failed',
        total: files.length,
        successful: 0,
        failed: files.length,
        results: [],
        errors: files.map((_, index) => ({ index, error: 'Analysis failed' }))
      };
    }
  }

  async validateFoodItems(items: string[], context?: string): Promise<string[]> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/food/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          items,
          context
        })
      });

      if (response.ok) {
        const data = await response.json();
        return data.validated_items || items;
      }
    } catch (error) {
      console.warn('Failed to validate food items via API:', error);
    }

    return items;
  }

  async searchFoods(query: string, category?: string, limit = 10): Promise<string[]> {
    try {
      const params = new URLSearchParams({
        q: query,
        limit: limit.toString()
      });

      if (category) {
        params.append('category', category);
      }

      const response = await fetch(`${this.apiBaseUrl}/food/search?${params}`);

      if (response.ok) {
        const data = await response.json();
        return data.results || [];
      }
    } catch (error) {
      console.warn('Failed to search foods via API:', error);
    }

    return [];
  }

  async getFoodDetails(foodName: string): Promise<any> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/food/${encodeURIComponent(foodName)}/details`);

      if (response.ok) {
        const data = await response.json();
        return data.details;
      }
    } catch (error) {
      console.warn('Failed to get food details via API:', error);
    }

    return {
      category: 'other',
      common_name: foodName.charAt(0).toUpperCase() + foodName.slice(1),
      nutritional_category: 'mixed'
    };
  }

  async calculateNutrition(foods: string[]): Promise<NutritionalData> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/nutrition/calculate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ foods })
      });

      if (response.ok) {
        const data = await response.json();
        return data.nutritional_data;
      }
    } catch (error) {
      console.warn('Failed to calculate nutrition via API:', error);
    }

    return {
      total_calories: 0,
      total_protein: 0,
      total_carbs: 0,
      total_fats: 0,
      items: []
    };
  }

  async compareFoods(foods: string[]): Promise<any> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/nutrition/compare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ foods })
      });

      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.warn('Failed to compare foods via API:', error);
    }

    return null;
  }

  async getDailyRecommendations(profile: {
    age?: number;
    gender?: string;
    weight?: number;
    height?: number;
    activity_level?: string;
  }): Promise<any> {
    try {
      const params = new URLSearchParams();
      Object.entries(profile).forEach(([key, value]) => {
        if (value !== undefined) {
          params.append(key, value.toString());
        }
      });

      const response = await fetch(`${this.apiBaseUrl}/nutrition/recommendations/daily?${params}`);

      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.warn('Failed to get daily recommendations via API:', error);
    }

    return null;
  }

  async analyzeMealBalance(foods: string[]): Promise<any> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/nutrition/analyze/balance`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ foods })
      });

      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.warn('Failed to analyze meal balance via API:', error);
    }

    return null;
  }

  async getModelStatus(): Promise<Record<string, boolean>> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/models/status`);

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
      const response = await fetch(`${this.apiBaseUrl.replace('/api', '')}/health`);

      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.warn('Failed to get service health:', error);
    }

    return { status: 'unknown' };
  }

  async analyzeManualFoodItems(foodItems: FoodItem[]): Promise<AnalysisResult> {
    try {
      // Calculate nutrition for the provided food items
      const foods = foodItems.map(item => item.name);
      const nutritionalData = await this.calculateNutrition(foods);

      // Generate analysis text
      const analysis = this.generateNutritionalAnalysis(nutritionalData);

      return {
        success: true,
        description: `Manual analysis of ${foodItems.length} food items`,
        analysis,
        nutritional_data: nutritionalData
      };
    } catch (error) {
      console.error('Manual food analysis failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Analysis failed',
        description: 'Manual analysis failed',
        analysis: 'Unable to process food items. Please check your connection and try again.',
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

  private generateNutritionalAnalysis(data: NutritionalData): string {
    const { total_calories, total_protein, total_carbs, total_fats } = data;

    if (total_calories === 0) {
      return 'No nutritional data available for the provided foods.';
    }

    let analysis = `Nutritional Analysis:\n\n`;
    analysis += `Total Calories: ${total_calories}\n`;
    analysis += `Protein: ${total_protein}g (${Math.round(total_protein * 4 / total_calories * 100)}%)\n`;
    analysis += `Carbohydrates: ${total_carbs}g (${Math.round(total_carbs * 4 / total_calories * 100)}%)\n`;
    analysis += `Fats: ${total_fats}g (${Math.round(total_fats * 9 / total_calories * 100)}%)\n\n`;

    // Add recommendations
    const proteinPercent = (total_protein * 4 / total_calories) * 100;
    const fatPercent = (total_fats * 9 / total_calories) * 100;
    const carbPercent = (total_carbs * 4 / total_calories) * 100;

    analysis += `Recommendations:\n`;

    if (proteinPercent < 15) {
      analysis += `• Consider adding more protein sources for muscle maintenance.\n`;
    } else if (proteinPercent > 25) {
      analysis += `• High protein content - ensure adequate hydration.\n`;
    } else {
      analysis += `• Good protein balance for most dietary needs.\n`;
    }

    if (fatPercent > 35) {
      analysis += `• This meal is high in fats. Consider balancing with more vegetables.\n`;
    } else if (fatPercent < 20) {
      analysis += `• Consider adding healthy fats for essential fatty acids.\n`;
    } else {
      analysis += `• Good fat balance for sustained energy.\n`;
    }

    if (carbPercent > 65) {
      analysis += `• High carbohydrate content. Balance with protein and healthy fats.\n`;
    } else if (carbPercent < 45) {
      analysis += `• Low carbohydrate content may limit energy availability.\n`;
    } else {
      analysis += `• Good carbohydrate balance for energy needs.\n`;
    }

    return analysis;
  }
}