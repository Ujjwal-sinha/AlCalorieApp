import type { AnalysisResult, FoodItem, NutritionalData } from '../types';
import { APP_CONFIG } from '../config';

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

  private async makeRequest(url: string, options: RequestInit, retries = APP_CONFIG.api.retries): Promise<Response> {
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const response = await fetch(url, {
          ...options,
          signal: AbortSignal.timeout(APP_CONFIG.api.timeout),
        });
        
        if (response.ok) {
          return response;
        }
        
        // Don't retry on client errors (4xx)
        if (response.status >= 400 && response.status < 500) {
          return response;
        }
        
        lastError = new Error(`HTTP ${response.status}: ${response.statusText}`);
      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Network error');
        
        if (attempt < retries) {
          await new Promise(resolve => setTimeout(resolve, APP_CONFIG.api.retryDelay * (attempt + 1)));
        }
      }
    }
    
    throw lastError || new Error('Request failed after all retries');
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

      const response = await this.makeRequest(`${this.apiBaseUrl}/analysis/advanced`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      return this.normalizeAnalysisResult(result);
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

      const response = await this.makeRequest(`${this.apiBaseUrl}/analysis/model/${modelType}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      return this.normalizeAnalysisResult(result);
    } catch (error) {
      console.error(`Model-specific analysis failed for ${modelType}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Model-specific analysis failed',
        description: `Unable to analyze image with ${modelType} model`,
        analysis: 'Please try again or use a different model',
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

  async analyzeBatch(files: File[]): Promise<AnalysisResult[]> {
    try {
      const results: AnalysisResult[] = [];
      
      for (const file of files) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('use_advanced_detection', 'true');

        const response = await this.makeRequest(`${this.apiBaseUrl}/analysis/advanced`, {
          method: 'POST',
          body: formData,
        });

        const result = this.normalizeAnalysisResult(response);
        results.push(result);
      }

      return results;
    } catch (error) {
      console.error('Batch analysis failed:', error);
      throw new Error('Batch analysis failed');
    }
  }

  async validateFoodItems(items: string[], context?: string): Promise<string[]> {
    try {
      const response = await this.makeRequest(`${this.apiBaseUrl}/food/validate`, {
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

      const response = await this.makeRequest(`${this.apiBaseUrl}/food/search?${params}`, {
        method: 'GET',
      });

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
      const response = await this.makeRequest(`${this.apiBaseUrl}/food/${encodeURIComponent(foodName)}/details`, {
        method: 'GET',
      });

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
      const response = await this.makeRequest(`${this.apiBaseUrl}/nutrition/calculate`, {
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
      const response = await this.makeRequest(`${this.apiBaseUrl}/nutrition/compare`, {
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

      const response = await this.makeRequest(`${this.apiBaseUrl}/nutrition/recommendations/daily?${params}`, {
        method: 'GET',
      });

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
      const response = await this.makeRequest(`${this.apiBaseUrl}/nutrition/analyze/balance`, {
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
      const response = await this.makeRequest(`${this.apiBaseUrl}/models/status`, {
        method: 'GET',
      });

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
      const response = await this.makeRequest(`${this.apiBaseUrl.replace('/api', '')}/health`, {
        method: 'GET',
      });

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

  private normalizeAnalysisResult(result: any): AnalysisResult {
    // Handle different response formats from the backend
    if (result.success !== undefined) {
      return result;
    }

    // If the backend returns a different format, normalize it
    return {
      success: true,
      description: result.description || 'Analysis completed',
      analysis: result.analysis || result.text || 'Analysis results available',
      nutritional_data: result.nutritional_data || result.nutrition || {
        total_calories: 0,
        total_protein: 0,
        total_carbs: 0,
        total_fats: 0,
        items: []
      },
      detected_foods: result.detected_foods || result.foods || [],
      confidence: result.confidence || 0,
      processing_time: result.processing_time || 0,
      model_used: result.model_used || 'ensemble'
    };
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