import axios from 'axios';
import type { NutritionalData, FoodItem, NutritionData } from '../types';

interface NutritionAPIResponse {
  foods: Array<{
    food_name: string;
    serving_qty: number;
    serving_unit: string;
    nix_item_id: string;
    nf_calories: number;
    nf_total_fat: number;
    nf_saturated_fat: number;
    nf_cholesterol: number;
    nf_sodium: number;
    nf_total_carbohydrate: number;
    nf_dietary_fiber: number;
    nf_sugars: number;
    nf_protein: number;
    nf_potassium: number;
  }>;
}

interface NutritionixConfig {
  appId: string;
  appKey: string;
  baseUrl: string;
}

export class NutritionService {
  private static instance: NutritionService;
  private nutritionDatabase: Map<string, NutritionData> = new Map();
  private nutritionixConfig: NutritionixConfig;
  private isInitialized = false;

  private constructor() {
    this.nutritionixConfig = {
      appId: process.env.NUTRITIONIX_APP_ID || '',
      appKey: process.env.NUTRITIONIX_APP_KEY || '',
      baseUrl: 'https://trackapi.nutritionix.com/v2'
    };
    this.initializeNutritionDatabase();
  }

  public static getInstance(): NutritionService {
    if (!NutritionService.instance) {
      NutritionService.instance = new NutritionService();
    }
    return NutritionService.instance;
  }

  private initializeNutritionDatabase(): void {
    // Comprehensive nutrition database (per 100g)
    const nutritionData: [string, NutritionData][] = [
      // Proteins
      ['chicken breast', { calories: 165, protein: 31, carbs: 0, fat: 3.6, fiber: 0 }],
      ['chicken', { calories: 165, protein: 31, carbs: 0, fat: 3.6, fiber: 0 }],
      ['beef', { calories: 250, protein: 26, carbs: 0, fat: 15, fiber: 0 }],
      ['pork', { calories: 242, protein: 27, carbs: 0, fat: 14, fiber: 0 }],
      ['fish', { calories: 206, protein: 22, carbs: 0, fat: 12, fiber: 0 }],
      ['salmon', { calories: 208, protein: 20, carbs: 0, fat: 13, fiber: 0 }],
      ['tuna', { calories: 144, protein: 30, carbs: 0, fat: 1, fiber: 0 }],
      ['egg', { calories: 155, protein: 13, carbs: 1.1, fat: 11, fiber: 0 }],
      ['eggs', { calories: 155, protein: 13, carbs: 1.1, fat: 11, fiber: 0 }],
      ['tofu', { calories: 76, protein: 8, carbs: 1.9, fat: 4.8, fiber: 0.3 }],
      
      // Grains & Starches
      ['rice', { calories: 130, protein: 2.7, carbs: 28, fat: 0.3, fiber: 0.4 }],
      ['white rice', { calories: 130, protein: 2.7, carbs: 28, fat: 0.3, fiber: 0.4 }],
      ['brown rice', { calories: 111, protein: 2.6, carbs: 23, fat: 0.9, fiber: 1.8 }],
      ['bread', { calories: 265, protein: 9, carbs: 49, fat: 3.2, fiber: 2.7 }],
      ['pasta', { calories: 131, protein: 5, carbs: 25, fat: 1.1, fiber: 1.8 }],
      ['quinoa', { calories: 120, protein: 4.4, carbs: 22, fat: 1.9, fiber: 2.8 }],
      ['oats', { calories: 389, protein: 17, carbs: 66, fat: 7, fiber: 10 }],
      
      // Vegetables
      ['broccoli', { calories: 34, protein: 2.8, carbs: 7, fat: 0.4, fiber: 2.6 }],
      ['carrot', { calories: 41, protein: 0.9, carbs: 10, fat: 0.2, fiber: 2.8 }],
      ['carrots', { calories: 41, protein: 0.9, carbs: 10, fat: 0.2, fiber: 2.8 }],
      ['tomato', { calories: 18, protein: 0.9, carbs: 3.9, fat: 0.2, fiber: 1.2 }],
      ['tomatoes', { calories: 18, protein: 0.9, carbs: 3.9, fat: 0.2, fiber: 1.2 }],
      ['potato', { calories: 77, protein: 2, carbs: 17, fat: 0.1, fiber: 2.2 }],
      ['potatoes', { calories: 77, protein: 2, carbs: 17, fat: 0.1, fiber: 2.2 }],
      ['spinach', { calories: 23, protein: 2.9, carbs: 3.6, fat: 0.4, fiber: 2.2 }],
      ['lettuce', { calories: 15, protein: 1.4, carbs: 2.9, fat: 0.1, fiber: 1.3 }],
      
      // Fruits
      ['apple', { calories: 52, protein: 0.3, carbs: 14, fat: 0.2, fiber: 2.4 }],
      ['banana', { calories: 89, protein: 1.1, carbs: 23, fat: 0.3, fiber: 2.6 }],
      ['orange', { calories: 47, protein: 0.9, carbs: 12, fat: 0.1, fiber: 2.4 }],
      ['strawberry', { calories: 32, protein: 0.7, carbs: 8, fat: 0.3, fiber: 2 }],
      ['blueberry', { calories: 57, protein: 0.7, carbs: 14, fat: 0.3, fiber: 2.4 }],
      
      // Dairy
      ['milk', { calories: 42, protein: 3.4, carbs: 5, fat: 1, fiber: 0 }],
      ['cheese', { calories: 402, protein: 25, carbs: 1.3, fat: 33, fiber: 0 }],
      ['yogurt', { calories: 59, protein: 10, carbs: 3.6, fat: 0.4, fiber: 0 }],
      ['butter', { calories: 717, protein: 0.9, carbs: 0.1, fat: 81, fiber: 0 }],
      
      // Prepared Foods
      ['pizza', { calories: 266, protein: 11, carbs: 33, fat: 10, fiber: 2.3 }],
      ['burger', { calories: 295, protein: 17, carbs: 24, fat: 15, fiber: 2 }],
      ['sandwich', { calories: 250, protein: 12, carbs: 30, fat: 8, fiber: 3 }],
      ['salad', { calories: 20, protein: 1.5, carbs: 4, fat: 0.2, fiber: 2 }],
      ['soup', { calories: 50, protein: 3, carbs: 8, fat: 1, fiber: 1 }],
      
      // Nuts & Seeds
      ['almond', { calories: 579, protein: 21, carbs: 22, fat: 50, fiber: 12 }],
      ['peanut', { calories: 567, protein: 26, carbs: 16, fat: 49, fiber: 8.5 }],
      ['walnut', { calories: 654, protein: 15, carbs: 14, fat: 65, fiber: 6.7 }],
      
      // Beverages
      ['coffee', { calories: 2, protein: 0.3, carbs: 0, fat: 0, fiber: 0 }],
      ['tea', { calories: 1, protein: 0, carbs: 0.2, fat: 0, fiber: 0 }],
      ['juice', { calories: 54, protein: 0.5, carbs: 13, fat: 0.1, fiber: 0.2 }],
      
      // Condiments
      ['olive oil', { calories: 884, protein: 0, carbs: 0, fat: 100, fiber: 0 }],
      ['honey', { calories: 304, protein: 0.3, carbs: 82, fat: 0, fiber: 0.2 }],
      ['sugar', { calories: 387, protein: 0, carbs: 100, fat: 0, fiber: 0 }]
    ];

    nutritionData.forEach(([food, nutrition]) => {
      this.nutritionDatabase.set(food.toLowerCase(), nutrition);
    });
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.log('NutritionService already initialized');
      return;
    }

    console.log('Initializing NutritionService...');
    
    try {
      // Test external API connection if credentials are available
      if (this.nutritionixConfig.appId && this.nutritionixConfig.appKey) {
        await this.testNutritionixConnection();
      } else {
        console.log('Nutritionix credentials not provided, using local database only');
      }
      
      this.isInitialized = true;
      console.log('NutritionService initialization completed');
    } catch (error) {
      console.error('NutritionService initialization failed:', error);
      // Continue with local database only
      this.isInitialized = true;
    }
  }

  private async testNutritionixConnection(): Promise<void> {
    try {
      const response = await axios.get(`${this.nutritionixConfig.baseUrl}/search/instant`, {
        headers: {
          'x-app-id': this.nutritionixConfig.appId,
          'x-app-key': this.nutritionixConfig.appKey,
          'x-remote-user-id': '0'
        },
        params: {
          query: 'apple',
          detailed: true
        },
        timeout: 5000
      });
      
      console.log('Nutritionix API connection successful');
    } catch (error) {
      console.warn('Nutritionix API connection failed, using local database only:', error);
    }
  }

  async calculateNutrition(foods: string[]): Promise<NutritionalData> {
    try {
      const foodItems: FoodItem[] = [];
      let totalCalories = 0;
      let totalProtein = 0;
      let totalCarbs = 0;
      let totalFats = 0;

      for (const food of foods) {
        const nutrition = await this.getNutritionForFood(food);
        if (nutrition) {
          const foodItem: FoodItem = {
            name: food,
            calories: nutrition.calories,
            protein: nutrition.protein,
            carbs: nutrition.carbs,
            fats: nutrition.fat,
            confidence: 0.8 // Default confidence
          };
          
          foodItems.push(foodItem);
          totalCalories += nutrition.calories;
          totalProtein += nutrition.protein;
          totalCarbs += nutrition.carbs;
          totalFats += nutrition.fat;
        }
      }

      return {
        total_calories: totalCalories,
        total_protein: totalProtein,
        total_carbs: totalCarbs,
        total_fats: totalFats,
        items: foodItems
      };
    } catch (error) {
      console.error('Nutrition calculation failed:', error);
      return {
        total_calories: 0,
        total_protein: 0,
        total_carbs: 0,
        total_fats: 0,
        items: []
      };
    }
  }

  private async getNutritionForFood(food: string): Promise<NutritionData | null> {
    const foodLower = food.toLowerCase().trim();
    
    // Check local database first
    if (this.nutritionDatabase.has(foodLower)) {
      return this.nutritionDatabase.get(foodLower)!;
    }
    
    // Try fuzzy matching
    const fuzzyMatch = this.findFuzzyMatch(foodLower);
    if (fuzzyMatch) {
      return this.nutritionDatabase.get(fuzzyMatch)!;
    }
    
    // Try external API if available
    if (this.nutritionixConfig.appId && this.nutritionixConfig.appKey) {
      try {
        const apiNutrition = await this.getNutritionFromAPI(food);
        if (apiNutrition) {
          // Cache the result
          this.nutritionDatabase.set(foodLower, apiNutrition);
          return apiNutrition;
        }
      } catch (error) {
        console.warn(`Failed to get nutrition from API for ${food}:`, error);
      }
    }
    
    // Return estimated nutrition based on food category
    return this.estimateNutrition(foodLower);
  }

  private findFuzzyMatch(food: string): string | null {
    const words = food.split(' ');
    
    for (const [dbFood, nutrition] of this.nutritionDatabase) {
      const dbWords = dbFood.split(' ');
      
      // Check if any word matches
      for (const word of words) {
        for (const dbWord of dbWords) {
          if (word.includes(dbWord) || dbWord.includes(word)) {
            return dbFood;
          }
        }
      }
    }
    
    return null;
  }

  private async getNutritionFromAPI(food: string): Promise<NutritionData | null> {
    try {
      const response = await axios.get<NutritionAPIResponse>(
        `${this.nutritionixConfig.baseUrl}/search/instant`,
        {
          headers: {
            'x-app-id': this.nutritionixConfig.appId,
            'x-app-key': this.nutritionixConfig.appKey,
            'x-remote-user-id': '0'
          },
          params: {
            query: food,
            detailed: true
          },
          timeout: 10000
        }
      );

      if (response.data.foods && response.data.foods.length > 0) {
        const foodData = response.data.foods[0];
        
        // Convert to per 100g values
        const servingGrams = this.convertServingToGrams(foodData.serving_qty, foodData.serving_unit);
        const multiplier = 100 / servingGrams;
        
        return {
          calories: Math.round(foodData.nf_calories * multiplier),
          protein: Math.round(foodData.nf_protein * multiplier * 10) / 10,
          carbs: Math.round(foodData.nf_total_carbohydrate * multiplier * 10) / 10,
          fat: Math.round(foodData.nf_total_fat * multiplier * 10) / 10,
          fiber: Math.round(foodData.nf_dietary_fiber * multiplier * 10) / 10
        };
      }
      
      return null;
    } catch (error) {
      console.error(`API nutrition lookup failed for ${food}:`, error);
      return null;
    }
  }

  private convertServingToGrams(quantity: number, unit: string): number {
    const unitMap: Record<string, number> = {
      'g': 1,
      'gram': 1,
      'grams': 1,
      'kg': 1000,
      'kilogram': 1000,
      'kilograms': 1000,
      'oz': 28.35,
      'ounce': 28.35,
      'ounces': 28.35,
      'lb': 453.59,
      'pound': 453.59,
      'pounds': 453.59,
      'cup': 240,
      'cups': 240,
      'tbsp': 15,
      'tablespoon': 15,
      'tablespoons': 15,
      'tsp': 5,
      'teaspoon': 5,
      'teaspoons': 5,
      'ml': 1,
      'milliliter': 1,
      'milliliters': 1,
      'l': 1000,
      'liter': 1000,
      'liters': 1000
    };
    
    return quantity * (unitMap[unit.toLowerCase()] || 100); // Default to 100g if unknown unit
  }

  private estimateNutrition(food: string): NutritionData | null {
    // Estimate nutrition based on food categories
    const foodLower = food.toLowerCase();
    
    // Protein foods
    if (foodLower.includes('chicken') || foodLower.includes('beef') || foodLower.includes('pork') || 
        foodLower.includes('fish') || foodLower.includes('meat') || foodLower.includes('protein')) {
      return { calories: 200, protein: 25, carbs: 0, fat: 10, fiber: 0 };
    }
    
    // Vegetables
    if (foodLower.includes('broccoli') || foodLower.includes('carrot') || foodLower.includes('spinach') ||
        foodLower.includes('lettuce') || foodLower.includes('vegetable') || foodLower.includes('green')) {
      return { calories: 30, protein: 2, carbs: 6, fat: 0.3, fiber: 2.5 };
    }
    
    // Fruits
    if (foodLower.includes('apple') || foodLower.includes('banana') || foodLower.includes('orange') ||
        foodLower.includes('fruit') || foodLower.includes('berry')) {
      return { calories: 50, protein: 0.5, carbs: 12, fat: 0.2, fiber: 2 };
    }
    
    // Grains/Carbs
    if (foodLower.includes('rice') || foodLower.includes('bread') || foodLower.includes('pasta') ||
        foodLower.includes('grain') || foodLower.includes('carb') || foodLower.includes('starch')) {
      return { calories: 130, protein: 3, carbs: 25, fat: 0.5, fiber: 1 };
    }
    
    // Dairy
    if (foodLower.includes('milk') || foodLower.includes('cheese') || foodLower.includes('yogurt') ||
        foodLower.includes('dairy') || foodLower.includes('cream')) {
      return { calories: 100, protein: 8, carbs: 5, fat: 5, fiber: 0 };
    }
    
    // Default estimation
    return { calories: 100, protein: 5, carbs: 15, fat: 3, fiber: 1 };
  }

  async getFoodDetails(food: string): Promise<any> {
    try {
      const nutrition = await this.getNutritionForFood(food);
      if (!nutrition) {
        return null;
      }
      
      return {
        name: food,
        category: this.categorizeFood(food),
        nutrition: nutrition,
        health_benefits: this.getHealthBenefits(food),
        serving_suggestions: this.getServingSuggestions(food)
      };
    } catch (error) {
      console.error(`Failed to get food details for ${food}:`, error);
      return null;
    }
  }

  private categorizeFood(food: string): string {
    const foodLower = food.toLowerCase();
    
    if (foodLower.includes('chicken') || foodLower.includes('beef') || foodLower.includes('fish') ||
        foodLower.includes('pork') || foodLower.includes('meat') || foodLower.includes('protein')) {
      return 'protein';
    }
    
    if (foodLower.includes('broccoli') || foodLower.includes('carrot') || foodLower.includes('spinach') ||
        foodLower.includes('lettuce') || foodLower.includes('vegetable')) {
      return 'vegetable';
    }
    
    if (foodLower.includes('apple') || foodLower.includes('banana') || foodLower.includes('orange') ||
        foodLower.includes('fruit') || foodLower.includes('berry')) {
      return 'fruit';
    }
    
    if (foodLower.includes('rice') || foodLower.includes('bread') || foodLower.includes('pasta') ||
        foodLower.includes('grain') || foodLower.includes('carb')) {
      return 'grain';
    }
    
    if (foodLower.includes('milk') || foodLower.includes('cheese') || foodLower.includes('yogurt') ||
        foodLower.includes('dairy')) {
      return 'dairy';
    }
    
    return 'other';
  }

  private getHealthBenefits(food: string): string[] {
    const foodLower = food.toLowerCase();
    const benefits: string[] = [];
    
    if (foodLower.includes('broccoli') || foodLower.includes('spinach')) {
      benefits.push('Rich in vitamins and minerals', 'High in fiber', 'Antioxidant properties');
    }
    
    if (foodLower.includes('salmon') || foodLower.includes('fish')) {
      benefits.push('High in omega-3 fatty acids', 'Good source of protein', 'Heart-healthy');
    }
    
    if (foodLower.includes('chicken') || foodLower.includes('beef')) {
      benefits.push('Excellent source of protein', 'Contains essential amino acids', 'Supports muscle growth');
    }
    
    if (foodLower.includes('apple') || foodLower.includes('banana')) {
      benefits.push('Good source of fiber', 'Contains vitamins and minerals', 'Natural energy boost');
    }
    
    return benefits;
  }

  private getServingSuggestions(food: string): string[] {
    const foodLower = food.toLowerCase();
    const suggestions: string[] = [];
    
    if (foodLower.includes('chicken') || foodLower.includes('beef')) {
      suggestions.push('3-4 oz serving (about the size of a deck of cards)', 'Grill or bake for healthier preparation');
    }
    
    if (foodLower.includes('rice') || foodLower.includes('pasta')) {
      suggestions.push('1/2 cup cooked serving', 'Choose whole grain varieties when possible');
    }
    
    if (foodLower.includes('vegetable')) {
      suggestions.push('1 cup raw or 1/2 cup cooked', 'Aim for variety and different colors');
    }
    
    if (foodLower.includes('fruit')) {
      suggestions.push('1 medium piece or 1/2 cup chopped', 'Eat with skin when possible for extra fiber');
    }
    
    return suggestions;
  }

  getNutritionDatabase(): Map<string, NutritionData> {
    return new Map(this.nutritionDatabase);
  }

  addCustomFood(name: string, nutrition: NutritionData): void {
    this.nutritionDatabase.set(name.toLowerCase(), nutrition);
  }

  removeCustomFood(name: string): boolean {
    return this.nutritionDatabase.delete(name.toLowerCase());
  }

  // Health check method
  async healthCheck(): Promise<{ healthy: boolean; databaseSize: number; apiAvailable: boolean }> {
    const databaseSize = this.nutritionDatabase.size;
    const apiAvailable = !!(this.nutritionixConfig.appId && this.nutritionixConfig.appKey);
    
    return {
      healthy: databaseSize > 0,
      databaseSize,
      apiAvailable
    };
  }
}