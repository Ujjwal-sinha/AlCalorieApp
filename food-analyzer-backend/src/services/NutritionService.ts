import type { NutritionalData, FoodItem } from '../types';

interface NutritionixConfig {
  baseUrl: string;
  appId: string;
  appKey: string;
}

interface NutritionixFood {
  food_name: string;
  serving_qty: number;
  serving_unit: string;
  nf_calories: number;
  nf_protein: number;
  nf_total_carbohydrate: number;
  nf_total_fat: number;
  nf_dietary_fiber: number;
}

export class NutritionService {
  private static instance: NutritionService;
  private nutritionDatabase: Map<string, NutritionixFood> = new Map();
  private nutritionixConfig: NutritionixConfig;

  private constructor() {
    this.nutritionixConfig = {
      baseUrl: 'https://trackapi.nutritionix.com/v2',
      appId: process.env['NUTRITIONIX_APP_ID'] || '',
      appKey: process.env['NUTRITIONIX_APP_KEY'] || ''
    };
    
    this.initializeNutritionDatabase();
  }

  public static getInstance(): NutritionService {
    if (!NutritionService.instance) {
      NutritionService.instance = new NutritionService();
    }
    return NutritionService.instance;
  }

  async initialize(): Promise<void> {
    console.log('Initializing NutritionService...');
    // Service is already initialized in constructor
    console.log('NutritionService initialized successfully');
  }

  private initializeNutritionDatabase(): void {
    // Initialize with common foods
    const commonFoods: NutritionixFood[] = [
      {
        food_name: 'apple',
        serving_qty: 1,
        serving_unit: 'medium',
        nf_calories: 95,
        nf_protein: 0.5,
        nf_total_carbohydrate: 25,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 4.4
      },
      {
        food_name: 'banana',
        serving_qty: 1,
        serving_unit: 'medium',
        nf_calories: 105,
        nf_protein: 1.3,
        nf_total_carbohydrate: 27,
        nf_total_fat: 0.4,
        nf_dietary_fiber: 3.1
      },
      {
        food_name: 'chicken breast',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 165,
        nf_protein: 31,
        nf_total_carbohydrate: 0,
        nf_total_fat: 3.6,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'rice',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 130,
        nf_protein: 2.7,
        nf_total_carbohydrate: 28,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 0.4
      },
      {
        food_name: 'salad',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 20,
        nf_protein: 2,
        nf_total_carbohydrate: 4,
        nf_total_fat: 0.2,
        nf_dietary_fiber: 1.5
      },
      {
        food_name: 'pizza',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 266,
        nf_protein: 11,
        nf_total_carbohydrate: 33,
        nf_total_fat: 10,
        nf_dietary_fiber: 2.5
      },
      {
        food_name: 'pasta',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 131,
        nf_protein: 5,
        nf_total_carbohydrate: 25,
        nf_total_fat: 1.1,
        nf_dietary_fiber: 1.8
      }
    ];

    commonFoods.forEach(food => {
      this.nutritionDatabase.set(food.food_name.toLowerCase(), food);
    });
  }

  async searchFood(query: string): Promise<string[]> {
    try {
      // First check local database
      const localResults = Array.from(this.nutritionDatabase.keys())
        .filter(food => food.toLowerCase().includes(query.toLowerCase()))
        .slice(0, 10);

      if (localResults.length > 0) {
        return localResults;
      }

      // If no local results and Nutritionix is configured, try external API
      if (this.nutritionixConfig.appId && this.nutritionixConfig.appKey) {
        try {
          const response = await fetch(`${this.nutritionixConfig.baseUrl}/search/instant`, {
            method: 'GET',
          headers: {
              'x-app-id': this.nutritionixConfig.appId,
              'x-app-key': this.nutritionixConfig.appKey,
              'x-remote-user-id': '0'
          }
        });

        if (response.ok) {
            const data = await response.json() as any;
            return data.common?.map((item: any) => item.food_name) || [];
          }
        } catch (error) {
          console.warn('Nutritionix API search failed:', error);
        }
      }

      return [];
    } catch (error) {
      console.error('Food search failed:', error);
      return [];
    }
  }

  async getFoodDetails(foodName: string): Promise<any> {
    try {
      const normalizedName = foodName.toLowerCase();
      const foodData = this.nutritionDatabase.get(normalizedName);

      if (foodData) {
        return {
          name: foodData.food_name,
          category: this.categorizeFood(foodData.food_name),
          serving_size: `${foodData.serving_qty} ${foodData.serving_unit}`,
          calories: foodData.nf_calories,
          protein: foodData.nf_protein,
          carbs: foodData.nf_total_carbohydrate,
          fat: foodData.nf_total_fat,
          fiber: foodData.nf_dietary_fiber
        };
      }

      // Fallback for unknown foods
      return {
        name: foodName,
        category: 'other',
        serving_size: '100g',
        calories: 100,
        protein: 5,
        carbs: 15,
        fat: 2,
        fiber: 2
      };
    } catch (error) {
      console.error('Get food details failed:', error);
      return null;
    }
  }

  async calculateNutrition(foods: string[]): Promise<NutritionalData> {
    try {
    const items: FoodItem[] = [];
    let totalCalories = 0;
    let totalProtein = 0;
    let totalCarbs = 0;
    let totalFats = 0;

      for (const foodName of foods) {
        const foodData = this.nutritionDatabase.get(foodName.toLowerCase());
        
        if (foodData) {
          const servingGrams = this.convertServingToGrams(foodData.serving_qty, foodData.serving_unit);
          const multiplier = servingGrams / 100; // Convert to per 100g basis

          const item: FoodItem = {
            name: foodData.food_name,
            calories: Math.round(foodData.nf_calories * multiplier),
            protein: Math.round(foodData.nf_protein * multiplier * 10) / 10,
            carbs: Math.round(foodData.nf_total_carbohydrate * multiplier * 10) / 10,
            fats: Math.round(foodData.nf_total_fat * multiplier * 10) / 10
          };

          items.push(item);
          totalCalories += item.calories;
          totalProtein += item.protein;
          totalCarbs += item.carbs;
          totalFats += item.fats;
      } else {
          // Fallback for unknown foods
          const fallbackItem: FoodItem = {
            name: foodName,
            calories: 100,
            protein: 5,
            carbs: 15,
            fats: 2
          };

          items.push(fallbackItem);
          totalCalories += fallbackItem.calories;
          totalProtein += fallbackItem.protein;
          totalCarbs += fallbackItem.carbs;
          totalFats += fallbackItem.fats;
        }
    }

    return {
      total_calories: totalCalories,
      total_protein: totalProtein,
      total_carbs: totalCarbs,
      total_fats: totalFats,
        items: items
      };
    } catch (error) {
      console.error('Calculate nutrition failed:', error);
      return {
        total_calories: 0,
        total_protein: 0,
        total_carbs: 0,
        total_fats: 0,
        items: []
      };
    }
  }

  private convertServingToGrams(quantity: number, unit: string): number {
    const unitMap: Record<string, number> = {
      'g': 1,
      'gram': 1,
      'grams': 1,
      'kg': 1000,
      'oz': 28.35,
      'lb': 453.59,
      'cup': 240,
      'tbsp': 15,
      'tsp': 5,
      'medium': 150,
      'large': 200,
      'small': 100
    };

    return quantity * (unitMap[unit.toLowerCase()] || 100);
  }

  private categorizeFood(foodName: string): string {
    const name = foodName.toLowerCase();
    
    if (['apple', 'banana', 'orange', 'grape', 'strawberry'].some(fruit => name.includes(fruit))) {
      return 'fruits';
    }
    if (['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna'].some(meat => name.includes(meat))) {
      return 'proteins';
    }
    if (['rice', 'pasta', 'bread', 'quinoa', 'oats'].some(grain => name.includes(grain))) {
      return 'grains';
    }
    if (['salad', 'lettuce', 'spinach', 'kale', 'broccoli'].some(veg => name.includes(veg))) {
      return 'vegetables';
    }
    if (['milk', 'cheese', 'yogurt', 'butter'].some(dairy => name.includes(dairy))) {
      return 'dairy';
    }
    
    return 'other';
  }

  getFoodCategories(): Record<string, string[]> {
    return {
      proteins: ['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna'],
      vegetables: ['salad', 'lettuce', 'spinach', 'kale', 'broccoli'],
      fruits: ['apple', 'banana', 'orange', 'grape', 'strawberry'],
      grains: ['rice', 'pasta', 'bread', 'quinoa', 'oats'],
      dairy: ['milk', 'cheese', 'yogurt', 'butter'],
      prepared: ['pizza', 'sandwich', 'burger', 'soup'],
      snacks: ['chips', 'nuts', 'crackers', 'popcorn'],
      desserts: ['cake', 'cookie', 'ice cream', 'chocolate'],
      beverages: ['coffee', 'tea', 'juice', 'water', 'soda']
    };
  }

  getFoodsByCategory(category: string): string[] {
    const categories = this.getFoodCategories();
    return categories[category] || [];
  }
}