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
    // Comprehensive nutrition database with per 100g values
    const comprehensiveFoods: NutritionixFood[] = [
      // Fruits
      {
        food_name: 'apple',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 52,
        nf_protein: 0.3,
        nf_total_carbohydrate: 14,
        nf_total_fat: 0.2,
        nf_dietary_fiber: 2.4
      },
      {
        food_name: 'banana',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 89,
        nf_protein: 1.1,
        nf_total_carbohydrate: 23,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 2.6
      },
      {
        food_name: 'orange',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 47,
        nf_protein: 0.9,
        nf_total_carbohydrate: 12,
        nf_total_fat: 0.1,
        nf_dietary_fiber: 2.4
      },
      {
        food_name: 'grape',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 62,
        nf_protein: 0.6,
        nf_total_carbohydrate: 16,
        nf_total_fat: 0.2,
        nf_dietary_fiber: 0.9
      },
      {
        food_name: 'strawberry',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 32,
        nf_protein: 0.7,
        nf_total_carbohydrate: 8,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 2.0
      },
      {
        food_name: 'blueberry',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 57,
        nf_protein: 0.7,
        nf_total_carbohydrate: 14,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 2.4
      },
      {
        food_name: 'pineapple',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 50,
        nf_protein: 0.5,
        nf_total_carbohydrate: 13,
        nf_total_fat: 0.1,
        nf_dietary_fiber: 1.4
      },
      {
        food_name: 'mango',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 60,
        nf_protein: 0.8,
        nf_total_carbohydrate: 15,
        nf_total_fat: 0.4,
        nf_dietary_fiber: 1.6
      },
      {
        food_name: 'peach',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 39,
        nf_protein: 0.9,
        nf_total_carbohydrate: 10,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 1.5
      },
      {
        food_name: 'cherry',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 50,
        nf_protein: 1.0,
        nf_total_carbohydrate: 12,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 1.6
      },
      {
        food_name: 'watermelon',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 30,
        nf_protein: 0.6,
        nf_total_carbohydrate: 8,
        nf_total_fat: 0.2,
        nf_dietary_fiber: 0.4
      },
      {
        food_name: 'avocado',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 160,
        nf_protein: 2.0,
        nf_total_carbohydrate: 9,
        nf_total_fat: 15,
        nf_dietary_fiber: 6.7
      },

      // Vegetables
      {
        food_name: 'carrot',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 41,
        nf_protein: 0.9,
        nf_total_carbohydrate: 10,
        nf_total_fat: 0.2,
        nf_dietary_fiber: 2.8
      },
      {
        food_name: 'broccoli',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 34,
        nf_protein: 2.8,
        nf_total_carbohydrate: 7,
        nf_total_fat: 0.4,
        nf_dietary_fiber: 2.6
      },
      {
        food_name: 'cauliflower',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 25,
        nf_protein: 1.9,
        nf_total_carbohydrate: 5,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 2.0
      },
      {
        food_name: 'corn',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 86,
        nf_protein: 3.2,
        nf_total_carbohydrate: 19,
        nf_total_fat: 1.2,
        nf_dietary_fiber: 2.7
      },
      {
        food_name: 'cucumber',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 16,
        nf_protein: 0.7,
        nf_total_carbohydrate: 4,
        nf_total_fat: 0.1,
        nf_dietary_fiber: 0.5
      },
      {
        food_name: 'tomato',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 18,
        nf_protein: 0.9,
        nf_total_carbohydrate: 4,
        nf_total_fat: 0.2,
        nf_dietary_fiber: 1.2
      },
      {
        food_name: 'potato',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 77,
        nf_protein: 2.0,
        nf_total_carbohydrate: 17,
        nf_total_fat: 0.1,
        nf_dietary_fiber: 2.2
      },
      {
        food_name: 'onion',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 40,
        nf_protein: 1.1,
        nf_total_carbohydrate: 9,
        nf_total_fat: 0.1,
        nf_dietary_fiber: 1.7
      },
      {
        food_name: 'garlic',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 149,
        nf_protein: 6.4,
        nf_total_carbohydrate: 33,
        nf_total_fat: 0.5,
        nf_dietary_fiber: 2.1
      },
      {
        food_name: 'pepper',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 20,
        nf_protein: 0.9,
        nf_total_carbohydrate: 5,
        nf_total_fat: 0.2,
        nf_dietary_fiber: 1.7
      },
      {
        food_name: 'lettuce',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 15,
        nf_protein: 1.4,
        nf_total_carbohydrate: 3,
        nf_total_fat: 0.1,
        nf_dietary_fiber: 1.3
      },
      {
        food_name: 'spinach',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 23,
        nf_protein: 2.9,
        nf_total_carbohydrate: 4,
        nf_total_fat: 0.4,
        nf_dietary_fiber: 2.2
      },
      {
        food_name: 'cabbage',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 25,
        nf_protein: 1.3,
        nf_total_carbohydrate: 6,
        nf_total_fat: 0.1,
        nf_dietary_fiber: 2.5
      },
      {
        food_name: 'asparagus',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 20,
        nf_protein: 2.2,
        nf_total_carbohydrate: 4,
        nf_total_fat: 0.1,
        nf_dietary_fiber: 2.1
      },
      {
        food_name: 'celery',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 16,
        nf_protein: 0.7,
        nf_total_carbohydrate: 3,
        nf_total_fat: 0.2,
        nf_dietary_fiber: 1.6
      },
      {
        food_name: 'eggplant',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 25,
        nf_protein: 1.0,
        nf_total_carbohydrate: 6,
        nf_total_fat: 0.2,
        nf_dietary_fiber: 3.0
      },
      {
        food_name: 'zucchini',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 17,
        nf_protein: 1.2,
        nf_total_carbohydrate: 3,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 1.0
      },
      {
        food_name: 'mushroom',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 22,
        nf_protein: 3.1,
        nf_total_carbohydrate: 3,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 1.0
      },
      {
        food_name: 'pea',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 84,
        nf_protein: 5.4,
        nf_total_carbohydrate: 14,
        nf_total_fat: 0.4,
        nf_dietary_fiber: 5.7
      },
      {
        food_name: 'bean',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 127,
        nf_protein: 9.0,
        nf_total_carbohydrate: 23,
        nf_total_fat: 0.5,
        nf_dietary_fiber: 6.4
      },

      // Grains and Bread
      {
        food_name: 'bread',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 265,
        nf_protein: 9.0,
        nf_total_carbohydrate: 49,
        nf_total_fat: 3.2,
        nf_dietary_fiber: 2.7
      },
      {
        food_name: 'toast',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 313,
        nf_protein: 8.5,
        nf_total_carbohydrate: 58,
        nf_total_fat: 4.2,
        nf_dietary_fiber: 2.8
      },
      {
        food_name: 'bagel',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 245,
        nf_protein: 10.0,
        nf_total_carbohydrate: 48,
        nf_total_fat: 1.5,
        nf_dietary_fiber: 2.7
      },
      {
        food_name: 'croissant',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 406,
        nf_protein: 8.2,
        nf_total_carbohydrate: 45,
        nf_total_fat: 21,
        nf_dietary_fiber: 2.6
      },
      {
        food_name: 'muffin',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 265,
        nf_protein: 5.0,
        nf_total_carbohydrate: 44,
        nf_total_fat: 8.0,
        nf_dietary_fiber: 1.5
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
        food_name: 'pasta',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 131,
        nf_protein: 5,
        nf_total_carbohydrate: 25,
        nf_total_fat: 1.1,
        nf_dietary_fiber: 1.8
      },
      {
        food_name: 'spaghetti',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 158,
        nf_protein: 5.8,
        nf_total_carbohydrate: 31,
        nf_total_fat: 0.9,
        nf_dietary_fiber: 1.8
      },
      {
        food_name: 'noodle',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 138,
        nf_protein: 4.5,
        nf_total_carbohydrate: 25,
        nf_total_fat: 2.1,
        nf_dietary_fiber: 1.2
      },
      {
        food_name: 'ramen',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 188,
        nf_protein: 6.0,
        nf_total_carbohydrate: 27,
        nf_total_fat: 7.0,
        nf_dietary_fiber: 1.2
      },

      // Proteins
      {
        food_name: 'chicken',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 165,
        nf_protein: 31,
        nf_total_carbohydrate: 0,
        nf_total_fat: 3.6,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'beef',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 250,
        nf_protein: 26,
        nf_total_carbohydrate: 0,
        nf_total_fat: 15,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'pork',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 242,
        nf_protein: 27,
        nf_total_carbohydrate: 0,
        nf_total_fat: 14,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'steak',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 271,
        nf_protein: 26,
        nf_total_carbohydrate: 0,
        nf_total_fat: 18,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'hamburger',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 295,
        nf_protein: 17,
        nf_total_carbohydrate: 30,
        nf_total_fat: 12,
        nf_dietary_fiber: 1.8
      },
      {
        food_name: 'hot dog',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 290,
        nf_protein: 12,
        nf_total_carbohydrate: 4,
        nf_total_fat: 26,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'bacon',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 541,
        nf_protein: 37,
        nf_total_carbohydrate: 1.4,
        nf_total_fat: 42,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'fish',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 206,
        nf_protein: 22,
        nf_total_carbohydrate: 0,
        nf_total_fat: 12,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'salmon',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 208,
        nf_protein: 25,
        nf_total_carbohydrate: 0,
        nf_total_fat: 12,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'tuna',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 144,
        nf_protein: 30,
        nf_total_carbohydrate: 0,
        nf_total_fat: 1,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'shrimp',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 99,
        nf_protein: 24,
        nf_total_carbohydrate: 0.2,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'egg',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 155,
        nf_protein: 13,
        nf_total_carbohydrate: 1.1,
        nf_total_fat: 11,
        nf_dietary_fiber: 0
      },

      // Dairy
      {
        food_name: 'milk',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 42,
        nf_protein: 3.4,
        nf_total_carbohydrate: 5,
        nf_total_fat: 1.0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'cheese',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 113,
        nf_protein: 25,
        nf_total_carbohydrate: 1.3,
        nf_total_fat: 0.3,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'yogurt',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 59,
        nf_protein: 10,
        nf_total_carbohydrate: 3.6,
        nf_total_fat: 0.4,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'butter',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 717,
        nf_protein: 0.9,
        nf_total_carbohydrate: 0.1,
        nf_total_fat: 81,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'cream',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 340,
        nf_protein: 2.1,
        nf_total_carbohydrate: 2.8,
        nf_total_fat: 36,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'mozzarella',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 280,
        nf_protein: 28,
        nf_total_carbohydrate: 2.2,
        nf_total_fat: 17,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'cheddar',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 403,
        nf_protein: 25,
        nf_total_carbohydrate: 1.3,
        nf_total_fat: 33,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'parmesan',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 431,
        nf_protein: 38,
        nf_total_carbohydrate: 4.1,
        nf_total_fat: 29,
        nf_dietary_fiber: 0
      },

      // Prepared Foods
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
        food_name: 'sandwich',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 257,
        nf_protein: 12,
        nf_total_carbohydrate: 28,
        nf_total_fat: 12,
        nf_dietary_fiber: 2.1
      },
      {
        food_name: 'taco',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 226,
        nf_protein: 9,
        nf_total_carbohydrate: 20,
        nf_total_fat: 12,
        nf_dietary_fiber: 2.8
      },
      {
        food_name: 'burrito',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 248,
        nf_protein: 8,
        nf_total_carbohydrate: 35,
        nf_total_fat: 9,
        nf_dietary_fiber: 3.2
      },
      {
        food_name: 'lasagna',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 132,
        nf_protein: 7,
        nf_total_carbohydrate: 15,
        nf_total_fat: 6,
        nf_dietary_fiber: 1.2
      },
      {
        food_name: 'soup',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 34,
        nf_protein: 2,
        nf_total_carbohydrate: 5,
        nf_total_fat: 1,
        nf_dietary_fiber: 0.8
      },
      {
        food_name: 'stew',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 73,
        nf_protein: 5,
        nf_total_carbohydrate: 8,
        nf_total_fat: 3,
        nf_dietary_fiber: 1.5
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
        food_name: 'sushi',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 150,
        nf_protein: 6,
        nf_total_carbohydrate: 30,
        nf_total_fat: 0.5,
        nf_dietary_fiber: 0.8
      },
      {
        food_name: 'dumpling',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 233,
        nf_protein: 8,
        nf_total_carbohydrate: 35,
        nf_total_fat: 8,
        nf_dietary_fiber: 1.5
      },

      // Desserts and Sweets
      {
        food_name: 'cake',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 257,
        nf_protein: 4,
        nf_total_carbohydrate: 45,
        nf_total_fat: 8,
        nf_dietary_fiber: 1.2
      },
      {
        food_name: 'cookie',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 502,
        nf_protein: 6,
        nf_total_carbohydrate: 65,
        nf_total_fat: 24,
        nf_dietary_fiber: 2.1
      },
      {
        food_name: 'donut',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 452,
        nf_protein: 5,
        nf_total_carbohydrate: 51,
        nf_total_fat: 25,
        nf_dietary_fiber: 1.8
      },
      {
        food_name: 'ice cream',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 207,
        nf_protein: 4,
        nf_total_carbohydrate: 24,
        nf_total_fat: 11,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'chocolate',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 546,
        nf_protein: 4.9,
        nf_total_carbohydrate: 61,
        nf_total_fat: 31,
        nf_dietary_fiber: 7.0
      },

      // Beverages
      {
        food_name: 'coffee',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 2,
        nf_protein: 0.3,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'tea',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 1,
        nf_protein: 0,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'juice',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 45,
        nf_protein: 0.5,
        nf_total_carbohydrate: 11,
        nf_total_fat: 0.1,
        nf_dietary_fiber: 0.2
      },
      {
        food_name: 'wine',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 83,
        nf_protein: 0.1,
        nf_total_carbohydrate: 2.6,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'beer',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 43,
        nf_protein: 0.5,
        nf_total_carbohydrate: 3.6,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },

      // Kitchen Items (for context, minimal nutrition)
      {
        food_name: 'plate',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 0,
        nf_protein: 0,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'bowl',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 0,
        nf_protein: 0,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'cup',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 0,
        nf_protein: 0,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'glass',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 0,
        nf_protein: 0,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'fork',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 0,
        nf_protein: 0,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'knife',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 0,
        nf_protein: 0,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'spoon',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 0,
        nf_protein: 0,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'person',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 0,
        nf_protein: 0,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'table',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 0,
        nf_protein: 0,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      },
      {
        food_name: 'chair',
        serving_qty: 100,
        serving_unit: 'g',
        nf_calories: 0,
        nf_protein: 0,
        nf_total_carbohydrate: 0,
        nf_total_fat: 0,
        nf_dietary_fiber: 0
      }
    ];

    comprehensiveFoods.forEach(food => {
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

      // Default serving size in grams for each food item
      const defaultServingSizes: { [key: string]: number } = {
        // Fruits (typical serving sizes)
        'apple': 150, 'banana': 120, 'orange': 130, 'grape': 100, 'strawberry': 100,
        'blueberry': 100, 'pineapple': 100, 'mango': 100, 'peach': 100, 'cherry': 100,
        'watermelon': 100, 'avocado': 100,
        
        // Vegetables (typical serving sizes)
        'carrot': 100, 'broccoli': 100, 'cauliflower': 100, 'corn': 100, 'cucumber': 100,
        'tomato': 100, 'potato': 150, 'onion': 100, 'garlic': 10, 'pepper': 100,
        'lettuce': 50, 'spinach': 50, 'cabbage': 100, 'asparagus': 100, 'celery': 100,
        'eggplant': 100, 'zucchini': 100, 'mushroom': 100, 'pea': 100, 'bean': 100,
        
        // Grains and Bread (typical serving sizes)
        'bread': 50, 'toast': 50, 'bagel': 95, 'croissant': 57, 'muffin': 113,
        'rice': 100, 'pasta': 100, 'spaghetti': 100, 'noodle': 100, 'ramen': 100,
        
        // Proteins (typical serving sizes)
        'chicken': 100, 'beef': 100, 'pork': 100, 'steak': 100, 'hamburger': 150,
        'hot dog': 100, 'bacon': 30, 'fish': 100, 'salmon': 100, 'tuna': 100,
        'shrimp': 100, 'egg': 50,
        
        // Dairy (typical serving sizes)
        'milk': 240, 'cheese': 30, 'yogurt': 170, 'butter': 15, 'cream': 30,
        'mozzarella': 30, 'cheddar': 30, 'parmesan': 30,
        
        // Prepared Foods (typical serving sizes)
        'pizza': 150, 'sandwich': 200, 'taco': 150, 'burrito': 200, 'lasagna': 200,
        'soup': 250, 'stew': 250, 'salad': 100, 'sushi': 100, 'dumpling': 100,
        
        // Desserts and Sweets (typical serving sizes)
        'cake': 100, 'cookie': 30, 'donut': 60, 'ice cream': 100, 'chocolate': 30,
        
        // Beverages (typical serving sizes)
        'coffee': 240, 'tea': 240, 'juice': 240, 'wine': 150, 'beer': 355,
        
        // Kitchen Items (no nutrition)
        'plate': 0, 'bowl': 0, 'cup': 0, 'glass': 0, 'fork': 0, 'knife': 0, 'spoon': 0,
        'person': 0, 'table': 0, 'chair': 0
      };

      for (const foodName of foods) {
        const normalizedName = foodName.toLowerCase().trim();
        const foodData = this.nutritionDatabase.get(normalizedName);
        
        if (foodData) {
          // Get default serving size for this food
          const servingSize = defaultServingSizes[normalizedName] || 100;
          
          // Calculate nutrition based on serving size (per 100g basis)
          const multiplier = servingSize / 100;
          
          const calories = Math.round(foodData.nf_calories * multiplier);
          const protein = Math.round(foodData.nf_protein * multiplier * 10) / 10;
          const carbs = Math.round(foodData.nf_total_carbohydrate * multiplier * 10) / 10;
          const fats = Math.round(foodData.nf_total_fat * multiplier * 10) / 10;

          const item: FoodItem = {
            name: foodData.food_name,
            calories: calories,
            protein: protein,
            carbs: carbs,
            fats: fats
          };

          items.push(item);
          totalCalories += calories;
          totalProtein += protein;
          totalCarbs += carbs;
          totalFats += fats;
          
          console.log(`Nutrition for ${foodData.food_name}: ${calories} cal, ${protein}g protein, ${carbs}g carbs, ${fats}g fats`);
        } else {
          // Fallback for unknown foods - use generic values
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
          
          console.log(`Fallback nutrition for ${foodName}: ${fallbackItem.calories} cal, ${fallbackItem.protein}g protein, ${fallbackItem.carbs}g carbs, ${fallbackItem.fats}g fats`);
        }
      }

      console.log(`Total nutrition: ${totalCalories} calories, ${totalProtein}g protein, ${totalCarbs}g carbs, ${totalFats}g fats`);

      return {
        total_calories: totalCalories,
        total_protein: Math.round(totalProtein * 10) / 10,
        total_carbs: Math.round(totalCarbs * 10) / 10,
        total_fats: Math.round(totalFats * 10) / 10,
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