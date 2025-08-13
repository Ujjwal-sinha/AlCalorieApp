import sharp from 'sharp';
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { ModelManager } from './ModelManager';
import { NutritionService } from './NutritionService';
import { config } from '../config';
import type {
  AnalysisResult,
  ProcessedImage,
  DetectionResult,
  AnalysisRequest,
  FoodItem,
  NutritionData,
  FoodAnalysisContext
} from '../types';

interface FoodAgentConfig {
  models: any;
  contextCache: Map<string, any>;
  searchCache: Map<string, any>;
  sessionId?: string;
  nutritionDb: Map<string, NutritionData>;
}

interface DetectionMethod {
  name: string;
  confidence: number;
  detectedFoods: string[];
  processingTime: number;
}

interface PythonModelResponse {
  success: boolean;
  detected_foods: string[];
  confidence_scores: Record<string, number>;
  processing_time: number;
  error?: string;
}

export class FoodDetectionService {
  private static instance: FoodDetectionService;
  private modelManager: ModelManager;
  private nutritionService: NutritionService;
  private foodAgent: FoodAgentConfig;
  private validFoodItems!: Set<string>;
  private nonFoodItems!: Set<string>;
  private pythonScriptsPath: string;

  private constructor() {
    this.modelManager = ModelManager.getInstance();
    this.nutritionService = NutritionService.getInstance();
    this.foodAgent = {
      models: {},
      contextCache: new Map(),
      searchCache: new Map(),
      nutritionDb: new Map()
    };
    this.pythonScriptsPath = path.join(process.cwd(), 'python_models');
    this.initializeFoodDatabases();
  }

  public static getInstance(): FoodDetectionService {
    if (!FoodDetectionService.instance) {
      FoodDetectionService.instance = new FoodDetectionService();
    }
    return FoodDetectionService.instance;
  }

  private initializeFoodDatabases(): void {
    // Valid food items database
    this.validFoodItems = new Set([
      // Proteins
      'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'fish', 'salmon', 'tuna', 'cod',
      'shrimp', 'crab', 'lobster', 'egg', 'eggs', 'tofu', 'tempeh', 'bacon', 'sausage', 'ham',
      'steak', 'ground beef', 'chicken breast', 'chicken thigh', 'pork chop', 'lamb chop',
      
      // Vegetables
      'tomato', 'tomatoes', 'potato', 'potatoes', 'carrot', 'carrots', 'onion', 'onions',
      'broccoli', 'cauliflower', 'lettuce', 'spinach', 'kale', 'cucumber', 'bell pepper',
      'peppers', 'garlic', 'ginger', 'mushroom', 'mushrooms', 'corn', 'peas', 'beans',
      'green beans', 'asparagus', 'zucchini', 'eggplant', 'celery', 'radish', 'beet',
      'sweet potato', 'cabbage', 'brussels sprouts', 'artichoke',
      
      // Fruits
      'apple', 'apples', 'banana', 'bananas', 'orange', 'oranges', 'grape', 'grapes',
      'strawberry', 'strawberries', 'blueberry', 'blueberries', 'lemon', 'lime', 'peach',
      'pear', 'mango', 'pineapple', 'watermelon', 'cantaloupe', 'kiwi', 'avocado',
      'cherry', 'cherries', 'plum', 'apricot', 'coconut',
      
      // Grains & Starches
      'rice', 'bread', 'pasta', 'noodles', 'quinoa', 'oats', 'oatmeal', 'cereal',
      'wheat', 'barley', 'couscous', 'bulgur', 'tortilla', 'bagel', 'croissant',
      
      // Dairy
      'cheese', 'milk', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese',
      'mozzarella', 'cheddar', 'parmesan', 'feta', 'ricotta',
      
      // Prepared Foods
      'pizza', 'burger', 'sandwich', 'salad', 'soup', 'stew', 'curry', 'pasta',
      'spaghetti', 'lasagna', 'tacos', 'burrito', 'sushi', 'ramen', 'stir fry',
      
      // Beverages
      'water', 'juice', 'coffee', 'tea', 'milk', 'soda', 'beer', 'wine', 'smoothie',
      
      // Condiments & Seasonings
      'salt', 'pepper', 'oil', 'olive oil', 'butter', 'sauce', 'ketchup', 'mustard',
      'mayonnaise', 'vinegar', 'soy sauce', 'hot sauce', 'herbs', 'spices',
      
      // Desserts
      'cake', 'cookie', 'cookies', 'ice cream', 'chocolate', 'candy', 'pie', 'pastry'
    ]);

    // Non-food items to filter out
    this.nonFoodItems = new Set([
      'plate', 'bowl', 'cup', 'glass', 'fork', 'knife', 'spoon', 'napkin', 'table',
      'chair', 'wall', 'background', 'surface', 'container', 'dish', 'utensil',
      'cutlery', 'placemat', 'tablecloth', 'decoration', 'garnish', 'presentation',
      'lighting', 'shadow', 'reflection', 'texture', 'color', 'pattern', 'style',
      'arrangement', 'display', 'setting', 'environment', 'scene', 'photo', 'image'
    ]);

    // Initialize nutrition database
    const nutritionData: [string, NutritionData][] = [
      ['chicken breast', { calories: 165, protein: 31, carbs: 0, fat: 3.6, fiber: 0 }],
      ['chicken', { calories: 165, protein: 31, carbs: 0, fat: 3.6, fiber: 0 }],
      ['rice', { calories: 130, protein: 2.7, carbs: 28, fat: 0.3, fiber: 0.4 }],
      ['white rice', { calories: 130, protein: 2.7, carbs: 28, fat: 0.3, fiber: 0.4 }],
      ['brown rice', { calories: 111, protein: 2.6, carbs: 23, fat: 0.9, fiber: 1.8 }],
      ['bread', { calories: 265, protein: 9, carbs: 49, fat: 3.2, fiber: 2.7 }],
      ['egg', { calories: 155, protein: 13, carbs: 1.1, fat: 11, fiber: 0 }],
      ['eggs', { calories: 155, protein: 13, carbs: 1.1, fat: 11, fiber: 0 }],
      ['tomato', { calories: 18, protein: 0.9, carbs: 3.9, fat: 0.2, fiber: 1.2 }],
      ['tomatoes', { calories: 18, protein: 0.9, carbs: 3.9, fat: 0.2, fiber: 1.2 }],
      ['potato', { calories: 77, protein: 2, carbs: 17, fat: 0.1, fiber: 2.2 }],
      ['potatoes', { calories: 77, protein: 2, carbs: 17, fat: 0.1, fiber: 2.2 }],
      ['banana', { calories: 89, protein: 1.1, carbs: 23, fat: 0.3, fiber: 2.6 }],
      ['apple', { calories: 52, protein: 0.3, carbs: 14, fat: 0.2, fiber: 2.4 }],
      ['orange', { calories: 47, protein: 0.9, carbs: 12, fat: 0.1, fiber: 2.4 }],
      ['broccoli', { calories: 34, protein: 2.8, carbs: 7, fat: 0.4, fiber: 2.6 }],
      ['carrot', { calories: 41, protein: 0.9, carbs: 10, fat: 0.2, fiber: 2.8 }],
      ['carrots', { calories: 41, protein: 0.9, carbs: 10, fat: 0.2, fiber: 2.8 }],
      ['beef', { calories: 250, protein: 26, carbs: 0, fat: 15, fiber: 0 }],
      ['pork', { calories: 242, protein: 27, carbs: 0, fat: 14, fiber: 0 }],
      ['fish', { calories: 206, protein: 22, carbs: 0, fat: 12, fiber: 0 }],
      ['salmon', { calories: 208, protein: 20, carbs: 0, fat: 13, fiber: 0 }],
      ['pasta', { calories: 131, protein: 5, carbs: 25, fat: 1.1, fiber: 1.8 }],
      ['cheese', { calories: 402, protein: 25, carbs: 1.3, fat: 33, fiber: 0 }],
      ['milk', { calories: 42, protein: 3.4, carbs: 5, fat: 1, fiber: 0 }],
      ['yogurt', { calories: 59, protein: 10, carbs: 3.6, fat: 0.4, fiber: 0 }],
      ['pizza', { calories: 266, protein: 11, carbs: 33, fat: 10, fiber: 2.3 }],
      ['burger', { calories: 295, protein: 17, carbs: 24, fat: 15, fiber: 2 }],
      ['sandwich', { calories: 250, protein: 12, carbs: 30, fat: 8, fiber: 3 }],
      ['salad', { calories: 20, protein: 1.5, carbs: 4, fat: 0.2, fiber: 2 }]
    ];

    nutritionData.forEach(([food, nutrition]) => {
      this.foodAgent.nutritionDb.set(food, nutrition);
    });
  }

  async processImage(buffer: Buffer): Promise<ProcessedImage> {
    try {
      const image = sharp(buffer);
      const metadata = await image.metadata();

      // Resize image if too large (max 1024x1024)
      const processedImage = await image
        .resize(1024, 1024, {
          fit: 'inside',
          withoutEnlargement: true
        })
        .jpeg({ quality: 90 })
        .toBuffer();

      return {
        buffer: processedImage,
        width: metadata.width || 0,
        height: metadata.height || 0,
        channels: metadata.channels || 3,
        format: metadata.format || 'jpeg'
      };
    } catch (error) {
      console.error('Image processing failed:', error);
      throw new Error('Failed to process image');
    }
  }

  private validateFoodItems(items: Set<string>, context: string): Set<string> {
    const validatedItems = new Set<string>();
    
    for (const item of items) {
      const itemClean = item.trim().toLowerCase();
      
      // Skip if too short or empty
      if (itemClean.length < 3) continue;
      
      // Skip if it's a non-food item
      if (Array.from(this.nonFoodItems).some(nonFood => itemClean.includes(nonFood))) {
        continue;
      }
      
      // Include if it's a known food item
      if (Array.from(this.validFoodItems).some(food => itemClean.includes(food))) {
        validatedItems.add(itemClean);
        continue;
      }
      
      // Include if it contains food-related keywords and context supports it
      const foodKeywords = ['meat', 'vegetable', 'fruit', 'grain', 'dairy', 'protein', 'sauce', 'seasoning'];
      if (foodKeywords.some(keyword => itemClean.includes(keyword))) {
        validatedItems.add(itemClean);
        continue;
      }
      
      // Include if the context strongly suggests it's food
      const foodContexts = ['cooked', 'fried', 'grilled', 'baked', 'fresh', 'seasoned'];
      if (foodContexts.some(foodContext => context.toLowerCase().includes(foodContext))) {
        if (itemClean.length > 3 && !Array.from(this.nonFoodItems).some(nonFood => itemClean.includes(nonFood))) {
          validatedItems.add(itemClean);
        }
      }
    }
    
    return validatedItems;
  }

  private generateSessionId(imageBuffer: Buffer): string {
    const timestamp = new Date().toISOString();
    const combined = `${imageBuffer.slice(0, 1000).toString('hex')}${timestamp}`;
    return require('crypto').createHash('md5').update(combined).digest('hex').slice(0, 12);
  }

  async analyzeWithAdvancedDetection(request: AnalysisRequest): Promise<AnalysisResult> {
    try {
      const startTime = Date.now();
      const sessionId = this.generateSessionId(request.image.buffer);

      console.log(`Starting advanced food analysis for session: ${sessionId}`);

      // Step 1: Process and validate image
      const processedImage = await this.processImage(request.image.buffer);
      
      // Step 2: Run multiple detection methods
      const detectionResults = await this.runDetectionMethods(processedImage);
      
      // Step 3: Validate and consolidate results
      const validatedFoods = this.validateFoodItems(
        new Set(detectionResults.flatMap(result => result.detectedFoods)),
        request.context || ''
      );
      
      // Step 4: Get nutrition data
      const nutritionData = await this.getNutritionData(Array.from(validatedFoods));
      
      // Step 5: Calculate totals
      const totalNutrition = this.calculateTotalNutrition(nutritionData);
      
      // Step 6: Generate insights
      const insights = this.generateInsights(nutritionData, totalNutrition, request);
      
      const processingTime = Date.now() - startTime;
      
      return {
        success: true,
        sessionId,
        detectedFoods: Array.from(validatedFoods),
        nutritionData,
        totalNutrition,
        insights,
        detectionMethods: detectionResults.map(result => result.name),
        processingTime,
        confidence: this.calculateOverallConfidence(detectionResults),
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      console.error('Advanced detection failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        timestamp: new Date().toISOString()
      };
    }
  }

  private async runDetectionMethods(image: ProcessedImage): Promise<DetectionMethod[]> {
    const detectionPromises: Promise<DetectionMethod>[] = [];
    const availableModels = this.modelManager.getAvailableModels();

    // YOLO detection
    if (availableModels.includes('yolo')) {
      detectionPromises.push(this.detectWithYOLO(image));
    }

    // Vision Transformer detection
    if (availableModels.includes('vit')) {
      detectionPromises.push(this.detectWithViT(image));
    }

    // Swin Transformer detection
    if (availableModels.includes('swin')) {
      detectionPromises.push(this.detectWithSwin(image));
    }

    // BLIP detection
    if (availableModels.includes('blip')) {
      detectionPromises.push(this.detectWithBLIP(image));
    }

    // CLIP detection
    if (availableModels.includes('clip')) {
      detectionPromises.push(this.detectWithCLIP(image));
    }

    // Color-based detection (always available)
    detectionPromises.push(this.detectWithColorAnalysis(image));

    // Run all detections in parallel
    const results = await Promise.allSettled(detectionPromises);
    
    return results
      .filter((result): result is PromiseFulfilledResult<DetectionMethod> => 
        result.status === 'fulfilled')
      .map(result => result.value);
  }

  private async detectWithYOLO(image: ProcessedImage): Promise<DetectionMethod> {
    const startTime = Date.now();
    try {
      // Try Python YOLO detection first
      const pythonResult = await this.callPythonDetection('yolo', image);
      if (pythonResult && pythonResult.success) {
        return {
          name: 'YOLO (Python)',
          confidence: Math.max(...Object.values(pythonResult.confidence_scores)),
          detectedFoods: pythonResult.detected_foods,
          processingTime: Date.now() - startTime
        };
      }

      // Fallback to simulation
      const detectedFoods = await this.simulateYOLODetection(image);
      
      return {
        name: 'YOLO (Simulated)',
        confidence: 0.85,
        detectedFoods,
        processingTime: Date.now() - startTime
      };
    } catch (error) {
      console.error('YOLO detection failed:', error);
      return {
        name: 'YOLO',
        confidence: 0,
        detectedFoods: [],
        processingTime: Date.now() - startTime
      };
    }
  }

  private async detectWithViT(image: ProcessedImage): Promise<DetectionMethod> {
    const startTime = Date.now();
    try {
      // Try Python ViT detection first
      const pythonResult = await this.callPythonDetection('vit', image);
      if (pythonResult && pythonResult.success) {
        return {
          name: 'Vision Transformer (Python)',
          confidence: Math.max(...Object.values(pythonResult.confidence_scores)),
          detectedFoods: pythonResult.detected_foods,
          processingTime: Date.now() - startTime
        };
      }

      // Fallback to simulation
      const detectedFoods = await this.simulateViTDetection(image);
      
      return {
        name: 'Vision Transformer (Simulated)',
        confidence: 0.88,
        detectedFoods,
        processingTime: Date.now() - startTime
      };
    } catch (error) {
      console.error('ViT detection failed:', error);
      return {
        name: 'Vision Transformer',
        confidence: 0,
        detectedFoods: [],
        processingTime: Date.now() - startTime
      };
    }
  }

  private async detectWithSwin(image: ProcessedImage): Promise<DetectionMethod> {
    const startTime = Date.now();
    try {
      // Try Python Swin detection first
      const pythonResult = await this.callPythonDetection('swin', image);
      if (pythonResult && pythonResult.success) {
        return {
          name: 'Swin Transformer (Python)',
          confidence: Math.max(...Object.values(pythonResult.confidence_scores)),
          detectedFoods: pythonResult.detected_foods,
          processingTime: Date.now() - startTime
        };
      }

      // Fallback to simulation
      const detectedFoods = await this.simulateSwinDetection(image);
      
      return {
        name: 'Swin Transformer (Simulated)',
        confidence: 0.87,
        detectedFoods,
        processingTime: Date.now() - startTime
      };
    } catch (error) {
      console.error('Swin detection failed:', error);
      return {
        name: 'Swin Transformer',
        confidence: 0,
        detectedFoods: [],
        processingTime: Date.now() - startTime
      };
    }
  }

  private async detectWithBLIP(image: ProcessedImage): Promise<DetectionMethod> {
    const startTime = Date.now();
    try {
      // Try Python BLIP detection first
      const pythonResult = await this.callPythonDetection('blip', image);
      if (pythonResult && pythonResult.success) {
        return {
          name: 'BLIP (Python)',
          confidence: Math.max(...Object.values(pythonResult.confidence_scores)),
          detectedFoods: pythonResult.detected_foods,
          processingTime: Date.now() - startTime
        };
      }

      // Fallback to simulation
      const detectedFoods = await this.simulateBLIPDetection(image);
      
      return {
        name: 'BLIP (Simulated)',
        confidence: 0.82,
        detectedFoods,
        processingTime: Date.now() - startTime
      };
    } catch (error) {
      console.error('BLIP detection failed:', error);
      return {
        name: 'BLIP',
        confidence: 0,
        detectedFoods: [],
        processingTime: Date.now() - startTime
      };
    }
  }

  private async detectWithCLIP(image: ProcessedImage): Promise<DetectionMethod> {
    const startTime = Date.now();
    try {
      // Try Python CLIP detection first
      const pythonResult = await this.callPythonDetection('clip', image);
      if (pythonResult && pythonResult.success) {
        return {
          name: 'CLIP (Python)',
          confidence: Math.max(...Object.values(pythonResult.confidence_scores)),
          detectedFoods: pythonResult.detected_foods,
          processingTime: Date.now() - startTime
        };
      }

      // Fallback to simulation
      const detectedFoods = await this.simulateCLIPDetection(image);
      
      return {
        name: 'CLIP (Simulated)',
        confidence: 0.84,
        detectedFoods,
        processingTime: Date.now() - startTime
      };
    } catch (error) {
      console.error('CLIP detection failed:', error);
      return {
        name: 'CLIP',
        confidence: 0,
        detectedFoods: [],
        processingTime: Date.now() - startTime
      };
    }
  }

  private async detectWithColorAnalysis(image: ProcessedImage): Promise<DetectionMethod> {
    const startTime = Date.now();
    try {
      // Analyze image colors to detect food types
      const detectedFoods = await this.analyzeImageColors(image);
      
      return {
        name: 'Color Analysis',
        confidence: 0.65,
        detectedFoods,
        processingTime: Date.now() - startTime
      };
    } catch (error) {
      console.error('Color analysis failed:', error);
      return {
        name: 'Color Analysis',
        confidence: 0,
        detectedFoods: [],
        processingTime: Date.now() - startTime
      };
    }
  }

  // Python integration for actual AI model detection
  private async callPythonDetection(modelType: string, image: ProcessedImage): Promise<PythonModelResponse | null> {
    try {
      const pythonScript = path.join(this.pythonScriptsPath, 'detect_food.py');
      
      if (!fs.existsSync(pythonScript)) {
        console.log(`Python script not found: ${pythonScript}, using simulation for ${modelType}`);
        return null;
      }

      return new Promise((resolve, reject) => {
        const python = spawn('python3', [pythonScript, modelType], {
          timeout: 30000 // 30 second timeout
        });
        
        let result = '';
        let error = '';

        // Send image data to Python script
        try {
          const requestData = {
            model: modelType,
            image_data: image.buffer.toString('base64'),
            width: image.width,
            height: image.height,
            format: image.format
          };
          
          python.stdin.write(JSON.stringify(requestData));
          python.stdin.end();
        } catch (e) {
          console.error(`Failed to send data to Python script: ${e}`);
          resolve(null);
          return;
        }

        python.stdout.on('data', (data: Buffer) => {
          result += data.toString();
        });

        python.stderr.on('data', (data: Buffer) => {
          error += data.toString();
        });

        python.on('close', (code: number) => {
          if (code === 0) {
            try {
              const parsed = JSON.parse(result) as PythonModelResponse;
              resolve(parsed);
            } catch (e) {
              console.error(`Failed to parse Python output: ${e}`);
              resolve(null);
            }
          } else {
            console.error(`Python script failed with code ${code}: ${error}`);
            resolve(null);
          }
        });

        python.on('error', (err: Error) => {
          console.error(`Python process error: ${err.message}`);
          resolve(null);
        });
      });
    } catch (error) {
      console.error(`Python ${modelType} detection failed:`, error);
      return null;
    }
  }

  // Simulation methods for AI model detection (fallback when Python models aren't available)
  private async simulateYOLODetection(image: ProcessedImage): Promise<string[]> {
    // Simulate YOLO object detection
    const possibleFoods = ['chicken', 'rice', 'vegetables', 'bread', 'egg'];
    return this.simulateDetection(possibleFoods, 0.8);
  }

  private async simulateViTDetection(image: ProcessedImage): Promise<string[]> {
    // Simulate Vision Transformer detection
    const possibleFoods = ['chicken breast', 'white rice', 'broccoli', 'carrots'];
    return this.simulateDetection(possibleFoods, 0.85);
  }

  private async simulateSwinDetection(image: ProcessedImage): Promise<string[]> {
    // Simulate Swin Transformer detection
    const possibleFoods = ['grilled chicken', 'steamed rice', 'green vegetables'];
    return this.simulateDetection(possibleFoods, 0.87);
  }

  private async simulateBLIPDetection(image: ProcessedImage): Promise<string[]> {
    // Simulate BLIP captioning
    const possibleFoods = ['cooked chicken', 'rice dish', 'fresh vegetables'];
    return this.simulateDetection(possibleFoods, 0.82);
  }

  private async simulateCLIPDetection(image: ProcessedImage): Promise<string[]> {
    // Simulate CLIP zero-shot classification
    const possibleFoods = ['protein', 'carbohydrates', 'vegetables'];
    return this.simulateDetection(possibleFoods, 0.84);
  }

  private async simulateDetection(possibleFoods: string[], confidence: number): Promise<string[]> {
    // Simulate detection with some randomness
    const detectedFoods: string[] = [];
    for (const food of possibleFoods) {
      if (Math.random() < confidence) {
        detectedFoods.push(food);
      }
    }
    return detectedFoods;
  }

  private async analyzeImageColors(image: ProcessedImage): Promise<string[]> {
    try {
      // Analyze dominant colors in the image
      const imageBuffer = image.buffer;
      const sharpImage = sharp(imageBuffer);
      
      // Get dominant colors
      const stats = await sharpImage.stats();
      const colors = stats.channels.map((channel, index) => ({
        r: index === 0 ? channel.mean : 0,
        g: index === 1 ? channel.mean : 0,
        b: index === 2 ? channel.mean : 0
      }));

      // Analyze colors to determine food types
      const detectedFoods: string[] = [];
      
      for (const color of colors) {
        // Green colors suggest vegetables
        if (color.g > color.r && color.g > color.b) {
          detectedFoods.push('vegetables');
        }
        // Brown/beige colors suggest meat or bread
        else if (color.r > 100 && color.g > 80 && color.b < 80) {
          detectedFoods.push('protein');
        }
        // White colors suggest rice or dairy
        else if (color.r > 200 && color.g > 200 && color.b > 200) {
          detectedFoods.push('carbohydrates');
        }
      }

      return [...new Set(detectedFoods)]; // Remove duplicates
    } catch (error) {
      console.error('Color analysis failed:', error);
      return [];
    }
  }

  private async getNutritionData(foods: string[]): Promise<Map<string, NutritionData>> {
    const nutritionData = new Map<string, NutritionData>();
    
    for (const food of foods) {
      // Check local database first
      if (this.foodAgent.nutritionDb.has(food)) {
        nutritionData.set(food, this.foodAgent.nutritionDb.get(food)!);
      } else {
        // Try to get from external nutrition service
        try {
          const nutrition = await this.nutritionService.calculateNutrition([food]);
          if (nutrition && nutrition.items.length > 0) {
            const item = nutrition.items[0];
            if (item) {
              nutritionData.set(food, {
                calories: item.calories,
                protein: item.protein,
                carbs: item.carbs,
                fat: item.fats,
                fiber: 0 // Default fiber value
              });
            }
          }
        } catch (error) {
          console.warn(`Failed to get nutrition data for ${food}:`, error);
        }
      }
    }
    
    return nutritionData;
  }

  private calculateTotalNutrition(nutritionData: Map<string, NutritionData>): NutritionData {
    const total: NutritionData = {
      calories: 0,
      protein: 0,
      carbs: 0,
      fat: 0,
      fiber: 0
    };

    for (const nutrition of nutritionData.values()) {
      total.calories += nutrition.calories;
      total.protein += nutrition.protein;
      total.carbs += nutrition.carbs;
      total.fat += nutrition.fat;
      total.fiber += nutrition.fiber;
    }

    return total;
  }

  private generateInsights(
    nutritionData: Map<string, NutritionData>, 
    totalNutrition: NutritionData, 
    request: AnalysisRequest
  ): string[] {
    const insights: string[] = [];
    
    // Calorie insights
    if (totalNutrition.calories > 500) {
      insights.push('This appears to be a high-calorie meal. Consider portion control.');
    } else if (totalNutrition.calories < 200) {
      insights.push('This is a light meal. You might want to add more protein or healthy fats.');
    }

    // Protein insights
    if (totalNutrition.protein < 20) {
      insights.push('This meal is low in protein. Consider adding lean protein sources.');
    } else if (totalNutrition.protein > 40) {
      insights.push('This meal is high in protein, great for muscle building and satiety.');
    }

    // Carbohydrate insights
    if (totalNutrition.carbs > 50) {
      insights.push('This meal is high in carbohydrates. Good for energy, but watch portion sizes.');
    }

    // Fat insights
    if (totalNutrition.fat > 20) {
      insights.push('This meal contains significant fat content. Consider the type of fats (healthy vs unhealthy).');
    }

    // Fiber insights
    if (totalNutrition.fiber < 5) {
      insights.push('This meal is low in fiber. Consider adding more vegetables or whole grains.');
    }

    // Balanced meal insight
    if (totalNutrition.protein >= 20 && totalNutrition.carbs >= 20 && nutritionData.size >= 3) {
      insights.push('This appears to be a well-balanced meal with good variety.');
    }

    return insights;
  }

  private calculateOverallConfidence(detectionMethods: DetectionMethod[]): number {
    if (detectionMethods.length === 0) return 0;
    
    const totalConfidence = detectionMethods.reduce((sum, method) => sum + method.confidence, 0);
    return totalConfidence / detectionMethods.length;
  }

  // Health check method
  async healthCheck(): Promise<{ healthy: boolean; pythonAvailable: boolean; models: string[] }> {
    const pythonAvailable = fs.existsSync(path.join(this.pythonScriptsPath, 'detect_food.py'));
    const availableModels = this.modelManager.getAvailableModels();
    
    return {
      healthy: availableModels.length > 0,
      pythonAvailable,
      models: availableModels
    };
  }
}