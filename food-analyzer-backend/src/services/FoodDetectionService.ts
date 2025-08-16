import sharp from 'sharp';
import { spawn } from 'child_process';
import { NutritionService } from './NutritionService';
import { GroqAnalysisService } from './GroqAnalysisService';
import { DietChatService } from './DietChatService';
import { ProcessedImage, AnalysisResult } from '../types';

interface PythonModelResponse {
  success: boolean;
  detected_foods: string[];
  confidence_scores: { [key: string]: number };
  processing_time: number;
  model_info?: {
    model_type: string;
    detection_count: number;
    confidence_threshold?: number;
    caption?: string;
  };
  error?: string;
}

/*
interface ExpertAnalysisResult {
  success: boolean;
  description: string;
  analysis: string;
  nutritional_data: {
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
  detected_foods: string[];
  confidence: number;
  processing_time: number;
  model_used: string;
  insights: string[];
  sessionId: string;
}
*/

export class FoodDetectionService {
  private static instance: FoodDetectionService;
  private nutritionService: NutritionService;
  private groqAnalysisService: GroqAnalysisService;
  private dietChatService: DietChatService;

  private constructor() {
    this.nutritionService = NutritionService.getInstance();
    this.groqAnalysisService = GroqAnalysisService.getInstance();
    this.dietChatService = DietChatService.getInstance();
  }

  public static getInstance(): FoodDetectionService {
    if (!FoodDetectionService.instance) {
      FoodDetectionService.instance = new FoodDetectionService();
    }
    return FoodDetectionService.instance;
  }

  async processImage(image: ProcessedImage | Buffer): Promise<ProcessedImage> {
    const buffer = Buffer.isBuffer(image) ? image : image.buffer;
    
    // Process image with sharp for optimization and enhancement
    const processedBuffer = await sharp(buffer)
      .resize(1024, 1024, { 
        fit: 'inside', 
        withoutEnlargement: true,
        kernel: sharp.kernel.lanczos3
      })
      .modulate({
        brightness: 1.1,  // Slightly increase brightness
        saturation: 1.1   // Slightly increase saturation
      })
      .sharpen(1, 1, 2)   // Sharpen the image with sigma, flat, jagged
      .jpeg({ 
        quality: 95,      // Higher quality
        progressive: true,
        mozjpeg: true
      })
      .toBuffer();

    return {
      buffer: processedBuffer,
      width: 1024,
      height: 1024,
      format: 'jpeg'
    };
  }

  private async callPythonDetection(modelType: string, imageBuffer: Buffer): Promise<PythonModelResponse> {
    try {
      const imageBase64 = imageBuffer.toString('base64');
      const inputData = {
        model_type: modelType,
        image_data: imageBase64,
        width: 1024,
        height: 1024
      };

      // Check if we should use remote Python service
      const pythonServiceUrl = process.env['PYTHON_MODELS_URL'];
      const useRemoteService = process.env['NODE_ENV'] === 'production' && pythonServiceUrl;

      if (useRemoteService) {
        console.log(`Using remote Python service for ${modelType} detection`);
        return await this.callRemotePythonService(pythonServiceUrl, inputData, modelType);
      } else {
        console.log(`Using local Python process for ${modelType} detection`);
        return await this.callLocalPythonProcess(modelType, inputData);
      }
    } catch (error) {
      console.error(`Error calling Python ${modelType} detection:`, error);
      throw error;
    }
  }

  private async callRemotePythonService(serviceUrl: string, inputData: any, modelType: string): Promise<PythonModelResponse> {
    try {
      const response = await fetch(`${serviceUrl}/detect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData),
        signal: AbortSignal.timeout(60000) // 60 second timeout
      });

      if (!response.ok) {
        throw new Error(`Remote Python service responded with status ${response.status}`);
      }

      const result = await response.json() as PythonModelResponse;
      console.log(`${modelType.toUpperCase()} detection result (remote):`, {
        success: result.success,
        detection_count: result.model_info?.detection_count || 0,
        detected_foods: result.detected_foods?.length || 0
      });
      return result;
    } catch (error) {
      console.error(`Remote Python service call failed for ${modelType}:`, error);
      throw error;
    }
  }

  private async callLocalPythonProcess(modelType: string, inputData: any): Promise<PythonModelResponse> {
    const pythonProcess = spawn('python3', ['python_models/detect_food.py'], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    return new Promise((resolve, reject) => {
      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
        console.log(`Python ${modelType} stderr:`, data.toString());
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error(`Python ${modelType} process exited with code ${code}`);
          console.error('Python stderr:', stderr);
          reject(new Error(`Python process failed with code ${code}`));
          return;
        }

        try {
          const result = JSON.parse(stdout);
          console.log(`${modelType.toUpperCase()} detection result (local):`, {
            success: result.success,
            detection_count: result.model_info?.detection_count || 0,
            detected_foods: result.detected_foods?.length || 0
          });
          resolve(result);
        } catch (error) {
          console.error(`Failed to parse Python ${modelType} response:`, error);
          console.error('Raw stdout:', stdout);
          reject(new Error(`Failed to parse Python response: ${error}`));
        }
      });

      pythonProcess.stdin.write(JSON.stringify(inputData));
      pythonProcess.stdin.end();
    });
  }

  async detectWithYOLO(image: ProcessedImage): Promise<{ foods: string[], confidence: number }> {
    try {
      const result = await this.callPythonDetection('yolo', image.buffer);
      if (result.success && result.detected_foods.length > 0) {
        // Calculate average confidence from all detected foods
        const avgConfidence = Object.values(result.confidence_scores).reduce((sum, conf) => sum + conf, 0) / Object.keys(result.confidence_scores).length;
        return { foods: result.detected_foods, confidence: avgConfidence };
      }
      return { foods: [], confidence: 0 };
    } catch (error) {
      console.error('YOLO detection failed:', error);
      return { foods: [], confidence: 0 };
    }
  }

  async detectWithViT(image: ProcessedImage): Promise<{ foods: string[], confidence: number }> {
    try {
      const result = await this.callPythonDetection('vit', image.buffer);
      if (result.success && result.detected_foods.length > 0) {
        // Calculate average confidence from all detected foods
        const avgConfidence = Object.values(result.confidence_scores).reduce((sum, conf) => sum + conf, 0) / Object.keys(result.confidence_scores).length;
        return { foods: result.detected_foods, confidence: avgConfidence };
      }
      return { foods: [], confidence: 0 };
    } catch (error) {
      console.error('ViT detection failed:', error);
      return { foods: [], confidence: 0 };
    }
  }

  async detectWithSwin(image: ProcessedImage): Promise<{ foods: string[], confidence: number }> {
    try {
      const result = await this.callPythonDetection('swin', image.buffer);
      if (result.success && result.detected_foods.length > 0) {
        // Calculate average confidence from all detected foods
        const avgConfidence = Object.values(result.confidence_scores).reduce((sum, conf) => sum + conf, 0) / Object.keys(result.confidence_scores).length;
        return { foods: result.detected_foods, confidence: avgConfidence };
      }
      return { foods: [], confidence: 0 };
    } catch (error) {
      console.error('Swin detection failed:', error);
      return { foods: [], confidence: 0 };
    }
  }

  async detectWithBLIP(image: ProcessedImage): Promise<{ foods: string[], confidence: number }> {
    try {
      const result = await this.callPythonDetection('blip', image.buffer);
      if (result.success && result.detected_foods.length > 0) {
        // Calculate average confidence from all detected foods
        const avgConfidence = Object.values(result.confidence_scores).reduce((sum, conf) => sum + conf, 0) / Object.keys(result.confidence_scores).length;
        return { foods: result.detected_foods, confidence: avgConfidence };
      }
      return { foods: [], confidence: 0 };
    } catch (error) {
      console.error('BLIP detection failed:', error);
      return { foods: [], confidence: 0 };
    }
  }

  async detectWithCLIP(image: ProcessedImage): Promise<{ foods: string[], confidence: number }> {
    try {
      const result = await this.callPythonDetection('clip', image.buffer);
      if (result.success && result.detected_foods.length > 0) {
        // Calculate average confidence from all detected foods
        const avgConfidence = Object.values(result.confidence_scores).reduce((sum, conf) => sum + conf, 0) / Object.keys(result.confidence_scores).length;
        return { foods: result.detected_foods, confidence: avgConfidence };
      }
      return { foods: [], confidence: 0 };
    } catch (error) {
      console.error('CLIP detection failed:', error);
      return { foods: [], confidence: 0 };
    }
  }

  /*
  private async detectWithLLM(image: ProcessedImage): Promise<{ foods: string[], confidence: number }> {
    try {
      const result = await this.callPythonDetection('llm', image.buffer);
      if (result.success && result.detected_foods.length > 0) {
        // Calculate average confidence from all detected foods
        const avgConfidence = Object.values(result.confidence_scores).reduce((sum: number, conf: number) => sum + conf, 0) / Object.values(result.confidence_scores).length;
        return {
          foods: result.detected_foods,
          confidence: avgConfidence
        };
      }
    } catch (error) {
      console.warn('LLM detection failed:', error);
    }

    // Return empty result instead of simulation
    return { foods: [], confidence: 0 };
  }
  */

  // Expert Analysis Implementation
  async performExpertAnalysis(request: { image: Express.Multer.File }): Promise<AnalysisResult> {
    const startTime = Date.now();
    const sessionId = this.generateSessionId(request.image.buffer);

    try {
      console.log(`Starting expert analysis for session ${sessionId}`);
      
      // Process image for better quality
      const processedImage = await this.processImage(request.image.buffer);
      console.log(`Image processed: ${processedImage.width}x${processedImage.height}`);

      // Run all detection models in parallel
      const detectionPromises = [
        this.detectWithYOLO(processedImage).then(result => ({ model: 'yolo', ...result })),
        this.detectWithViT(processedImage).then(result => ({ model: 'vit', ...result })),
        this.detectWithSwin(processedImage).then(result => ({ model: 'swin', ...result })),
        this.detectWithBLIP(processedImage).then(result => ({ model: 'blip', ...result })),
        this.detectWithCLIP(processedImage).then(result => ({ model: 'clip', ...result }))
      ];

      const detectionResults = await Promise.allSettled(detectionPromises);
      
      // Collect all detections with model information
      const allDetections = new Map<string, { count: number, totalConfidence: number, methods: string[], modelDetails: any[] }>();
      const modelPerformance: { [key: string]: { success: boolean, detection_count: number, error?: string } } = {};

      const modelNames = ['yolo', 'vit', 'swin', 'blip', 'clip'];
      detectionResults.forEach((result, index) => {
        const modelName = modelNames[index];
        if (!modelName) return;
        
        if (result.status === 'fulfilled') {
          const { model, foods, confidence } = result.value;
          
          modelPerformance[modelName] = {
            success: true,
            detection_count: foods.length
          };

          foods.forEach(food => {
            const normalizedFood = food.toLowerCase().trim();
            if (!allDetections.has(normalizedFood)) {
              allDetections.set(normalizedFood, { count: 0, totalConfidence: 0, methods: [], modelDetails: [] });
            }
            
            const detection = allDetections.get(normalizedFood);
            if (detection) {
              detection.count++;
              detection.totalConfidence += confidence;
              detection.methods.push(model);
              detection.modelDetails.push({ model, confidence, food });
            }
          });
          
          console.log(`${modelName.toUpperCase()} detected ${foods.length} items:`, foods);
        } else {
          modelPerformance[modelName] = {
            success: false,
            detection_count: 0,
            error: result.reason?.message || 'Unknown error'
          };
          console.error(`${modelName.toUpperCase()} detection failed:`, result.reason);
        }
      });

      // Apply expert filtering
      const filteredFoods = this.applyExpertFiltering(allDetections);
      
      console.log(`Expert filtering completed. Found ${filteredFoods.length} food items.`);

      if (filteredFoods.length === 0) {
        return {
          success: false,
          sessionId,
          detectedFoods: [],
          nutritionalData: null,
          totalNutrition: null,
          insights: [
            "No food items detected with sufficient confidence by AI models.",
            "Try uploading a clearer image with better lighting.",
            "Ensure the food items are clearly visible in the image.",
            "Consider taking the photo from a different angle."
          ],
          detectionMethods: Object.keys(modelPerformance).filter(m => modelPerformance[m]?.success),
          processingTime: Date.now() - startTime,
          confidence: 0,
          model_used: 'expert_ensemble',
          error: "No food items detected with sufficient confidence by AI models"
        };
      }

      // Get nutrition data for detected foods
      const nutritionPromises = filteredFoods.map(food => 
        this.nutritionService.searchFood(food.name)
      );
      
      const nutritionResults = await Promise.allSettled(nutritionPromises);
      const nutritionData = nutritionResults
        .map((result) => result.status === 'fulfilled' ? result.value : null)
        .filter(Boolean);

      // Calculate total nutrition
      const totalNutrition = await this.nutritionService.calculateNutrition(
        filteredFoods.map(food => food.name)
      );

      // Generate insights
      const insights = this.generateInsights(filteredFoods, modelPerformance, allDetections);

      // Generate GROQ analysis
      let groqAnalysis = null;
      try {
        const groqRequest = {
          detectedFoods: filteredFoods.map(food => food.name),
          nutritionalData: totalNutrition,
          foodItems: totalNutrition.items || [],
          imageDescription: `Meal containing: ${filteredFoods.map(food => food.name).join(', ')}`,
          mealContext: 'AI-detected meal analysis'
        };

        groqAnalysis = await this.groqAnalysisService.generateComprehensiveAnalysis(groqRequest);
        console.log('GROQ analysis completed successfully');
      } catch (error) {
        console.warn('GROQ analysis failed:', error);
        // Continue without GROQ analysis
      }

      // Generate automatic diet chat response
      let dietChatResponse = null;
      try {
        const detectedFoodNames = filteredFoods.map(food => food.name);
        const mealDescription = `I just uploaded a photo of my meal which contains: ${detectedFoodNames.join(', ')}. The total calories are ${totalNutrition.total_calories} with ${totalNutrition.total_protein}g protein, ${totalNutrition.total_carbs}g carbs, and ${totalNutrition.total_fats}g fats.`;
        
        const dietQuery = {
          question: `Can you analyze this meal and give me nutrition advice? ${mealDescription}`,
          context: `Meal analysis: ${detectedFoodNames.join(', ')} (${totalNutrition.total_calories} calories)`,
          userHistory: []
        };

        dietChatResponse = await this.dietChatService.answerDietQuery(dietQuery);
        console.log('Automatic diet chat response generated successfully');
      } catch (error) {
        console.warn('Automatic diet chat generation failed:', error);
        // Generate fallback diet chat response
        dietChatResponse = {
          answer: 'I analyzed your meal and found it contains a variety of foods. For personalized nutrition advice, please ensure your GROQ API key is configured.',
          suggestions: ['Consider adding more vegetables', 'Balance your protein and carbohydrate intake', 'Stay hydrated throughout the day'],
          relatedTopics: ['Meal Planning', 'Nutrition Basics', 'Healthy Eating'],
          confidence: 0.5
        };
      }

      const processingTime = Date.now() - startTime;
      const avgConfidence = filteredFoods.reduce((sum, food) => sum + food.confidence, 0) / filteredFoods.length;

      console.log(`Expert analysis completed in ${processingTime}ms. Detected ${filteredFoods.length} items with ${avgConfidence.toFixed(3)} average confidence.`);
      
      // Debug diet chat response
      if (dietChatResponse) {
        console.log('✅ Diet chat response generated:', {
          answer: dietChatResponse.answer?.substring(0, 100) + '...',
          suggestions: dietChatResponse.suggestions?.length || 0,
          relatedTopics: dietChatResponse.relatedTopics?.length || 0,
          confidence: dietChatResponse.confidence
        });
      } else {
        console.log('❌ No diet chat response generated');
      }

      return {
        success: true,
        sessionId,
        detectedFoods: filteredFoods.map(food => ({
          name: food.name,
          confidence: food.confidence,
          detectionMethods: food.methods
        })),
        nutritionalData: nutritionData,
        nutritional_data: totalNutrition, // Add this for frontend compatibility
        totalNutrition,
        insights,
        detectionMethods: Object.keys(modelPerformance).filter(m => modelPerformance[m]?.success),
        processingTime,
        confidence: avgConfidence,
        model_used: 'expert_ensemble',
        groq_analysis: groqAnalysis?.success ? {
          summary: groqAnalysis.summary,
          detailedAnalysis: groqAnalysis.detailedAnalysis,
          healthScore: groqAnalysis.healthScore,
          recommendations: groqAnalysis.recommendations,
          dietaryConsiderations: groqAnalysis.dietaryConsiderations
        } : undefined,
        diet_chat_response: dietChatResponse ? {
          answer: dietChatResponse.answer,
          suggestions: dietChatResponse.suggestions,
          relatedTopics: dietChatResponse.relatedTopics,
          confidence: dietChatResponse.confidence
        } : {
          answer: 'Diet chat analysis is currently unavailable. Please try again later.',
          suggestions: ['Upload a clear food image', 'Ensure good lighting', 'Try different angles'],
          relatedTopics: ['Food Photography', 'Nutrition Basics'],
          confidence: 0.3
        },
        model_info: {
          detection_count: filteredFoods.length,
          total_confidence: avgConfidence,
          model_performance: modelPerformance,
          detailed_detections: Array.from(allDetections.entries()).map(([food, details]) => ({
            food,
            count: details.count,
            methods: details.methods,
            avg_confidence: details.totalConfidence / details.count,
            model_details: details.modelDetails
          }))
        }
      };

    } catch (error) {
      console.error('Expert analysis failed:', error);
      return {
        success: false,
        sessionId,
        detectedFoods: [],
        nutritionalData: null,
        totalNutrition: null,
        insights: [
          "An error occurred during food detection.",
          "Please try uploading the image again.",
          "Ensure the image format is supported (JPEG, PNG)."
        ],
        detectionMethods: [],
        processingTime: Date.now() - startTime,
        confidence: 0,
        model_used: 'expert_ensemble',
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  /*
  private normalizeFoodName(foodName: string): string {
    return foodName.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .trim()
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }
  */

  private applyExpertFiltering(allDetections: Map<string, { count: number, totalConfidence: number, methods: string[] }>): Array<{ name: string, confidence: number, methods: string[] }> {
    const minConfidence = 0.08; // Much lower threshold for better detection
    const filteredFoods: Array<{ name: string, confidence: number, methods: string[] }> = [];

    for (const [foodName, detection] of allDetections) {
      let finalConfidence = detection.totalConfidence / detection.count;

      // Boost confidence for multi-model agreement
      if (detection.methods.length >= 3) {
        finalConfidence = Math.min(0.95, finalConfidence * 1.4);
      } else if (detection.methods.length >= 2) {
        finalConfidence = Math.min(0.95, finalConfidence * 1.3);
      }

      // Ensure minimum confidence for single detections
      finalConfidence = Math.max(finalConfidence, 0.15);

      if (finalConfidence >= minConfidence) {
        filteredFoods.push({
          name: foodName,
          confidence: finalConfidence,
          methods: detection.methods
        });
      }
    }

    // Sort by confidence and return top results
    return filteredFoods
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 20); // Increased from 15 to 20 for more comprehensive results
  }





  private generateInsights(filteredFoods: Array<{ name: string, confidence: number, methods: string[] }>, 
                          modelPerformance: { [key: string]: { success: boolean, detection_count: number, error?: string } },
                          _allDetections: Map<string, any>): string[] {
    const insights: string[] = [];
    
    // Model performance insights
    const successfulModels = Object.keys(modelPerformance).filter(m => modelPerformance[m]?.success);
    const totalDetections = successfulModels.reduce((sum, model) => sum + (modelPerformance[model]?.detection_count || 0), 0);
    
    insights.push(`Analysis completed using ${successfulModels.length} AI models: ${successfulModels.join(', ')}`);
    insights.push(`Total raw detections across all models: ${totalDetections}`);
    
    // Multi-model agreement insights
    const multiModelFoods = filteredFoods.filter(food => food.methods.length >= 2);
    if (multiModelFoods.length > 0) {
      insights.push(`${multiModelFoods.length} items detected by multiple models (high confidence)`);
    }
    
    // Model-specific insights
    successfulModels.forEach(model => {
      const count = modelPerformance[model]?.detection_count || 0;
      if (count > 0) {
        insights.push(`${model.toUpperCase()} detected ${count} items`);
      }
    });
    
    // Confidence insights
    const highConfidenceFoods = filteredFoods.filter(food => food.confidence > 0.7);
    if (highConfidenceFoods.length > 0) {
      insights.push(`${highConfidenceFoods.length} items with high confidence (>70%)`);
    }
    
    return insights;
  }

  private generateSessionId(imageBuffer: Buffer): string {
    const hash = require('crypto').createHash('md5').update(imageBuffer).digest('hex');
    return `sess_${hash.substring(0, 8)}_${Date.now()}`;
  }

  async healthCheck(): Promise<{ status: string; models: Record<string, boolean>; pythonAvailable: boolean }> {
    const models = {
      yolo: false,
      vit: false,
      swin: false,
      blip: false,
      clip: false,
      llm: false
    };

    try {
      // Test Python availability
      const testResult = await this.callPythonDetection('test', Buffer.from('test'));
      models.yolo = testResult.success;
      models.vit = testResult.success;
      models.swin = testResult.success;
      models.blip = testResult.success;
      models.clip = testResult.success;
      models.llm = testResult.success;
    } catch (error) {
      console.warn('Python models not available:', error);
    }

    return {
      status: 'healthy',
      models,
      pythonAvailable: Object.values(models).some(v => v)
    };
  }
}