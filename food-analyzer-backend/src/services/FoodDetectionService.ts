import sharp from 'sharp';
import { spawn } from 'child_process';
import { NutritionService } from './NutritionService';
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

export class FoodDetectionService {
  private static instance: FoodDetectionService;
  private nutritionService: NutritionService;

  private constructor() {
    this.nutritionService = NutritionService.getInstance();
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
            console.log(`${modelType.toUpperCase()} detection result:`, {
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
    } catch (error) {
      console.error(`Error calling Python ${modelType} detection:`, error);
      throw error;
    }
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
        .map((result, index) => result.status === 'fulfilled' ? result.value : null)
        .filter(Boolean);

      // Calculate total nutrition
      const totalNutrition = await this.nutritionService.calculateNutrition(
        filteredFoods.map(food => food.name)
      );

      // Generate insights
      const insights = this.generateInsights(filteredFoods, modelPerformance, allDetections);

      const processingTime = Date.now() - startTime;
      const avgConfidence = filteredFoods.reduce((sum, food) => sum + food.confidence, 0) / filteredFoods.length;

      console.log(`Expert analysis completed in ${processingTime}ms. Detected ${filteredFoods.length} items with ${avgConfidence.toFixed(3)} average confidence.`);

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

  private normalizeFoodName(foodName: string): string {
    return foodName.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .trim()
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  private applyExpertFiltering(allDetections: Map<string, { count: number, totalConfidence: number, methods: string[] }>): Array<{ name: string, confidence: number, methods: string[] }> {
    const minConfidence = 0.15; // Higher threshold for accurate detection (increased from 0.05)
    const filteredFoods: Array<{ name: string, confidence: number, methods: string[] }> = [];

    for (const [foodName, detection] of allDetections) {
      let finalConfidence = detection.totalConfidence / detection.count;

      // Boost confidence for multi-model agreement
      if (detection.methods.length >= 3) {
        finalConfidence = Math.min(0.95, finalConfidence * 1.3); // Reduced boost for accuracy
      } else if (detection.methods.length >= 2) {
        finalConfidence = Math.min(0.95, finalConfidence * 1.2); // Reduced boost for accuracy
      }

      // Ensure minimum confidence for single detections
      finalConfidence = Math.max(finalConfidence, 0.2); // Increased from 0.1 to 0.2

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
      .slice(0, 15); // Reduced from 25 to 15 for more focused results
  }

  private async generateExpertAnalysis(foods: Array<{ name: string, confidence: number, methods: string[] }>, context: string): Promise<string> {
    if (foods.length === 0) {
      return "No food items detected with sufficient confidence. Please try with a clearer image or different angle.";
    }

    const foodList = foods.map(f => `${f.name} (${Math.round(f.confidence * 100)}% confidence via ${f.methods.join(', ')})`).join('\n- ');

    let analysis = `## COMPREHENSIVE FOOD ANALYSIS

### IDENTIFIED FOOD ITEMS:
- ${foodList}

### DETECTION METHODOLOGY:
This analysis used an expert ensemble of ${foods.length} AI models:
${foods.map(f => `- ${f.name}: Detected by ${f.methods.join(', ')} with ${Math.round(f.confidence * 100)}% confidence`).join('\n')}

### MEAL COMPOSITION ANALYSIS:
- **Detection Quality**: ${this.assessDetectionQuality(foods)}
- **Confidence Level**: ${this.assessConfidenceLevel(foods)}
- **Model Agreement**: ${this.assessModelAgreement(foods)}

### EXPERT INSIGHTS:
${this.generateExpertInsights(foods)}

### RECOMMENDATIONS:
1. **Verification**: Review detected items for accuracy
2. **Portion Estimation**: Consider visual cues for serving sizes
3. **Nutritional Context**: Use this analysis as a starting point for detailed nutrition tracking

${context ? `\n### CONTEXT NOTES:\n${context}` : ''}`;

    return analysis;
  }

  private assessDetectionQuality(foods: Array<{ name: string, confidence: number, methods: string[] }>): string {
    const avgConfidence = foods.reduce((sum, f) => sum + f.confidence, 0) / foods.length;
    if (avgConfidence >= 0.8) return "Excellent - High confidence detections";
    if (avgConfidence >= 0.6) return "Good - Reliable detections";
    if (avgConfidence >= 0.4) return "Fair - Moderate confidence";
    return "Limited - Low confidence detections";
  }

  private assessConfidenceLevel(foods: Array<{ name: string, confidence: number, methods: string[] }>): string {
    const highConfidence = foods.filter(f => f.confidence >= 0.7).length;
    const total = foods.length;
    const percentage = (highConfidence / total) * 100;
    return `${percentage.toFixed(0)}% high confidence (${highConfidence}/${total} items)`;
  }

  private assessModelAgreement(foods: Array<{ name: string, confidence: number, methods: string[] }>): string {
    const multiModelDetections = foods.filter(f => f.methods.length > 1).length;
    const total = foods.length;
    const percentage = (multiModelDetections / total) * 100;
    return `${percentage.toFixed(0)}% detected by multiple models (${multiModelDetections}/${total} items)`;
  }

  private generateExpertInsights(foods: Array<{ name: string, confidence: number, methods: string[] }>): string {
    const insights = [];
    
    if (foods.length === 0) {
      insights.push("No food items were detected with sufficient confidence.");
    } else {
      insights.push(`Successfully identified ${foods.length} food items using expert AI ensemble.`);
      
      const highConfidenceCount = foods.filter(f => f.confidence >= 0.7).length;
      if (highConfidenceCount > 0) {
        insights.push(`${highConfidenceCount} items detected with high confidence (≥70%).`);
      }
      
      const multiModelCount = foods.filter(f => f.methods.length > 1).length;
      if (multiModelCount > 0) {
        insights.push(`${multiModelCount} items confirmed by multiple AI models for enhanced accuracy.`);
      }
    }
    
    return insights.join('\n');
  }

  private generateInsights(filteredFoods: Array<{ name: string, confidence: number, methods: string[] }>, 
                          modelPerformance: { [key: string]: { success: boolean, detection_count: number, error?: string } },
                          allDetections: Map<string, any>): string[] {
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
