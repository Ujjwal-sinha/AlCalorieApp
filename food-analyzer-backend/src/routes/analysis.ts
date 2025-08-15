import { Router, Request, Response } from 'express';
import multer from 'multer';
import { FoodDetectionService } from '../services/FoodDetectionService';
import { GroqAnalysisService } from '../services/GroqAnalysisService';
import { DietChatService } from '../services/DietChatService';
// import { validateAnalysisRequest } from '../middleware/validation';
import { asyncHandler } from '../middleware/asyncHandler';
import { config } from '../config';

const router = Router();
const foodDetectionService = FoodDetectionService.getInstance();
const groqAnalysisService = GroqAnalysisService.getInstance();
const dietChatService = DietChatService.getInstance();

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: config.maxFileSize,
    files: 1
  },
  fileFilter: (_req, file, cb) => {
    if (config.allowedMimeTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`Invalid file type. Allowed types: ${config.allowedMimeTypes.join(', ')}`));
    }
  }
});

// Expert analysis endpoint (main endpoint)
router.post('/advanced', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'Image file is required'
      });
    }

    console.log('Starting expert analysis...');
    
    // Perform expert analysis with the uploaded file
    const result = await foodDetectionService.performExpertAnalysis({ image: req.file });
    
    console.log('Expert analysis completed:', {
      success: result.success,
      detection_count: result.model_info?.detection_count || 0,
      models_used: result.detectionMethods?.length || 0
    });

    console.log('Sending response with nutrition data:', {
      has_nutritional_data: !!result.nutritional_data,
      has_totalNutrition: !!result.totalNutrition,
      nutrition_calories: result.nutritional_data?.total_calories || result.totalNutrition?.total_calories,
      nutrition_protein: result.nutritional_data?.total_protein || result.totalNutrition?.total_protein,
      nutrition_carbs: result.nutritional_data?.total_carbs || result.totalNutrition?.total_carbs,
      nutrition_fats: result.nutritional_data?.total_fats || result.totalNutrition?.total_fats
    });

    return res.json(result);
  } catch (error) {
    console.error('Expert analysis error:', error);
    return res.status(500).json({
      success: false,
      error: 'Expert analysis failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Model-specific analysis endpoints
router.post('/model/:modelType', upload.single('image'), async (req, res) => {
  try {
    const modelType = req.params['modelType'];
    
    if (!modelType) {
      return res.status(400).json({
        success: false,
        error: 'Model type is required'
      });
    }
    
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'Image file is required'
      });
    }

    const validModels = ['yolo', 'vit', 'swin', 'blip', 'clip'];
    if (!validModels.includes(modelType)) {
      return res.status(400).json({
        success: false,
        error: `Invalid model type. Must be one of: ${validModels.join(', ')}`
      });
    }

    console.log(`Starting ${modelType} analysis...`);
    
    // Process the image
    const processedImage = await foodDetectionService.processImage(req.file.buffer);
    
    // Perform model-specific detection
    let result;
    switch (modelType) {
      case 'yolo':
        result = await foodDetectionService.detectWithYOLO(processedImage);
        break;
      case 'vit':
        result = await foodDetectionService.detectWithViT(processedImage);
        break;
      case 'swin':
        result = await foodDetectionService.detectWithSwin(processedImage);
        break;
      case 'blip':
        result = await foodDetectionService.detectWithBLIP(processedImage);
        break;
      case 'clip':
        result = await foodDetectionService.detectWithCLIP(processedImage);
        break;
      default:
        return res.status(400).json({
          success: false,
          error: 'Invalid model type'
        });
    }

    return res.json({
      success: true,
      model_type: modelType,
      detected_foods: result.foods,
      confidence: result.confidence,
      processing_time: Date.now()
    });
  } catch (error) {
    console.error(`${req.params['modelType']} analysis error:`, error);
    return res.status(500).json({
      success: false,
      error: `${req.params['modelType']} analysis failed`,
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Batch analysis endpoint
router.post('/batch',
  upload.array('images', 10),
  asyncHandler(async (req: Request, res: Response) => {
    const files = req.files as Express.Multer.File[];
    
    if (!files || files.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No image files provided'
      });
    }

    try {
      const results = await Promise.allSettled(
        files.map(async (file, index) => {
          // const processedImage = await foodDetectionService.processImage(file.buffer);
          
          const result = await foodDetectionService.performExpertAnalysis({
            image: file
          });

          return {
            index,
            filename: file.originalname,
            result: result
          };
        })
      );

      const successful = results
        .filter((result): result is PromiseFulfilledResult<any> => result.status === 'fulfilled')
        .map(result => result.value);

      const failed = results
        .map((result, index) => result.status === 'rejected' ? { index, error: result.reason } : null)
        .filter(Boolean);

      return res.json({
        success: true,
        total: files.length,
        successful: successful.length,
        failed: failed.length,
        results: successful,
        errors: failed
      });
    } catch (error) {
      console.error('Batch analysis failed:', error);
      return res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Batch analysis failed'
      });
    }
  })
);

// Analysis status endpoint
router.get('/status', asyncHandler(async (_req: Request, res: Response) => {
  const healthCheck = await foodDetectionService.healthCheck();
  
  return res.json({
    service: 'Expert Food Detection Service',
    status: 'operational',
    capabilities: {
      expert_analysis: true,
      multi_model_ensemble: true,
      model_specific_analysis: true,
      batch_processing: true,
      supported_formats: config.allowedMimeTypes,
      max_file_size: config.maxFileSize,
      max_batch_size: 10,
      python_integration: healthCheck.pythonAvailable
    },
    models: healthCheck.models,
    configuration: {
      confidence_threshold: config.detection.confidence_threshold,
      ensemble_threshold: config.detection.ensemble_threshold,
      max_detection_time: config.detection.max_detection_time,
      fallback_enabled: config.detection.fallback_enabled
    }
  });
}));

// GROQ analysis endpoint
router.post('/groq', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { detectedFoods, nutritionalData, foodItems, imageDescription, mealContext } = req.body;

    if (!detectedFoods || !Array.isArray(detectedFoods)) {
      return res.status(400).json({
        success: false,
        error: 'detectedFoods array is required'
      });
    }

    if (!nutritionalData) {
      return res.status(400).json({
        success: false,
        error: 'nutritionalData is required'
      });
    }

    console.log('Starting GROQ analysis...');
    
    const result = await groqAnalysisService.generateComprehensiveAnalysis({
      detectedFoods,
      nutritionalData,
      foodItems: foodItems || [],
      imageDescription,
      mealContext
    });

    console.log('GROQ analysis completed:', {
      success: result.success,
      healthScore: result.healthScore,
      recommendationsCount: result.recommendations.length
    });

    return res.json(result);
  } catch (error) {
    console.error('GROQ analysis error:', error);
    return res.status(500).json({
      success: false,
      error: 'GROQ analysis failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}));

// GROQ health check endpoint
router.get('/groq/health', asyncHandler(async (_req: Request, res: Response) => {
  const healthCheck = await groqAnalysisService.healthCheck();
  
  return res.json({
    service: 'GROQ Analysis Service',
    status: healthCheck.status,
    available: healthCheck.available,
    error: healthCheck.error
  });
}));

// Diet Chat endpoints
router.post('/diet-chat', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { question, context, userHistory } = req.body;

    if (!question || typeof question !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'Question is required and must be a string'
      });
    }

    console.log('Starting diet chat query...');
    
    const result = await dietChatService.answerDietQuery({
      question,
      context,
      userHistory
    });

    console.log('Diet chat completed:', {
      confidence: result.confidence,
      suggestionsCount: result.suggestions.length,
      topicsCount: result.relatedTopics.length
    });

    return res.json({
      success: true,
      ...result
    });
  } catch (error) {
    console.error('Diet chat error:', error);
    return res.status(500).json({
      success: false,
      error: 'Diet chat failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}));

// Diet chat health check endpoint
router.get('/diet-chat/health', asyncHandler(async (_req: Request, res: Response) => {
  const healthCheck = await dietChatService.healthCheck();
  
  return res.json({
    service: 'Diet Chat Service',
    status: healthCheck.status,
    available: healthCheck.available,
    error: healthCheck.error
  });
}));

// Get sample questions for diet chat
router.get('/diet-chat/sample-questions', asyncHandler(async (_req: Request, res: Response) => {
  const sampleQuestions = dietChatService.getSampleQuestions();
  
  return res.json({
    success: true,
    questions: sampleQuestions
  });
}));

// Generate diet plan from detected foods
router.post('/generate-diet-plan', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { detectedFoods, nutritionalData, userPreferences: _userPreferences } = req.body;

    if (!detectedFoods || !Array.isArray(detectedFoods) || detectedFoods.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Detected foods array is required'
      });
    }

    console.log('Generating diet plan for foods:', detectedFoods);

    // Generate diet plan using GROQ analysis service
    const result = await groqAnalysisService.generateComprehensiveAnalysis({
      detectedFoods: detectedFoods.map(food => typeof food === 'string' ? food : food.name),
      nutritionalData: nutritionalData || {},
      foodItems: [],
      imageDescription: `Meal containing: ${detectedFoods.map(food => typeof food === 'string' ? food : food.name).join(', ')}`,
      mealContext: 'Diet plan generation from detected foods'
    });

    const dietPlan = result.dailyMealPlan;

    console.log('Generated diet plan structure:', dietPlan);

    return res.json({
      success: true,
      dietPlan
    });
  } catch (error) {
    console.error('Diet plan generation error:', error);
    return res.status(500).json({
      success: false,
      error: 'Diet plan generation failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}));

export default router;