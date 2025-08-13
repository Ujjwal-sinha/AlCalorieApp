import { Router, Request, Response } from 'express';
import multer from 'multer';
import { FoodDetectionService } from '../services/FoodDetectionService';
import { validateAnalysisRequest } from '../middleware/validation';
import { asyncHandler } from '../middleware/asyncHandler';
import { config } from '../config';

const router = Router();
const foodDetectionService = FoodDetectionService.getInstance();

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
          const processedImage = await foodDetectionService.processImage(file.buffer);
          
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

export default router;