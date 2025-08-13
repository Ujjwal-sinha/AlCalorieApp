import { Router, Request, Response } from 'express';
import multer from 'multer';
import { FoodDetectionService } from '../services/FoodDetectionService';
import { validateAnalysisRequest } from '../middleware/validation';
import { asyncHandler } from '../middleware/asyncHandler';
import { config } from '../config';
import type { AnalysisRequest } from '../types';

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

// Advanced analysis endpoint
router.post('/advanced', 
  upload.single('image'),
  validateAnalysisRequest,
  asyncHandler(async (req: Request, res: Response) => {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No image file provided'
      });
    }

    try {
      // Process the uploaded image
      const processedImage = await foodDetectionService.processImage(req.file.buffer);

      // Create analysis request
      const analysisRequest: AnalysisRequest = {
        image: processedImage,
        context: req.body.context,
        confidence_threshold: parseFloat(req.body.confidence_threshold) || config.detection.confidence_threshold,
        ensemble_threshold: parseFloat(req.body.ensemble_threshold) || config.detection.ensemble_threshold,
        use_advanced_detection: req.body.use_advanced_detection === 'true'
      };

      // Perform analysis
      const result = await foodDetectionService.analyzeWithAdvancedDetection(analysisRequest);

      res.json(result);
    } catch (error) {
      console.error('Advanced analysis failed:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Analysis failed'
      });
    }
  })
);

// Model-specific analysis endpoints
router.post('/model/:modelType',
  upload.single('image'),
  validateAnalysisRequest,
  asyncHandler(async (req: Request, res: Response) => {
    const { modelType } = req.params;
    
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No image file provided'
      });
    }

    if (!modelType) {
      return res.status(400).json({
        success: false,
        error: 'Model type is required'
      });
    }

    const validModels = ['yolo', 'vit', 'swin', 'blip', 'clip', 'lightweight', 'robust', 'simple'];
    if (!validModels.includes(modelType)) {
      return res.status(400).json({
        success: false,
        error: `Invalid model type. Valid types: ${validModels.join(', ')}`
      });
    }

    try {
      // Process the uploaded image
      const processedImage = await foodDetectionService.processImage(req.file.buffer);

      // Create analysis request for specific model
      const analysisRequest: AnalysisRequest = {
        image: processedImage,
        context: req.body.context,
        model_type: modelType,
        confidence_threshold: parseFloat(req.body.confidence_threshold) || config.detection.confidence_threshold
      };

      // For now, use advanced detection (in real implementation, you'd have model-specific methods)
      const result = await foodDetectionService.analyzeWithAdvancedDetection(analysisRequest);

      res.json({
        ...result,
        model_used: modelType
      });
    } catch (error) {
      console.error(`${modelType} analysis failed:`, error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : `${modelType} analysis failed`
      });
    }
  })
);

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
          
          const analysisRequest: AnalysisRequest = {
            image: processedImage,
            context: req.body.context,
            confidence_threshold: parseFloat(req.body.confidence_threshold) || config.detection.confidence_threshold,
            ensemble_threshold: parseFloat(req.body.ensemble_threshold) || config.detection.ensemble_threshold,
            use_advanced_detection: req.body.use_advanced_detection === 'true'
          };

          return {
            index,
            filename: file.originalname,
            result: await foodDetectionService.analyzeWithAdvancedDetection(analysisRequest)
          };
        })
      );

      const successful = results
        .filter((result): result is PromiseFulfilledResult<any> => result.status === 'fulfilled')
        .map(result => result.value);

      const failed = results
        .map((result, index) => result.status === 'rejected' ? { index, error: result.reason } : null)
        .filter(Boolean);

      res.json({
        success: true,
        total: files.length,
        successful: successful.length,
        failed: failed.length,
        results: successful,
        errors: failed
      });
    } catch (error) {
      console.error('Batch analysis failed:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Batch analysis failed'
      });
    }
  })
);

// Analysis status endpoint
router.get('/status', asyncHandler(async (_req: Request, res: Response) => {
  res.json({
    service: 'Food Detection Service',
    status: 'operational',
    capabilities: {
      advanced_detection: true,
      model_specific_analysis: true,
      batch_processing: true,
      supported_formats: config.allowedMimeTypes,
      max_file_size: config.maxFileSize,
      max_batch_size: 10
    },
    configuration: {
      confidence_threshold: config.detection.confidence_threshold,
      ensemble_threshold: config.detection.ensemble_threshold,
      max_detection_time: config.detection.max_detection_time,
      fallback_enabled: config.detection.fallback_enabled
    }
  });
}));

export default router;