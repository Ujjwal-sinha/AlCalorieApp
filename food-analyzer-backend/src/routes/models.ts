import { Router, Request, Response } from 'express';
import { ModelManager } from '../services/ModelManager';
import { asyncHandler } from '../middleware/asyncHandler';

const router = Router();
const modelManager = ModelManager.getInstance();

// Get model status
router.get('/status', asyncHandler(async (req: Request, res: Response) => {
  const status = modelManager.getModelStatus();
  const models = modelManager.getAllModels();

  const detailedStatus = Array.from(models.entries()).map(([name, model]) => ({
    name,
    type: model.type,
    loaded: model.loaded,
    config: model.config
  }));

  res.json({
    summary: status,
    models: detailedStatus,
    available_models: modelManager.getAvailableModels(),
    total_models: models.size,
    loaded_models: Object.values(status).filter(Boolean).length
  });
}));

// Get specific model info
router.get('/:modelName', asyncHandler(async (req: Request, res: Response) => {
  const { modelName } = req.params;

  if (!modelName) {
    return res.status(400).json({
      success: false,
      error: 'Model name is required'
    });
  }

  const model = modelManager.getModel(modelName);

  if (!model) {
    return res.status(404).json({
      success: false,
      error: `Model '${modelName}' not found`
    });
  }

  res.json({
    name: model.name,
    type: model.type,
    loaded: model.loaded,
    config: model.config,
    capabilities: getModelCapabilities(model.name)
  });
}));

// Reload specific model
router.post('/:modelName/reload', asyncHandler(async (req: Request, res: Response) => {
  const { modelName } = req.params;

  if (!modelName) {
    return res.status(400).json({
      success: false,
      error: 'Model name is required'
    });
  }

  try {
    const success = await modelManager.reloadModel(modelName);

    if (success) {
      res.json({
        success: true,
        message: `Model '${modelName}' reloaded successfully`,
        status: modelManager.isModelLoaded(modelName)
      });
    } else {
      res.status(500).json({
        success: false,
        error: `Failed to reload model '${modelName}'`
      });
    }
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Reload failed'
    });
  }
}));

// Get model capabilities
router.get('/:modelName/capabilities', asyncHandler(async (req: Request, res: Response) => {
  const { modelName } = req.params;

  if (!modelName) {
    return res.status(400).json({
      success: false,
      error: 'Model name is required'
    });
  }

  const model = modelManager.getModel(modelName);

  if (!model) {
    return res.status(404).json({
      success: false,
      error: `Model '${modelName}' not found`
    });
  }

  const capabilities = getModelCapabilities(modelName);

  res.json({
    model: modelName,
    capabilities,
    loaded: model.loaded,
    type: model.type
  });
}));

// Initialize all models
router.post('/initialize', asyncHandler(async (req: Request, res: Response) => {
  try {
    await modelManager.initialize();

    res.json({
      success: true,
      message: 'All models initialized',
      status: modelManager.getModelStatus()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Initialization failed'
    });
  }
}));

// Get available model types
router.get('/types/available', asyncHandler(async (req: Request, res: Response) => {
  const models = modelManager.getAllModels();
  const types = new Set<string>();

  for (const model of models.values()) {
    types.add(model.type);
  }

  res.json({
    types: Array.from(types),
    models_by_type: {
      vision: Array.from(models.values()).filter(m => m.type === 'vision').map(m => m.name),
      detection: Array.from(models.values()).filter(m => m.type === 'detection').map(m => m.name),
      language: Array.from(models.values()).filter(m => m.type === 'language').map(m => m.name)
    }
  });
}));

// Health check for models
router.get('/health/check', asyncHandler(async (req: Request, res: Response) => {
  const models = modelManager.getAllModels();
  const health = {
    overall: 'healthy',
    models: {} as Record<string, string>,
    issues: [] as string[]
  };

  for (const [name, model] of models) {
    if (model.loaded) {
      health.models[name] = 'healthy';
    } else {
      health.models[name] = 'unhealthy';
      health.issues.push(`Model '${name}' is not loaded`);
    }
  }

  if (health.issues.length > 0) {
    health.overall = 'degraded';
  }

  const statusCode = health.overall === 'healthy' ? 200 : 503;
  res.status(statusCode).json(health);
}));

function getModelCapabilities(modelName: string): Record<string, any> {
  const capabilities: Record<string, Record<string, any>> = {
    vit: {
      type: 'Vision Transformer',
      tasks: ['image_classification', 'food_detection'],
      input_formats: ['jpeg', 'png', 'webp'],
      max_resolution: '1024x1024',
      confidence_threshold: 0.05,
      batch_processing: true
    },
    swin: {
      type: 'Swin Transformer',
      tasks: ['image_classification', 'food_detection'],
      input_formats: ['jpeg', 'png', 'webp'],
      max_resolution: '1024x1024',
      confidence_threshold: 0.05,
      batch_processing: true
    },
    blip: {
      type: 'BLIP Image Captioning',
      tasks: ['image_captioning', 'food_description'],
      input_formats: ['jpeg', 'png', 'webp'],
      max_resolution: '1024x1024',
      max_tokens: 50,
      batch_processing: false
    },
    clip: {
      type: 'CLIP Vision-Language',
      tasks: ['image_text_matching', 'food_similarity'],
      input_formats: ['jpeg', 'png', 'webp'],
      max_resolution: '1024x1024',
      similarity_threshold: 0.28,
      batch_processing: true
    },
    yolo: {
      type: 'YOLO Object Detection',
      tasks: ['object_detection', 'food_localization'],
      input_formats: ['jpeg', 'png', 'webp'],
      max_resolution: '1024x1024',
      confidence_levels: [0.15, 0.25, 0.35, 0.45],
      batch_processing: true
    },
    llm: {
      type: 'Language Model',
      tasks: ['text_generation', 'nutrition_analysis'],
      max_tokens: 1000,
      temperature: 0.7,
      batch_processing: false
    }
  };

  return capabilities[modelName] || {
    type: 'Unknown',
    tasks: [],
    batch_processing: false
  };
}

export default router;