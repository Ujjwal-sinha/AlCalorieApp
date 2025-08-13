import { Router, Request, Response } from 'express';
import { ModelManager } from '../services/ModelManager';
import { asyncHandler } from '../middleware/asyncHandler';

const router = Router();
const modelManager = ModelManager.getInstance();

// Get all models status
router.get('/status', asyncHandler(async (_req: Request, res: Response) => {
  try {
    const status = modelManager.getModelStatus();
    
    return res.json({
      success: true,
      models: status,
      total_models: Object.keys(status).length,
      loaded_models: Object.values(status).filter(Boolean).length
    });
  } catch (error) {
    console.error('Failed to get model status:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve model status'
    });
  }
}));

// Get specific model details
router.get('/:modelName', asyncHandler(async (req: Request, res: Response) => {
  try {
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
        error: `Model "${modelName}" not found`
      });
    }

    return res.json({
      success: true,
      model: {
        name: model.name,
        type: model.type,
        loaded: model.loaded,
        config: model.config
      }
    });
  } catch (error) {
    console.error('Failed to get model details:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve model details'
    });
  }
}));

// Reload specific model
router.post('/:modelName/reload', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { modelName } = req.params;
    
    if (!modelName) {
      return res.status(400).json({
        success: false,
        error: 'Model name is required'
      });
    }

    const success = await modelManager.reloadModel(modelName);
    
    if (!success) {
      return res.status(500).json({
        success: false,
        error: `Failed to reload model "${modelName}"`
      });
    }

    return res.json({
      success: true,
      message: `Model "${modelName}" reloaded successfully`,
      status: modelManager.isModelLoaded(modelName)
    });
  } catch (error) {
    console.error('Failed to reload model:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to reload model'
    });
  }
}));

// Get model capabilities
router.get('/:modelName/capabilities', asyncHandler(async (req: Request, res: Response) => {
  try {
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
        error: `Model "${modelName}" not found`
      });
    }

    const capabilities = {
      name: model.name,
      type: model.type,
      loaded: model.loaded,
      features: getModelCapabilities(model.type),
      config: model.config
    };

    return res.json({
      success: true,
      capabilities: capabilities
    });
  } catch (error) {
    console.error('Failed to get model capabilities:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve model capabilities'
    });
  }
}));

// Initialize all models
router.post('/initialize', asyncHandler(async (_req: Request, res: Response) => {
  try {
    await modelManager.initialize();
    
    return res.json({
      success: true,
      message: 'All models initialized successfully',
      status: modelManager.getModelStatus()
    });
  } catch (error) {
    console.error('Failed to initialize models:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to initialize models'
    });
  }
}));

// Get available model types
router.get('/types/available', asyncHandler(async (_req: Request, res: Response) => {
  try {
    const models = modelManager.getLoadedModels();
    const modelTypes = {
      vision: ['vit', 'swin', 'clip', 'blip', 'cnn'],
      detection: ['yolo'],
      language: ['llm']
    };

    return res.json({
      success: true,
      model_types: modelTypes,
      loaded_models: models
    });
  } catch (error) {
    console.error('Failed to get model types:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve model types'
    });
  }
}));

// Health check for models
router.get('/health/check', asyncHandler(async (_req: Request, res: Response) => {
  try {
    const health = await modelManager.healthCheck();
    
    return res.json({
      success: true,
      health: health
    });
  } catch (error) {
    console.error('Model health check failed:', error);
    return res.status(500).json({
      success: false,
      error: 'Model health check failed'
    });
  }
}));

// Helper function to get model capabilities
function getModelCapabilities(modelType: string): string[] {
  switch (modelType) {
    case 'vision':
      return ['image_classification', 'feature_extraction', 'visual_analysis'];
    case 'detection':
      return ['object_detection', 'bounding_box_prediction', 'confidence_scoring'];
    case 'language':
      return ['text_generation', 'analysis', 'insights'];
    default:
      return ['basic_processing'];
  }
}

export default router;