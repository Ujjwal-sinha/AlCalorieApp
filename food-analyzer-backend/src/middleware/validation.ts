import { Request, Response, NextFunction } from 'express';
import Joi from 'joi';

// Validation schemas
const analysisRequestSchema = Joi.object({
  context: Joi.string().optional().max(500),
  confidence_threshold: Joi.number().min(0).max(1).optional(),
  ensemble_threshold: Joi.number().min(0).max(1).optional(),
  use_advanced_detection: Joi.string().valid('true', 'false').optional(),
  model_type: Joi.string().valid('yolo', 'vit', 'swin', 'blip', 'clip', 'lightweight', 'robust', 'simple').optional()
});

const nutritionCalculateSchema = Joi.object({
  foods: Joi.array().items(Joi.string().min(1).max(50)).min(1).max(20).required()
});

const nutritionCompareSchema = Joi.object({
  foods: Joi.array().items(Joi.string().min(1).max(50)).min(2).max(10).required()
});

const dailyRecommendationsSchema = Joi.object({
  age: Joi.number().integer().min(1).max(120).optional(),
  gender: Joi.string().valid('male', 'female').optional(),
  weight: Joi.number().min(20).max(300).optional(),
  height: Joi.number().min(100).max(250).optional(),
  activity_level: Joi.string().valid('sedentary', 'light', 'moderate', 'active', 'very_active').optional()
});

// Validation middleware factory
const createValidationMiddleware = (schema: Joi.ObjectSchema, source: 'body' | 'query' | 'params' = 'body') => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const data = req[source];
    const { error, value } = schema.validate(data, {
      abortEarly: false,
      stripUnknown: true,
      convert: true
    });

    if (error) {
      const errorMessages = error.details.map(detail => detail.message);
      res.status(400).json({
        success: false,
        error: 'Validation failed',
        details: errorMessages
      });
      return;
    }

    // Replace the original data with validated and sanitized data
    req[source] = value;
    next();
  };
};

// Specific validation middleware
export const validateAnalysisRequest = createValidationMiddleware(analysisRequestSchema, 'body');
export const validateNutritionCalculate = createValidationMiddleware(nutritionCalculateSchema, 'body');
export const validateNutritionCompare = createValidationMiddleware(nutritionCompareSchema, 'body');
export const validateDailyRecommendations = createValidationMiddleware(dailyRecommendationsSchema, 'query');

// File validation middleware
export const validateImageFile = (req: Request, res: Response, next: NextFunction) => {
  if (!req.file) {
    return res.status(400).json({
      success: false,
      error: 'No image file provided'
    });
  }

  const allowedMimeTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  if (!allowedMimeTypes.includes(req.file.mimetype)) {
    return res.status(400).json({
      success: false,
      error: `Invalid file type. Allowed types: ${allowedMimeTypes.join(', ')}`
    });
  }

  const maxSize = 10 * 1024 * 1024; // 10MB
  if (req.file.size > maxSize) {
    return res.status(400).json({
      success: false,
      error: `File too large. Maximum size: ${maxSize / (1024 * 1024)}MB`
    });
  }

  next();
};

// Rate limiting validation
export const validateRateLimit = (maxRequests: number, windowMs: number) => {
  const requests = new Map<string, { count: number; resetTime: number }>();

  return (req: Request, res: Response, next: NextFunction) => {
    const clientId = req.ip || 'unknown';
    const now = Date.now();
    const windowStart = now - windowMs;

    // Clean up old entries
    for (const [id, data] of requests.entries()) {
      if (data.resetTime < windowStart) {
        requests.delete(id);
      }
    }

    // Check current client
    const clientData = requests.get(clientId);
    if (!clientData) {
      requests.set(clientId, { count: 1, resetTime: now + windowMs });
      next();
    } else if (clientData.count < maxRequests) {
      clientData.count++;
      next();
    } else {
      res.status(429).json({
        success: false,
        error: 'Too many requests',
        retry_after: Math.ceil((clientData.resetTime - now) / 1000)
      });
    }
  };
};

// Custom validation helpers
export const validateObjectId = (id: string): boolean => {
  return /^[0-9a-fA-F]{24}$/.test(id);
};

export const sanitizeString = (str: string): string => {
  return str.trim().replace(/[<>]/g, '');
};

export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};