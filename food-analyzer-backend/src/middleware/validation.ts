import { Request, Response, NextFunction } from 'express';
import Joi from 'joi';

// Validation schemas
const analysisRequestSchema = Joi.object({
  context: Joi.string().optional(),
  confidence_threshold: Joi.number().min(0).max(1).optional(),
  ensemble_threshold: Joi.number().min(0).max(1).optional(),
  use_advanced_detection: Joi.boolean().optional()
});

const nutritionRequestSchema = Joi.object({
  foods: Joi.array().items(Joi.string()).min(1).required()
});

const nutritionComparisonSchema = Joi.object({
  foods: Joi.array().items(Joi.string()).min(2).max(10).required()
});

const balanceAnalysisSchema = Joi.object({
  foods: Joi.array().items(Joi.string()).min(1).required()
});

// Validation middleware
export const validateAnalysisRequest = (req: Request, res: Response, next: NextFunction) => {
  const { error } = analysisRequestSchema.validate(req.body);
  
  if (error) {
    res.status(400).json({
      success: false,
      error: 'Validation error',
      details: error.details.map(detail => detail.message)
    });
    return;
  }
  
  next();
};

export const validateNutritionRequest = (req: Request, res: Response, next: NextFunction) => {
  const { error } = nutritionRequestSchema.validate(req.body);
  
  if (error) {
    res.status(400).json({
      success: false,
      error: 'Validation error',
      details: error.details.map(detail => detail.message)
    });
    return;
  }
  
  next();
};

export const validateNutritionComparison = (req: Request, res: Response, next: NextFunction) => {
  const { error } = nutritionComparisonSchema.validate(req.body);
  
  if (error) {
    res.status(400).json({
      success: false,
      error: 'Validation error',
      details: error.details.map(detail => detail.message)
    });
    return;
  }
  
  next();
};

export const validateBalanceAnalysis = (req: Request, res: Response, next: NextFunction) => {
  const { error } = balanceAnalysisSchema.validate(req.body);
  
  if (error) {
    res.status(400).json({
      success: false,
      error: 'Validation error',
      details: error.details.map(detail => detail.message)
    });
    return;
  }
  
  next();
};

// File validation middleware
export const validateImageFile = (req: Request, res: Response, next: NextFunction) => {
  if (!req.file) {
    res.status(400).json({
      success: false,
      error: 'No image file provided'
    });
    return;
  }

  const allowedMimeTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  
  if (!allowedMimeTypes.includes(req.file.mimetype)) {
    res.status(400).json({
      success: false,
      error: 'Invalid file type. Only JPEG, PNG, and WebP images are allowed'
    });
    return;
  }

  const maxSize = 10 * 1024 * 1024; // 10MB
  
  if (req.file.size > maxSize) {
    res.status(400).json({
      success: false,
      error: 'File size too large. Maximum size is 10MB'
    });
    return;
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