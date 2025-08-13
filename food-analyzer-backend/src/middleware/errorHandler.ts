import { Request, Response, NextFunction } from 'express';

export const errorHandler = (
  error: Error,
  _req: Request,
  res: Response,
  _next: NextFunction
): void => {
  console.error('Error:', error);

  // Default error response
  const errorResponse = {
    success: false,
    error: error.message || 'Internal server error',
    timestamp: new Date().toISOString()
  };

  // Handle specific error types
  if (error.name === 'ValidationError') {
    res.status(400).json({
      ...errorResponse,
      error: 'Validation error',
      details: error.message
    });
    return;
  }

  if (error.name === 'UnauthorizedError') {
    res.status(401).json({
      ...errorResponse,
      error: 'Unauthorized'
    });
    return;
  }

  if (error.name === 'NotFoundError') {
    res.status(404).json({
      ...errorResponse,
      error: 'Resource not found'
    });
    return;
  }

  // Generic server error
  res.status(500).json(errorResponse);
};

export const notFoundHandler = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  const error: Error = new Error(`Route ${req.method} ${req.originalUrl} not found`);
  error.name = 'NotFoundError';
  next(error);
};

export const createError = (message: string, statusCode: number = 500): Error => {
  const error: Error = new Error(message);
  error.name = 'ValidationError'; // Assuming a common error name for validation
  return error;
};