import { Router } from 'express';
import analysisRoutes from './analysis';
import modelsRoutes from './models';
import foodRoutes from './food';
import nutritionRoutes from './nutrition';

export function setupRoutes(): Router {
  const router = Router();

  // API version info
  router.get('/', (req, res) => {
    res.json({
      name: 'Food Analyzer API',
      version: '1.0.0',
      description: 'AI-powered food analysis and nutrition tracking',
      endpoints: {
        analysis: '/analyze',
        models: '/models',
        food: '/food',
        nutrition: '/nutrition'
      },
      documentation: '/docs'
    });
  });

  // Mount route modules
  router.use('/analyze', analysisRoutes);
  router.use('/models', modelsRoutes);
  router.use('/food', foodRoutes);
  router.use('/nutrition', nutritionRoutes);

  return router;
}