import { Router, Request, Response } from 'express';
import analysisRoutes from './analysis';
import foodRoutes from './food';
import nutritionRoutes from './nutrition';
import modelsRoutes from './models';
import dietRoutes from './diet';

const router = Router();

// Main API endpoint
router.get('/', (_req: Request, res: Response) => {
  return res.json({
    message: 'Food Analyzer API',
    version: '1.0.0',
    endpoints: {
      analysis: '/api/analysis',
      food: '/api/food',
      nutrition: '/api/nutrition',
      models: '/api/models',
      diet: '/api/diet'
    },
    documentation: '/api/docs'
  });
});

// Debug endpoint to test Python integration
router.get('/debug/python', async (_req, res) => {
  try {
    const { spawn } = require('child_process');
    const pythonProcess = spawn('python3', ['--version']);
    
    pythonProcess.stdout.on('data', (data: Buffer) => {
      res.json({ 
        success: true, 
        python_version: data.toString().trim(),
        message: 'Python is available'
      });
    });
    
    pythonProcess.stderr.on('data', (data: Buffer) => {
      res.status(500).json({ 
        success: false, 
        error: data.toString(),
        message: 'Python not available'
      });
    });
  } catch (error) {
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Mount route modules
router.use('/analysis', analysisRoutes);
router.use('/food', foodRoutes);
router.use('/nutrition', nutritionRoutes);
router.use('/models', modelsRoutes);
router.use('/diet', dietRoutes);

export default router;