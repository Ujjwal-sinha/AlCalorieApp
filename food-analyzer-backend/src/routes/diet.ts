import { Router, Request, Response } from 'express';
import { DietChatService } from '../services/DietChatService';
import { asyncHandler } from '../middleware/asyncHandler';

const router = Router();
const dietChatService = DietChatService.getInstance();

// Health check endpoint
router.get('/health', asyncHandler(async (_req: Request, res: Response) => {
  const health = await dietChatService.healthCheck();
  res.json(health);
}));

// Get sample questions
router.get('/sample-questions', asyncHandler(async (_req: Request, res: Response) => {
  const questions = await dietChatService.getSampleQuestions();
  res.json({ questions });
}));

// Answer diet query
router.post('/query', asyncHandler(async (req: Request, res: Response) => {
  const { question, context, userHistory } = req.body;
  
  if (!question) {
    return res.status(400).json({ 
      error: 'Question is required' 
    });
  }

  const response = await dietChatService.answerDietQuery({
    question,
    context,
    userHistory
  });

  return res.json(response);
}));

export default router;
