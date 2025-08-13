import { Router, Request, Response } from 'express';
import { NutritionService } from '../services/NutritionService';
import { asyncHandler } from '../middleware/asyncHandler';

const router = Router();
const nutritionService = NutritionService.getInstance();

// Calculate nutrition for food items
router.post('/calculate', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { foods } = req.body;

    if (!foods || !Array.isArray(foods)) {
      return res.status(400).json({
        success: false,
        error: 'Foods array is required'
      });
    }

    const nutrition = await nutritionService.calculateNutrition(foods);

    return res.json({
      success: true,
      nutrition: nutrition,
      total_foods: foods.length
    });
  } catch (error) {
    console.error('Nutrition calculation failed:', error);
    return res.status(500).json({
      success: false,
      error: 'Nutrition calculation failed'
    });
  }
}));

// Get nutrition for specific food
router.get('/:foodName', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { foodName } = req.params;

    if (!foodName) {
      return res.status(400).json({
        success: false,
        error: 'Food name is required'
      });
    }

    const details = await nutritionService.getFoodDetails(foodName);

    if (!details) {
      return res.status(404).json({
        success: false,
        error: `Food "${foodName}" not found`
      });
    }

    return res.json({
      success: true,
      food: details
    });
  } catch (error) {
    console.error('Failed to get food nutrition:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve food nutrition'
    });
  }
}));

// Compare nutrition between foods
router.post('/compare', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { foods } = req.body;

    if (!foods || !Array.isArray(foods) || foods.length < 2) {
      return res.status(400).json({
        success: false,
        error: 'At least 2 foods are required for comparison'
      });
    }

    const nutritionResults = await Promise.all(
      foods.map(async (food: string) => {
        const nutrition = await nutritionService.calculateNutrition([food]);
        return {
          food,
          nutrition: nutrition
        };
      })
    );

    // Calculate totals for comparison
    const totalNutrition = nutritionResults.reduce(
      (acc, result) => ({
        total_calories: acc.total_calories + result.nutrition.total_calories,
        total_protein: acc.total_protein + result.nutrition.total_protein,
        total_carbs: acc.total_carbs + result.nutrition.total_carbs,
        total_fats: acc.total_fats + result.nutrition.total_fats
      }),
      { total_calories: 0, total_protein: 0, total_carbs: 0, total_fats: 0 }
    );

    return res.json({
      success: true,
      comparison: {
        foods: nutritionResults,
        totals: totalNutrition,
        analysis: generateComparisonAnalysis(nutritionResults)
      }
    });
  } catch (error) {
    console.error('Nutrition comparison failed:', error);
    return res.status(500).json({
      success: false,
      error: 'Nutrition comparison failed'
    });
  }
}));

// Get daily nutrition recommendations
router.get('/recommendations/daily', asyncHandler(async (_req: Request, res: Response) => {
  try {
    const recommendations = {
      calories: {
        male: { sedentary: 2000, moderate: 2400, active: 2800 },
        female: { sedentary: 1600, moderate: 2000, active: 2400 }
      },
      protein: {
        grams: { minimum: 50, recommended: 80, maximum: 120 },
        percentage: 15
      },
      carbs: {
        grams: { minimum: 130, recommended: 250, maximum: 325 },
        percentage: 45
      },
      fats: {
        grams: { minimum: 44, recommended: 65, maximum: 78 },
        percentage: 25
      },
      fiber: {
        grams: { minimum: 25, recommended: 30, maximum: 35 }
      },
      vitamins: {
        vitamin_c: { mg: 90, source: 'Citrus fruits, vegetables' },
        vitamin_d: { mcg: 15, source: 'Sunlight, fatty fish' },
        vitamin_b12: { mcg: 2.4, source: 'Animal products' }
      },
      minerals: {
        calcium: { mg: 1000, source: 'Dairy, leafy greens' },
        iron: { mg: 18, source: 'Red meat, beans' },
        potassium: { mg: 3500, source: 'Bananas, potatoes' }
      }
    };

    return res.json({
      success: true,
      recommendations: recommendations,
      notes: [
        'Recommendations vary based on age, weight, activity level, and health goals',
        'Consult with a healthcare provider for personalized nutrition advice',
        'These are general guidelines and may need adjustment for specific dietary needs'
      ]
    });
  } catch (error) {
    console.error('Failed to get nutrition recommendations:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve nutrition recommendations'
    });
  }
}));

// Analyze nutritional balance
router.post('/analyze/balance', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { foods } = req.body;

    if (!foods || !Array.isArray(foods)) {
      return res.status(400).json({
        success: false,
        error: 'Foods array is required'
      });
    }

    const nutrition = await nutritionService.calculateNutrition(foods);
    const balance = analyzeNutritionalBalance(nutrition);

    return res.json({
      success: true,
      nutrition: nutrition,
      balance: balance,
      recommendations: generateBalanceRecommendations(balance)
    });
  } catch (error) {
    console.error('Nutritional balance analysis failed:', error);
    return res.status(500).json({
      success: false,
      error: 'Nutritional balance analysis failed'
    });
  }
}));

// Helper functions
function generateComparisonAnalysis(results: any[]): string[] {
  const analysis: string[] = [];
  
  if (results.length === 0) return analysis;

  const totalCalories = results.reduce((sum, result) => sum + result.nutrition.total_calories, 0);
  const totalProtein = results.reduce((sum, result) => sum + result.nutrition.total_protein, 0);
  const totalCarbs = results.reduce((sum, result) => sum + result.nutrition.total_carbs, 0);
  const totalFats = results.reduce((sum, result) => sum + result.nutrition.total_fats, 0);

  analysis.push(`Total calories: ${totalCalories} kcal`);
  analysis.push(`Total protein: ${totalProtein}g (${Math.round((totalProtein * 4 / totalCalories) * 100)}% of calories)`);
  analysis.push(`Total carbs: ${totalCarbs}g (${Math.round((totalCarbs * 4 / totalCalories) * 100)}% of calories)`);
  analysis.push(`Total fats: ${totalFats}g (${Math.round((totalFats * 9 / totalCalories) * 100)}% of calories)`);

  return analysis;
}

function analyzeNutritionalBalance(nutrition: any): any {
  const totalCalories = nutrition.total_calories;
  const proteinPercentage = (nutrition.total_protein * 4 / totalCalories) * 100;
  const carbsPercentage = (nutrition.total_carbs * 4 / totalCalories) * 100;
  const fatsPercentage = (nutrition.total_fats * 9 / totalCalories) * 100;

  const balance = {
    score: 0,
    protein_balance: 'balanced',
    carbs_balance: 'balanced',
    fats_balance: 'balanced',
    overall_balance: 'balanced'
  };

  // Analyze protein balance
  if (proteinPercentage < 10) balance.protein_balance = 'low';
  else if (proteinPercentage > 35) balance.protein_balance = 'high';
  else balance.protein_balance = 'balanced';

  // Analyze carbs balance
  if (carbsPercentage < 45) balance.carbs_balance = 'low';
  else if (carbsPercentage > 65) balance.carbs_balance = 'high';
  else balance.carbs_balance = 'balanced';

  // Analyze fats balance
  if (fatsPercentage < 20) balance.fats_balance = 'low';
  else if (fatsPercentage > 35) balance.fats_balance = 'high';
  else balance.fats_balance = 'balanced';

  // Calculate overall balance score
  const balancedCount = [balance.protein_balance, balance.carbs_balance, balance.fats_balance]
    .filter(b => b === 'balanced').length;
  balance.score = (balancedCount / 3) * 100;

  if (balance.score >= 80) balance.overall_balance = 'excellent';
  else if (balance.score >= 60) balance.overall_balance = 'good';
  else if (balance.score >= 40) balance.overall_balance = 'fair';
  else balance.overall_balance = 'poor';

  return balance;
}

function generateBalanceRecommendations(balance: any): string[] {
  const recommendations: string[] = [];

  if (balance.protein_balance === 'low') {
    recommendations.push('Consider adding more protein-rich foods like lean meats, fish, eggs, or legumes');
  } else if (balance.protein_balance === 'high') {
    recommendations.push('Consider reducing protein intake and increasing carbohydrates or healthy fats');
  }

  if (balance.carbs_balance === 'low') {
    recommendations.push('Consider adding more complex carbohydrates like whole grains, fruits, and vegetables');
  } else if (balance.carbs_balance === 'high') {
    recommendations.push('Consider reducing carbohydrate intake and increasing protein or healthy fats');
  }

  if (balance.fats_balance === 'low') {
    recommendations.push('Consider adding healthy fats like nuts, avocados, or olive oil');
  } else if (balance.fats_balance === 'high') {
    recommendations.push('Consider reducing fat intake and increasing protein or complex carbohydrates');
  }

  if (balance.overall_balance === 'excellent') {
    recommendations.push('Excellent nutritional balance! Maintain this variety in your diet.');
  } else if (balance.overall_balance === 'good') {
    recommendations.push('Good nutritional balance. Minor adjustments could optimize your diet further.');
  } else {
    recommendations.push('Consider consulting with a nutritionist for personalized dietary recommendations.');
  }

  return recommendations;
}

export default router;