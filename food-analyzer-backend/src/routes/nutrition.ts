import { Router, Request, Response } from 'express';
import { NutritionService } from '../services/NutritionService';
import { asyncHandler } from '../middleware/asyncHandler';

const router = Router();
const nutritionService = NutritionService.getInstance();

// Calculate nutrition for a list of foods
router.post('/calculate', asyncHandler(async (req: Request, res: Response) => {
  const { foods } = req.body;

  if (!foods || !Array.isArray(foods) || foods.length === 0) {
    return res.status(400).json({
      success: false,
      error: 'Foods array is required and must not be empty'
    });
  }

  try {
    const nutritionalData = await nutritionService.calculateNutrition(foods);

    res.json({
      success: true,
      foods,
      nutritional_data: nutritionalData,
      calculated_at: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Nutrition calculation failed'
    });
  }
}));

// Get nutrition info for a specific food
router.get('/:foodName', asyncHandler(async (req: Request, res: Response) => {
  const { foodName } = req.params;
  const { serving_size = 100 } = req.query;

  if (!foodName) {
    return res.status(400).json({
      success: false,
      error: 'Food name is required'
    });
  }

  try {
    const servingSize = parseInt(serving_size as string, 10);
    if (isNaN(servingSize) || servingSize <= 0) {
      return res.status(400).json({
        success: false,
        error: 'Invalid serving size'
      });
    }

    const nutritionalData = await nutritionService.calculateNutrition([foodName]);

    if (nutritionalData.items.length === 0) {
      return res.status(404).json({
        success: false,
        error: `Nutrition data not found for '${foodName}'`
      });
    }

    const foodItem = nutritionalData.items[0];
    if (!foodItem) {
      return res.status(404).json({
        success: false,
        error: `Nutrition data not found for '${foodName}'`
      });
    }

    const details = await nutritionService.getFoodDetails(foodName);

    // Adjust for serving size (default calculation is for 100g)
    const adjustmentFactor = servingSize / 100;

    res.json({
      food: foodName,
      serving_size: `${servingSize}g`,
      nutrition: {
        calories: Math.round(foodItem.calories * adjustmentFactor),
        protein: Math.round(foodItem.protein * adjustmentFactor),
        carbs: Math.round(foodItem.carbs * adjustmentFactor),
        fats: Math.round(foodItem.fats * adjustmentFactor)
      },
      details,
      per_100g: {
        calories: foodItem.calories,
        protein: foodItem.protein,
        carbs: foodItem.carbs,
        fats: foodItem.fats
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get nutrition info'
    });
  }
}));

// Compare nutrition between foods
router.post('/compare', asyncHandler(async (req: Request, res: Response) => {
  const { foods } = req.body;

  if (!foods || !Array.isArray(foods) || foods.length < 2) {
    return res.status(400).json({
      success: false,
      error: 'At least 2 foods are required for comparison'
    });
  }

  if (foods.length > 10) {
    return res.status(400).json({
      success: false,
      error: 'Maximum 10 foods can be compared at once'
    });
  }

  try {
    const comparisons = await Promise.all(
      foods.map(async (food: string) => {
        const nutritionalData = await nutritionService.calculateNutrition([food]);
        const details = await nutritionService.getFoodDetails(food);

        return {
          food,
          nutrition: nutritionalData.items[0] || null,
          details
        };
      })
    );

    // Calculate averages for comparison
    const validComparisons = comparisons.filter(c => c.nutrition);
    const averages = {
      calories: Math.round(validComparisons.reduce((sum, c) => sum + c.nutrition!.calories, 0) / validComparisons.length),
      protein: Math.round(validComparisons.reduce((sum, c) => sum + c.nutrition!.protein, 0) / validComparisons.length),
      carbs: Math.round(validComparisons.reduce((sum, c) => sum + c.nutrition!.carbs, 0) / validComparisons.length),
      fats: Math.round(validComparisons.reduce((sum, c) => sum + c.nutrition!.fats, 0) / validComparisons.length)
    };

    res.json({
      success: true,
      foods,
      comparisons,
      averages,
      highest: {
        calories: validComparisons.reduce((max, c) => c.nutrition!.calories > max.nutrition!.calories ? c : max),
        protein: validComparisons.reduce((max, c) => c.nutrition!.protein > max.nutrition!.protein ? c : max),
        carbs: validComparisons.reduce((max, c) => c.nutrition!.carbs > max.nutrition!.carbs ? c : max),
        fats: validComparisons.reduce((max, c) => c.nutrition!.fats > max.nutrition!.fats ? c : max)
      },
      lowest: {
        calories: validComparisons.reduce((min, c) => c.nutrition!.calories < min.nutrition!.calories ? c : min),
        protein: validComparisons.reduce((min, c) => c.nutrition!.protein < min.nutrition!.protein ? c : min),
        carbs: validComparisons.reduce((min, c) => c.nutrition!.carbs < min.nutrition!.carbs ? c : min),
        fats: validComparisons.reduce((min, c) => c.nutrition!.fats < min.nutrition!.fats ? c : min)
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Nutrition comparison failed'
    });
  }
}));

// Get daily nutrition recommendations
router.get('/recommendations/daily', asyncHandler(async (req: Request, res: Response) => {
  const {
    age = 30,
    gender = 'male',
    weight = 70,
    height = 175,
    activity_level = 'moderate'
  } = req.query;

  try {
    const ageNum = parseInt(age as string, 10);
    const weightNum = parseFloat(weight as string);
    const heightNum = parseFloat(height as string);

    if (isNaN(ageNum) || isNaN(weightNum) || isNaN(heightNum)) {
      return res.status(400).json({
        success: false,
        error: 'Invalid age, weight, or height values'
      });
    }

    // Calculate BMR using Mifflin-St Jeor Equation
    let bmr: number;
    if (gender === 'male') {
      bmr = 10 * weightNum + 6.25 * heightNum - 5 * ageNum + 5;
    } else {
      bmr = 10 * weightNum + 6.25 * heightNum - 5 * ageNum - 161;
    }

    // Apply activity multiplier
    const activityMultipliers: Record<string, number> = {
      sedentary: 1.2,
      light: 1.375,
      moderate: 1.55,
      active: 1.725,
      very_active: 1.9
    };

    const multiplier = activityMultipliers[activity_level as string] || 1.55;
    const dailyCalories = Math.round(bmr * multiplier);

    // Calculate macronutrient recommendations
    const proteinCalories = dailyCalories * 0.15; // 15% protein
    const carbCalories = dailyCalories * 0.55; // 55% carbs
    const fatCalories = dailyCalories * 0.30; // 30% fats

    const recommendations = {
      calories: dailyCalories,
      protein: Math.round(proteinCalories / 4), // 4 calories per gram
      carbs: Math.round(carbCalories / 4), // 4 calories per gram
      fats: Math.round(fatCalories / 9), // 9 calories per gram
      fiber: Math.round(ageNum < 50 ? (gender === 'male' ? 38 : 25) : (gender === 'male' ? 30 : 21)),
      water: Math.round(weightNum * 35) // 35ml per kg body weight
    };

    res.json({
      profile: {
        age: ageNum,
        gender,
        weight: weightNum,
        height: heightNum,
        activity_level,
        bmr: Math.round(bmr)
      },
      daily_recommendations: recommendations,
      macronutrient_percentages: {
        protein: 15,
        carbs: 55,
        fats: 30
      },
      notes: [
        'These are general recommendations and may vary based on individual needs',
        'Consult with a healthcare provider for personalized nutrition advice',
        'Recommendations based on Dietary Guidelines for Americans'
      ]
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to calculate recommendations'
    });
  }
}));

// Analyze meal balance
router.post('/analyze/balance', asyncHandler(async (req: Request, res: Response) => {
  const { foods } = req.body;

  if (!foods || !Array.isArray(foods) || foods.length === 0) {
    return res.status(400).json({
      success: false,
      error: 'Foods array is required and must not be empty'
    });
  }

  try {
    const nutritionalData = await nutritionService.calculateNutrition(foods);
    const totalCalories = nutritionalData.total_calories;

    if (totalCalories === 0) {
      return res.status(400).json({
        success: false,
        error: 'No nutritional data found for provided foods'
      });
    }

    // Calculate percentages
    const proteinPercent = Math.round((nutritionalData.total_protein * 4 / totalCalories) * 100);
    const carbsPercent = Math.round((nutritionalData.total_carbs * 4 / totalCalories) * 100);
    const fatsPercent = Math.round((nutritionalData.total_fats * 9 / totalCalories) * 100);

    // Analyze balance
    const analysis = {
      overall_score: 0,
      recommendations: [] as string[],
      warnings: [] as string[]
    };

    // Protein analysis
    if (proteinPercent >= 10 && proteinPercent <= 20) {
      analysis.overall_score += 25;
      analysis.recommendations.push('Good protein balance');
    } else if (proteinPercent < 10) {
      analysis.warnings.push('Low protein content - consider adding protein sources');
    } else {
      analysis.warnings.push('High protein content - ensure adequate hydration');
    }

    // Carbs analysis
    if (carbsPercent >= 45 && carbsPercent <= 65) {
      analysis.overall_score += 25;
      analysis.recommendations.push('Good carbohydrate balance');
    } else if (carbsPercent < 45) {
      analysis.warnings.push('Low carbohydrate content - may lack energy sources');
    } else {
      analysis.warnings.push('High carbohydrate content - consider balancing with protein and fats');
    }

    // Fats analysis
    if (fatsPercent >= 20 && fatsPercent <= 35) {
      analysis.overall_score += 25;
      analysis.recommendations.push('Good fat balance');
    } else if (fatsPercent < 20) {
      analysis.warnings.push('Low fat content - may lack essential fatty acids');
    } else {
      analysis.warnings.push('High fat content - consider reducing portion sizes');
    }

    // Variety analysis
    const categories = new Set();
    for (const item of nutritionalData.items) {
      const details = await nutritionService.getFoodDetails(item.name);
      categories.add(details.category);
    }

    if (categories.size >= 3) {
      analysis.overall_score += 25;
      analysis.recommendations.push('Good variety of food categories');
    } else {
      analysis.warnings.push('Limited food variety - try to include more food groups');
    }

    res.json({
      success: true,
      foods,
      nutritional_data: nutritionalData,
      macronutrient_breakdown: {
        protein: { grams: nutritionalData.total_protein, percentage: proteinPercent },
        carbs: { grams: nutritionalData.total_carbs, percentage: carbsPercent },
        fats: { grams: nutritionalData.total_fats, percentage: fatsPercent }
      },
      balance_analysis: analysis,
      food_categories: Array.from(categories),
      meal_type_suggestion: getMealTypeSuggestion(totalCalories)
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Balance analysis failed'
    });
  }
}));

function getMealTypeSuggestion(calories: number): string {
  if (calories < 200) return 'snack';
  if (calories < 400) return 'light meal';
  if (calories < 700) return 'regular meal';
  if (calories < 1000) return 'large meal';
  return 'very large meal';
}

export default router;