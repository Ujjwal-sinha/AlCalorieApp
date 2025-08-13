import { Router, Request, Response } from 'express';
import { NutritionService } from '../services/NutritionService';
import { asyncHandler } from '../middleware/asyncHandler';

const router = Router();
const nutritionService = NutritionService.getInstance();

// Get food vocabulary
router.get('/vocabulary', asyncHandler(async (_req: Request, res: Response) => {
  try {
    const categories = nutritionService.getFoodCategories();
    const allFoods: string[] = [];
    
    Object.values(categories).forEach(categoryFoods => {
      allFoods.push(...categoryFoods);
    });

    return res.json({
      success: true,
      vocabulary: allFoods,
      categories: await nutritionService.getFoodCategories()
    });
  } catch (error) {
    console.error('Failed to get food vocabulary:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve food vocabulary'
    });
  }
}));

// Get food categories
router.get('/categories', asyncHandler(async (_req: Request, res: Response) => {
  try {
    const categories = nutritionService.getFoodCategories();
    
    return res.json({
      success: true,
      categories: categories
    });
  } catch (error) {
    console.error('Failed to get food categories:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve food categories'
    });
  }
}));

// Get visual features
router.get('/visual-features', asyncHandler(async (_req: Request, res: Response) => {
  try {
    const visualFeatures = {
      color_profiles: {
        'red_foods': { red: [0.6, 1.0], green: [0.0, 0.4], blue: [0.0, 0.3] },
        'green_foods': { red: [0.0, 0.4], green: [0.5, 1.0], blue: [0.0, 0.3] },
        'yellow_foods': { red: [0.7, 1.0], green: [0.6, 1.0], blue: [0.0, 0.3] },
        'brown_foods': { red: [0.4, 0.7], green: [0.2, 0.5], blue: [0.0, 0.3] }
      },
      texture_patterns: {
        'smooth': 'Low texture variation, uniform surface',
        'rough': 'High texture variation, irregular surface',
        'grainy': 'Medium texture with visible grain patterns',
        'layered': 'Multiple texture layers visible'
      },
      shape_characteristics: {
        'round': 'Circular or spherical shapes',
        'elongated': 'Long, narrow shapes',
        'irregular': 'Non-uniform, organic shapes',
        'geometric': 'Regular geometric shapes'
      }
    };

    return res.json({
      success: true,
      visual_features: visualFeatures
    });
  } catch (error) {
    console.error('Failed to get visual features:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve visual features'
    });
  }
}));

// Search foods
router.get('/search', asyncHandler(async (req: Request, res: Response) => {
  try {
    const query = req.query['q'] as string;
    const category = req.query['category'] as string;

    if (!query) {
      return res.status(400).json({
        success: false,
        error: 'Query parameter "q" is required'
      });
    }

    const results = await nutritionService.searchFood(query);
    
    let filteredResults = results;
    
    if (category) {
      const categoryFoods = nutritionService.getFoodsByCategory(category);
      filteredResults = results.filter((food: string) => categoryFoods.includes(food));
    }

    return res.json({
      success: true,
      query: query,
      category: category || 'all',
      results: filteredResults,
      total: filteredResults.length
    });
  } catch (error) {
    console.error('Food search failed:', error);
    return res.status(500).json({
      success: false,
      error: 'Food search failed'
    });
  }
}));

// Get foods by category
router.get('/category/:categoryName', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { categoryName } = req.params;
    
    if (!categoryName) {
      return res.status(400).json({
        success: false,
        error: 'Category name is required'
      });
    }

    const foods = nutritionService.getFoodsByCategory(categoryName);
    
    if (foods.length === 0) {
      return res.status(404).json({
        success: false,
        error: `Category "${categoryName}" not found`
      });
    }

    return res.json({
      success: true,
      category: categoryName,
      foods: foods,
      total: foods.length
    });
  } catch (error) {
    console.error('Failed to get foods by category:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve foods by category'
    });
  }
}));

// Get food details
router.get('/:foodName/details', asyncHandler(async (req: Request, res: Response) => {
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
    console.error('Failed to get food details:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve food details'
    });
  }
}));

// Get similar foods
router.get('/:foodName/similar', asyncHandler(async (req: Request, res: Response) => {
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

    const categoryFoods = nutritionService.getFoodsByCategory(details.category);
    const similarFoods = categoryFoods
      .filter((food: string) => food !== foodName.toLowerCase())
      .slice(0, 5);

    return res.json({
      success: true,
      food: foodName,
      category: details.category,
      similar_foods: similarFoods
    });
  } catch (error) {
    console.error('Failed to get similar foods:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve similar foods'
    });
  }
}));

export default router;