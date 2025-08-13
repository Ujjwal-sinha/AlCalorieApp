import { Router, Request, Response } from 'express';
import { NutritionService } from '../services/NutritionService';
import { asyncHandler } from '../middleware/asyncHandler';

const router = Router();
const nutritionService = NutritionService.getInstance();

// Get food vocabulary
router.get('/vocabulary', asyncHandler(async (req: Request, res: Response) => {
  // In a real implementation, this would come from the nutrition service
  // For now, return a comprehensive food vocabulary
  const vocabulary = [
    // Proteins
    'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'fish', 'salmon', 'tuna', 'cod',
    'shrimp', 'crab', 'lobster', 'egg', 'tofu', 'tempeh', 'beans', 'lentils', 'chickpeas',
    'black beans', 'kidney beans', 'nuts', 'almonds', 'walnuts', 'cashews', 'peanuts',

    // Vegetables
    'tomato', 'potato', 'sweet potato', 'carrot', 'onion', 'garlic', 'ginger', 'broccoli',
    'cauliflower', 'cabbage', 'lettuce', 'spinach', 'kale', 'arugula', 'bell pepper',
    'jalapeño', 'cucumber', 'zucchini', 'eggplant', 'mushroom', 'corn', 'peas',
    'green beans', 'asparagus', 'celery', 'radish', 'beet',

    // Fruits
    'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'raspberry',
    'blackberry', 'cherry', 'peach', 'pear', 'plum', 'apricot', 'kiwi', 'mango',
    'pineapple', 'papaya', 'watermelon', 'cantaloupe', 'honeydew', 'lemon', 'lime',
    'grapefruit', 'pomegranate', 'fig', 'date', 'coconut', 'avocado',

    // Grains
    'rice', 'bread', 'pasta', 'quinoa', 'oats', 'barley', 'wheat', 'cereal',
    'bagel', 'croissant', 'muffin', 'pancake', 'waffle', 'tortilla', 'naan',

    // Dairy
    'milk', 'cheese', 'yogurt', 'butter', 'cream', 'ice cream', 'sour cream',

    // Prepared Foods
    'pizza', 'burger', 'sandwich', 'salad', 'soup', 'curry', 'stir fry',
    'pasta dish', 'rice bowl', 'sushi', 'tacos', 'burrito', 'quesadilla',

    // Snacks & Desserts
    'chips', 'crackers', 'popcorn', 'pretzels', 'cake', 'cookie', 'pie',
    'chocolate', 'candy', 'donut', 'brownie',

    // Beverages
    'water', 'coffee', 'tea', 'juice', 'soda', 'smoothie', 'wine', 'beer'
  ];

  res.json({
    vocabulary,
    total_items: vocabulary.length,
    categories: await nutritionService.getFoodCategories()
  });
}));

// Get food categories
router.get('/categories', asyncHandler(async (req: Request, res: Response) => {
  const categories = {
    proteins: ['chicken', 'beef', 'pork', 'fish', 'egg', 'tofu', 'beans', 'lentils'],
    vegetables: ['tomato', 'potato', 'carrot', 'broccoli', 'spinach', 'lettuce', 'onion'],
    fruits: ['apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'mango'],
    grains: ['rice', 'bread', 'pasta', 'quinoa', 'oats', 'cereal', 'wheat'],
    dairy: ['cheese', 'milk', 'yogurt', 'butter', 'cream', 'ice cream'],
    prepared: ['pizza', 'burger', 'sandwich', 'salad', 'soup', 'curry', 'stir fry'],
    snacks: ['chips', 'nuts', 'crackers', 'popcorn', 'pretzels', 'cookies'],
    desserts: ['cake', 'cookie', 'ice cream', 'chocolate', 'pie', 'donut'],
    beverages: ['coffee', 'tea', 'juice', 'water', 'soda', 'smoothie', 'wine']
  };

  res.json({
    categories,
    total_categories: Object.keys(categories).length,
    total_items: Object.values(categories).flat().length
  });
}));

// Get visual features for food recognition
router.get('/visual-features', asyncHandler(async (req: Request, res: Response) => {
  const features = {
    color_profiles: {
      tomato: { red: [200, 255], green: [0, 100], blue: [0, 100] },
      broccoli: { red: [0, 100], green: [100, 200], blue: [0, 100] },
      banana: { red: [200, 255], green: [200, 255], blue: [0, 150] },
      orange: { red: [200, 255], green: [100, 200], blue: [0, 100] },
      carrot: { red: [200, 255], green: [100, 150], blue: [0, 50] },
      spinach: { red: [0, 50], green: [100, 150], blue: [0, 50] },
      apple: { red: [150, 255], green: [0, 150], blue: [0, 100] },
      lemon: { red: [200, 255], green: [200, 255], blue: [0, 100] }
    },
    texture_patterns: {
      bread: 'porous_texture',
      rice: 'granular_texture',
      pasta: 'smooth_cylindrical',
      meat: 'fibrous_texture',
      cheese: 'smooth_dense',
      nuts: 'rough_irregular',
      fruits: 'smooth_curved',
      vegetables: 'varied_organic'
    },
    shape_characteristics: {
      pizza: 'circular_flat',
      burger: 'layered_cylindrical',
      sandwich: 'rectangular_layered',
      apple: 'spherical_smooth',
      banana: 'elongated_curved',
      carrot: 'conical_tapered',
      broccoli: 'tree_like_clustered',
      bread: 'rectangular_loaf'
    }
  };

  res.json({
    features,
    color_profiles_count: Object.keys(features.color_profiles).length,
    texture_patterns_count: Object.keys(features.texture_patterns).length,
    shape_characteristics_count: Object.keys(features.shape_characteristics).length
  });
}));

// Search foods
router.get('/search', asyncHandler(async (req: Request, res: Response) => {
  const { q: query, category, limit = 10 } = req.query;

  if (!query || typeof query !== 'string') {
    return res.status(400).json({
      success: false,
      error: 'Query parameter "q" is required'
    });
  }

  try {
    const results = await nutritionService.searchFood(query);
    let filteredResults = results;

    // Filter by category if specified
    if (category && typeof category === 'string') {
      const categoryFoods = nutritionService.getFoodsByCategory(category);
      filteredResults = results.filter(food => categoryFoods.includes(food));
    }

    // Apply limit
    const limitNum = parseInt(limit as string, 10);
    if (!isNaN(limitNum) && limitNum > 0) {
      filteredResults = filteredResults.slice(0, limitNum);
    }

    res.json({
      query,
      category: category || null,
      results: filteredResults,
      total: filteredResults.length,
      limit: limitNum
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Search failed'
    });
  }
}));

// Get foods by category
router.get('/category/:categoryName', asyncHandler(async (req: Request, res: Response) => {
  const { categoryName } = req.params;
  const { limit = 50 } = req.query;

  if (!categoryName) {
    return res.status(400).json({
      success: false,
      error: 'Category name is required'
    });
  }

  try {
    const foods = nutritionService.getFoodsByCategory(categoryName);

    if (foods.length === 0) {
      return res.status(404).json({
        success: false,
        error: `Category '${categoryName}' not found or empty`
      });
    }

    const limitNum = parseInt(limit as string, 10);
    const limitedFoods = !isNaN(limitNum) && limitNum > 0 ? foods.slice(0, limitNum) : foods;

    res.json({
      category: categoryName,
      foods: limitedFoods,
      total: foods.length,
      returned: limitedFoods.length
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get category foods'
    });
  }
}));

// Get food details
router.get('/:foodName/details', asyncHandler(async (req: Request, res: Response) => {
  const { foodName } = req.params;

  if (!foodName) {
    return res.status(400).json({
      success: false,
      error: 'Food name is required'
    });
  }

  try {
    const details = await nutritionService.getFoodDetails(foodName);

    res.json({
      food: foodName,
      details,
      found: details.category !== 'other'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get food details'
    });
  }
}));

// Get similar foods
router.get('/:foodName/similar', asyncHandler(async (req: Request, res: Response) => {
  const { foodName } = req.params;
  const { limit = 5 } = req.query;

  if (!foodName) {
    return res.status(400).json({
      success: false,
      error: 'Food name is required'
    });
  }

  try {
    // Simple similarity based on category
    const details = await nutritionService.getFoodDetails(foodName);
    const categoryFoods = nutritionService.getFoodsByCategory(details.category);

    // Remove the original food and limit results
    const similarFoods = categoryFoods
      .filter(food => food !== foodName.toLowerCase())
      .slice(0, parseInt(limit as string, 10) || 5);

    res.json({
      food: foodName,
      category: details.category,
      similar_foods: similarFoods,
      total: similarFoods.length
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to find similar foods'
    });
  }
}));

export default router;