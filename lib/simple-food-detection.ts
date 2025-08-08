// Simple Food Detection for Next.js
// Works entirely in the browser without external dependencies

export interface SimpleDetectionResult {
  success: boolean
  detected_items: string[]
  description: string
  confidence: number
}

export class SimpleFoodDetector {
  private foodKeywords: Set<string>
  private commonFoodPatterns: RegExp[]

  constructor() {
    this.foodKeywords = this.loadFoodKeywords()
    this.commonFoodPatterns = this.loadFoodPatterns()
  }

  private loadFoodKeywords(): Set<string> {
    const keywords = [
      // Fruits
      'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'raspberry', 'blackberry',
      'pineapple', 'mango', 'peach', 'pear', 'plum', 'cherry', 'lemon', 'lime', 'kiwi',
      'watermelon', 'cantaloupe', 'honeydew', 'pomegranate', 'fig', 'date', 'prune',
      
      // Vegetables
      'tomato', 'potato', 'carrot', 'onion', 'garlic', 'broccoli', 'cauliflower', 'cabbage',
      'lettuce', 'spinach', 'kale', 'arugula', 'cucumber', 'bell pepper', 'jalapeno',
      'mushroom', 'eggplant', 'zucchini', 'squash', 'pumpkin', 'corn', 'peas', 'beans',
      'asparagus', 'celery', 'radish', 'beet', 'turnip', 'parsnip', 'rutabaga',
      
      // Proteins
      'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'fish', 'salmon', 'tuna',
      'shrimp', 'crab', 'lobster', 'clam', 'mussel', 'oyster', 'egg', 'tofu', 'tempeh',
      'beans', 'lentils', 'chickpeas', 'black beans', 'kidney beans', 'pinto beans',
      
      // Grains and carbs
      'rice', 'bread', 'pasta', 'noodle', 'quinoa', 'oats', 'wheat', 'barley', 'rye',
      'corn', 'potato', 'sweet potato', 'yam', 'cereal', 'granola', 'muesli',
      
      // Dairy
      'milk', 'cheese', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese',
      'cream cheese', 'mozzarella', 'cheddar', 'parmesan', 'feta', 'blue cheese',
      
      // Beverages
      'water', 'coffee', 'tea', 'juice', 'smoothie', 'milkshake', 'soda', 'beer',
      'wine', 'cocktail', 'lemonade', 'iced tea', 'hot chocolate', 'milk',
      
      // Common dishes
      'pizza', 'burger', 'sandwich', 'salad', 'soup', 'stew', 'curry', 'stir fry',
      'pasta', 'rice dish', 'noodle dish', 'casserole', 'lasagna', 'enchilada',
      'taco', 'burrito', 'sushi', 'sashimi', 'roll', 'dumpling', 'spring roll',
      
      // Desserts and sweets
      'cake', 'cookie', 'brownie', 'pie', 'ice cream', 'pudding', 'custard',
      'chocolate', 'candy', 'donut', 'muffin', 'cupcake', 'pastry', 'croissant',
      
      // Condiments and sauces
      'ketchup', 'mustard', 'mayonnaise', 'hot sauce', 'soy sauce', 'teriyaki',
      'barbecue sauce', 'ranch', 'blue cheese dressing', 'vinaigrette', 'oil',
      'vinegar', 'salt', 'pepper', 'herbs', 'spices', 'garlic', 'ginger',
      
      // Nuts and seeds
      'almond', 'walnut', 'pecan', 'cashew', 'peanut', 'pistachio', 'sunflower seed',
      'pumpkin seed', 'chia seed', 'flax seed', 'sesame seed',
      
      // Cooking methods
      'grilled', 'fried', 'baked', 'roasted', 'steamed', 'boiled', 'sauteed',
      'stir-fried', 'smoked', 'cured', 'pickled', 'fermented'
    ]
    
    return new Set(keywords.map(k => k.toLowerCase()))
  }

  private loadFoodPatterns(): RegExp[] {
    return [
      /\b(food|dish|meal|cuisine|recipe)\b/i,
      /\b(ingredient|component|element)\b/i,
      /\b(edible|consumable|nourishing)\b/i,
      /\b(breakfast|lunch|dinner|snack)\b/i,
      /\b(plate|bowl|serving|portion)\b/i
    ]
  }

  /**
   * Detect food items from image filename and context
   */
  async detectFoodFromContext(filename: string, context: string = ''): Promise<SimpleDetectionResult> {
    try {
      console.log('🔍 Starting simple food detection...')
      
      const detectedItems = new Set<string>()
      let confidence = 0

      // Method 1: Extract from filename
      const filenameItems = this.extractFromText(filename)
      filenameItems.forEach(item => detectedItems.add(item))
      if (filenameItems.length > 0) confidence += 20

      // Method 2: Extract from context
      const contextItems = this.extractFromText(context)
      contextItems.forEach(item => detectedItems.add(item))
      if (contextItems.length > 0) confidence += 30

      // Method 3: Generate based on common patterns
      const generatedItems = this.generateFromPatterns(filename, context)
      generatedItems.forEach(item => detectedItems.add(item))
      if (generatedItems.length > 0) confidence += 25

      // Method 4: Smart fallback detection
      if (detectedItems.size === 0) {
        const fallbackItems = this.generateFallbackItems(filename, context)
        fallbackItems.forEach(item => detectedItems.add(item))
        confidence = 15 // Lower confidence for fallback
      }

      const finalItems = Array.from(detectedItems).slice(0, 10) // Limit to top 10
      const description = this.createDescription(finalItems)
      
      // Ensure confidence is between 0-100
      confidence = Math.min(Math.max(confidence, 0), 100)

      console.log(`✅ Simple detection completed. Found ${finalItems.length} items with ${confidence}% confidence.`)

      return {
        success: true,
        detected_items: finalItems,
        description,
        confidence
      }

    } catch (error) {
      console.error('❌ Simple food detection failed:', error)
      return {
        success: false,
        detected_items: [],
        description: 'Food detection failed. Please try again.',
        confidence: 0
      }
    }
  }

  /**
   * Extract food items from text
   */
  private extractFromText(text: string): string[] {
    if (!text.trim()) return []

    const items = new Set<string>()
    const cleanText = text.toLowerCase().trim()

    // Check for exact keyword matches
    this.foodKeywords.forEach((keyword: string) => {
      if (cleanText.includes(keyword)) {
        items.add(keyword)
      }
    });

    // Check for food patterns
    for (const pattern of this.commonFoodPatterns) {
      if (pattern.test(cleanText)) {
        // Extract surrounding words as potential food items
        const words = cleanText.split(/\s+/)
        words.forEach(word => {
          if (word.length > 2 && this.foodKeywords.has(word)) {
            items.add(word)
          }
        })
      }
    }

    return Array.from(items)
  }

  /**
   * Generate food items based on common patterns
   */
  private generateFromPatterns(filename: string, context: string): string[] {
    const items = new Set<string>()
    const combinedText = `${filename} ${context}`.toLowerCase()

    // Common food combinations
    const foodCombinations = [
      { pattern: /chicken|poultry/i, items: ['chicken', 'rice', 'vegetables'] },
      { pattern: /salmon|fish/i, items: ['salmon', 'vegetables', 'rice'] },
      { pattern: /pasta|noodle/i, items: ['pasta', 'sauce', 'cheese'] },
      { pattern: /salad|lettuce/i, items: ['salad', 'vegetables', 'dressing'] },
      { pattern: /sandwich|burger/i, items: ['bread', 'meat', 'cheese'] },
      { pattern: /pizza/i, items: ['pizza', 'cheese', 'tomato'] },
      { pattern: /soup|stew/i, items: ['soup', 'vegetables', 'broth'] },
      { pattern: /rice|grain/i, items: ['rice', 'vegetables', 'protein'] },
      { pattern: /breakfast/i, items: ['eggs', 'bread', 'milk'] },
      { pattern: /lunch/i, items: ['sandwich', 'salad', 'soup'] },
      { pattern: /dinner/i, items: ['meat', 'vegetables', 'grains'] },
      { pattern: /snack/i, items: ['fruit', 'nuts', 'crackers'] }
    ]

    for (const combo of foodCombinations) {
      if (combo.pattern.test(combinedText)) {
        combo.items.forEach(item => items.add(item))
      }
    }

    return Array.from(items)
  }

  /**
   * Generate fallback items when no specific detection
   */
  private generateFallbackItems(filename: string, context: string): string[] {
    const combinedText = `${filename} ${context}`.toLowerCase()
    
    // Default food items based on common scenarios
    const defaultItems = [
      'mixed vegetables',
      'protein source',
      'grains',
      'sauce',
      'seasoning'
    ]

    // If we have any context, try to be more specific
    if (context.trim()) {
      return ['mixed food items', 'vegetables', 'protein', 'grains']
    }

    // If filename suggests it's a food image
    if (filename.toLowerCase().includes('food') || 
        filename.toLowerCase().includes('meal') ||
        filename.toLowerCase().includes('dish')) {
      return ['main dish', 'side dish', 'vegetables', 'protein']
    }

    return defaultItems
  }

  /**
   * Create a comprehensive description
   */
  private createDescription(items: string[]): string {
    if (items.length === 0) {
      return 'Food items detected from image'
    }

    // Group items by category
    const categories: Record<string, string[]> = {
      'proteins': [],
      'vegetables': [],
      'fruits': [],
      'grains': [],
      'dairy': [],
      'beverages': [],
      'condiments': [],
      'dishes': [],
      'other': []
    }

    for (const item of items) {
      const category = this.categorizeItem(item)
      if (category in categories) {
        categories[category].push(item)
      } else {
        categories['other'].push(item)
      }
    }

    // Build description
    const descriptionParts: string[] = []

    for (const [category, categoryItems] of Object.entries(categories)) {
      if (categoryItems.length > 0) {
        const categoryName = category.replace(/\b\w/g, l => l.toUpperCase())
        descriptionParts.push(`${categoryName}: ${categoryItems.join(', ')}`)
      }
    }

    let description = descriptionParts.join('. ')
    
    if (description) {
      description += `. Total items detected: ${items.length}.`
    } else {
      description = 'Mixed food items detected from image'
    }

    return description
  }

  /**
   * Categorize food items
   */
  private categorizeItem(item: string): string {
    const itemLower = item.toLowerCase()

    // Protein sources
    if (['chicken', 'beef', 'pork', 'fish', 'egg', 'tofu', 'beans', 'lentils', 'turkey', 'lamb', 'shrimp', 'salmon', 'tuna', 'meat', 'protein'].some(protein => itemLower.includes(protein))) {
      return 'proteins'
    }

    // Vegetables
    if (['lettuce', 'tomato', 'onion', 'carrot', 'broccoli', 'spinach', 'pepper', 'cucumber', 'cabbage', 'vegetable'].some(veg => itemLower.includes(veg))) {
      return 'vegetables'
    }

    // Fruits
    if (['apple', 'banana', 'orange', 'berry', 'grape', 'lemon', 'lime', 'mango', 'pineapple', 'fruit'].some(fruit => itemLower.includes(fruit))) {
      return 'fruits'
    }

    // Grains/Carbs
    if (['rice', 'bread', 'pasta', 'noodle', 'potato', 'quinoa', 'oats', 'cereal', 'grain'].some(grain => itemLower.includes(grain))) {
      return 'grains'
    }

    // Dairy
    if (['milk', 'cheese', 'yogurt', 'butter', 'cream', 'dairy'].some(dairy => itemLower.includes(dairy))) {
      return 'dairy'
    }

    // Beverages
    if (['water', 'juice', 'coffee', 'tea', 'soda', 'beer', 'wine', 'smoothie', 'drink', 'beverage'].some(drink => itemLower.includes(drink))) {
      return 'beverages'
    }

    // Main dishes
    if (['pizza', 'burger', 'sandwich', 'salad', 'soup', 'curry', 'dish', 'meal'].some(dish => itemLower.includes(dish))) {
      return 'dishes'
    }

    // Condiments
    if (['sauce', 'ketchup', 'mustard', 'mayonnaise', 'condiment'].some(condiment => itemLower.includes(condiment))) {
      return 'condiments'
    }

    return 'other'
  }
}

// Global instance for reuse
let _simpleDetector: SimpleFoodDetector | null = null

export function getSimpleDetector(): SimpleFoodDetector {
  if (!_simpleDetector) {
    _simpleDetector = new SimpleFoodDetector()
  }
  return _simpleDetector
}

export async function detectFoodSimple(filename: string, context: string = ''): Promise<string> {
  try {
    const detector = getSimpleDetector()
    const result = await detector.detectFoodFromContext(filename, context)

    if (result.success) {
      return result.description
    } else {
      return 'Food items detected from image'
    }

  } catch (error) {
    console.error('Simple food detection failed:', error)
    return 'Food items detected from image'
  }
}

export default SimpleFoodDetector
