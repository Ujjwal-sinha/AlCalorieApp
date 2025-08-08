// Enhanced Food Detection for Next.js
// Implements multiple detection strategies for comprehensive food recognition

import { AnalysisResult, FoodItem, NutritionData } from '../types'

export interface DetectionResult {
  success: boolean
  detected_items: string[]
  description: string
  detection_methods: Record<string, string[]>
  total_items: number
  confidence: number
}

export interface FoodKeywords {
  fruits: string[]
  vegetables: string[]
  proteins: string[]
  grains: string[]
  dairy: string[]
  beverages: string[]
  dishes: string[]
  desserts: string[]
  condiments: string[]
  nuts: string[]
  cooking_methods: string[]
}

export class EnhancedFoodDetector {
  private foodKeywords: FoodKeywords
  private detectionPrompts: string[]
  private confidenceThresholds: number[]

  constructor() {
    this.foodKeywords = this.loadFoodKeywords()
    this.detectionPrompts = this.loadDetectionPrompts()
    this.confidenceThresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
  }

  private loadFoodKeywords(): FoodKeywords {
    return {
      fruits: [
        'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'raspberry', 'blackberry',
        'pineapple', 'mango', 'peach', 'pear', 'plum', 'cherry', 'lemon', 'lime', 'kiwi',
        'watermelon', 'cantaloupe', 'honeydew', 'pomegranate', 'fig', 'date', 'prune'
      ],
      vegetables: [
        'tomato', 'potato', 'carrot', 'onion', 'garlic', 'broccoli', 'cauliflower', 'cabbage',
        'lettuce', 'spinach', 'kale', 'arugula', 'cucumber', 'bell pepper', 'jalapeno',
        'mushroom', 'eggplant', 'zucchini', 'squash', 'pumpkin', 'corn', 'peas', 'beans',
        'asparagus', 'celery', 'radish', 'beet', 'turnip', 'parsnip', 'rutabaga'
      ],
      proteins: [
        'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'fish', 'salmon', 'tuna',
        'shrimp', 'crab', 'lobster', 'clam', 'mussel', 'oyster', 'egg', 'tofu', 'tempeh',
        'beans', 'lentils', 'chickpeas', 'black beans', 'kidney beans', 'pinto beans'
      ],
      grains: [
        'rice', 'bread', 'pasta', 'noodle', 'quinoa', 'oats', 'wheat', 'barley', 'rye',
        'corn', 'potato', 'sweet potato', 'yam', 'cereal', 'granola', 'muesli'
      ],
      dairy: [
        'milk', 'cheese', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese',
        'cream cheese', 'mozzarella', 'cheddar', 'parmesan', 'feta', 'blue cheese'
      ],
      beverages: [
        'water', 'coffee', 'tea', 'juice', 'smoothie', 'milkshake', 'soda', 'beer',
        'wine', 'cocktail', 'lemonade', 'iced tea', 'hot chocolate', 'milk'
      ],
      dishes: [
        'pizza', 'burger', 'sandwich', 'salad', 'soup', 'stew', 'curry', 'stir fry',
        'pasta', 'rice dish', 'noodle dish', 'casserole', 'lasagna', 'enchilada',
        'taco', 'burrito', 'sushi', 'sashimi', 'roll', 'dumpling', 'spring roll'
      ],
      desserts: [
        'cake', 'cookie', 'brownie', 'pie', 'ice cream', 'pudding', 'custard',
        'chocolate', 'candy', 'donut', 'muffin', 'cupcake', 'pastry', 'croissant'
      ],
      condiments: [
        'ketchup', 'mustard', 'mayonnaise', 'hot sauce', 'soy sauce', 'teriyaki',
        'barbecue sauce', 'ranch', 'blue cheese dressing', 'vinaigrette', 'oil',
        'vinegar', 'salt', 'pepper', 'herbs', 'spices', 'garlic', 'ginger'
      ],
      nuts: [
        'almond', 'walnut', 'pecan', 'cashew', 'peanut', 'pistachio', 'sunflower seed',
        'pumpkin seed', 'chia seed', 'flax seed', 'sesame seed'
      ],
      cooking_methods: [
        'grilled', 'fried', 'baked', 'roasted', 'steamed', 'boiled', 'sauteed',
        'stir-fried', 'smoked', 'cured', 'pickled', 'fermented'
      ]
    }
  }

  private loadDetectionPrompts(): string[] {
    return [
      "List every food item, ingredient, dish, sauce, and beverage visible in this image:",
      "What are all the foods, vegetables, fruits, meats, grains, and drinks you can see?",
      "Identify each food component including main dishes, sides, garnishes, and condiments:",
      "Describe all edible items, ingredients, and food products in this image:",
      "What food items, dishes, ingredients, and beverages are present?",
      "List every food, drink, sauce, seasoning, and edible component:",
      "Identify all food elements including main courses, sides, toppings, and beverages:",
      "What are all the edible items, ingredients, and food products visible?",
      "Describe every food component, dish, ingredient, and beverage in detail:",
      "List all food items, dishes, sauces, garnishes, and drinks present:",
      "What specific foods, ingredients, and dishes can you identify in this image?",
      "Identify all food components, including main dishes, sides, and accompaniments:",
      "List every edible item, ingredient, and food product visible in this image:",
      "What are all the food items, beverages, and condiments you can see?",
      "Describe all food elements, dishes, and ingredients present in this image:"
    ]
  }

  /**
   * Enhanced food detection using multiple strategies
   */
  async detectFoodComprehensive(imageFile: File, context: string = ''): Promise<DetectionResult> {
    try {
      console.log('🔍 Starting comprehensive food detection...')

      const allDetectedItems = new Set<string>()
      const detectionMethods: Record<string, string[]> = {}

      // Method 1: Enhanced BLIP detection with multiple prompts
      console.log('Running enhanced BLIP detection...')
      const blipItems = await this.detectWithEnhancedBLIP(imageFile)
      blipItems.forEach(item => allDetectedItems.add(item))
      detectionMethods['blip'] = blipItems

      // Method 2: Context-aware detection
      console.log('Running context-aware detection...')
      const contextItems = this.detectFromContext(context)
      contextItems.forEach(item => allDetectedItems.add(item))
      detectionMethods['context'] = contextItems

      // Method 3: Keyword-based detection from image description
      console.log('Running keyword-based detection...')
      const keywordItems = await this.detectWithKeywords(imageFile)
      keywordItems.forEach(item => allDetectedItems.add(item))
      detectionMethods['keywords'] = keywordItems

      // Method 4: Pattern-based detection
      console.log('Running pattern-based detection...')
      const patternItems = this.detectWithPatterns(Array.from(allDetectedItems))
      patternItems.forEach(item => allDetectedItems.add(item))
      detectionMethods['patterns'] = patternItems

      // Filter and rank items
      const finalItems = this.filterAndRankItems(Array.from(allDetectedItems))

      // Create comprehensive description
      const description = this.createComprehensiveDescription(finalItems, detectionMethods)

      // Calculate confidence score
      const confidence = this.calculateConfidence(finalItems, detectionMethods)

      console.log(`✅ Comprehensive detection completed. Found ${finalItems.length} items with ${confidence}% confidence.`)

      return {
        success: true,
        detected_items: finalItems,
        description,
        detection_methods: detectionMethods,
        total_items: finalItems.length,
        confidence
      }

    } catch (error) {
      console.error('❌ Comprehensive food detection failed:', error)
      return {
        success: false,
        detected_items: [],
        description: 'Food detection failed. Please try again.',
        detection_methods: {},
        total_items: 0,
        confidence: 0
      }
    }
  }

  /**
   * Enhanced BLIP detection with multiple prompts
   */
  private async detectWithEnhancedBLIP(imageFile: File): Promise<string[]> {
    const detectedItems = new Set<string>()

    try {
      // Convert image to base64
      const base64Image = await this.fileToBase64(imageFile)

      // Use multiple prompts for better detection
      for (const prompt of this.detectionPrompts.slice(0, 5)) { // Use first 5 prompts for speed
        try {
          const response = await fetch('/api/blip-analyze', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image: base64Image,
              prompt: prompt,
              format: imageFile.type
            })
          })

          if (response.ok) {
            const result = await response.json()
            if (result.success && result.description) {
              const items = this.extractFoodItemsFromText(result.description)
              items.forEach(item => detectedItems.add(item))
            }
          }
        } catch (error) {
          console.warn(`BLIP detection failed for prompt: ${error}`)
        }
      }

    } catch (error) {
      console.warn('Enhanced BLIP detection failed:', error)
    }

    return Array.from(detectedItems)
  }

  /**
   * Detect food items from context
   */
  private detectFromContext(context: string): string[] {
    if (!context.trim()) return []

    const detectedItems = new Set<string>()
    const contextLower = context.toLowerCase()

    // Check all food categories
    Object.values(this.foodKeywords).flat().forEach(keyword => {
      if (contextLower.includes(keyword.toLowerCase())) {
        detectedItems.add(keyword)
      }
    })

    // Extract food items using text parsing
    const extractedItems = this.extractFoodItemsFromText(context)
    extractedItems.forEach(item => detectedItems.add(item))

    return Array.from(detectedItems)
  }

  /**
   * Keyword-based detection
   */
  private async detectWithKeywords(imageFile: File): Promise<string[]> {
    const detectedItems = new Set<string>()

    try {
      // Get basic image description
      const base64Image = await this.fileToBase64(imageFile)
      
      const response = await fetch('/api/blip-analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          prompt: "What food items are in this image?",
          format: imageFile.type
        })
      })

      if (response.ok) {
        const result = await response.json()
        if (result.success && result.description) {
          const description = result.description.toLowerCase()
          
          // Check against all food keywords
          Object.values(this.foodKeywords).flat().forEach(keyword => {
            if (description.includes(keyword.toLowerCase())) {
              detectedItems.add(keyword)
            }
          })
        }
      }

    } catch (error) {
      console.warn('Keyword-based detection failed:', error)
    }

    return Array.from(detectedItems)
  }

  /**
   * Pattern-based detection
   */
  private detectWithPatterns(items: string[]): string[] {
    const detectedItems = new Set<string>()

    // Common food patterns and combinations
    const foodPatterns = [
      { pattern: /chicken.*rice/i, items: ['chicken', 'rice'] },
      { pattern: /salmon.*vegetables/i, items: ['salmon', 'vegetables'] },
      { pattern: /pasta.*sauce/i, items: ['pasta', 'sauce'] },
      { pattern: /salad.*dressing/i, items: ['salad', 'dressing'] },
      { pattern: /sandwich.*bread/i, items: ['sandwich', 'bread'] },
      { pattern: /pizza.*cheese/i, items: ['pizza', 'cheese'] },
      { pattern: /burger.*bun/i, items: ['burger', 'bun'] },
      { pattern: /soup.*broth/i, items: ['soup', 'broth'] }
    ]

    const combinedText = items.join(' ').toLowerCase()

    for (const foodPattern of foodPatterns) {
      if (foodPattern.pattern.test(combinedText)) {
        foodPattern.items.forEach(item => detectedItems.add(item))
      }
    }

    return Array.from(detectedItems)
  }

  /**
   * Extract food items from text
   */
  private extractFoodItemsFromText(text: string): string[] {
    const items = new Set<string>()
    let cleanText = text.toLowerCase().trim()

    // Remove common prefixes
    const prefixesToRemove = [
      'a photo of', 'an image of', 'this image shows', 'i can see', 'there is', 'there are',
      'the image contains', 'visible in the image', 'in this image', 'this appears to be',
      'looking at this', 'from what i can see', 'it looks like', 'this seems to be'
    ]

    for (const prefix of prefixesToRemove) {
      if (cleanText.startsWith(prefix)) {
        cleanText = cleanText.replace(prefix, '').trim()
      }
    }

    // Enhanced separators
    const separators = [
      ',', ';', ' and ', ' with ', ' including ', ' plus ', ' also ', ' as well as ',
      ' along with ', ' together with ', ' accompanied by ', ' served with ', ' topped with ',
      ' garnished with ', ' mixed with ', ' combined with ', ' containing ', ' featuring ',
      ' such as ', ' like ', ' including ', ' especially ', ' particularly '
    ]

    // Split text by separators
    let parts = [cleanText]
    for (const sep of separators) {
      const newParts: string[] = []
      for (const part of parts) {
        newParts.push(...part.split(sep))
      }
      parts = newParts
    }

    // Clean and filter parts
    const skipWords = new Set([
      'the', 'and', 'with', 'on', 'in', 'of', 'a', 'an', 'is', 'are', 'was', 'were',
      'this', 'that', 'these', 'those', 'some', 'many', 'few', 'several', 'various',
      'different', 'other', 'another', 'each', 'every', 'all', 'both', 'either',
      'neither', 'one', 'two', 'three', 'first', 'second', 'third', 'next', 'last',
      'here', 'there', 'where', 'when', 'how', 'what', 'which', 'who', 'why',
      'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might', 'must',
      'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been', 'being', 'am',
      'very', 'quite', 'rather', 'pretty', 'really', 'truly', 'actually', 'certainly',
      'probably', 'possibly', 'maybe', 'perhaps', 'likely', 'unlikely'
    ])

    for (let part of parts) {
      // Clean the part
      part = part.trim().replace(/[.,!?:;]+$/, '')
      part = part.replace(/\s+/g, ' ') // Remove extra whitespace

      // Skip if too short or is a skip word
      if (part.length <= 2 || skipWords.has(part)) {
        continue
      }

      // Remove quantity descriptors but keep the food item
      const quantityPatterns = [
        /^(a|an|some|many|few|several|various|different|fresh|cooked|raw|fried|grilled|baked|roasted|steamed|boiled)\s+/i,
        /^(small|medium|large|big|huge|tiny|little|sliced|diced|chopped|minced|whole|half|quarter)\s+/i,
        /^(hot|cold|warm|cool|spicy|mild|sweet|sour|salty|bitter|savory|delicious|tasty)\s+/i,
        /^\d+\s*(pieces?|slices?|cups?|tablespoons?|teaspoons?|ounces?|grams?|pounds?|lbs?|oz|g|kg)\s+(of\s+)?/i
      ]

      for (const pattern of quantityPatterns) {
        part = part.replace(pattern, '').trim()
      }

      // Skip if became too short after cleaning
      if (part.length <= 2) {
        continue
      }

      // Check if it's a food item
      if (this.isFoodItem(part)) {
        items.add(part)
      }
    }

    return Array.from(items)
  }

  /**
   * Check if an item is food-related
   */
  private isFoodItem(item: string): boolean {
    const itemLower = item.toLowerCase()

    // Check against food keywords
    if (Object.values(this.foodKeywords).flat().some(keyword => itemLower.includes(keyword.toLowerCase()))) {
      return true
    }

    // Check for common food patterns
    const foodPatterns = [
      /\b(food|dish|meal|cuisine|recipe)\b/,
      /\b(ingredient|component|element)\b/,
      /\b(edible|consumable|nourishing)\b/
    ]

    return foodPatterns.some(pattern => pattern.test(itemLower))
  }

  /**
   * Filter and rank detected items
   */
  private filterAndRankItems(items: string[]): string[] {
    const scoredItems: Array<{ item: string; score: number }> = []

    for (const item of items) {
      let score = 0

      // Base score for being a food item
      if (this.isFoodItem(item)) {
        score += 10
      }

      // Bonus for specific food keywords
      if (Object.values(this.foodKeywords).flat().some(keyword => item.toLowerCase().includes(keyword.toLowerCase()))) {
        score += 5
      }

      // Bonus for longer, more specific items
      if (item.split(' ').length > 1) {
        score += 3
      }

      // Penalty for very short items
      if (item.length < 3) {
        score -= 5
      }

      // Penalty for common non-food words
      const nonFoodWords = new Set(['plate', 'bowl', 'cup', 'glass', 'table', 'chair', 'fork', 'spoon', 'knife'])
      if (nonFoodWords.has(item.toLowerCase())) {
        score -= 10
      }

      if (score > 0) {
        scoredItems.push({ item, score })
      }
    }

    // Sort by score and return top items
    scoredItems.sort((a, b) => b.score - a.score)
    return scoredItems.slice(0, 20).map(item => item.item) // Return top 20 items
  }

  /**
   * Create comprehensive description
   */
  private createComprehensiveDescription(items: string[], detectionMethods: Record<string, string[]>): string {
    if (items.length === 0) {
      return 'No food items detected'
    }

    // Group items by category
    const categories: Record<string, string[]> = {
      'main_dishes': [],
      'vegetables': [],
      'fruits': [],
      'proteins': [],
      'grains': [],
      'dairy': [],
      'beverages': [],
      'condiments': [],
      'desserts': [],
      'other': []
    }

    for (const item of items) {
      const category = this.categorizeFoodItem(item)
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
        const categoryName = category.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
        descriptionParts.push(`${categoryName}: ${categoryItems.join(', ')}`)
      }
    }

    let description = descriptionParts.join('. ')

    // Add detection confidence info
    const methodCounts = Object.fromEntries(
      Object.entries(detectionMethods).map(([method, items]) => [method, items.length])
    )
    const totalDetections = Object.values(methodCounts).reduce((sum, count) => sum + count, 0)

    if (totalDetections > 0) {
      description += `. Detection confidence: ${items.length} items confirmed across ${Object.keys(detectionMethods).length} methods.`
    }

    return description
  }

  /**
   * Categorize food items
   */
  private categorizeFoodItem(item: string): string {
    const itemLower = item.toLowerCase()

    // Protein sources
    if (this.foodKeywords.proteins.some(protein => itemLower.includes(protein))) {
      return 'proteins'
    }

    // Vegetables
    if (this.foodKeywords.vegetables.some(veg => itemLower.includes(veg))) {
      return 'vegetables'
    }

    // Fruits
    if (this.foodKeywords.fruits.some(fruit => itemLower.includes(fruit))) {
      return 'fruits'
    }

    // Grains/Carbs
    if (this.foodKeywords.grains.some(grain => itemLower.includes(grain))) {
      return 'grains'
    }

    // Dairy
    if (this.foodKeywords.dairy.some(dairy => itemLower.includes(dairy))) {
      return 'dairy'
    }

    // Beverages
    if (this.foodKeywords.beverages.some(drink => itemLower.includes(drink))) {
      return 'beverages'
    }

    // Desserts
    if (this.foodKeywords.desserts.some(dessert => itemLower.includes(dessert))) {
      return 'desserts'
    }

    // Condiments
    if (this.foodKeywords.condiments.some(condiment => itemLower.includes(condiment))) {
      return 'condiments'
    }

    // Main dishes
    if (this.foodKeywords.dishes.some(dish => itemLower.includes(dish))) {
      return 'main_dishes'
    }

    return 'other'
  }

  /**
   * Calculate confidence score
   */
  private calculateConfidence(items: string[], detectionMethods: Record<string, string[]>): number {
    let confidence = 0

    // Base confidence from number of items
    if (items.length > 0) {
      confidence += Math.min(items.length * 5, 30) // Max 30 points for items
    }

    // Confidence from multiple detection methods
    const methodCount = Object.keys(detectionMethods).length
    confidence += Math.min(methodCount * 10, 40) // Max 40 points for methods

    // Confidence from method agreement
    const allMethodItems = Object.values(detectionMethods).flat()
    const itemFrequency = new Map<string, number>()
    
    for (const item of allMethodItems) {
      itemFrequency.set(item, (itemFrequency.get(item) || 0) + 1)
    }

    const averageFrequency = Array.from(itemFrequency.values()).reduce((sum, freq) => sum + freq, 0) / itemFrequency.size
    confidence += Math.min(averageFrequency * 5, 30) // Max 30 points for agreement

    return Math.min(confidence, 100) // Cap at 100%
  }

  /**
   * Convert file to base64 - works in both browser and Node.js
   */
  private async fileToBase64(file: File): Promise<string> {
    // Check if we're in a browser environment
    if (typeof window !== 'undefined' && typeof FileReader !== 'undefined') {
      return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.readAsDataURL(file)
        reader.onload = () => {
          const result = reader.result as string
          const base64 = result.split(',')[1]
          resolve(base64)
        }
        reader.onerror = (error) => reject(error)
      })
    } else {
      // Server-side: convert buffer to base64
      try {
        const arrayBuffer = await file.arrayBuffer()
        const buffer = Buffer.from(arrayBuffer)
        return buffer.toString('base64')
      } catch (error) {
        console.error('Error converting file to base64:', error)
        throw error
      }
    }
  }
}

// Global instance for reuse
let _enhancedDetector: EnhancedFoodDetector | null = null

export function getEnhancedDetector(): EnhancedFoodDetector {
  if (!_enhancedDetector) {
    _enhancedDetector = new EnhancedFoodDetector()
  }
  return _enhancedDetector
}

export async function detectFoodEnhanced(imageFile: File, context: string = ''): Promise<string> {
  try {
    const detector = getEnhancedDetector()
    const result = await detector.detectFoodComprehensive(imageFile, context)

    if (result.success) {
      return result.description
    } else {
      return 'Food detection failed. Please try again.'
    }

  } catch (error) {
    console.error('Enhanced food detection failed:', error)
    return 'Food detection error. Please try again.'
  }
}

export default EnhancedFoodDetector
