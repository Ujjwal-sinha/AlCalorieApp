// TypeScript equivalent of app.py for Next.js
// Main food analysis functionality - Exact match to Python implementation

import { AnalysisResult, FoodItem, NutritionData } from '../types'
import { FoodDetectionAgent, FoodSearchAgent } from './agents'
import { config, getUploadConfig, getNutritionConfig, getMockDataConfig, getApiConfig } from './config'
import { PythonEquivalentFoodAnalyzer, createPythonEquivalentAnalyzer } from './python-equivalent-analyzer'

export interface FoodAnalyzerConfig {
  groqApiKey?: string
  apiEndpoint?: string
  enableMockMode?: boolean
}

// Image enhancement options (for future use)
export interface ImageEnhancementOptions {
  contrast?: number
  sharpness?: number
  brightness?: number
}

export interface AnalysisOptions {
  includeVisualizations?: boolean
  detailedAnalysis?: boolean
  context?: string
}

export class FoodAnalyzer {
  private config: FoodAnalyzerConfig
  private foodAgent: FoodDetectionAgent
  private searchAgent: FoodSearchAgent
  private pythonAnalyzer: PythonEquivalentFoodAnalyzer
  private mockMode: boolean
  private apiConfig = getApiConfig()

  constructor(config: FoodAnalyzerConfig = {}) {
    this.config = {
      apiEndpoint: this.apiConfig.baseUrl,
      enableMockMode: !config.groqApiKey,
      ...config
    }

    this.mockMode = this.config.enableMockMode || false
    this.foodAgent = new FoodDetectionAgent(config)
    this.searchAgent = new FoodSearchAgent(config)
    this.pythonAnalyzer = createPythonEquivalentAnalyzer({
      groqApiKey: config.groqApiKey,
      enableMockMode: this.mockMode
    })
  }

  /**
   * Main food analysis function - uses Python-equivalent analyzer
   */
  async analyzeFoodWithEnhancedPrompt(
    foodDescription: string,
    context: string = '',
    options: AnalysisOptions = {}
  ): Promise<AnalysisResult> {
    return await this.pythonAnalyzer.analyzeFoodWithEnhancedPrompt(foodDescription, context)
  }

  /**
   * Enhanced image description function - uses Python-equivalent analyzer
   */
  async describeImageEnhanced(imageFile: File): Promise<string> {
    return await this.pythonAnalyzer.describeImageEnhanced(imageFile)
  }

  /**
   * Extract food items from text - equivalent to Python's extract_food_items_from_text
   */
  extractFoodItemsFromText(text: string): Set<string> {
    const items = new Set<string>()
    let cleanText = text.toLowerCase().trim()

    // Remove common prefixes and phrases
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

    // Enhanced separators for better splitting
    const separators = [
      ',', ';', ' and ', ' with ', ' including ', ' plus ', ' also ', ' as well as ',
      ' along with ', ' together with ', ' accompanied by ', ' served with ', ' topped with ',
      ' garnished with ', ' mixed with ', ' combined with ', ' containing ', ' featuring '
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

      // Add the cleaned food item
      items.add(part)
    }

    return items
  }

  /**
   * Extract items and nutrients from analysis text
   */
  extractItemsAndNutrients(text: string): { items: FoodItem[], totals: NutritionData } {
    const items: FoodItem[] = []

    try {
      // Enhanced patterns to capture more detailed nutritional information
      const patterns = [
        // Standard format with fiber
        /Item:\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fiber:\s*(\d+\.?\d*)\s*g)?/g,

        // Bullet point format with enhanced nutrients
        /-\s*Item:\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fiber:\s*(\d+\.?\d*)\s*g)?/g,

        // Simple bullet format
        /-\s*([^:,]+):\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?/g,

        // Alternative format without "Item:" prefix
        /-\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?/g
      ]

      for (const pattern of patterns) {
        // Reset regex lastIndex to avoid issues with global flag
        pattern.lastIndex = 0
        let match
        while ((match = pattern.exec(text)) !== null) {
          if (match.length >= 3) {
            // Handle different match group structures
            let item: string
            let calories: number
            let protein: number
            let carbs: number
            let fats: number
            let fiber: number

            if (match.length === 7 && match[1] && match[2]) {
              // Pattern with item description
              item = `${match[1].trim()}: ${match[2].trim()}`
              calories = parseInt(match[3]) || 0
              protein = parseFloat(match[4]) || 0
              carbs = parseFloat(match[5]) || 0
              fats = parseFloat(match[6]) || 0
              fiber = parseFloat(match[7]) || 0
            } else {
              // Standard patterns
              item = match[1].trim()
              calories = parseInt(match[2]) || 0
              protein = parseFloat(match[3]) || 0
              carbs = parseFloat(match[4]) || 0
              fats = parseFloat(match[5]) || 0
              fiber = parseFloat(match[6]) || 0
            }

            // Avoid duplicates
            if (!items.some(existingItem => existingItem.item.toLowerCase() === item.toLowerCase())) {
              items.push({
                item,
                description: `${item} - ${calories} calories`,
                calories,
                protein,
                carbs,
                fats,
                fiber
              })
            }
          }
        }
      }

      // Calculate totals
      const totals: NutritionData = {
        total_calories: items.reduce((sum, item) => sum + (item.calories || 0), 0),
        total_protein: items.reduce((sum, item) => sum + (item.protein || 0), 0),
        total_carbs: items.reduce((sum, item) => sum + (item.carbs || 0), 0),
        total_fats: items.reduce((sum, item) => sum + (item.fats || 0), 0),
        items: items.map(item => ({
          item: item.item,
          description: item.description || `${item.item} - ${item.calories || 0} calories`,
          calories: item.calories || 0,
          protein: item.protein || 0,
          carbs: item.carbs || 0,
          fats: item.fats || 0,
          fiber: item.fiber || getNutritionConfig().defaultPortionSizes.small / 100
        }))
      }

      // Try to extract totals from summary sections if individual items weren't found
      if (items.length === 0) {
        const totalPatterns = [
          /Total Calories?:\s*(\d{1,4})\s*(?:kcal|cal|calories)?/i,
          /Total Protein:\s*(\d+\.?\d*)\s*g/i,
          /Total Carbohydrates?:\s*(\d+\.?\d*)\s*g/i,
          /Total Fats?:\s*(\d+\.?\d*)\s*g/i
        ]

        const calorieMatch = text.match(totalPatterns[0])
        const proteinMatch = text.match(totalPatterns[1])
        const carbsMatch = text.match(totalPatterns[2])
        const fatsMatch = text.match(totalPatterns[3])

        if (calorieMatch) {
          const totalCalories = parseInt(calorieMatch[1])
          const totalProtein = proteinMatch ? parseFloat(proteinMatch[1]) : totalCalories * 0.15 / 4
          const totalCarbs = carbsMatch ? parseFloat(carbsMatch[1]) : totalCalories * 0.50 / 4
          const totalFats = fatsMatch ? parseFloat(fatsMatch[1]) : totalCalories * 0.35 / 9

          items.push({
            item: 'Complete meal (from totals)',
            description: `Complete meal (from totals) - ${totalCalories} calories`,
            calories: totalCalories,
            protein: totalProtein,
            carbs: totalCarbs,
            fats: totalFats,
            fiber: 5 // Estimated
          })

          totals.total_calories = totalCalories
          totals.total_protein = totalProtein
          totals.total_carbs = totalCarbs
          totals.total_fats = totalFats
          totals.items = [{
            item: 'Complete meal (from totals)',
            calories: totalCalories,
            protein: totalProtein,
            carbs: totalCarbs,
            fats: totalFats,
            fiber: getNutritionConfig().defaultPortionSizes.medium / 100,
            description: ''
          }]
        }
      }

      // Final fallback: extract any calorie numbers
      if (items.length === 0 && text.trim().length > 10) {
        const calorieMatches = text.match(/(\d{2,4})\s*(?:cal|kcal|calories)/gi)
        if (calorieMatches && calorieMatches.length > 0) {
          const calorieNumbers = calorieMatches
            .map(match => {
              const numMatch = match.match(/\d+/)
              return numMatch ? parseInt(numMatch[0]) : 0
            })
            .filter(num => num > 0)

          if (calorieNumbers.length > 0) {
            const estimatedCalories = Math.max(...calorieNumbers)
            items.push({
              item: 'Meal items (detected but not fully parsed)',
              description: `Meal items (detected but not fully parsed) - ${estimatedCalories} calories`,
                          calories: estimatedCalories,
            protein: estimatedCalories * 0.15 / getNutritionConfig().macroCaloriesPerGram.protein,
            carbs: estimatedCalories * 0.50 / getNutritionConfig().macroCaloriesPerGram.carbs,
            fats: estimatedCalories * 0.35 / getNutritionConfig().macroCaloriesPerGram.fats,
            fiber: getNutritionConfig().defaultPortionSizes.small / 100
            })

            totals.total_calories = estimatedCalories
            totals.total_protein = estimatedCalories * 0.15 / getNutritionConfig().macroCaloriesPerGram.protein
            totals.total_carbs = estimatedCalories * 0.50 / getNutritionConfig().macroCaloriesPerGram.carbs
            totals.total_fats = estimatedCalories * 0.35 / getNutritionConfig().macroCaloriesPerGram.fats
            totals.items = [{
              item: 'Meal items (detected but not fully parsed)',
              calories: estimatedCalories,
              protein: estimatedCalories * 0.15 / getNutritionConfig().macroCaloriesPerGram.protein,
              carbs: estimatedCalories * 0.50 / getNutritionConfig().macroCaloriesPerGram.carbs,
              fats: estimatedCalories * 0.35 / getNutritionConfig().macroCaloriesPerGram.fats,
              fiber: getNutritionConfig().defaultPortionSizes.small / 100,
              description: ''
            }]
          }
        }
      }

      console.log(`📊 Extracted ${items.length} food items with ${totals.total_calories} total calories`)
      return { items, totals }

    } catch (error) {
      console.error('Error extracting items and nutrients:', error)
      return {
        items: [],
        totals: {
          total_calories: 0,
          total_protein: 0,
          total_carbs: 0,
          total_fats: 0,
          items: []
        }
      }
    }
  }

  /**
   * Convert file to base64
   */
  private async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => {
        const result = reader.result as string
        // Remove data:image/jpeg;base64, prefix
        const base64 = result.split(',')[1]
        resolve(base64)
      }
      reader.onerror = (error) => reject(error)
    })
  }

  /**
   * Create improved description from food items
   */
  private createImprovedDescription(foodItems: FoodItem[]): string {
    return foodItems.map(item => item.item.toLowerCase()).join(', ')
  }

  /**
   * Generate mock analysis for development - uses input parameters dynamically
   */
  public generateMockAnalysis(foodDescription: string, context: string): AnalysisResult {
    // Generate dynamic mock data based on input
    const mockFoodItems = this.generateMockFoodItems(foodDescription)
    const mockNutrition = this.calculateMockNutrition(mockFoodItems)
    const contextInfo = context ? `\n\nCONTEXT PROVIDED: ${context}` : ''

    return {
      success: true,
      analysis: `## COMPREHENSIVE FOOD ANALYSIS

### FOOD DESCRIPTION ANALYZED:
${foodDescription}${contextInfo}

### IDENTIFIED FOOD ITEMS:
${mockFoodItems.map(item => `- Item: ${item.name} (${item.portion}), Calories: ${item.calories}, Protein: ${item.protein}g, Carbs: ${item.carbs}g, Fats: ${item.fats}g`).join('\n')}

### NUTRITIONAL TOTALS:
- Total Calories: ${mockNutrition.totalCalories} kcal
- Total Protein: ${mockNutrition.totalProtein}g (${mockNutrition.totalCalories > 0 ? Math.round((mockNutrition.totalProtein * getNutritionConfig().macroCaloriesPerGram.protein / mockNutrition.totalCalories) * 100) : 0}% of calories)
- Total Carbohydrates: ${mockNutrition.totalCarbs}g (${mockNutrition.totalCalories > 0 ? Math.round((mockNutrition.totalCarbs * getNutritionConfig().macroCaloriesPerGram.carbs / mockNutrition.totalCalories) * 100) : 0}% of calories)
- Total Fats: ${mockNutrition.totalFats}g (${mockNutrition.totalCalories > 0 ? Math.round((mockNutrition.totalFats * getNutritionConfig().macroCaloriesPerGram.fats / mockNutrition.totalCalories) * 100) : 0}% of calories)

### MEAL COMPOSITION ANALYSIS:
- **Meal Type**: ${this.determineMealType(context)}
- **Cuisine Style**: ${this.determineCuisineStyle(foodDescription)}
- **Portion Size**: ${this.determinePortionSize(mockNutrition.totalCalories)}
- **Main Macronutrient**: ${this.determinePrimaryMacro(mockNutrition)}

### NUTRITIONAL QUALITY ASSESSMENT:
- **Strengths**: Balanced nutritional profile based on identified components
- **Areas for Improvement**: Analysis based on mock data - actual analysis may vary
- **Missing Nutrients**: Recommendation based on typical nutritional patterns

### HEALTH RECOMMENDATIONS:
1. **Mock Analysis Note** - This is generated mock data for development purposes
2. **Balanced Approach** - Actual analysis would provide personalized recommendations
3. **Nutritional Variety** - Consider including diverse food groups for optimal nutrition`,
      error: '',
      food_items: mockFoodItems.map(item => ({
        item: item.name,
        description: `${item.name} - ${item.calories} calories`,
        calories: item.calories,
        protein: item.protein,
        carbs: item.carbs,
        fats: item.fats,
        fiber: getNutritionConfig().defaultPortionSizes.small / 100
      })),
      nutritional_data: {
        total_calories: mockNutrition.totalCalories,
        total_protein: mockNutrition.totalProtein,
        total_carbs: mockNutrition.totalCarbs,
        total_fats: mockNutrition.totalFats,
        items: mockFoodItems.map(item => ({
          item: item.name,
          description: `${item.name} - ${item.calories} calories`,
          calories: item.calories,
          protein: item.protein,
          carbs: item.carbs,
          fats: item.fats,
          fiber: getNutritionConfig().defaultPortionSizes.small / 100
        }))
      },
      improved_description: mockFoodItems.map(item => item.name.toLowerCase()).join(', '),
      detailed: true
    }
  }

  /**
   * Generate mock food items based on description
   */
  private generateMockFoodItems(description: string) {
    const descLower = description.toLowerCase()
    const mockItems: Array<{
      name: string
      portion: string
      calories: number
      protein: number
      carbs: number
      fats: number
    }> = []

    // Basic food detection patterns
    const foodPatterns = [
      { keywords: ['chicken', 'poultry'], name: 'Grilled chicken breast', portion: '150g', calories: 250, protein: 46, carbs: 0, fats: 5 },
      { keywords: ['salmon', 'fish'], name: 'Salmon fillet', portion: '120g', calories: 280, protein: 39, carbs: 0, fats: 12 },
      { keywords: ['salad', 'lettuce', 'greens'], name: 'Mixed green salad', portion: '100g', calories: 25, protein: 2, carbs: 5, fats: 0.3 },
      { keywords: ['rice', 'grain'], name: 'Brown rice', portion: '80g', calories: 180, protein: 4, carbs: 36, fats: 1.5 },
      { keywords: ['quinoa'], name: 'Quinoa', portion: '85g', calories: 120, protein: 4, carbs: 22, fats: 2 },
      { keywords: ['vegetables', 'veggie'], name: 'Roasted vegetables', portion: '100g', calories: 45, protein: 2, carbs: 10, fats: 0.5 },
      { keywords: ['bread', 'toast'], name: 'Whole grain bread', portion: '30g', calories: 80, protein: 3, carbs: 15, fats: 1 },
      { keywords: ['pasta'], name: 'Whole wheat pasta', portion: '85g', calories: 220, protein: 8, carbs: 44, fats: 1.5 }
    ]

    // Find matching patterns
    for (const pattern of foodPatterns) {
      if (pattern.keywords.some(keyword => descLower.includes(keyword))) {
        // Extract only the needed properties
        const { keywords, ...foodItem } = pattern
        mockItems.push(foodItem)
      }
    }

    // If no specific matches, add generic items
    if (mockItems.length === 0) {
      mockItems.push(
        { name: 'Mixed food items', portion: 'estimated', calories: 300, protein: 15, carbs: 30, fats: 12 },
        { name: 'Side dish', portion: 'estimated', calories: 120, protein: 3, carbs: 20, fats: 4 }
      )
    }

    return mockItems
  }

  /**
   * Calculate mock nutrition totals
   */
  private calculateMockNutrition(items: Array<{
    name: string
    portion: string
    calories: number
    protein: number
    carbs: number
    fats: number
  }>) {
    return {
      totalCalories: items.reduce((sum, item) => sum + item.calories, 0),
      totalProtein: items.reduce((sum, item) => sum + item.protein, 0),
      totalCarbs: items.reduce((sum, item) => sum + item.carbs, 0),
      totalFats: items.reduce((sum, item) => sum + item.fats, 0)
    }
  }

  /**
   * Determine meal type from context
   */
  private determineMealType(context: string): string {
    const contextLower = context.toLowerCase()
    if (contextLower.includes('breakfast') || contextLower.includes('morning')) return 'Breakfast'
    if (contextLower.includes('lunch') || contextLower.includes('noon')) return 'Lunch'
    if (contextLower.includes('dinner') || contextLower.includes('evening')) return 'Dinner'
    if (contextLower.includes('snack')) return 'Snack'
    return 'Mixed meal'
  }

  /**
   * Determine cuisine style from description
   */
  private determineCuisineStyle(description: string): string {
    const descLower = description.toLowerCase()
    if (descLower.includes('mediterranean') || descLower.includes('olive')) return 'Mediterranean'
    if (descLower.includes('asian') || descLower.includes('stir')) return 'Asian'
    if (descLower.includes('mexican') || descLower.includes('salsa')) return 'Mexican'
    if (descLower.includes('italian') || descLower.includes('pasta')) return 'Italian'
    return 'Mixed/International'
  }

  /**
   * Determine portion size from calories
   */
  private determinePortionSize(calories: number): string {
    const nutritionConfig = getNutritionConfig()
    if (calories < nutritionConfig.defaultPortionSizes.small) return 'Small'
    if (calories < nutritionConfig.defaultPortionSizes.medium) return 'Medium'
    if (calories < nutritionConfig.defaultPortionSizes.large) return 'Large'
    return 'Extra Large'
  }

  /**
   * Determine primary macronutrient
   */
  private determinePrimaryMacro(nutrition: {
    totalCalories: number
    totalProtein: number
    totalCarbs: number
    totalFats: number
  }): string {
    const nutritionConfig = getNutritionConfig()
    const proteinCals = nutrition.totalProtein * getNutritionConfig().macroCaloriesPerGram.protein
    const carbsCals = nutrition.totalCarbs * getNutritionConfig().macroCaloriesPerGram.carbs
    const fatsCals = nutrition.totalFats * getNutritionConfig().macroCaloriesPerGram.fats

    if (proteinCals > carbsCals && proteinCals > fatsCals) return 'Protein-rich'
    if (carbsCals > proteinCals && carbsCals > fatsCals) return 'Carbohydrate-rich'
    if (fatsCals > proteinCals && fatsCals > carbsCals) return 'Fat-rich'
    return 'Balanced'
  }

  /**
   * Generate mock image description
   */
  private generateMockImageDescription(fileName: string): string {
    const mockDescriptions = [
      'grilled chicken breast with steamed broccoli and brown rice',
      'mixed green salad with cherry tomatoes, cucumber, and olive oil dressing',
      'salmon fillet with quinoa and roasted vegetables',
      'turkey sandwich with lettuce, tomato, and whole grain bread',
      'vegetable stir-fry with tofu and jasmine rice',
      'greek yogurt with mixed berries and granola'
    ]

    return mockDescriptions[Math.floor(Math.random() * mockDescriptions.length)]
  }

  /**
   * Generate fallback description when analysis fails
   */
  private generateFallbackDescription(fileName: string): string {
    return `food items from ${fileName.replace(/\.[^/.]+$/, '')}`
  }

  /**
   * Validate image file
   */
  validateImageFile(file: File): { valid: boolean; error?: string } {
    const uploadConfig = getUploadConfig()

    if (!uploadConfig.allowedTypes.includes(file.type)) {
      return {
        valid: false,
        error: `Please upload a valid image file (${uploadConfig.allowedTypes.map(type => type.replace('image/', '').toUpperCase()).join(', ')})`
      }
    }

    if (file.size > uploadConfig.maxFileSize) {
      return {
        valid: false,
        error: `Image file size must be less than ${uploadConfig.maxFileSizeMB}MB`
      }
    }

    return { valid: true }
  }
}

// Utility functions
export const foodAnalyzerUtils = {
  // Format calories
  formatCalories(calories: number): string {
    return new Intl.NumberFormat('en-US').format(Math.round(calories))
  },

  // Format macronutrients
  formatMacro(value: number, unit: string = 'g'): string {
    return `${value.toFixed(1)}${unit}`
  },

  // Calculate macro percentages
  calculateMacroPercentages(protein: number, carbs: number, fats: number) {
    const nutritionConfig = getNutritionConfig()
    const totalCalories = (protein * nutritionConfig.macroCaloriesPerGram.protein) + 
                         (carbs * nutritionConfig.macroCaloriesPerGram.carbs) + 
                         (fats * nutritionConfig.macroCaloriesPerGram.fats)

    if (totalCalories === 0) {
      return { protein: 0, carbs: 0, fats: 0 }
    }

    return {
      protein: Math.round(((protein * nutritionConfig.macroCaloriesPerGram.protein) / totalCalories) * 100),
      carbs: Math.round(((carbs * nutritionConfig.macroCaloriesPerGram.carbs) / totalCalories) * 100),
      fats: Math.round(((fats * nutritionConfig.macroCaloriesPerGram.fats) / totalCalories) * 100)
    }
  }
}

// Export default instance
export default FoodAnalyzer