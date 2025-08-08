// Mock Food Analyzer for Next.js
// Provides realistic food analysis without external dependencies

import { AnalysisResult, FoodItem, NutritionData } from '../types'

export interface MockAnalysisConfig {
  enableDetailedAnalysis?: boolean
  includeVisualizations?: boolean
}

export class MockFoodAnalyzer {
  private config: MockAnalysisConfig

  constructor(config: MockAnalysisConfig = {}) {
    this.config = {
      enableDetailedAnalysis: true,
      includeVisualizations: false,
      ...config
    }
  }

  /**
   * Analyze food with mock data based on description
   */
  async analyzeFoodWithMockData(foodDescription: string, context: string = ''): Promise<AnalysisResult> {
    try {
      console.log('🔍 Starting mock food analysis...')

      // Generate realistic food items based on description
      const foodItems = this.generateFoodItems(foodDescription, context)
      
      // Calculate nutritional data
      const nutritionalData = this.calculateNutritionalData(foodItems)
      
      // Generate comprehensive analysis
      const analysis = this.generateComprehensiveAnalysis(foodItems, nutritionalData, context)
      
      // Create improved description
      const improvedDescription = this.createImprovedDescription(foodItems)

      console.log('✅ Mock analysis completed successfully')

      return {
        success: true,
        analysis,
        food_items: foodItems,
        nutritional_data: nutritionalData,
        improved_description: improvedDescription,
        detailed: this.config.enableDetailedAnalysis ?? true,
        error: ''
      }

    } catch (error) {
      console.error('❌ Mock analysis failed:', error)
      return this.generateFallbackResult(foodDescription)
    }
  }

  /**
   * Generate realistic food items based on description
   */
  private generateFoodItems(description: string, context: string): FoodItem[] {
    const items: FoodItem[] = []
    const combinedText = `${description} ${context}`.toLowerCase()

    // Common food combinations with realistic nutritional values
    const foodDatabase = {
      // Proteins
      chicken: { calories: 165, protein: 31, carbs: 0, fats: 3.6, fiber: 0 },
      salmon: { calories: 208, protein: 25, carbs: 0, fats: 12, fiber: 0 },
      beef: { calories: 250, protein: 26, carbs: 0, fats: 15, fiber: 0 },
      eggs: { calories: 70, protein: 6, carbs: 0, fats: 5, fiber: 0 },
      tofu: { calories: 76, protein: 8, carbs: 1.9, fats: 4.8, fiber: 0.3 },
      
      // Vegetables
      broccoli: { calories: 34, protein: 2.8, carbs: 7, fats: 0.4, fiber: 2.6 },
      spinach: { calories: 23, protein: 2.9, carbs: 3.6, fats: 0.4, fiber: 2.2 },
      carrots: { calories: 41, protein: 0.9, carbs: 10, fats: 0.2, fiber: 2.8 },
      tomatoes: { calories: 18, protein: 0.9, carbs: 3.9, fats: 0.2, fiber: 1.2 },
      onions: { calories: 40, protein: 1.1, carbs: 9.3, fats: 0.1, fiber: 1.7 },
      
      // Grains
      rice: { calories: 130, protein: 2.7, carbs: 28, fats: 0.3, fiber: 0.4 },
      pasta: { calories: 131, protein: 5, carbs: 25, fats: 1.1, fiber: 1.8 },
      bread: { calories: 79, protein: 3.1, carbs: 14, fats: 1, fiber: 1.2 },
      quinoa: { calories: 120, protein: 4.4, carbs: 22, fats: 1.9, fiber: 2.8 },
      
      // Dairy
      cheese: { calories: 113, protein: 7, carbs: 0.4, fats: 9, fiber: 0 },
      milk: { calories: 42, protein: 3.4, carbs: 5, fats: 1, fiber: 0 },
      yogurt: { calories: 59, protein: 10, carbs: 3.6, fats: 0.4, fiber: 0 },
      
      // Fruits
      apple: { calories: 52, protein: 0.3, carbs: 14, fats: 0.2, fiber: 2.4 },
      banana: { calories: 89, protein: 1.1, carbs: 23, fats: 0.3, fiber: 2.6 },
      orange: { calories: 47, protein: 0.9, carbs: 12, fats: 0.1, fiber: 2.4 },
      
      // Common dishes
      pizza: { calories: 266, protein: 11, carbs: 33, fats: 10, fiber: 2.5 },
      burger: { calories: 354, protein: 16, carbs: 30, fats: 17, fiber: 1.2 },
      salad: { calories: 25, protein: 2, carbs: 5, fats: 0.3, fiber: 2 },
      soup: { calories: 120, protein: 8, carbs: 15, fats: 4, fiber: 3 },
      
      // Condiments
      oil: { calories: 120, protein: 0, carbs: 0, fats: 14, fiber: 0 },
      sauce: { calories: 30, protein: 0.5, carbs: 6, fats: 0.2, fiber: 0.5 },
      dressing: { calories: 45, protein: 0.3, carbs: 2, fats: 4, fiber: 0.1 }
    }

    // Extract food items from description
    const detectedItems = this.extractFoodItems(combinedText)
    
    // Generate items with realistic portions
    for (const itemName of detectedItems) {
      const baseItem = foodDatabase[itemName as keyof typeof foodDatabase]
      if (baseItem) {
        // Apply realistic portion sizes
        const portionMultiplier = this.getPortionMultiplier(itemName)
        const item: FoodItem = {
          item: itemName,
          description: `${itemName} - ${Math.round(baseItem.calories * portionMultiplier)} calories`,
          calories: Math.round(baseItem.calories * portionMultiplier),
          protein: Math.round(baseItem.protein * portionMultiplier * 10) / 10,
          carbs: Math.round(baseItem.carbs * portionMultiplier * 10) / 10,
          fats: Math.round(baseItem.fats * portionMultiplier * 10) / 10,
          fiber: Math.round(baseItem.fiber * portionMultiplier * 10) / 10
        }
        items.push(item)
      }
    }

    // If no specific items found, generate generic items
    if (items.length === 0) {
      items.push(
        {
          item: 'Mixed vegetables',
          description: 'Mixed vegetables - 45 calories',
          calories: 45,
          protein: 2.5,
          carbs: 8,
          fats: 0.3,
          fiber: 3
        },
        {
          item: 'Protein source',
          description: 'Protein source - 180 calories',
          calories: 180,
          protein: 25,
          carbs: 2,
          fats: 8,
          fiber: 0
        },
        {
          item: 'Grains',
          description: 'Grains - 150 calories',
          calories: 150,
          protein: 4,
          carbs: 30,
          fats: 1,
          fiber: 2
        }
      )
    }

    return items
  }

  /**
   * Extract food items from text
   */
  private extractFoodItems(text: string): string[] {
    const items = new Set<string>()
    const foodKeywords = [
      'chicken', 'salmon', 'beef', 'eggs', 'tofu', 'broccoli', 'spinach', 'carrots',
      'tomatoes', 'onions', 'rice', 'pasta', 'bread', 'quinoa', 'cheese', 'milk',
      'yogurt', 'apple', 'banana', 'orange', 'pizza', 'burger', 'salad', 'soup',
      'oil', 'sauce', 'dressing'
    ]

    for (const keyword of foodKeywords) {
      if (text.includes(keyword)) {
        items.add(keyword)
      }
    }

    return Array.from(items)
  }

  /**
   * Get realistic portion multiplier for food items
   */
  private getPortionMultiplier(itemName: string): number {
    const itemLower = itemName.toLowerCase()
    
    // Proteins - typically 3-6 oz portions
    if (['chicken', 'salmon', 'beef', 'tofu'].includes(itemLower)) {
      return 1.5 // 150g portion
    }
    
    // Vegetables - typically 1 cup portions
    if (['broccoli', 'spinach', 'carrots', 'tomatoes', 'onions'].includes(itemLower)) {
      return 1.0 // 100g portion
    }
    
    // Grains - typically 1/2 cup cooked portions
    if (['rice', 'pasta', 'quinoa'].includes(itemLower)) {
      return 0.8 // 80g portion
    }
    
    // Dairy - typical serving sizes
    if (['cheese', 'milk', 'yogurt'].includes(itemLower)) {
      return 1.0 // 100g portion
    }
    
    // Fruits - typically 1 medium piece
    if (['apple', 'banana', 'orange'].includes(itemLower)) {
      return 1.0 // 1 medium piece
    }
    
    // Dishes - full meal portions
    if (['pizza', 'burger', 'salad', 'soup'].includes(itemLower)) {
      return 1.2 // 120% of base
    }
    
    // Condiments - small amounts
    if (['oil', 'sauce', 'dressing'].includes(itemLower)) {
      return 0.3 // 30g portion
    }
    
    return 1.0 // Default
  }

  /**
   * Calculate nutritional data from food items
   */
  private calculateNutritionalData(foodItems: FoodItem[]): NutritionData {
    const totalCalories = foodItems.reduce((sum, item) => sum + item.calories, 0)
    const totalProtein = foodItems.reduce((sum, item) => sum + (item.protein || 0), 0)
    const totalCarbs = foodItems.reduce((sum, item) => sum + (item.carbs || 0), 0)
    const totalFats = foodItems.reduce((sum, item) => sum + (item.fats || 0), 0)

    return {
      total_calories: totalCalories,
      total_protein: Math.round(totalProtein * 10) / 10,
      total_carbs: Math.round(totalCarbs * 10) / 10,
      total_fats: Math.round(totalFats * 10) / 10,
      items: foodItems.map(item => ({
        item: item.item,
        description: item.description,
        calories: item.calories,
        protein: item.protein || 0,
        carbs: item.carbs || 0,
        fats: item.fats || 0,
        fiber: item.fiber || 0
      }))
    }
  }

  /**
   * Generate comprehensive analysis
   */
  private generateComprehensiveAnalysis(foodItems: FoodItem[], nutritionalData: NutritionData, context: string): string {
    const mealType = this.determineMealType(context)
    const cuisineStyle = this.determineCuisineStyle(foodItems)
    const portionSize = this.determinePortionSize(nutritionalData.total_calories)
    const mainMacro = this.determineMainMacro(nutritionalData)
    
    const proteinPercentage = Math.round((nutritionalData.total_protein * 4 / nutritionalData.total_calories) * 100)
    const carbsPercentage = Math.round((nutritionalData.total_carbs * 4 / nutritionalData.total_calories) * 100)
    const fatsPercentage = Math.round((nutritionalData.total_fats * 9 / nutritionalData.total_calories) * 100)

    return `## COMPREHENSIVE FOOD ANALYSIS

### IDENTIFIED FOOD ITEMS:
${foodItems.map(item => 
  `- Item: ${item.item} (estimated portion), Calories: ${item.calories}, Protein: ${item.protein}g, Carbs: ${item.carbs}g, Fats: ${item.fats}g, Fiber: ${item.fiber || 0}g`
).join('\n')}

### NUTRITIONAL TOTALS:
- Total Calories: ${nutritionalData.total_calories} kcal
- Total Protein: ${nutritionalData.total_protein}g (${proteinPercentage}% of calories)
- Total Carbohydrates: ${nutritionalData.total_carbs}g (${carbsPercentage}% of calories)
- Total Fats: ${nutritionalData.total_fats}g (${fatsPercentage}% of calories)
- Total Fiber: ${foodItems.reduce((sum, item) => sum + (item.fiber || 0), 0).toFixed(1)}g
- Estimated Sodium: ${Math.round(nutritionalData.total_calories * 0.4)}mg

### MEAL COMPOSITION ANALYSIS:
- **Meal Type**: ${mealType}
- **Cuisine Style**: ${cuisineStyle}
- **Portion Size**: ${portionSize}
- **Cooking Methods**: Mixed preparation methods
- **Main Macronutrient**: ${mainMacro}

### NUTRITIONAL QUALITY ASSESSMENT:
- **Strengths**: ${this.generateStrengths(foodItems, nutritionalData)}
- **Areas for Improvement**: ${this.generateImprovements(foodItems, nutritionalData)}
- **Missing Nutrients**: ${this.generateMissingNutrients(foodItems)}
- **Calorie Density**: ${this.determineCalorieDensity(nutritionalData.total_calories)}

### HEALTH RECOMMENDATIONS:
1. **Balanced Nutrition**: This meal provides a good mix of macronutrients
2. **Portion Awareness**: Consider portion sizes for calorie management
3. **Variety**: Include diverse food groups for optimal nutrition
4. **Hydration**: Remember to drink water with your meal

### DIETARY CONSIDERATIONS:
- **Allergen Information**: Check for common allergens in ingredients
- **Dietary Restrictions**: Verify compatibility with your dietary needs
- **Blood Sugar Impact**: ${this.determineGlycemicImpact(nutritionalData.total_carbs)}

### ANALYSIS NOTE:
This is a mock analysis based on detected food items. For precise nutritional information, consider using a food scale and verified nutritional databases.`
  }

  /**
   * Create improved description
   */
  private createImprovedDescription(foodItems: FoodItem[]): string {
    if (foodItems.length === 0) {
      return 'Food items detected from image'
    }
    
    const categories: Record<string, string[]> = {
      'proteins': [],
      'vegetables': [],
      'grains': [],
      'dairy': [],
      'fruits': [],
      'dishes': [],
      'condiments': []
    }

    for (const item of foodItems) {
      const category = this.categorizeItem(item.item)
      if (category in categories) {
        categories[category].push(item.item)
      }
    }

    const descriptionParts: string[] = []
    for (const [category, items] of Object.entries(categories)) {
      if (items.length > 0) {
        const categoryName = category.replace(/\b\w/g, l => l.toUpperCase())
        descriptionParts.push(`${categoryName}: ${items.join(', ')}`)
      }
    }

    return descriptionParts.join('. ') || 'Mixed food items detected from image'
  }

  /**
   * Helper methods for analysis generation
   */
  private determineMealType(context: string): string {
    const contextLower = context.toLowerCase()
    if (contextLower.includes('breakfast')) return 'Breakfast'
    if (contextLower.includes('lunch')) return 'Lunch'
    if (contextLower.includes('dinner')) return 'Dinner'
    if (contextLower.includes('snack')) return 'Snack'
    return 'Mixed meal'
  }

  private determineCuisineStyle(foodItems: FoodItem[]): string {
    const items = foodItems.map(item => item.item.toLowerCase())
    if (items.some(item => ['pizza', 'pasta'].includes(item))) return 'Italian'
    if (items.some(item => ['sushi', 'rice'].includes(item))) return 'Asian'
    if (items.some(item => ['taco', 'burrito'].includes(item))) return 'Mexican'
    if (items.some(item => ['curry', 'rice'].includes(item))) return 'Indian'
    return 'Mixed/International'
  }

  private determinePortionSize(calories: number): string {
    if (calories < 300) return 'Small'
    if (calories < 600) return 'Medium'
    if (calories < 900) return 'Large'
    return 'Extra Large'
  }

  private determineMainMacro(nutritionalData: NutritionData): string {
    const proteinCals = nutritionalData.total_protein * 4
    const carbsCals = nutritionalData.total_carbs * 4
    const fatsCals = nutritionalData.total_fats * 9

    if (proteinCals > carbsCals && proteinCals > fatsCals) return 'Protein-rich'
    if (carbsCals > proteinCals && carbsCals > fatsCals) return 'Carbohydrate-rich'
    if (fatsCals > proteinCals && fatsCals > carbsCals) return 'Fat-rich'
    return 'Balanced'
  }

  private generateStrengths(foodItems: FoodItem[], nutritionalData: NutritionData): string {
    const strengths: string[] = []
    
    if (nutritionalData.total_protein > 20) strengths.push('Good protein content')
    if (foodItems.some(item => item.fiber && item.fiber > 2)) strengths.push('Good fiber content')
    if (foodItems.length > 2) strengths.push('Variety of food groups')
    if (nutritionalData.total_calories < 800) strengths.push('Reasonable calorie content')
    
    return strengths.length > 0 ? strengths.join(', ') : 'Balanced nutritional profile'
  }

  private generateImprovements(foodItems: FoodItem[], nutritionalData: NutritionData): string {
    const improvements: string[] = []
    
    if (nutritionalData.total_fats > 30) improvements.push('Consider reducing fat content')
    if (nutritionalData.total_carbs > 60) improvements.push('Monitor carbohydrate intake')
    if (!foodItems.some(item => item.fiber && item.fiber > 1)) improvements.push('Add more fiber-rich foods')
    
    return improvements.length > 0 ? improvements.join(', ') : 'Well-balanced meal'
  }

  private generateMissingNutrients(foodItems: FoodItem[]): string {
    const missing: string[] = []
    
    if (!foodItems.some(item => item.item.toLowerCase().includes('vegetable'))) {
      missing.push('More vegetables')
    }
    if (!foodItems.some(item => item.item.toLowerCase().includes('fruit'))) {
      missing.push('Fruits')
    }
    
    return missing.length > 0 ? missing.join(', ') : 'Good variety of nutrients'
  }

  private determineCalorieDensity(calories: number): string {
    if (calories < 400) return 'Low'
    if (calories < 700) return 'Medium'
    return 'High'
  }

  private determineGlycemicImpact(carbs: number): string {
    if (carbs < 30) return 'Low'
    if (carbs < 60) return 'Medium'
    return 'High'
  }

  private categorizeItem(itemName: string): string {
    const itemLower = itemName.toLowerCase()
    
    if (['chicken', 'salmon', 'beef', 'eggs', 'tofu'].includes(itemLower)) return 'proteins'
    if (['broccoli', 'spinach', 'carrots', 'tomatoes', 'onions'].includes(itemLower)) return 'vegetables'
    if (['rice', 'pasta', 'bread', 'quinoa'].includes(itemLower)) return 'grains'
    if (['cheese', 'milk', 'yogurt'].includes(itemLower)) return 'dairy'
    if (['apple', 'banana', 'orange'].includes(itemLower)) return 'fruits'
    if (['pizza', 'burger', 'salad', 'soup'].includes(itemLower)) return 'dishes'
    if (['oil', 'sauce', 'dressing'].includes(itemLower)) return 'condiments'
    
    return 'other'
  }

  /**
   * Generate fallback result
   */
  private generateFallbackResult(description: string): AnalysisResult {
    return {
      success: false,
      error: 'Analysis failed',
      analysis: 'Unable to analyze food items. Please try again with a clearer image or more context.',
      food_items: [],
      nutritional_data: {
        total_calories: 0,
        total_protein: 0,
        total_carbs: 0,
        total_fats: 0,
        items: []
      },
      improved_description: description || 'Food items from image',
      detailed: false
    }
  }
}

// Global instance for reuse
let _mockAnalyzer: MockFoodAnalyzer | null = null

export function getMockAnalyzer(config?: MockAnalysisConfig): MockFoodAnalyzer {
  if (!_mockAnalyzer) {
    _mockAnalyzer = new MockFoodAnalyzer(config)
  }
  return _mockAnalyzer
}

export default MockFoodAnalyzer
