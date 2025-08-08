// TypeScript equivalent of agents.py for Next.js
// Food detection and analysis agents

import { AnalysisResult, FoodItem, NutritionData } from '../types'

export interface FoodDetectionConfig {
  groqApiKey?: string
  temperature?: number
  maxTokens?: number
  enableMockMode?: boolean
}

export interface SearchResult {
  success: boolean
  data?: string
  error?: string
}

export interface FoodAnalysisResult {
  success: boolean
  analysis: string
  food_items: FoodItem[]
  nutritional_data: NutritionData
  comprehensive: boolean
  error?: string
}

export class FoodDetectionAgent {
  private config: FoodDetectionConfig
  private apiEndpoint: string

  constructor(config: FoodDetectionConfig = {}) {
    this.config = {
      temperature: 0.1,
      maxTokens: 1000,
      enableMockMode: !config.groqApiKey, // Default to mock mode if no API key
      ...config
    }
    this.apiEndpoint = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  }

  /**
   * Search for food information using external APIs
   */
  async searchFoodInformation(query: string): Promise<SearchResult> {
    try {
      // In a real implementation, you would use a search API like DuckDuckGo or Google
      // For now, we'll simulate the search functionality
      const searchQuery = encodeURIComponent(`food nutrition ${query}`)
      
      // Mock search results - in production, replace with actual search API
      const mockResults = this.generateMockSearchResults(query)
      
      return {
        success: true,
        data: mockResults
      }
    } catch (error) {
      console.error('Search failed:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Search failed'
      }
    }
  }

  /**
   * Enhanced food detection from image description with comprehensive analysis
   */
  async detectFoodFromImageDescription(
    imageDescription: string, 
    context: string = ''
  ): Promise<FoodAnalysisResult> {
    try {
      // Enhanced prompt for better food detection and analysis
      const prompt = this.createComprehensiveAnalysisPrompt(imageDescription, context)
      
      // Get analysis from LLM (via API or mock)
      const analysis = await this.queryLLM(prompt)
      
      // Enhanced extraction of food items and nutritional data
      const foodItems = this.extractFoodItemsEnhanced(analysis)
      const nutritionalData = this.extractNutritionalDataEnhanced(analysis)
      
      // Search for additional information on unidentified items
      if (foodItems.length < 3) {
        const searchResults = await this.searchAdditionalFoodInfo(imageDescription)
        if (searchResults.success && searchResults.data) {
          // Append search results to analysis
          // In a real implementation, you'd re-analyze with this additional info
        }
      }
      
      return {
        success: true,
        analysis,
        food_items: foodItems,
        nutritional_data: nutritionalData,
        comprehensive: true
      }
      
    } catch (error) {
      console.error('Error in enhanced food detection agent:', error)
      return {
        success: false,
        analysis: 'Failed to analyze food items comprehensively',
        food_items: [],
        nutritional_data: {
          total_calories: 0,
          total_protein: 0,
          total_carbs: 0,
          total_fats: 0,
          items: []
        },
        comprehensive: false,
        error: error instanceof Error ? error.message : 'Analysis failed'
      }
    }
  }

  /**
   * Create comprehensive analysis prompt
   */
  private createComprehensiveAnalysisPrompt(imageDescription: string, context: string): string {
    return `You are an expert nutritionist and food identification specialist. Analyze this food description comprehensively:

FOOD DESCRIPTION: ${imageDescription}
ADDITIONAL CONTEXT: ${context || 'None provided'}

TASK: Provide a detailed analysis following this exact format:

## IDENTIFIED FOOD ITEMS:
[List every food item you can identify, even if mentioned briefly. Include:]
- Main dishes
- Side dishes  
- Beverages
- Condiments/sauces
- Garnishes
- Individual ingredients visible

## DETAILED NUTRITIONAL BREAKDOWN:
For each identified item, provide:
- Item: [Name with estimated portion size]
- Calories: [Amount]
- Protein: [Amount in grams]  
- Carbohydrates: [Amount in grams]
- Fats: [Amount in grams]
- Key nutrients: [Vitamins, minerals, fiber]

## MEAL TOTALS:
- Total Calories: [Sum]
- Total Protein: [Sum in grams]
- Total Carbohydrates: [Sum in grams] 
- Total Fats: [Sum in grams]

## MEAL ASSESSMENT:
- Meal type: [Breakfast/Lunch/Dinner/Snack]
- Cuisine style: [If identifiable]
- Nutritional balance: [Assessment of macro balance]
- Portion size: [Small/Medium/Large/Extra Large]

## HEALTH INSIGHTS:
- Positive aspects: [What's nutritionally good]
- Areas for improvement: [Suggestions]
- Missing nutrients: [What could be added]

## RECOMMENDATIONS:
- Healthier alternatives: [If applicable]
- Portion adjustments: [If needed]
- Complementary foods: [What would complete the meal]

IMPORTANT: Be thorough in identifying ALL food items, even small garnishes or condiments. Provide realistic portion estimates and accurate nutritional values.`
  }

  /**
   * Enhanced extraction of food items from analysis
   */
  private extractFoodItemsEnhanced(analysis: string): FoodItem[] {
    try {
      const lines = analysis.split('\n')
      const foodItems: FoodItem[] = []
      let inFoodSection = false
      
      for (const line of lines) {
        const trimmedLine = line.trim()
        
        // Check if we're in the food items section
        if (trimmedLine.toUpperCase().includes('IDENTIFIED FOOD ITEMS') || 
            trimmedLine.toUpperCase().includes('FOOD ITEMS')) {
          inFoodSection = true
          continue
        } else if (trimmedLine.startsWith('##') && inFoodSection) {
          inFoodSection = false
          continue
        }
        
        // Extract food items
        if (inFoodSection && (trimmedLine.startsWith('-') || trimmedLine.startsWith('•') || trimmedLine.startsWith('*'))) {
          const item = trimmedLine.substring(1).trim()
          if (item && item.length > 2) {
            foodItems.push({
              item,
              description: item,
              calories: this.estimateCalories(item),
              category: this.categorizeFoodItem(item)
            })
          }
        }
        
        // Also look for "Item:" format in nutritional breakdown
        else if (trimmedLine.toLowerCase().startsWith('- item:') || trimmedLine.toLowerCase().startsWith('item:')) {
          const item = trimmedLine.split(':', 2)[1]?.trim()
          if (item && item.length > 2) {
            foodItems.push({
              item,
              description: item,
              calories: this.estimateCalories(item),
              category: this.categorizeFoodItem(item)
            })
          }
        }
      }
      
      // If no items found in structured format, try to extract from text
      if (foodItems.length === 0) {
        return this.extractFoodItems(analysis)
      }
      
      return foodItems
      
    } catch (error) {
      console.error('Error in enhanced food item extraction:', error)
      return this.extractFoodItems(analysis) // Fallback to original method
    }
  }

  /**
   * Enhanced extraction of nutritional data from analysis
   */
  private extractNutritionalDataEnhanced(analysis: string): NutritionData {
    try {
      const nutritionalData: NutritionData = {
        total_calories: 0,
        total_protein: 0,
        total_carbs: 0,
        total_fats: 0,
        items: []
      }
      
      const lines = analysis.split('\n')
      
      // Extract totals using regex
      for (const line of lines) {
        const lowerLine = line.trim().toLowerCase()
        
        // Extract total calories
        if (lowerLine.includes('total calories') || lowerLine.includes('total calorie')) {
          const calories = lowerLine.match(/\d+/)
          if (calories) {
            nutritionalData.total_calories = parseInt(calories[0])
          }
        }
        
        // Extract total protein
        else if (lowerLine.includes('total protein')) {
          const protein = lowerLine.match(/\d+\.?\d*/)
          if (protein) {
            nutritionalData.total_protein = parseFloat(protein[0])
          }
        }
        
        // Extract total carbs
        else if (lowerLine.includes('total carbohydrate') || lowerLine.includes('total carbs')) {
          const carbs = lowerLine.match(/\d+\.?\d*/)
          if (carbs) {
            nutritionalData.total_carbs = parseFloat(carbs[0])
          }
        }
        
        // Extract total fats
        else if (lowerLine.includes('total fats') || lowerLine.includes('total fat')) {
          const fats = lowerLine.match(/\d+\.?\d*/)
          if (fats) {
            nutritionalData.total_fats = parseFloat(fats[0])
          }
        }
      }
      
      return nutritionalData
      
    } catch (error) {
      console.error('Error in enhanced nutritional data extraction:', error)
      return this.extractNutritionalData(analysis) // Fallback to original method
    }
  }

  /**
   * Categorize food items for better organization
   */
  private categorizeFoodItem(item: string): string {
    const itemLower = item.toLowerCase()
    
    // Protein sources
    if (['chicken', 'beef', 'pork', 'fish', 'egg', 'tofu', 'beans', 'lentils', 'turkey', 'lamb', 'shrimp', 'salmon']
        .some(protein => itemLower.includes(protein))) {
      return 'protein'
    }
    
    // Vegetables
    else if (['lettuce', 'tomato', 'onion', 'carrot', 'broccoli', 'spinach', 'pepper', 'cucumber', 'cabbage']
        .some(veg => itemLower.includes(veg))) {
      return 'vegetable'
    }
    
    // Fruits
    else if (['apple', 'banana', 'orange', 'berry', 'grape', 'lemon', 'lime', 'mango', 'pineapple']
        .some(fruit => itemLower.includes(fruit))) {
      return 'fruit'
    }
    
    // Grains/Carbs
    else if (['rice', 'bread', 'pasta', 'noodle', 'potato', 'quinoa', 'oats', 'cereal']
        .some(grain => itemLower.includes(grain))) {
      return 'grain/carb'
    }
    
    // Dairy
    else if (['milk', 'cheese', 'yogurt', 'butter', 'cream']
        .some(dairy => itemLower.includes(dairy))) {
      return 'dairy'
    }
    
    // Beverages
    else if (['water', 'juice', 'coffee', 'tea', 'soda', 'beer', 'wine', 'smoothie']
        .some(drink => itemLower.includes(drink))) {
      return 'beverage'
    }
    
    else {
      return 'other'
    }
  }

  /**
   * Search for additional information about food items that might have been missed
   */
  private async searchAdditionalFoodInfo(description: string): Promise<SearchResult> {
    try {
      const searchQuery = `food ingredients nutrition ${description}`
      return await this.searchFoodInformation(searchQuery)
    } catch (error) {
      console.error('Error searching for additional food info:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Search failed'
      }
    }
  }

  /**
   * Extract food items from analysis (fallback method)
   */
  private extractFoodItems(analysis: string): FoodItem[] {
    try {
      const lines = analysis.split('\n')
      const foodItems: FoodItem[] = []
      
      for (const line of lines) {
        const trimmedLine = line.trim()
        if (trimmedLine.startsWith('-') || trimmedLine.startsWith('•') || trimmedLine.startsWith('*')) {
          const item = trimmedLine.substring(1).trim()
          if (item && item.length > 3) {
            foodItems.push({
              item,
              description: item,
              calories: this.estimateCalories(item)
            })
          }
        }
      }
      
      return foodItems
    } catch (error) {
      console.error('Error extracting food items:', error)
      return []
    }
  }

  /**
   * Extract nutritional data from analysis (fallback method)
   */
  private extractNutritionalData(analysis: string): NutritionData {
    try {
      const lines = analysis.split('\n')
      const nutritionalData: NutritionData = {
        total_calories: 0,
        total_protein: 0,
        total_carbs: 0,
        total_fats: 0,
        items: []
      }
      
      for (const line of lines) {
        const lowerLine = line.trim().toLowerCase()
        if (lowerLine.includes('calorie') && /\d+/.test(lowerLine)) {
          const calories = lowerLine.match(/\d+/)
          if (calories) {
            nutritionalData.total_calories = parseInt(calories[0])
          }
        } else if (lowerLine.includes('protein') && /\d+/.test(lowerLine)) {
          const protein = lowerLine.match(/\d+/)
          if (protein) {
            nutritionalData.total_protein = parseFloat(protein[0])
          }
        } else if (lowerLine.includes('carb') && /\d+/.test(lowerLine)) {
          const carbs = lowerLine.match(/\d+/)
          if (carbs) {
            nutritionalData.total_carbs = parseFloat(carbs[0])
          }
        } else if (lowerLine.includes('fat') && /\d+/.test(lowerLine)) {
          const fats = lowerLine.match(/\d+/)
          if (fats) {
            nutritionalData.total_fats = parseFloat(fats[0])
          }
        }
      }
      
      return nutritionalData
    } catch (error) {
      console.error('Error extracting nutritional data:', error)
      return {
        total_calories: 0,
        total_protein: 0,
        total_carbs: 0,
        total_fats: 0,
        items: []
      }
    }
  }

  /**
   * Estimate calories for a food item (basic estimation)
   */
  private estimateCalories(item: string): number {
    const itemLower = item.toLowerCase()
    
    // Basic calorie estimation based on food type
    if (itemLower.includes('salad')) return 150
    if (itemLower.includes('chicken')) return 250
    if (itemLower.includes('beef')) return 300
    if (itemLower.includes('fish')) return 200
    if (itemLower.includes('rice')) return 180
    if (itemLower.includes('bread')) return 80
    if (itemLower.includes('pasta')) return 220
    if (itemLower.includes('pizza')) return 400
    if (itemLower.includes('burger')) return 500
    if (itemLower.includes('sandwich')) return 350
    if (itemLower.includes('soup')) return 120
    if (itemLower.includes('cake')) return 300
    if (itemLower.includes('cookie')) return 150
    
    // Default estimation
    return 200
  }

  /**
   * Query LLM (mock implementation - replace with actual API call)
   */
  private async queryLLM(prompt: string): Promise<string> {
    try {
      // In a real implementation, this would call your LLM API
      // For now, we'll return a mock response
      return this.generateMockAnalysis(prompt)
    } catch (error) {
      console.error('Error querying LLM:', error)
      throw error
    }
  }

  /**
   * Generate mock search results
   */
  private generateMockSearchResults(query: string): string {
    return `Search results for "${query}": Found nutritional information and common ingredients for this food item. Typical preparation methods and cultural context available.`
  }

  /**
   * Generate mock analysis (replace with actual LLM call)
   */
  private generateMockAnalysis(prompt: string): string {
    return `## IDENTIFIED FOOD ITEMS:
- Grilled chicken breast (150g)
- Mixed green salad (100g)
- Brown rice (80g)
- Olive oil dressing (10ml)

## DETAILED NUTRITIONAL BREAKDOWN:
- Item: Grilled chicken breast (150g), Calories: 231, Protein: 43.5g, Carbohydrates: 0g, Fats: 5g
- Item: Mixed green salad (100g), Calories: 20, Protein: 2g, Carbohydrates: 4g, Fats: 0.2g
- Item: Brown rice (80g), Calories: 216, Protein: 5g, Carbohydrates: 45g, Fats: 1.8g
- Item: Olive oil dressing (10ml), Calories: 90, Protein: 0g, Carbohydrates: 0g, Fats: 10g

## MEAL TOTALS:
- Total Calories: 557
- Total Protein: 50.5g
- Total Carbohydrates: 49g
- Total Fats: 17g

## MEAL ASSESSMENT:
- Meal type: Lunch/Dinner
- Cuisine style: Healthy Mediterranean
- Nutritional balance: Well-balanced with high protein
- Portion size: Medium

## HEALTH INSIGHTS:
- Positive aspects: High protein content, good fiber from vegetables, healthy fats from olive oil
- Areas for improvement: Well-balanced meal
- Missing nutrients: Could add more colorful vegetables for additional vitamins

## RECOMMENDATIONS:
- Healthier alternatives: Already a healthy choice
- Portion adjustments: Appropriate portion sizes
- Complementary foods: Add colorful vegetables like bell peppers or tomatoes`
  }
}

export class FoodSearchAgent {
  private config: FoodDetectionConfig
  private apiEndpoint: string

  constructor(config: FoodDetectionConfig = {}) {
    this.config = {
      temperature: 0.1,
      maxTokens: 1000,
      enableMockMode: !config.groqApiKey, // Default to mock mode if no API key
      ...config
    }
    this.apiEndpoint = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  }

  /**
   * Search for food information globally
   */
  async searchFoodInformation(query: string): Promise<string> {
    try {
      // In a real implementation, this would use a search API
      // For now, we'll return mock search results
      const searchResults = this.generateMockSearchResults(query)
      
      // Process search results with LLM
      const prompt = `
Search Query: ${query}
Search Results: ${searchResults}

Please analyze the search results and provide:
1. Accurate information about the food item/dish
2. Common ingredients and preparation methods
3. Typical nutritional information
4. Cultural context (if applicable)
5. Alternative names or variations

Provide a comprehensive summary of the food information found.
`

      // In a real implementation, this would call your LLM API
      return this.generateMockAnalysisResponse(prompt)
      
    } catch (error) {
      console.error('Error in food search:', error)
      return `Error searching for food information: ${error instanceof Error ? error.message : 'Unknown error'}`
    }
  }

  /**
   * Identify unknown food items using global search
   */
  async identifyUnknownFood(description: string): Promise<string> {
    try {
      const searchQuery = `food dish ${description} ingredients preparation`
      const searchResults = this.generateMockSearchResults(searchQuery)
      
      const prompt = `
Unknown Food Description: ${description}
Search Results: ${searchResults}

Please identify what this food item/dish is:
1. What is the name of this food/dish?
2. What are the main ingredients?
3. How is it typically prepared?
4. What cuisine does it belong to?
5. What are the typical nutritional values?

Provide a detailed identification and description.
`

      return this.generateMockAnalysisResponse(prompt)
      
    } catch (error) {
      console.error('Error identifying unknown food:', error)
      return `Error identifying food: ${error instanceof Error ? error.message : 'Unknown error'}`
    }
  }

  private generateMockSearchResults(query: string): string {
    return `Mock search results for "${query}": Found comprehensive information about ingredients, preparation methods, nutritional values, and cultural context.`
  }

  private generateMockAnalysisResponse(prompt: string): string {
    return `Based on the search results, this appears to be a common food item with well-documented nutritional information. The dish typically contains standard ingredients and follows traditional preparation methods. Nutritional values are within expected ranges for this type of food.`
  }
}

/**
 * Initialize food detection agents
 */
export function initializeAgents(config: FoodDetectionConfig = {}) {
  try {
    const foodAgent = new FoodDetectionAgent(config)
    const searchAgent = new FoodSearchAgent(config)
    return { foodAgent, searchAgent }
  } catch (error) {
    console.error('Error initializing agents:', error)
    return { foodAgent: null, searchAgent: null }
  }
}