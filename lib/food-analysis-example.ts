// Example usage of the TypeScript food analysis system
// This demonstrates how to use the new TypeScript equivalents of the Python code

import FoodAnalyzer from './food-analyzer'
import { initializeAgents } from './agents'
import { apiClient, utils } from './api'

/**
 * Example 1: Direct food analysis without Python backend
 */
export async function exampleDirectAnalysis(imageFile: File, context?: string) {
  console.log('🍱 Example: Direct TypeScript Food Analysis')
  console.log('==========================================')
  
  try {
    // Initialize the food analyzer
    const analyzer = new FoodAnalyzer({
      enableMockMode: true // Set to false if you have GROQ API key
    })
    
    // Validate the image file
    const validation = analyzer.validateImageFile(imageFile)
    if (!validation.valid) {
      throw new Error(validation.error)
    }
    
    // Step 1: Get enhanced image description
    console.log('📸 Step 1: Analyzing image...')
    const imageDescription = await analyzer.describeImageEnhanced(imageFile)
    console.log('📝 Image description:', imageDescription)
    
    // Step 2: Perform comprehensive food analysis
    console.log('🔍 Step 2: Performing nutritional analysis...')
    const analysisResult = await analyzer.analyzeFoodWithEnhancedPrompt(
      imageDescription, 
      context || ''
    )
    
    if (analysisResult.success) {
      console.log('✅ Analysis completed successfully!')
      console.log('📊 Results:')
      console.log(`   - Food items found: ${analysisResult.food_items.length}`)
      console.log(`   - Total calories: ${analysisResult.nutritional_data.total_calories}`)
      console.log(`   - Total protein: ${analysisResult.nutritional_data.total_protein}g`)
      console.log(`   - Total carbs: ${analysisResult.nutritional_data.total_carbs}g`)
      console.log(`   - Total fats: ${analysisResult.nutritional_data.total_fats}g`)
      
      return analysisResult
    } else {
      throw new Error(analysisResult.error || 'Analysis failed')
    }
    
  } catch (error) {
    console.error('❌ Direct analysis failed:', error)
    throw error
  }
}

/**
 * Example 2: Using the API client with TypeScript implementation
 */
export async function exampleApiClientAnalysis(imageFile: File, context?: string) {
  console.log('🌐 Example: API Client Food Analysis')
  console.log('====================================')
  
  try {
    // Use the new direct analysis method
    const result = await apiClient.analyzeFoodDirect(imageFile, context)
    
    if (result.success && result.data) {
      console.log('✅ API analysis completed successfully!')
      console.log('📊 Results:', result.data)
      return result.data
    } else {
      throw new Error(result.error || 'API analysis failed')
    }
    
  } catch (error) {
    console.error('❌ API analysis failed:', error)
    
    // Fallback to description analysis
    console.log('🔄 Trying fallback analysis...')
    const fallbackResult = await apiClient.analyzeFoodDescription(
      'mixed food items', 
      context
    )
    
    if (fallbackResult.success && fallbackResult.data) {
      console.log('✅ Fallback analysis completed!')
      return fallbackResult.data
    }
    
    throw error
  }
}

/**
 * Example 3: Using agents for specialized analysis
 */
export async function exampleAgentAnalysis(foodDescription: string, context?: string) {
  console.log('🤖 Example: Agent-Based Food Analysis')
  console.log('=====================================')
  
  try {
    // Initialize agents
    const { foodAgent, searchAgent } = initializeAgents({
      enableMockMode: true
    })
    
    if (!foodAgent || !searchAgent) {
      throw new Error('Failed to initialize agents')
    }
    
    // Step 1: Search for additional food information
    console.log('🔍 Step 1: Searching for food information...')
    const searchResult = await searchAgent.searchFoodInformation(foodDescription)
    console.log('📝 Search result:', searchResult)
    
    // Step 2: Perform comprehensive food detection
    console.log('🍽️ Step 2: Detecting food items...')
    const detectionResult = await foodAgent.detectFoodFromImageDescription(
      foodDescription, 
      context || ''
    )
    
    if (detectionResult.success) {
      console.log('✅ Agent analysis completed successfully!')
      console.log('📊 Results:')
      console.log(`   - Food items: ${detectionResult.food_items.length}`)
      console.log(`   - Comprehensive: ${detectionResult.comprehensive}`)
      console.log(`   - Total calories: ${detectionResult.nutritional_data.total_calories}`)
      
      return detectionResult
    } else {
      throw new Error(detectionResult.error || 'Agent analysis failed')
    }
    
  } catch (error) {
    console.error('❌ Agent analysis failed:', error)
    throw error
  }
}

/**
 * Example 4: Complete workflow demonstration
 */
export async function exampleCompleteWorkflow(imageFile: File) {
  console.log('🔄 Example: Complete Food Analysis Workflow')
  console.log('===========================================')
  
  try {
    // Step 1: Try direct TypeScript analysis first (fastest)
    console.log('⚡ Attempting direct TypeScript analysis...')
    try {
      const directResult = await exampleDirectAnalysis(imageFile)
      console.log('✅ Direct analysis succeeded!')
      return { method: 'direct', result: directResult }
    } catch (error) {
      console.log('⚠️ Direct analysis failed, trying API client...')
    }
    
    // Step 2: Try API client analysis (with Python backend)
    console.log('🌐 Attempting API client analysis...')
    try {
      const apiResult = await exampleApiClientAnalysis(imageFile)
      console.log('✅ API client analysis succeeded!')
      return { method: 'api', result: apiResult }
    } catch (error) {
      console.log('⚠️ API client analysis failed, trying agent analysis...')
    }
    
    // Step 3: Try agent-based analysis (fallback)
    console.log('🤖 Attempting agent-based analysis...')
    const agentResult = await exampleAgentAnalysis('food items from uploaded image')
    console.log('✅ Agent analysis succeeded!')
    return { method: 'agent', result: agentResult }
    
  } catch (error) {
    console.error('❌ All analysis methods failed:', error)
    throw new Error('Complete workflow failed - no analysis method succeeded')
  }
}

/**
 * Example 5: Utility functions demonstration
 */
export function exampleUtilityFunctions() {
  console.log('🛠️ Example: Utility Functions')
  console.log('=============================')
  
  // Example nutritional data
  const protein = 45
  const carbs = 32
  const fats = 14.5
  const calories = 445
  
  // Format calories
  const formattedCalories = utils.formatCalories(calories)
  console.log(`📊 Formatted calories: ${formattedCalories}`)
  
  // Format macronutrients
  const formattedProtein = utils.formatMacro(protein)
  const formattedCarbs = utils.formatMacro(carbs)
  const formattedFats = utils.formatMacro(fats)
  
  console.log(`🥩 Protein: ${formattedProtein}`)
  console.log(`🍞 Carbs: ${formattedCarbs}`)
  console.log(`🥑 Fats: ${formattedFats}`)
  
  // Calculate macro percentages
  const macroPercentages = utils.calculateMacroPercentages(protein, carbs, fats)
  
  console.log('📈 Macro percentages:')
  console.log(`   - Protein: ${macroPercentages.protein}%`)
  console.log(`   - Carbs: ${macroPercentages.carbs}%`)
  console.log(`   - Fats: ${macroPercentages.fats}%`)
  
  return {
    formattedCalories,
    formattedMacros: { formattedProtein, formattedCarbs, formattedFats },
    macroPercentages
  }
}

// Export all examples
export const foodAnalysisExamples = {
  directAnalysis: exampleDirectAnalysis,
  apiClientAnalysis: exampleApiClientAnalysis,
  agentAnalysis: exampleAgentAnalysis,
  completeWorkflow: exampleCompleteWorkflow,
  utilityFunctions: exampleUtilityFunctions
}