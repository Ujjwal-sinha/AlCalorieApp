import { ChatGroq } from '@langchain/groq';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { AIMessage } from '@langchain/core/messages';
import { FoodItem, NutritionalData } from '../types';

interface GroqAnalysisRequest {
  detectedFoods: string[];
  nutritionalData: NutritionalData;
  foodItems: FoodItem[];
  imageDescription?: string;
  mealContext?: string;
}

interface GroqAnalysisResponse {
  success: boolean;
  summary: string;
  detailedAnalysis: string;
  healthScore: number;
  recommendations: string[];
  dietaryConsiderations: string[];
  foodItemReports?: {
    [foodName: string]: {
      nutritionProfile: string;
      healthBenefits: string;
      nutritionalHistory: string;
      cookingMethods: string;
      servingSuggestions: string;
      potentialConcerns: string;
      alternatives: string;
    };
  };
  dailyMealPlan?: {
    breakfast: string[];
    lunch: string[];
    dinner: string[];
    snacks: string[];
    hydration: string[];
    totalCalories: number;
    notes: string;
  };
  error?: string;
}

export class GroqAnalysisService {
  private static instance: GroqAnalysisService;
  private apiKey: string;
  private groqModel: ChatGroq | null = null;
  private requestQueue: Array<() => Promise<any>> = [];
  private isProcessing = false;
  private lastRequestTime = 0;
  private readonly MIN_REQUEST_INTERVAL = 2000; // 2 seconds between requests

  private constructor() {
    this.apiKey = process.env['GROQ_API_KEY'] || '';
    if (!this.apiKey) {
      console.warn('GROQ_API_KEY not found in environment variables');
    } else {
      this.groqModel = new ChatGroq({
        apiKey: this.apiKey,
        modelName: 'llama3-8b-8192', // Using a more efficient model
        temperature: 0.3,
        maxTokens: 2000, // Reduced token limit
      });
    }
  }

  public static getInstance(): GroqAnalysisService {
    if (!GroqAnalysisService.instance) {
      GroqAnalysisService.instance = new GroqAnalysisService();
    }
    return GroqAnalysisService.instance;
  }

  private async rateLimitedRequest<T>(requestFn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.requestQueue.push(async () => {
        try {
          const result = await requestFn();
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });

      this.processQueue();
    });
  }

  private async processQueue() {
    if (this.isProcessing || this.requestQueue.length === 0) {
      return;
    }

    this.isProcessing = true;

    while (this.requestQueue.length > 0) {
      const now = Date.now();
      const timeSinceLastRequest = now - this.lastRequestTime;

      if (timeSinceLastRequest < this.MIN_REQUEST_INTERVAL) {
        const delay = this.MIN_REQUEST_INTERVAL - timeSinceLastRequest;
        await new Promise(resolve => setTimeout(resolve, delay));
      }

      const requestFn = this.requestQueue.shift();
      if (requestFn) {
        try {
          this.lastRequestTime = Date.now();
          await requestFn();
        } catch (error) {
          console.error('Request failed:', error);
          // Continue processing other requests
        }
      }
    }

    this.isProcessing = false;
  }

  async generateComprehensiveAnalysis(request: GroqAnalysisRequest): Promise<GroqAnalysisResponse> {
    if (!this.apiKey || !this.groqModel) {
      return {
        success: false,
        summary: 'GROQ API key not configured',
        detailedAnalysis: '',
        healthScore: 0,
        recommendations: ['Configure GROQ_API_KEY environment variable'],
        dietaryConsiderations: [],
        error: 'GROQ_API_KEY not configured'
      };
    }

    try {
      // Step 1: Generate detailed food item reports
      const foodItemReports = await this.generateFoodItemReports(request.detectedFoods);
      
      // Step 2: Generate comprehensive meal analysis
      const mealAnalysis = await this.generateMealAnalysis(request, foodItemReports);
      
      // Step 3: Generate daily meal plan
      const dailyMealPlan = await this.generateDailyMealPlan(request, foodItemReports);

      return {
        success: true,
        summary: mealAnalysis.summary,
        detailedAnalysis: mealAnalysis.detailedAnalysis,
        healthScore: mealAnalysis.healthScore,
        recommendations: mealAnalysis.recommendations,
        dietaryConsiderations: mealAnalysis.dietaryConsiderations,
        foodItemReports,
        dailyMealPlan
      };

    } catch (error) {
      console.error('Comprehensive analysis failed:', error);
      return {
        success: false,
        summary: 'Failed to generate comprehensive analysis',
        detailedAnalysis: '',
        healthScore: 0,
        recommendations: ['Try again later'],
        dietaryConsiderations: [],
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  private async generateFoodItemReports(detectedFoods: string[]): Promise<any> {
    const reports: any = {};
    
    // Limit to first 3 foods to avoid rate limits
    const foodsToAnalyze = detectedFoods.slice(0, 3);
    
    for (const foodItem of foodsToAnalyze) {
      try {
        console.log(`Generating report for: ${foodItem}`);
        const report = await this.generateIndividualFoodReport(foodItem);
        reports[foodItem] = report;
        
        // Add delay between requests
        await new Promise(resolve => setTimeout(resolve, 1000));
        
      } catch (error) {
        console.error(`Failed to generate report for ${foodItem}:`, error);
        reports[foodItem] = this.getFallbackFoodReport(foodItem);
      }
    }
    
    // Add fallback reports for remaining foods
    for (const foodItem of detectedFoods.slice(3)) {
      reports[foodItem] = this.getFallbackFoodReport(foodItem);
    }
    
    return reports;
  }

  private async generateIndividualFoodReport(foodItem: string): Promise<any> {
    const prompt = `Analyze "${foodItem}" nutritionally. Provide concise, structured response:

NUTRITION PROFILE: [Key nutrients, calories, macros]
HEALTH BENEFITS: [Main health advantages]
NUTRITIONAL HISTORY: [Brief cultural significance]
COOKING METHODS: [Best cooking practices]
SERVING SUGGESTIONS: [Portion and preparation tips]
POTENTIAL CONCERNS: [Allergies, restrictions]
ALTERNATIVES: [Healthy substitutes]

Keep each section brief but informative.`;

    return this.rateLimitedRequest(async () => {
      const promptTemplate = ChatPromptTemplate.fromTemplate(prompt);
      const messages = await promptTemplate.formatMessages({});
      const response = await this.groqModel!.invoke(messages);
      const analysisText = response.content as string;

      return this.parseFoodItemReport(analysisText, foodItem);
    });
  }

  private parseFoodItemReport(analysisText: string, foodItem: string): any {
    try {
      const sections = {
        nutritionProfile: this.extractSection(analysisText, 'NUTRITION PROFILE'),
        healthBenefits: this.extractSection(analysisText, 'HEALTH BENEFITS'),
        nutritionalHistory: this.extractSection(analysisText, 'NUTRITIONAL HISTORY'),
        cookingMethods: this.extractSection(analysisText, 'COOKING METHODS'),
        servingSuggestions: this.extractSection(analysisText, 'SERVING SUGGESTIONS'),
        potentialConcerns: this.extractSection(analysisText, 'POTENTIAL CONCERNS'),
        alternatives: this.extractSection(analysisText, 'ALTERNATIVES')
      };

      return sections;
    } catch (error) {
      console.error(`Error parsing report for ${foodItem}:`, error);
      return this.getFallbackFoodReport(foodItem);
    }
  }

  private extractSection(text: string, sectionName: string): string {
    const pattern = new RegExp(`${sectionName}:\\s*\\n([\\s\\S]*?)(?=\\n###|$)`, 'i');
    const match = text.match(pattern);
    return match && match[1] ? match[1].trim() : `Detailed ${sectionName.toLowerCase()} information for this food item.`;
  }

  private getFallbackFoodReport(foodItem: string): any {
    return {
      nutritionProfile: `Comprehensive nutritional profile for ${foodItem} including macronutrients, vitamins, and minerals.`,
      healthBenefits: `Various health benefits of ${foodItem} including its positive impact on overall wellness.`,
      nutritionalHistory: `Historical significance and traditional uses of ${foodItem} across different cultures.`,
      cookingMethods: `Different cooking methods for ${foodItem} and their nutritional impact.`,
      servingSuggestions: `Practical serving suggestions and portion recommendations for ${foodItem}.`,
      potentialConcerns: `Potential health considerations and dietary restrictions related to ${foodItem}.`,
      alternatives: `Healthy alternatives and substitutes for ${foodItem}.`
    };
  }

  private async generateMealAnalysis(request: GroqAnalysisRequest, _foodItemReports: any): Promise<any> {
    return this.rateLimitedRequest(async () => {
      try {
        const prompt = this.buildAnalysisPrompt(request);
        
        const promptTemplate = ChatPromptTemplate.fromTemplate(`
          You are an expert nutritionist. Provide concise analysis:

          {prompt}
        `);

        const messages = await promptTemplate.formatMessages({
          prompt: prompt
        });
        const response = await this.groqModel!.invoke(messages);
        const analysisText = response.content as string;
        
        if (!analysisText) {
          throw new Error('No response content from GROQ API');
        }
        
        return this.parseAnalysisResponse(analysisText, request);

      } catch (error) {
        console.error('Meal analysis failed:', error);
        return {
          success: false,
          summary: 'Failed to generate meal analysis',
          detailedAnalysis: '',
          healthScore: 0,
          recommendations: ['Try again later'],
          dietaryConsiderations: [],
          error: error instanceof Error ? error.message : 'Unknown error'
        };
      }
    });
  }

  private async generateDailyMealPlan(request: GroqAnalysisRequest, _foodItemReports: any): Promise<any> {
    return this.rateLimitedRequest(async () => {
      try {
        const prompt = this.buildMealPlanPrompt(request);
        
        const promptTemplate = ChatPromptTemplate.fromTemplate(`
          You are an expert nutritionist. Provide meal plan:

          {prompt}
        `);

        const messages = await promptTemplate.formatMessages({
          prompt: prompt
        });
        const response = await this.groqModel!.invoke(messages);
        const analysisText = response.content as string;
        
        if (!analysisText) {
          throw new Error('No response content from GROQ API');
        }
        
        return this.extractDailyMealPlan(analysisText);

      } catch (error) {
        console.error('Daily meal plan generation failed:', error);
        return this.getFallbackMealPlan();
      }
    });
  }

  private buildAnalysisPrompt(request: GroqAnalysisRequest): string {
    const { detectedFoods, nutritionalData, foodItems, mealContext } = request;
    
    const foodList = foodItems.map(item => 
      `${item.name}: ${item.calories}cal, ${item.protein}g protein, ${item.carbs}g carbs, ${item.fats}g fats`
    ).join(', ');

    return `Analyze this meal: ${detectedFoods.join(', ')} (${nutritionalData.total_calories} cal total)

Nutrition: ${foodList}
Context: ${mealContext || 'General meal'}

Provide brief analysis:
EXECUTIVE SUMMARY: [2-3 sentence summary]
NUTRITIONAL QUALITY SCORE: [1-10] with brief justification
STRENGTHS: [2-3 good points]
AREAS FOR IMPROVEMENT: [2-3 suggestions]
HEALTH RECOMMENDATIONS: [2-3 actionable tips]
DIETARY CONSIDERATIONS: [Key concerns and restrictions]`;
  }

  private buildMealPlanPrompt(request: GroqAnalysisRequest): string {
    const { detectedFoods, nutritionalData, mealContext } = request;
    
    return `Current meal: ${detectedFoods.join(', ')} (${nutritionalData.total_calories} cal)
Context: ${mealContext || 'General meal'}

Provide meal plan for rest of day:
BREAKFAST: [2-3 options with calories]
LUNCH: [2-3 options with calories] 
DINNER: [2-3 options with calories]
SNACKS: [2-3 healthy snacks with calories]
HYDRATION: [beverage recommendations]
TOTAL DAILY CALORIES: [total for complete day]
NOTES: [brief considerations]`;
  }

  private parseAnalysisResponse(analysisText: string, request: GroqAnalysisRequest): any {
    try {
      // Extract summary (first paragraph after "EXECUTIVE SUMMARY:")
      const summaryMatch = analysisText.match(/### EXECUTIVE SUMMARY:\s*\n([\s\S]*?)(?=\n###|$)/);
      const summary = summaryMatch && summaryMatch[1] ? summaryMatch[1].trim() : 'Analysis completed successfully.';

      // Extract health score
      const scoreMatch = analysisText.match(/NUTRITIONAL QUALITY SCORE:\s*\[1-10\]\s*\n\*\*Score\*\*:\s*(\d+)\/10/);
      const healthScore = scoreMatch && scoreMatch[1] ? parseInt(scoreMatch[1]) : this.calculateDefaultHealthScore(request.nutritionalData);

      // Extract recommendations
      const recommendationsMatch = analysisText.match(/### HEALTH RECOMMENDATIONS:\s*\n([\s\S]*?)(?=\n###|$)/);
      const recommendations = recommendationsMatch && recommendationsMatch[1]
        ? this.extractListItems(recommendationsMatch[1])
        : ['Maintain balanced nutrition', 'Monitor portion sizes'];

      // Extract dietary considerations
      const dietaryMatch = analysisText.match(/### DIETARY CONSIDERATIONS:\s*\n([\s\S]*?)(?=\n###|$)/);
      const dietaryConsiderations = dietaryMatch && dietaryMatch[1]
        ? this.extractListItems(dietaryMatch[1])
        : ['Consider individual dietary needs'];

      return {
        success: true,
        summary,
        detailedAnalysis: analysisText,
        healthScore,
        recommendations,
        dietaryConsiderations
      };

    } catch (error) {
      console.error('Error parsing analysis response:', error);
      return {
        success: false,
        summary: 'Analysis completed with basic information',
        detailedAnalysis: analysisText,
        healthScore: this.calculateDefaultHealthScore(request.nutritionalData),
        recommendations: ['Review the detailed analysis for specific recommendations'],
        dietaryConsiderations: ['Consider consulting a nutritionist for personalized advice'],
        error: 'Failed to parse analysis response'
      };
    }
  }

  private extractListItems(text: string): string[] {
    const items: string[] = [];
    const lines = text.split('\n');
    
    for (const line of lines) {
      const trimmed = line.trim();
      
      // Skip empty lines and section headers
      if (!trimmed || trimmed.startsWith('**') || trimmed.startsWith('###')) {
        continue;
      }
      
      // Check for various list formats
      if (trimmed.startsWith('-') || trimmed.startsWith('•') || trimmed.match(/^\d+\./)) {
        const item = trimmed.replace(/^[-•\d\.\s]+/, '').trim();
        if (item && item.length > 5) { // Ensure meaningful content
          items.push(item);
        }
      } else if (trimmed.includes('cal') || trimmed.includes('calories') || 
                 trimmed.includes('~') || trimmed.includes('(')) {
        // This might be a meal item without bullet points
        if (trimmed.length > 5) {
          items.push(trimmed);
        }
      }
    }
    
    // If no items found, try to extract from the text more broadly
    if (items.length === 0) {
      const sentences = text.split(/[.!?]/);
      for (const sentence of sentences) {
        const trimmed = sentence.trim();
        if (trimmed.length > 10 && (trimmed.includes('cal') || trimmed.includes('~'))) {
          items.push(trimmed);
        }
      }
    }
    
    return items.length > 0 ? items : ['Review detailed analysis for specific recommendations'];
  }

  private extractDailyMealPlan(analysisText: string): any {
    try {
      const mealPlan: any = {
        breakfast: [],
        lunch: [],
        dinner: [],
        snacks: [],
        hydration: [],
        totalCalories: 0,
        notes: ''
      };

      console.log('Extracting meal plan from:', analysisText.substring(0, 500) + '...');

      // Extract breakfast - look for both **BREAKFAST** and BREAKFAST patterns
      let breakfastMatch = analysisText.match(/\*\*BREAKFAST\*\*.*?\n([\s\S]*?)(?=\*\*LUNCH\*\*|\*\*DINNER\*\*|\*\*SNACKS\*\*|\*\*HYDRATION\*\*|$)/i);
      if (!breakfastMatch) {
        breakfastMatch = analysisText.match(/BREAKFAST.*?\n([\s\S]*?)(?=LUNCH|DINNER|SNACKS|HYDRATION|$)/i);
      }
      if (breakfastMatch && breakfastMatch[1]) {
        mealPlan.breakfast = this.extractListItems(breakfastMatch[1]);
        console.log('Extracted breakfast:', mealPlan.breakfast);
      }

      // Extract lunch
      let lunchMatch = analysisText.match(/\*\*LUNCH\*\*.*?\n([\s\S]*?)(?=\*\*DINNER\*\*|\*\*SNACKS\*\*|\*\*HYDRATION\*\*|$)/i);
      if (!lunchMatch) {
        lunchMatch = analysisText.match(/LUNCH.*?\n([\s\S]*?)(?=DINNER|SNACKS|HYDRATION|$)/i);
      }
      if (lunchMatch && lunchMatch[1]) {
        mealPlan.lunch = this.extractListItems(lunchMatch[1]);
        console.log('Extracted lunch:', mealPlan.lunch);
      }

      // Extract dinner
      let dinnerMatch = analysisText.match(/\*\*DINNER\*\*.*?\n([\s\S]*?)(?=\*\*SNACKS\*\*|\*\*HYDRATION\*\*|$)/i);
      if (!dinnerMatch) {
        dinnerMatch = analysisText.match(/DINNER.*?\n([\s\S]*?)(?=SNACKS|HYDRATION|$)/i);
      }
      if (dinnerMatch && dinnerMatch[1]) {
        mealPlan.dinner = this.extractListItems(dinnerMatch[1]);
        console.log('Extracted dinner:', mealPlan.dinner);
      }

      // Extract snacks
      let snacksMatch = analysisText.match(/\*\*SNACKS\*\*.*?\n([\s\S]*?)(?=\*\*HYDRATION\*\*|\*\*TOTAL DAILY CALORIES\*\*|$)/i);
      if (!snacksMatch) {
        snacksMatch = analysisText.match(/SNACKS.*?\n([\s\S]*?)(?=HYDRATION|TOTAL DAILY CALORIES|$)/i);
      }
      if (snacksMatch && snacksMatch[1]) {
        mealPlan.snacks = this.extractListItems(snacksMatch[1]);
        console.log('Extracted snacks:', mealPlan.snacks);
      }

      // Extract hydration
      let hydrationMatch = analysisText.match(/\*\*HYDRATION\*\*.*?\n([\s\S]*?)(?=\*\*TOTAL DAILY CALORIES\*\*|$)/i);
      if (!hydrationMatch) {
        hydrationMatch = analysisText.match(/HYDRATION.*?\n([\s\S]*?)(?=TOTAL DAILY CALORIES|$)/i);
      }
      if (hydrationMatch && hydrationMatch[1]) {
        mealPlan.hydration = this.extractListItems(hydrationMatch[1]);
        console.log('Extracted hydration:', mealPlan.hydration);
      }

      // Extract total calories - look for multiple patterns
      let caloriesMatch = analysisText.match(/\*\*TOTAL DAILY CALORIES\*\*:\s*(\d+)/i);
      if (!caloriesMatch) {
        caloriesMatch = analysisText.match(/TOTAL DAILY CALORIES:\s*(\d+)/i);
      }
      if (!caloriesMatch) {
        caloriesMatch = analysisText.match(/TOTAL.*?CALORIES.*?(\d+)/i);
      }
      if (caloriesMatch && caloriesMatch[1]) {
        mealPlan.totalCalories = parseInt(caloriesMatch[1]);
        console.log('Extracted total calories:', mealPlan.totalCalories);
      }

      // Extract notes
      let notesMatch = analysisText.match(/\*\*NOTES\*\*:\s*\n([\s\S]*?)(?=\n\n|$)/i);
      if (!notesMatch) {
        notesMatch = analysisText.match(/NOTES:\s*\n([\s\S]*?)(?=\n\n|$)/i);
      }
      if (notesMatch && notesMatch[1]) {
        mealPlan.notes = notesMatch[1].trim();
        console.log('Extracted notes:', mealPlan.notes);
      }

      // Validate that we extracted at least some meal data
      const hasMealData = mealPlan.breakfast.length > 0 || mealPlan.lunch.length > 0 || 
                         mealPlan.dinner.length > 0 || mealPlan.snacks.length > 0;
      
      if (!hasMealData) {
        console.warn('No meal data extracted, using fallback');
        return this.getFallbackMealPlan();
      }

      console.log('Final meal plan:', mealPlan);
      return mealPlan;
    } catch (error) {
      console.error('Error extracting meal plan:', error);
      return this.getFallbackMealPlan();
    }
  }

  private getFallbackMealPlan(): any {
    return {
      breakfast: ['Oatmeal with berries and nuts (~300 cal)', 'Greek yogurt with granola (~250 cal)', 'Whole grain toast with avocado (~280 cal)'],
      lunch: ['Grilled chicken salad with mixed greens (~400 cal)', 'Quinoa bowl with vegetables (~350 cal)', 'Turkey sandwich on whole grain bread (~380 cal)'],
      dinner: ['Salmon with quinoa and roasted vegetables (~500 cal)', 'Lean beef stir-fry with brown rice (~450 cal)', 'Vegetarian pasta with tomato sauce (~420 cal)'],
      snacks: ['Apple with almond butter (~200 cal)', 'Carrot sticks with hummus (~150 cal)', 'Mixed nuts (~180 cal)'],
      hydration: ['8-10 glasses of water', 'Herbal tea', 'Green tea'],
      totalCalories: 1800,
      notes: 'Balanced meal plan with adequate protein, complex carbohydrates, and healthy fats. Remember to stay hydrated throughout the day.'
    };
  }



  private calculateDefaultHealthScore(nutritionalData: NutritionalData): number {
    const { total_calories, total_protein, total_carbs, total_fats } = nutritionalData;
    
    if (total_calories === 0) return 5;

    const proteinPercent = (total_protein * 4 / total_calories) * 100;
    const carbPercent = (total_carbs * 4 / total_calories) * 100;
    const fatPercent = (total_fats * 9 / total_calories) * 100;

    let score = 5; // Base score

    // Protein balance (ideal: 10-35%)
    if (proteinPercent >= 10 && proteinPercent <= 35) score += 2;
    else if (proteinPercent >= 5 && proteinPercent <= 40) score += 1;

    // Carb balance (ideal: 45-65%)
    if (carbPercent >= 45 && carbPercent <= 65) score += 2;
    else if (carbPercent >= 35 && carbPercent <= 75) score += 1;

    // Fat balance (ideal: 20-35%)
    if (fatPercent >= 20 && fatPercent <= 35) score += 1;
    else if (fatPercent > 40) score -= 1;

    return Math.max(1, Math.min(10, score));
  }

  async healthCheck(): Promise<{ status: string; available: boolean; error?: string }> {
    if (!this.apiKey || !this.groqModel) {
      return {
        status: 'unavailable',
        available: false,
        error: 'GROQ_API_KEY not configured'
      };
    }

    try {
      // Test the GROQ model with a simple prompt
      await this.groqModel.invoke([
        new AIMessage({ content: 'Test' })
      ]);

      return {
        status: 'healthy',
        available: true
      };
    } catch (error) {
      return {
        status: 'error',
        available: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
}
