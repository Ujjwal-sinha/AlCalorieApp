import { ChatGroq } from '@langchain/groq';
import { ChatPromptTemplate } from '@langchain/core/prompts';

interface DietQuery {
  question: string;
  context?: string;
  userHistory?: string[];
}

interface DietResponse {
  answer: string;
  suggestions: string[];
  relatedTopics: string[];
  confidence: number;
}

export class DietChatService {
  private static instance: DietChatService;
  private apiKey: string;
  private groqModel: ChatGroq | null = null;
  private readonly MAX_RETRIES = 3;

  private constructor() {
    this.apiKey = process.env['GROQ_API_KEY'] || '';
    if (!this.apiKey) {
      console.warn('GROQ_API_KEY not found in environment variables');
    } else {
      this.initializeGroqModel();
    }
  }

  public static getInstance(): DietChatService {
    if (!DietChatService.instance) {
      DietChatService.instance = new DietChatService();
    }
    return DietChatService.instance;
  }

  private initializeGroqModel(): void {
    try {
      this.groqModel = new ChatGroq({
        apiKey: this.apiKey,
        modelName: 'llama-3.3-70b-versatile',
        temperature: 0.7,
        maxTokens: 1500,
        timeout: 30000, // 30 second timeout
      });
      console.log('GROQ model initialized successfully');
    } catch (error) {
      console.error('Failed to initialize GROQ model:', error);
      this.groqModel = null;
    }
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
      // Test the GROQ model with a simple prompt using proper format
      const testPrompt = ChatPromptTemplate.fromTemplate('Hello, this is a health check test.');
      const testMessages = await testPrompt.formatMessages({});
      const testResponse = await this.groqModel.invoke(testMessages);

      if (testResponse && testResponse.content) {
        return {
          status: 'healthy',
          available: true
        };
      } else {
        return {
          status: 'error',
          available: false,
          error: 'No response from GROQ API'
        };
      }
    } catch (error) {
      console.error('Health check failed:', error);
      return {
        status: 'error',
        available: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async answerDietQuery(query: DietQuery): Promise<DietResponse> {
    if (!this.apiKey || !this.groqModel) {
      return {
        answer: 'I apologize, but the diet chat service is not properly configured. Please ensure the GROQ API key is set up correctly.',
        suggestions: ['Check your API configuration', 'Verify your internet connection', 'Try again in a few moments'],
        relatedTopics: ['API Setup', 'Technical Support'],
        confidence: 0
      };
    }

    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= this.MAX_RETRIES; attempt++) {
      try {
        // Create a proper prompt template
        const promptTemplate = ChatPromptTemplate.fromTemplate(`
          You are an expert nutritionist, registered dietitian, and certified nutrition coach with over 15 years of experience helping people achieve their health and nutrition goals. You have deep knowledge of:

          - Clinical nutrition and dietary guidelines
          - Sports nutrition and athletic performance
          - Weight management and metabolism
          - Medical nutrition therapy
          - Food science and nutrient bioavailability
          - Meal planning and recipe development
          - Dietary restrictions and special diets
          - Supplement science and evidence-based recommendations

          Your expertise covers all aspects of nutrition including:
          • Macronutrients (proteins, carbohydrates, fats) and their roles
          • Micronutrients (vitamins, minerals, antioxidants)
          • Hydration and electrolyte balance
          • Gut health and microbiome
          • Food allergies and intolerances
          • Chronic disease management through nutrition
          • Performance nutrition for athletes
          • Pediatric and geriatric nutrition
          • Pregnancy and breastfeeding nutrition

          When answering nutrition questions, you should:

          1. **Provide Evidence-Based Information**: Base all recommendations on current scientific research and established nutritional guidelines from organizations like WHO, FDA, USDA, and Academy of Nutrition and Dietetics.

          2. **Be Comprehensive Yet Practical**: Give detailed, actionable advice that people can implement in their daily lives. Include specific food suggestions, portion sizes, and meal timing when relevant.

          3. **Consider Individual Context**: Take into account the person's age, activity level, health status, dietary preferences, and goals when making recommendations.

          4. **Address Safety and Precautions**: Mention any potential risks, contraindications, or when someone should consult a healthcare provider.

          5. **Provide Multiple Options**: Offer various approaches and alternatives to accommodate different preferences, budgets, and dietary restrictions.

          6. **Include Practical Tips**: Share meal prep ideas, cooking techniques, shopping tips, and lifestyle factors that support good nutrition.

          7. **Explain the "Why"**: Help people understand the science behind recommendations so they can make informed decisions.

          8. **Be Encouraging and Supportive**: Use a positive, motivating tone that empowers people to make healthy choices.

          **IMPORTANT: Format your response in clear bullet points using • or - symbols. Make it easy to read and scan. Structure your answer with main points and sub-points where appropriate.**

          Question: {question}
          
          Additional Context: {context}
          
          Please provide a comprehensive, helpful, and accurate response that addresses the question thoroughly. Format your response in clear bullet points for easy reading.
        `);
        
        // Format the messages properly
        const formattedPrompt = await promptTemplate.formatMessages({
          question: query.question,
          context: query.context || 'No additional context provided.'
        });
        
        // Invoke the model with proper message format
        const response = await this.groqModel!.invoke(formattedPrompt);
        
        const answerText = response.content as string;
        
        if (!answerText) {
          throw new Error('No response content from GROQ API');
        }
        
        return this.parseDietResponse(answerText, query.question);

      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Unknown error');
        console.error(`Diet query attempt ${attempt} failed:`, error);
        
        if (attempt < this.MAX_RETRIES) {
          // Wait before retrying (exponential backoff)
          await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        }
      }
    }

    // All retries failed
    console.error('All retry attempts failed for diet query');
    return this.generateFallbackResponse(query.question, lastError);
  }


  private parseDietResponse(responseText: string, originalQuestion: string): DietResponse {
    try {
      // Clean and format the response text
      let formattedResponse = responseText.trim();
      
      // Ensure bullet points are properly formatted
      formattedResponse = formattedResponse
        .replace(/^[-•*]\s*/gm, '• ') // Convert any bullet styles to •
        .replace(/\n\s*[-•*]\s*/g, '\n• ') // Convert inline bullets
        .replace(/\n{3,}/g, '\n\n') // Remove excessive line breaks
        .replace(/\s+$/gm, ''); // Remove trailing spaces
      
      // Extract the main answer (first few paragraphs for comprehensive responses)
      const paragraphs = formattedResponse.split('\n\n').filter(p => p.trim());
      const answer = paragraphs.slice(0, 4).join('\n\n') || formattedResponse.substring(0, 600) + '...';

      // Generate context-aware suggestions based on the response and question
      const suggestions = this.generateEnhancedSuggestions(responseText, originalQuestion);

      // Generate more comprehensive related topics
      const relatedTopics = this.generateEnhancedRelatedTopics(originalQuestion, responseText);

      // Calculate confidence based on response quality and comprehensiveness
      const confidence = this.calculateEnhancedConfidence(responseText, originalQuestion);

      return {
        answer: answer.trim(),
        suggestions,
        relatedTopics,
        confidence
      };
    } catch (error) {
      console.error('Error parsing diet response:', error);
      return {
        answer: responseText.substring(0, 400) + '...',
        suggestions: ['Try asking a more specific question', 'Consider your dietary goals and preferences'],
        relatedTopics: ['Nutrition Basics', 'Healthy Eating', 'Diet Planning'],
        confidence: 0.7
      };
    }
  }

  private generateEnhancedSuggestions(responseText: string, question: string): string[] {
    const suggestions: string[] = [];
    const questionLower = question.toLowerCase();
    const responseLower = responseText.toLowerCase();
    
    // Weight management suggestions
    if (questionLower.includes('weight') || questionLower.includes('lose') || questionLower.includes('gain')) {
      suggestions.push('Track your daily calorie intake and macronutrients');
      suggestions.push('Focus on whole, unprocessed foods and vegetables');
      suggestions.push('Stay consistent with your exercise routine and sleep schedule');
      suggestions.push('Consider working with a registered dietitian for personalized guidance');
    }
    
    // Protein-related suggestions
    else if (questionLower.includes('protein') || questionLower.includes('muscle')) {
      suggestions.push('Include lean protein sources in every meal (chicken, fish, legumes)');
      suggestions.push('Aim for 0.8-1.2g protein per kg body weight daily');
      suggestions.push('Consider plant-based protein options for variety');
      suggestions.push('Time protein intake around workouts for optimal muscle recovery');
    }
    
    // Vitamin and mineral suggestions
    else if (questionLower.includes('vitamin') || questionLower.includes('mineral') || questionLower.includes('nutrient')) {
      suggestions.push('Eat a variety of colorful fruits and vegetables daily');
      suggestions.push('Consider a multivitamin if you have dietary restrictions');
      suggestions.push('Get regular blood work to check your nutrient levels');
      suggestions.push('Focus on food sources over supplements when possible');
    }
    
    // Gut health suggestions
    else if (questionLower.includes('gut') || questionLower.includes('digest') || questionLower.includes('microbiome')) {
      suggestions.push('Include probiotic-rich foods like yogurt, kefir, and sauerkraut');
      suggestions.push('Eat plenty of fiber from fruits, vegetables, and whole grains');
      suggestions.push('Stay hydrated and manage stress levels');
      suggestions.push('Consider eliminating trigger foods if you have sensitivities');
    }
    
    // Meal planning suggestions
    else if (questionLower.includes('meal') || questionLower.includes('plan') || questionLower.includes('breakfast') || questionLower.includes('dinner')) {
      suggestions.push('Plan your meals and snacks ahead of time');
      suggestions.push('Prep ingredients in advance to save time during the week');
      suggestions.push('Keep healthy staples stocked in your pantry and freezer');
      suggestions.push('Use a meal planning app or template to stay organized');
    }
    
    // General nutrition suggestions
    else {
      suggestions.push('Maintain a balanced diet with variety and moderation');
      suggestions.push('Stay hydrated throughout the day with water and herbal teas');
      suggestions.push('Listen to your body\'s hunger and fullness cues');
      suggestions.push('Make gradual, sustainable changes rather than drastic diets');
    }

    // Add context-specific suggestions based on response content
    if (responseLower.includes('supplement')) {
      suggestions.push('Consult with a healthcare provider before starting any supplements');
    }
    
    if (responseLower.includes('exercise') || responseLower.includes('workout')) {
      suggestions.push('Combine good nutrition with regular physical activity');
    }
    
    if (responseLower.includes('sleep') || responseLower.includes('rest')) {
      suggestions.push('Prioritize quality sleep for optimal health and recovery');
    }

    return suggestions.slice(0, 4);
  }

  private generateEnhancedRelatedTopics(question: string, responseText: string): string[] {
    const topics: string[] = [];
    const questionLower = question.toLowerCase();
    const responseLower = responseText.toLowerCase();
    
    // Weight and body composition topics
    if (questionLower.includes('weight') || questionLower.includes('lose') || questionLower.includes('gain')) {
      topics.push('Weight Management', 'Calorie Counting', 'Metabolism', 'Body Composition');
    }
    
    // Macronutrient topics
    else if (questionLower.includes('protein') || questionLower.includes('carb') || questionLower.includes('fat')) {
      topics.push('Macronutrients', 'Protein Sources', 'Carbohydrate Timing', 'Healthy Fats');
    }
    
    // Micronutrient topics
    else if (questionLower.includes('vitamin') || questionLower.includes('mineral') || questionLower.includes('nutrient')) {
      topics.push('Vitamins & Minerals', 'Nutrient Absorption', 'Supplements', 'Antioxidants');
    }
    
    // Gut health topics
    else if (questionLower.includes('gut') || questionLower.includes('digest') || questionLower.includes('microbiome')) {
      topics.push('Gut Health', 'Digestive Health', 'Probiotics', 'Fiber');
    }
    
    // Meal planning topics
    else if (questionLower.includes('meal') || questionLower.includes('plan') || questionLower.includes('breakfast')) {
      topics.push('Meal Planning', 'Healthy Recipes', 'Portion Control', 'Food Prep');
    }
    
    // Performance and sports topics
    else if (questionLower.includes('muscle') || questionLower.includes('recovery') || questionLower.includes('athlete')) {
      topics.push('Sports Nutrition', 'Muscle Building', 'Recovery Nutrition', 'Performance');
    }
    
    // Health condition topics
    else if (questionLower.includes('diabetes') || questionLower.includes('heart') || questionLower.includes('blood pressure')) {
      topics.push('Medical Nutrition Therapy', 'Chronic Disease Management', 'Heart Health', 'Blood Sugar Control');
    }
    
    // General topics
    else {
      topics.push('Nutrition Basics', 'Healthy Eating', 'Wellness Tips', 'Lifestyle Medicine');
    }

    // Add context-specific topics based on response content
    if (responseLower.includes('supplement')) {
      topics.push('Supplementation', 'Nutrient Deficiencies');
    }
    
    if (responseLower.includes('exercise') || responseLower.includes('workout')) {
      topics.push('Exercise Nutrition', 'Pre/Post Workout Nutrition');
    }
    
    if (responseLower.includes('sleep') || responseLower.includes('stress')) {
      topics.push('Sleep & Recovery', 'Stress Management');
    }
    
    if (responseLower.includes('budget') || responseLower.includes('cost')) {
      topics.push('Budget-Friendly Nutrition', 'Affordable Healthy Eating');
    }

    return topics.slice(0, 5);
  }

  private calculateEnhancedConfidence(responseText: string, question: string): number {
    // Base confidence score
    let confidence = 0.6;
    
    // Length and comprehensiveness
    const length = responseText.length;
    if (length > 200) confidence += 0.1;
    if (length > 400) confidence += 0.1;
    if (length > 600) confidence += 0.1;
    
    // Content quality indicators
    const hasSpecificInfo = responseText.includes('gram') || responseText.includes('calorie') || responseText.includes('vitamin') || responseText.includes('protein');
    const hasPracticalTips = responseText.includes('try') || responseText.includes('include') || responseText.includes('add') || responseText.includes('eat');
    const hasScientificInfo = responseText.includes('research') || responseText.includes('study') || responseText.includes('evidence') || responseText.includes('guidelines');
    const hasSafetyInfo = responseText.includes('consult') || responseText.includes('doctor') || responseText.includes('healthcare') || responseText.includes('precaution');
    
    if (hasSpecificInfo) confidence += 0.1;
    if (hasPracticalTips) confidence += 0.1;
    if (hasScientificInfo) confidence += 0.1;
    if (hasSafetyInfo) confidence += 0.05;
    
    // Question-specific confidence adjustments
    const questionLower = question.toLowerCase();
    if (questionLower.includes('weight') && responseText.includes('calorie')) confidence += 0.05;
    if (questionLower.includes('protein') && responseText.includes('gram')) confidence += 0.05;
    if (questionLower.includes('vitamin') && responseText.includes('food')) confidence += 0.05;
    
    // Ensure confidence is within reasonable bounds
    return Math.min(0.95, Math.max(0.4, confidence));
  }

  private generateFallbackResponse(_question: string, _error: Error | null): DietResponse {
    const fallbackResponses: DietResponse[] = [
      {
        answer: "I understand you're asking about nutrition, but I'm having trouble connecting to my knowledge base right now. This could be due to a temporary network issue or high demand. In the meantime, here are some general nutrition principles you can follow:\n\n• Focus on whole, unprocessed foods\n• Eat plenty of colorful fruits and vegetables\n• Stay hydrated throughout the day\n• Maintain a balanced diet with adequate protein, healthy fats, and complex carbohydrates\n• Listen to your body's hunger and fullness cues",
        suggestions: [
          'Try asking your question again in a moment',
          'Check your internet connection',
          'Consider rephrasing your question',
          'Focus on whole, unprocessed foods in the meantime'
        ],
        relatedTopics: ['Nutrition Basics', 'Healthy Eating', 'Whole Foods'],
        confidence: 0.4
      },
      {
        answer: "I'm experiencing some technical difficulties at the moment. While I can't provide a detailed response right now, I'd be happy to help once the connection is restored. For general nutrition guidance, remember:\n\n• Eat a variety of nutrient-dense foods\n• Prioritize vegetables and fruits in your meals\n• Include lean proteins with every meal\n• Stay hydrated throughout the day\n• Choose whole grains over refined grains",
        suggestions: [
          'Wait a few minutes and try again',
          'Ensure you have a stable internet connection',
          'Try a simpler question format',
          'Focus on balanced meals with protein, carbs, and healthy fats'
        ],
        relatedTopics: ['Balanced Nutrition', 'Healthy Eating', 'Nutrient-Dense Foods'],
        confidence: 0.3
      },
      {
        answer: "I apologize for the connection issue. As a temporary solution, here are some evidence-based nutrition tips:\n\n• Aim for 5-9 servings of fruits and vegetables daily\n• Include protein with every meal\n• Choose whole grains over refined grains\n• Limit added sugars and processed foods\n• These principles form the foundation of good nutrition",
        suggestions: [
          'Try refreshing the page and asking again',
          'Check if your internet connection is stable',
          'Consider asking a more specific question',
          'Apply these basic nutrition principles to your daily eating'
        ],
        relatedTopics: ['Evidence-Based Nutrition', 'Whole Foods', 'Balanced Diet'],
        confidence: 0.5
      }
    ];

    const randomIndex = Math.floor(Math.random() * fallbackResponses.length);
    return fallbackResponses[randomIndex]!;
  }

  async getSampleQuestions(): Promise<string[]> {
    return [
      "What should I eat to lose weight healthily?",
      "How much protein do I need daily?",
      "What are the best sources of vitamin D?",
      "How can I improve my gut health?",
      "What's a good breakfast for energy?",
      "How do I read nutrition labels?",
      "What foods help with muscle recovery?",
      "How can I reduce my sugar intake?",
      "What's the best diet for heart health?",
      "How do I plan healthy meals for the week?"
    ];
  }
}
