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
  private requestQueue: Array<() => Promise<any>> = [];
  private isProcessing = false;
  private lastRequestTime = 0;
  private readonly MIN_REQUEST_INTERVAL = 1000; // 1 second between requests

  private constructor() {
    this.apiKey = process.env['GROQ_API_KEY'] || '';
    if (!this.apiKey) {
      console.warn('GROQ_API_KEY not found in environment variables');
    } else {
      this.groqModel = new ChatGroq({
        apiKey: this.apiKey,
        modelName: 'llama-3.3-70b-versatile',
        temperature: 0.7,
        maxTokens: 1500,
      });
    }
  }

  public static getInstance(): DietChatService {
    if (!DietChatService.instance) {
      DietChatService.instance = new DietChatService();
    }
    return DietChatService.instance;
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
        }
      }
    }

    this.isProcessing = false;
  }

  async answerDietQuery(query: DietQuery): Promise<DietResponse> {
    if (!this.apiKey || !this.groqModel) {
      return {
        answer: 'Diet chat service is not configured. Please set GROQ_API_KEY.',
        suggestions: ['Check your API configuration'],
        relatedTopics: ['API Setup'],
        confidence: 0
      };
    }

    return this.rateLimitedRequest(async () => {
      try {
        const prompt = this.buildDietQueryPrompt(query);
        
        const promptTemplate = ChatPromptTemplate.fromTemplate(`
          You are an expert nutritionist and dietitian. Answer the user's question about diet and nutrition.

          {prompt}
        `);

        const messages = await promptTemplate.formatMessages({
          prompt: prompt
        });
        
        const response = await this.groqModel!.invoke(messages);
        const answerText = response.content as string;
        
        if (!answerText) {
          throw new Error('No response from GROQ API');
        }
        
        return this.parseDietResponse(answerText, query.question);

      } catch (error) {
        console.error('Diet query failed:', error);
        return {
          answer: 'I apologize, but I encountered an error while processing your question. Please try again.',
          suggestions: ['Try rephrasing your question', 'Check your internet connection'],
          relatedTopics: ['General Nutrition', 'Healthy Eating'],
          confidence: 0.3
        };
      }
    });
  }

  private buildDietQueryPrompt(query: DietQuery): string {
    const { question, context, userHistory } = query;
    
    let prompt = `Question: ${question}\n\n`;
    
    if (context) {
      prompt += `Context: ${context}\n\n`;
    }
    
    if (userHistory && userHistory.length > 0) {
      prompt += `Previous questions: ${userHistory.slice(-3).join(', ')}\n\n`;
    }
    
    prompt += `Please provide a comprehensive answer in this format:

ANSWER: [Provide a detailed, helpful response to the question]

SUGGESTIONS: [List 2-3 actionable suggestions related to the question]

RELATED TOPICS: [List 2-3 related nutrition topics the user might be interested in]

CONFIDENCE: [Rate your confidence in this answer from 0.1 to 1.0]

Be informative, practical, and encouraging. Focus on evidence-based nutrition advice.`;

    return prompt;
  }

  private parseDietResponse(responseText: string, _originalQuestion: string): DietResponse {
    try {
      // Extract answer
      const answerMatch = responseText.match(/ANSWER:\s*(.*?)(?=\nSUGGESTIONS:|$)/s);
      const answer = answerMatch && answerMatch[1] ? answerMatch[1].trim() : responseText.substring(0, 200) + '...';

      // Extract suggestions
      const suggestionsMatch = responseText.match(/SUGGESTIONS:\s*(.*?)(?=\nRELATED TOPICS:|$)/s);
      const suggestions = suggestionsMatch && suggestionsMatch[1] 
        ? this.extractListItems(suggestionsMatch[1])
        : ['Consider consulting a nutritionist', 'Focus on balanced meals'];

      // Extract related topics
      const topicsMatch = responseText.match(/RELATED TOPICS:\s*(.*?)(?=\nCONFIDENCE:|$)/s);
      const relatedTopics = topicsMatch && topicsMatch[1]
        ? this.extractListItems(topicsMatch[1])
        : ['General Nutrition', 'Healthy Eating'];

      // Extract confidence
      const confidenceMatch = responseText.match(/CONFIDENCE:\s*([0-9]*\.?[0-9]+)/);
      const confidence = confidenceMatch && confidenceMatch[1] 
        ? parseFloat(confidenceMatch[1])
        : 0.8;

      return {
        answer,
        suggestions,
        relatedTopics,
        confidence: Math.max(0.1, Math.min(1.0, confidence))
      };

    } catch (error) {
      console.error('Error parsing diet response:', error);
      return {
        answer: responseText.substring(0, 300) + '...',
        suggestions: ['Consider consulting a nutritionist'],
        relatedTopics: ['General Nutrition'],
        confidence: 0.5
      };
    }
  }

  private extractListItems(text: string): string[] {
    const items: string[] = [];
    const lines = text.split('\n');
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('-') || trimmed.startsWith('•') || trimmed.match(/^\d+\./)) {
        const item = trimmed.replace(/^[-•\d\.\s]+/, '').trim();
        if (item && item.length > 5) {
          items.push(item);
        }
      } else if (trimmed.length > 10 && !trimmed.includes(':')) {
        items.push(trimmed);
      }
    }
    
    return items.length > 0 ? items : ['Focus on balanced nutrition'];
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
      await this.groqModel.invoke([
        { content: 'Test diet query' } as any
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

  // Get sample questions for the chat interface
  getSampleQuestions(): string[] {
    return [
      "What should I eat to lose weight healthily?",
      "How can I build muscle through diet?",
      "What foods are good for heart health?",
      "How much protein do I need daily?",
      "What's the best diet for diabetes?",
      "How can I improve my gut health?",
      "What should I eat before and after workouts?",
      "How can I reduce inflammation through diet?",
      "What are the best sources of omega-3?",
      "How can I boost my immune system with food?"
    ];
  }
}
