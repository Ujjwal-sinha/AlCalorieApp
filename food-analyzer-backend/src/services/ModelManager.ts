import type { ModelStatus } from '../types';

export class ModelManager {
  private static instance: ModelManager;
  private models: Map<string, any> = new Map();
  private modelStatus: ModelStatus = {
    'BLIP (Image Analysis)': false,
    'ViT-B/16 (Vision Transformer)': false,
    'Swin Transformer': false,
    'CLIP (Similarity Scoring)': false,
    'LLM (Nutrition Analysis)': false,
    'YOLO (Object Detection)': false,
    'CNN (Visualizations)': false
  };

  private constructor() {}

  public static getInstance(): ModelManager {
    if (!ModelManager.instance) {
      ModelManager.instance = new ModelManager();
    }
    return ModelManager.instance;
  }

  async initialize(): Promise<void> {
    console.log('Initializing ModelManager...');
    
    try {
      // Initialize models with retry logic
      await this.loadModelsWithRetry();
      console.log('ModelManager initialized successfully');
    } catch (error) {
      console.error('Failed to initialize ModelManager:', error);
      throw error;
    }
  }

  private async loadModelsWithRetry(maxRetries = 3): Promise<void> {
    let lastError: Error | null = null;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Attempt ${attempt}/${maxRetries} to load models...`);
        await this.loadModels();
        return; // Success, exit retry loop
      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Unknown error');
        console.warn(`Model loading attempt ${attempt} failed:`, lastError.message);
        
        if (attempt < maxRetries) {
          // Wait before retrying (exponential backoff)
          const delay = Math.pow(2, attempt) * 1000;
          console.log(`Waiting ${delay}ms before retry...`);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    // All retries failed
    throw lastError || new Error('Failed to load models after all retries');
  }

  private async loadModels(): Promise<void> {
    // Load YOLO model
    try {
      await this.loadYOLOModel();
      this.modelStatus['YOLO (Object Detection)'] = true;
    } catch (error) {
      console.warn('Failed to load YOLO model:', error);
    }

    // Load Vision Transformer models
    try {
      await this.loadViTModel();
      this.modelStatus['ViT-B/16 (Vision Transformer)'] = true;
    } catch (error) {
      console.warn('Failed to load ViT model:', error);
    }

    try {
      await this.loadSwinModel();
      this.modelStatus['Swin Transformer'] = true;
    } catch (error) {
      console.warn('Failed to load Swin model:', error);
    }

    // Load CLIP model
    try {
      await this.loadCLIPModel();
      this.modelStatus['CLIP (Similarity Scoring)'] = true;
    } catch (error) {
      console.warn('Failed to load CLIP model:', error);
    }

    // Load BLIP model
    try {
      await this.loadBLIPModel();
      this.modelStatus['BLIP (Image Analysis)'] = true;
    } catch (error) {
      console.warn('Failed to load BLIP model:', error);
    }

    // Load LLM model
    try {
      await this.loadLLMModel();
      this.modelStatus['LLM (Nutrition Analysis)'] = true;
    } catch (error) {
      console.warn('Failed to load LLM model:', error);
    }

    // Load CNN model
    try {
      await this.loadCNNModel();
      this.modelStatus['CNN (Visualizations)'] = true;
    } catch (error) {
      console.warn('Failed to load CNN model:', error);
    }
  }

  private async loadYOLOModel(): Promise<void> {
    // Simulate YOLO model loading
    await new Promise(resolve => setTimeout(resolve, 100));
    this.models.set('yolo', {
      name: 'YOLO',
      type: 'detection',
      loaded: true,
      config: {
        model_path: 'yolo11m.pt',
        confidence_threshold: 0.25
      }
    });
  }

  private async loadViTModel(): Promise<void> {
    // Simulate ViT model loading
    await new Promise(resolve => setTimeout(resolve, 150));
    this.models.set('vit', {
      name: 'ViT',
      type: 'vision',
      loaded: true,
      config: {
        model_name: 'google/vit-base-patch16-224',
        confidence_threshold: 0.05
      }
    });
  }

  private async loadSwinModel(): Promise<void> {
    // Simulate Swin model loading
    await new Promise(resolve => setTimeout(resolve, 120));
    this.models.set('swin', {
      name: 'Swin',
      type: 'vision',
      loaded: true,
      config: {
        model_name: 'microsoft/swin-base-patch4-window7-224',
        confidence_threshold: 0.05
      }
    });
  }

  private async loadCLIPModel(): Promise<void> {
    // Simulate CLIP model loading
    await new Promise(resolve => setTimeout(resolve, 200));
    this.models.set('clip', {
      name: 'CLIP',
      type: 'vision',
      loaded: true,
      config: {
        model_name: 'openai/clip-vit-base-patch32',
        similarity_threshold: 0.28
      }
    });
  }

  private async loadBLIPModel(): Promise<void> {
    // Simulate BLIP model loading
    await new Promise(resolve => setTimeout(resolve, 180));
    this.models.set('blip', {
      name: 'BLIP',
      type: 'vision',
      loaded: true,
      config: {
        model_name: 'Salesforce/blip-image-captioning-base',
        max_new_tokens: 50
      }
    });
  }

  private async loadLLMModel(): Promise<void> {
    // Simulate LLM model loading
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const apiKey = process.env['GROQ_API_KEY'];
    if (!apiKey) {
      throw new Error('GROQ_API_KEY environment variable is required for LLM model');
    }

    this.models.set('llm', {
      name: 'LLM',
      type: 'language',
      loaded: true,
      config: {
        api_key: apiKey,
        model_name: 'mixtral-8x7b-32768',
        api_endpoint: process.env['GROQ_API_ENDPOINT'] || 'https://api.groq.com'
      }
    });
  }

  private async loadCNNModel(): Promise<void> {
    // Simulate CNN model loading
    await new Promise(resolve => setTimeout(resolve, 80));
    this.models.set('cnn', {
      name: 'CNN',
      type: 'vision',
      loaded: true,
      config: {
        model_name: 'resnet50',
        confidence_threshold: 0.1
      }
    });
  }

  getModel(name: string): any | undefined {
    return this.models.get(name);
  }

  getModelStatus(): ModelStatus {
    return { ...this.modelStatus };
  }

  async reloadModel(modelName: string): Promise<boolean> {
    try {
      console.log(`Reloading model: ${modelName}`);
      
      // Remove existing model
      this.models.delete(modelName);
      
      // Reload based on model type
      switch (modelName.toLowerCase()) {
        case 'yolo':
          await this.loadYOLOModel();
          this.modelStatus['YOLO (Object Detection)'] = true;
          break;
        case 'vit':
          await this.loadViTModel();
          this.modelStatus['ViT-B/16 (Vision Transformer)'] = true;
          break;
        case 'swin':
          await this.loadSwinModel();
          this.modelStatus['Swin Transformer'] = true;
          break;
        case 'clip':
          await this.loadCLIPModel();
          this.modelStatus['CLIP (Similarity Scoring)'] = true;
          break;
        case 'blip':
          await this.loadBLIPModel();
          this.modelStatus['BLIP (Image Analysis)'] = true;
          break;
        case 'llm':
          await this.loadLLMModel();
          this.modelStatus['LLM (Nutrition Analysis)'] = true;
          break;
        case 'cnn':
          await this.loadCNNModel();
          this.modelStatus['CNN (Visualizations)'] = true;
          break;
        default:
          throw new Error(`Unknown model: ${modelName}`);
      }
      
      return true;
    } catch (error) {
      console.error(`Failed to reload model ${modelName}:`, error);
      return false;
    }
  }

  async unloadModel(modelName: string): Promise<boolean> {
    try {
      this.models.delete(modelName);
      
      // Update status
      switch (modelName.toLowerCase()) {
        case 'yolo':
          this.modelStatus['YOLO (Object Detection)'] = false;
          break;
        case 'vit':
          this.modelStatus['ViT-B/16 (Vision Transformer)'] = false;
          break;
        case 'swin':
          this.modelStatus['Swin Transformer'] = false;
          break;
        case 'clip':
          this.modelStatus['CLIP (Similarity Scoring)'] = false;
          break;
        case 'blip':
          this.modelStatus['BLIP (Image Analysis)'] = false;
          break;
        case 'llm':
          this.modelStatus['LLM (Nutrition Analysis)'] = false;
          break;
        case 'cnn':
          this.modelStatus['CNN (Visualizations)'] = false;
          break;
      }
      
      return true;
    } catch (error) {
      console.error(`Failed to unload model ${modelName}:`, error);
      return false;
    }
  }

  getLoadedModels(): string[] {
    return Array.from(this.models.keys());
  }

  isModelLoaded(modelName: string): boolean {
    const model = this.models.get(modelName);
    return model ? model.loaded : false;
  }

  getModelConfig(modelName: string): any {
    const model = this.models.get(modelName);
    return model ? model.config : null;
  }

  async healthCheck(): Promise<{ status: string; models: Record<string, boolean>; pythonAvailable: boolean }> {
    const models = {
      yolo: this.isModelLoaded('yolo'),
      vit: this.isModelLoaded('vit'),
      swin: this.isModelLoaded('swin'),
      blip: this.isModelLoaded('blip'),
      clip: this.isModelLoaded('clip'),
      llm: this.isModelLoaded('llm'),
      cnn: this.isModelLoaded('cnn')
    };

    return {
      status: 'healthy',
      models,
      pythonAvailable: Object.values(models).some(v => v)
    };
  }
}