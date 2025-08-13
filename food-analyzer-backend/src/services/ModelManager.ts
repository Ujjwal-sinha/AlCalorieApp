import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import type { AIModel, ModelStatus, ModelConfig } from '../types';

export class ModelManager {
  private static instance: ModelManager;
  private models: Map<string, AIModel> = new Map();
  private modelConfigs: Map<string, ModelConfig> = new Map();
  private isInitialized = false;

  private constructor() {
    this.initializeModelConfigs();
  }

  public static getInstance(): ModelManager {
    if (!ModelManager.instance) {
      ModelManager.instance = new ModelManager();
    }
    return ModelManager.instance;
  }

  private initializeModelConfigs(): void {
    // Initialize model configurations
    const configs: ModelConfig[] = [
      {
        name: 'yolo',
        type: 'detection',
        enabled: true,
        confidence_threshold: 0.5,
        model_path: 'yolov8n.pt'
      },
      {
        name: 'vit',
        type: 'vision',
        enabled: true,
        confidence_threshold: 0.6,
        model_path: 'vit_model'
      },
      {
        name: 'swin',
        type: 'vision',
        enabled: true,
        confidence_threshold: 0.6,
        model_path: 'swin_model'
      },
      {
        name: 'blip',
        type: 'vision',
        enabled: true,
        confidence_threshold: 0.5,
        model_path: 'blip_model'
      },
      {
        name: 'clip',
        type: 'vision',
        enabled: true,
        confidence_threshold: 0.6,
        model_path: 'clip_model'
      },
      {
        name: 'llm',
        type: 'language',
        enabled: true,
        confidence_threshold: 0.7,
        api_endpoint: process.env.GROQ_API_ENDPOINT || 'https://api.groq.com'
      }
    ];

    configs.forEach(config => {
      this.modelConfigs.set(config.name, config);
    });
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.log('ModelManager already initialized');
      return;
    }

    console.log('Initializing ModelManager...');
    
    try {
      // Initialize models in parallel with timeout and retry logic
      const initPromises = Array.from(this.modelConfigs.values())
        .filter(config => config.enabled)
        .map(config => this.loadModelWithRetry(config));

      await Promise.allSettled(initPromises);
      
      this.isInitialized = true;
      console.log('ModelManager initialization completed');
      this.logModelStatus();
    } catch (error) {
      console.error('ModelManager initialization failed:', error);
      throw error;
    }
  }

  private async loadModelWithRetry(config: ModelConfig, maxRetries = 3): Promise<void> {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Loading model ${config.name} (attempt ${attempt}/${maxRetries})`);
        
        const model = await this.loadModel(config);
        this.models.set(config.name, model);
        
        console.log(`Successfully loaded model: ${config.name}`);
        return;
      } catch (error) {
        console.warn(`Failed to load model ${config.name} (attempt ${attempt}/${maxRetries}):`, error);
        
        if (attempt === maxRetries) {
          console.error(`Failed to load model ${config.name} after ${maxRetries} attempts`);
          // Create a placeholder model to prevent crashes
          this.models.set(config.name, {
            name: config.name,
            type: config.type,
            loaded: false,
            config
          });
        } else {
          // Wait before retrying
          await this.delay(1000 * attempt);
        }
      }
    }
  }

  private async loadModel(config: ModelConfig): Promise<AIModel> {
    const startTime = Date.now();
    
    try {
      switch (config.type) {
        case 'detection':
          return await this.loadDetectionModel(config);
        case 'vision':
          return await this.loadVisionModel(config);
        case 'language':
          return await this.loadLanguageModel(config);
        default:
          throw new Error(`Unknown model type: ${config.type}`);
      }
    } catch (error) {
      console.error(`Error loading model ${config.name}:`, error);
      throw error;
    }
  }

  private async loadDetectionModel(config: ModelConfig): Promise<AIModel> {
    // For YOLO and other detection models
    if (config.name === 'yolo') {
      return await this.loadYOLOModel(config);
    }
    
    throw new Error(`Unknown detection model: ${config.name}`);
  }

  private async loadVisionModel(config: ModelConfig): Promise<AIModel> {
    // For Vision Transformer, Swin, BLIP, CLIP models
    switch (config.name) {
      case 'vit':
        return await this.loadViTModel(config);
      case 'swin':
        return await this.loadSwinModel(config);
      case 'blip':
        return await this.loadBLIPModel(config);
      case 'clip':
        return await this.loadCLIPModel(config);
      default:
        throw new Error(`Unknown vision model: ${config.name}`);
    }
  }

  private async loadLanguageModel(config: ModelConfig): Promise<AIModel> {
    // For LLM models like Groq
    if (config.name === 'llm') {
      return await this.loadLLMModel(config);
    }
    
    throw new Error(`Unknown language model: ${config.name}`);
  }

  private async loadYOLOModel(config: ModelConfig): Promise<AIModel> {
    try {
      // Check if YOLO model file exists
      const modelPath = config.model_path;
      if (modelPath && !fs.existsSync(modelPath)) {
        console.warn(`YOLO model file not found: ${modelPath}`);
        return {
          name: config.name,
          type: config.type,
          loaded: false,
          config
        };
      }

      // In a real implementation, this would load the actual YOLO model
      // For now, we'll simulate the loading process
      console.log(`Loading YOLO model from: ${modelPath}`);
      
      // Simulate model loading time
      await this.delay(2000);
      
      return {
        name: config.name,
        type: config.type,
        loaded: true,
        config,
        instance: { type: 'yolo_simulation' }
      };
    } catch (error) {
      console.error('YOLO model loading failed:', error);
      throw error;
    }
  }

  private async loadViTModel(config: ModelConfig): Promise<AIModel> {
    try {
      console.log(`Loading Vision Transformer model: ${config.name}`);
      
      // Simulate model loading time
      await this.delay(1500);
      
      return {
        name: config.name,
        type: config.type,
        loaded: true,
        config,
        instance: { type: 'vit_simulation' }
      };
    } catch (error) {
      console.error('ViT model loading failed:', error);
      throw error;
    }
  }

  private async loadSwinModel(config: ModelConfig): Promise<AIModel> {
    try {
      console.log(`Loading Swin Transformer model: ${config.name}`);
      
      // Simulate model loading time
      await this.delay(1800);
      
      return {
        name: config.name,
        type: config.type,
        loaded: true,
        config,
        instance: { type: 'swin_simulation' }
      };
    } catch (error) {
      console.error('Swin model loading failed:', error);
      throw error;
    }
  }

  private async loadBLIPModel(config: ModelConfig): Promise<AIModel> {
    try {
      console.log(`Loading BLIP model: ${config.name}`);
      
      // Simulate model loading time
      await this.delay(1200);
      
      return {
        name: config.name,
        type: config.type,
        loaded: true,
        config,
        instance: { type: 'blip_simulation' }
      };
    } catch (error) {
      console.error('BLIP model loading failed:', error);
      throw error;
    }
  }

  private async loadCLIPModel(config: ModelConfig): Promise<AIModel> {
    try {
      console.log(`Loading CLIP model: ${config.name}`);
      
      // Simulate model loading time
      await this.delay(1000);
      
      return {
        name: config.name,
        type: config.type,
        loaded: true,
        config,
        instance: { type: 'clip_simulation' }
      };
    } catch (error) {
      console.error('CLIP model loading failed:', error);
      throw error;
    }
  }

  private async loadLLMModel(config: ModelConfig): Promise<AIModel> {
    try {
      console.log(`Loading LLM model: ${config.name}`);
      
      // Check for API key
      const apiKey = process.env.GROQ_API_KEY;
      if (!apiKey) {
        console.warn('GROQ_API_KEY not found, LLM model will not be available');
        return {
          name: config.name,
          type: config.type,
          loaded: false,
          config
        };
      }
      
      // Simulate model loading time
      await this.delay(500);
      
      return {
        name: config.name,
        type: config.type,
        loaded: true,
        config,
        instance: { 
          type: 'llm_simulation',
          apiKey: apiKey.substring(0, 10) + '...' // Log partial key for debugging
        }
      };
    } catch (error) {
      console.error('LLM model loading failed:', error);
      throw error;
    }
  }

  getModel(name: string): AIModel | undefined {
    return this.models.get(name);
  }

  getAvailableModels(): string[] {
    return Array.from(this.models.values())
      .filter(model => model.loaded)
      .map(model => model.name);
  }

  getModelStatus(): ModelStatus {
    const status: ModelStatus = {};
    
    for (const [name, model] of this.models) {
      status[name] = model.loaded;
    }
    
    return status;
  }

  async reloadModel(name: string): Promise<boolean> {
    try {
      console.log(`Reloading model: ${name}`);
      
      const config = this.modelConfigs.get(name);
      if (!config) {
        throw new Error(`Model config not found: ${name}`);
      }
      
      // Remove existing model
      this.models.delete(name);
      
      // Reload model
      await this.loadModelWithRetry(config);
      
      console.log(`Successfully reloaded model: ${name}`);
      return true;
    } catch (error) {
      console.error(`Failed to reload model ${name}:`, error);
      return false;
    }
  }

  async unloadModel(name: string): Promise<boolean> {
    try {
      console.log(`Unloading model: ${name}`);
      
      const model = this.models.get(name);
      if (!model) {
        console.warn(`Model not found: ${name}`);
        return false;
      }
      
      // Clean up model resources
      if (model.instance && typeof model.instance.cleanup === 'function') {
        await model.instance.cleanup();
      }
      
      this.models.delete(name);
      
      console.log(`Successfully unloaded model: ${name}`);
      return true;
    } catch (error) {
      console.error(`Failed to unload model ${name}:`, error);
      return false;
    }
  }

  getModelConfig(name: string): ModelConfig | undefined {
    return this.modelConfigs.get(name);
  }

  updateModelConfig(name: string, updates: Partial<ModelConfig>): boolean {
    const config = this.modelConfigs.get(name);
    if (!config) {
      return false;
    }
    
    Object.assign(config, updates);
    return true;
  }

  private logModelStatus(): void {
    console.log('\n=== Model Status ===');
    for (const [name, model] of this.models) {
      const status = model.loaded ? '✅ Loaded' : '❌ Failed';
      console.log(`${name}: ${status}`);
    }
    console.log('==================\n');
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Python integration methods for actual model loading
  private async callPythonModelLoader(modelType: string, modelPath?: string): Promise<any> {
    try {
      const pythonScript = path.join(process.cwd(), 'python_models', 'load_models.py');
      
      if (!fs.existsSync(pythonScript)) {
        console.warn(`Python script not found: ${pythonScript}`);
        return null;
      }

      return new Promise((resolve, reject) => {
        const python = spawn('python3', [pythonScript, modelType, modelPath || ''], {
          timeout: 30000 // 30 second timeout
        });
        
        let result = '';
        let error = '';

        python.stdout.on('data', (data: Buffer) => {
          result += data.toString();
        });

        python.stderr.on('data', (data: Buffer) => {
          error += data.toString();
        });

        python.on('close', (code: number) => {
          if (code === 0) {
            try {
              const parsed = JSON.parse(result);
              resolve(parsed);
            } catch (e) {
              console.error(`Failed to parse Python output: ${e}`);
              reject(new Error('Invalid Python output'));
            }
          } else {
            console.error(`Python script failed with code ${code}: ${error}`);
            reject(new Error(`Python script failed: ${error}`));
          }
        });

        python.on('error', (err: Error) => {
          console.error(`Python process error: ${err.message}`);
          reject(err);
        });
      });
    } catch (error) {
      console.error(`Python model loading failed for ${modelType}:`, error);
      throw error;
    }
  }

  // Health check method
  async healthCheck(): Promise<{ healthy: boolean; models: ModelStatus; errors: string[] }> {
    const errors: string[] = [];
    const modelStatus = this.getModelStatus();
    
    // Check if any required models failed to load
    const requiredModels = ['yolo', 'vit'];
    for (const modelName of requiredModels) {
      if (!modelStatus[modelName]) {
        errors.push(`Required model ${modelName} is not loaded`);
      }
    }
    
    const healthy = errors.length === 0;
    
    return {
      healthy,
      models: modelStatus,
      errors
    };
  }

  // Cleanup method
  async cleanup(): Promise<void> {
    console.log('Cleaning up ModelManager...');
    
    const unloadPromises = Array.from(this.models.keys()).map(name => this.unloadModel(name));
    await Promise.allSettled(unloadPromises);
    
    this.models.clear();
    this.isInitialized = false;
    
    console.log('ModelManager cleanup completed');
  }
}