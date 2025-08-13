import dotenv from 'dotenv';

dotenv.config();

export const config = {
  // Server Configuration
  port: parseInt(process.env['PORT'] || '8000', 10),
  host: process.env['HOST'] || '0.0.0.0',
  nodeEnv: process.env['NODE_ENV'] || 'development',
  
  // API Configuration
  apiPrefix: '/api',
  corsOrigin: process.env['CORS_ORIGIN'] || 'http://localhost:5173',
  
  // File Upload Configuration
  maxFileSize: parseInt(process.env['MAX_FILE_SIZE'] || '10485760', 10), // 10MB
  allowedMimeTypes: [
    'image/jpeg',
    'image/jpg', 
    'image/png',
    'image/webp'
  ],
  
  // AI Models Configuration
  models: {
    // Vision Transformer Models
    vit: {
      enabled: process.env['VIT_ENABLED'] !== 'false',
      model_name: 'google/vit-base-patch16-224',
      confidence_threshold: 0.05,
      max_predictions: 10
    },
    
    // Swin Transformer
    swin: {
      enabled: process.env['SWIN_ENABLED'] !== 'false',
      model_name: 'microsoft/swin-base-patch4-window7-224',
      confidence_threshold: 0.05,
      max_predictions: 10
    },
    
    // BLIP Model
    blip: {
      enabled: process.env['BLIP_ENABLED'] !== 'false',
      model_name: 'Salesforce/blip-image-captioning-base',
      max_new_tokens: 50,
      num_beams: 3
    },
    
    // CLIP Model
    clip: {
      enabled: process.env['CLIP_ENABLED'] !== 'false',
      model_name: 'openai/clip-vit-base-patch32',
      similarity_threshold: 0.28
    },
    
    // YOLO Model
    yolo: {
      enabled: process.env['YOLO_ENABLED'] !== 'false',
      model_path: process.env['YOLO_MODEL_PATH'] || 'yolov8n.pt',
      confidence_levels: [0.15, 0.25, 0.35, 0.45],
      iou_threshold: 0.4
    },
    
    // Language Model
    llm: {
      enabled: process.env['LLM_ENABLED'] !== 'false',
      api_key: process.env['GROQ_API_KEY'],
      model_name: process.env['LLM_MODEL'] || 'mixtral-8x7b-32768',
      max_tokens: 1000,
      temperature: 0.7
    }
  },
  
  // Detection Configuration
  detection: {
    confidence_threshold: 0.3,
    ensemble_threshold: 0.6,
    max_detection_time: 15000, // 15 seconds
    fallback_enabled: true,
    use_ensemble: true
  },
  
  // Cache Configuration
  cache: {
    enabled: process.env['CACHE_ENABLED'] !== 'false',
    ttl: parseInt(process.env['CACHE_TTL'] || '3600', 10), // 1 hour
    max_size: parseInt(process.env['CACHE_MAX_SIZE'] || '100', 10)
  },
  
  // Logging Configuration
  logging: {
    level: process.env['LOG_LEVEL'] || 'info',
    format: process.env['LOG_FORMAT'] || 'combined'
  },
  
  // Database Configuration (for future use)
  database: {
    url: process.env['DATABASE_URL'],
    max_connections: parseInt(process.env['DB_MAX_CONNECTIONS'] || '10', 10)
  },
  
  // External APIs
  external_apis: {
    nutrition_api: {
      enabled: process.env['NUTRITION_API_ENABLED'] === 'true',
      url: process.env['NUTRITION_API_URL'],
      api_key: process.env['NUTRITION_API_KEY']
    },
    
    food_database: {
      enabled: process.env['FOOD_DB_ENABLED'] === 'true',
      url: process.env['FOOD_DB_URL'],
      api_key: process.env['FOOD_DB_API_KEY']
    }
  }
};

export default config;