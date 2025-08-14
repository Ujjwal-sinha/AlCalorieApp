export const APP_CONFIG = {
  // App Information
  name: 'AI Calorie Analyzer',
  version: '1.0.0',
  description: 'Advanced food recognition and nutritional analysis',
  
  // API Configuration
  api: {
    baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
    timeout: 45000, // Increased from 30000 to 45000 (45 seconds)
    retries: 3,
    retryDelay: 1000,
  },
  
  // Feature Flags
  features: {
    enableTensorFlow: true,
    enableOfflineMode: false,
    enableCameraCapture: true,
    enableExport: true,
    enableSocialSharing: false,
    enableAdvancedDetection: true,
    enableModelSelection: true,
  },
  
  // UI Configuration
  ui: {
    maxImageSize: 10 * 1024 * 1024, // 10MB
    supportedImageTypes: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
    maxHistoryEntries: 100,
    defaultChartColors: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9CA24', '#6C5CE7'],
    uploadTimeout: 45000, // Increased from 30000 to 45000 (45 seconds)
  },
  
  // Analysis Configuration
  analysis: {
    confidenceThreshold: 0.3,
    ensembleThreshold: 0.6,
    maxDetectionTime: 30000, // Increased from 15000 to 30000 (30 seconds)
    fallbackEnabled: true,
    enableProgressTracking: true,
  },
  
  // Storage Configuration
  storage: {
    historyKey: 'food-analyzer-history',
    settingsKey: 'food-analyzer-settings',
    cacheKey: 'food-analyzer-cache',
    maxCacheSize: 50 * 1024 * 1024, // 50MB
  },
  
  // Nutrition Defaults
  nutrition: {
    caloriesPerGramProtein: 4,
    caloriesPerGramCarbs: 4,
    caloriesPerGramFat: 9,
    defaultPortionSize: 100, // grams
  },
  
  // Chart Configuration
  charts: {
    animationDuration: 300,
    colors: {
      protein: '#FF6B6B',
      carbs: '#4ECDC4',
      fats: '#45B7D1',
      calories: '#F9CA24',
    },
  },

  // Model Configuration
  models: {
    available: ['yolo', 'vit', 'swin', 'blip', 'clip', 'llm'],
    default: 'ensemble',
    fallback: 'simulated',
  },
};

export const FOOD_CATEGORIES = {
  proteins: ['chicken', 'beef', 'pork', 'fish', 'egg', 'tofu', 'beans', 'lentils'],
  vegetables: ['tomato', 'potato', 'carrot', 'broccoli', 'spinach', 'lettuce', 'onion'],
  fruits: ['apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'mango'],
  grains: ['rice', 'bread', 'pasta', 'quinoa', 'oats', 'cereal', 'wheat'],
  dairy: ['cheese', 'milk', 'yogurt', 'butter', 'cream', 'ice cream'],
  prepared: ['pizza', 'burger', 'sandwich', 'salad', 'soup', 'curry', 'stir fry'],
  snacks: ['chips', 'nuts', 'crackers', 'popcorn', 'pretzels', 'cookies'],
  desserts: ['cake', 'cookie', 'ice cream', 'chocolate', 'pie', 'donut'],
  beverages: ['coffee', 'tea', 'juice', 'water', 'soda', 'smoothie', 'wine'],
};

export const CALORIE_DATABASE: Record<string, number> = {
  // Proteins (per 100g)
  chicken: 165,
  beef: 250,
  pork: 242,
  fish: 206,
  salmon: 208,
  tuna: 144,
  egg: 155,
  tofu: 76,
  beans: 127,
  lentils: 116,
  
  // Vegetables (per 100g)
  tomato: 18,
  potato: 77,
  'sweet potato': 86,
  carrot: 41,
  broccoli: 34,
  spinach: 23,
  lettuce: 15,
  onion: 40,
  
  // Fruits (per 100g)
  apple: 52,
  banana: 89,
  orange: 47,
  grape: 62,
  strawberry: 32,
  blueberry: 57,
  mango: 60,
  
  // Grains (per 100g)
  rice: 130,
  bread: 265,
  pasta: 131,
  quinoa: 120,
  oats: 389,
  
  // Dairy (per 100g)
  cheese: 113,
  milk: 42,
  yogurt: 59,
  butter: 717,
  
  // Prepared Foods (per serving)
  pizza: 266,
  burger: 540,
  sandwich: 300,
  salad: 150,
  soup: 120,
  
  // Snacks (per 100g)
  chips: 536,
  nuts: 607,
  crackers: 502,
  
  // Desserts (per serving)
  cake: 257,
  cookie: 502,
  'ice cream': 207,
  chocolate: 546,
  
  // Beverages (per 100ml)
  coffee: 2,
  tea: 1,
  juice: 45,
  soda: 42,
  wine: 83,
};

export default APP_CONFIG;