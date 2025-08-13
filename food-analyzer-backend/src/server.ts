import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import { config } from './config';
import { ModelManager } from './services/ModelManager';
import { FoodDetectionService } from './services/FoodDetectionService';
import { NutritionService } from './services/NutritionService';
import { setupRoutes } from './routes';
import { errorHandler, notFoundHandler } from './middleware/errorHandler';

class Server {
  private app: express.Application;
  private modelManager: ModelManager;
  private foodDetectionService: FoodDetectionService;
  private nutritionService: NutritionService;

  constructor() {
    this.app = express();
    this.modelManager = ModelManager.getInstance();
    this.foodDetectionService = FoodDetectionService.getInstance();
    this.nutritionService = NutritionService.getInstance();
  }

  private setupMiddleware(): void {
    // Security middleware
    this.app.use(helmet({
      crossOriginEmbedderPolicy: false,
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "blob:"],
        },
      },
    }));

    // CORS configuration
    this.app.use(cors({
      origin: config.corsOrigin,
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
    }));

    // Compression
    this.app.use(compression());

    // Logging
    if (config.nodeEnv !== 'test') {
      this.app.use(morgan(config.logging.format));
    }

    // Body parsing
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        environment: config.nodeEnv,
        models: this.modelManager.getModelStatus()
      });
    });
  }

  private async initializeServices(): Promise<void> {
    console.log('Initializing services...');
    
    try {
      // Initialize services in parallel
      await Promise.all([
        this.modelManager.initialize(),
        this.nutritionService.initialize()
      ]);
      
      console.log('All services initialized successfully');
    } catch (error) {
      console.error('Failed to initialize services:', error);
      throw error;
    }
  }

  private setupRoutes(): void {
    // API routes
    this.app.use(config.apiPrefix, setupRoutes());

    // Error handling
    this.app.use(notFoundHandler);
    this.app.use(errorHandler);
  }

  private setupGracefulShutdown(): void {
    const gracefulShutdown = async (signal: string) => {
      console.log(`Received ${signal}. Starting graceful shutdown...`);
      
      try {
        // Close server
        if (this.server) {
          await new Promise<void>((resolve) => {
            this.server.close(() => {
              console.log('HTTP server closed');
              resolve();
            });
          });
        }

        // Shutdown services
        await this.modelManager.shutdown();
        
        console.log('Graceful shutdown completed');
        process.exit(0);
      } catch (error) {
        console.error('Error during shutdown:', error);
        process.exit(1);
      }
    };

    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));
  }

  private server?: any;

  public async start(): Promise<void> {
    try {
      // Setup middleware
      this.setupMiddleware();

      // Initialize services
      await this.initializeServices();

      // Setup routes
      this.setupRoutes();

      // Setup graceful shutdown
      this.setupGracefulShutdown();

      // Start server
      this.server = this.app.listen(config.port, config.host, () => {
        console.log(`🚀 Food Analyzer Backend running on http://${config.host}:${config.port}`);
        console.log(`📊 Health check: http://${config.host}:${config.port}/health`);
        console.log(`🔧 API endpoints: http://${config.host}:${config.port}${config.apiPrefix}`);
        console.log(`🌍 Environment: ${config.nodeEnv}`);
        
        const modelStatus = this.modelManager.getModelStatus();
        const loadedModels = Object.entries(modelStatus)
          .filter(([, loaded]) => loaded)
          .map(([name]) => name);
        
        console.log(`🤖 Loaded models: ${loadedModels.join(', ') || 'None'}`);
      });

      // Handle server errors
      this.server.on('error', (error: any) => {
        if (error.code === 'EADDRINUSE') {
          console.error(`Port ${config.port} is already in use`);
        } else {
          console.error('Server error:', error);
        }
        process.exit(1);
      });

    } catch (error) {
      console.error('Failed to start server:', error);
      process.exit(1);
    }
  }

  public getApp(): express.Application {
    return this.app;
  }
}

// Start server if this file is run directly
if (require.main === module) {
  const server = new Server();
  server.start().catch((error) => {
    console.error('Failed to start server:', error);
    process.exit(1);
  });
}

export default Server;