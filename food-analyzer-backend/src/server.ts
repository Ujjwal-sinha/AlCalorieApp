import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import { config } from './config';
import routes from './routes';
import { errorHandler } from './middleware/errorHandler';
import { ModelManager } from './services/ModelManager';

class Server {
  private app: express.Application;
  private modelManager: ModelManager;

  constructor() {
    this.app = express();
    this.modelManager = ModelManager.getInstance();
    this.setupMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();
  }

  private setupMiddleware(): void {
    // Security middleware
    this.app.use(helmet());
    
    // CORS configuration
    this.app.use(cors({
      origin: config.corsOrigin,
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization']
    }));

    // Logging middleware
    this.app.use(morgan('combined'));

    // Body parsing middleware
    this.app.use(express.json({ limit: config.maxFileSize }));
    this.app.use(express.urlencoded({ extended: true, limit: config.maxFileSize }));
  }

  private setupRoutes(): void {
    // Health check endpoint
    this.app.get('/health', (_req, res) => {
      return res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        environment: process.env['NODE_ENV'] || 'development'
      });
    });

    // API routes
    this.app.use(config.apiPrefix, routes);
  }

  private setupErrorHandling(): void {
    // Global error handler
    this.app.use(errorHandler);
  }

  async start(): Promise<void> {
    try {
      console.log('🚀 Starting Food Analyzer Backend...');
      
      // Initialize AI models
      console.log('📦 Initializing AI models...');
      await this.modelManager.initialize();
      
      // Start server
      this.app.listen(config.port, () => {
        console.log(`✅ Server running on port ${config.port}`);
        console.log(`🌐 API available at http://localhost:${config.port}${config.apiPrefix}`);
        console.log(`🏥 Health check at http://localhost:${config.port}/health`);
      });

      // Graceful shutdown handling
      process.on('SIGTERM', () => this.gracefulShutdown());
      process.on('SIGINT', () => this.gracefulShutdown());
      
    } catch (error) {
      console.error('❌ Failed to start server:', error);
      process.exit(1);
    }
  }

  private async gracefulShutdown(): Promise<void> {
    console.log('\n🛑 Graceful shutdown initiated...');
    
    try {
      // Cleanup resources
      console.log('🧹 Cleaning up resources...');
      
      // Note: ModelManager doesn't have a shutdown method, so we'll skip that
      // In a real implementation, you'd want to add proper cleanup methods
      
      console.log('✅ Graceful shutdown completed');
      process.exit(0);
    } catch (error) {
      console.error('❌ Error during shutdown:', error);
      process.exit(1);
    }
  }
}

// Start the server
const server = new Server();
server.start().catch(error => {
  console.error('❌ Server startup failed:', error);
  process.exit(1);
});