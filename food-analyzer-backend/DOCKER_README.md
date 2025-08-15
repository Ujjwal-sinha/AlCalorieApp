# Food Analyzer Backend - Docker Setup

This document provides comprehensive instructions for building and running the Food Analyzer Backend using Docker.

## 🏗️ Architecture

The application consists of:
- **Node.js/TypeScript Express Backend**: Handles HTTP requests and spawns Python processes
- **Python AI Models**: YOLO, Vision Transformers, Swin, BLIP, and CLIP for food detection
- **YOLO Model**: `yolo11m.pt` for object detection

## 📋 Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- At least 4GB RAM available for Docker
- 10GB+ free disk space

## 🚀 Quick Start

### Option 1: Using the Build Script (Recommended)

```bash
# Navigate to the backend directory
cd food-analyzer-backend

# Build and run (clean build)
./build.sh --clean

# Or just build and run
./build.sh
```

### Option 2: Using Docker Compose

```bash
# Build and start the service
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Option 3: Manual Docker Commands

```bash
# Build the image
docker build --target production -t food-analyzer-backend:latest .

# Run the container
docker run -p 3000:3000 --name food-analyzer-backend food-analyzer-backend:latest

# Run with custom port
docker run -p 8080:3000 --name food-analyzer-backend food-analyzer-backend:latest
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Port the application runs on |
| `NODE_ENV` | `production` | Node.js environment |
| `PYTHONUNBUFFERED` | `1` | Python output buffering |
| `PYTHONDONTWRITEBYTECODE` | `1` | Python bytecode generation |

### Custom Configuration

Create a `.env` file in the project root:

```env
PORT=8080
NODE_ENV=production
```

## 🏥 Health Checks

The application includes built-in health checks:

```bash
# Check application health
curl http://localhost:3000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "uptime": 123.456,
  "environment": "production"
}
```

## 📊 Monitoring

### Container Metrics

```bash
# View container stats
docker stats food-analyzer-backend-container

# View container logs
docker logs -f food-analyzer-backend-container
```

### Resource Usage

The container is configured with:
- **Memory**: 2GB limit, 1GB reservation
- **CPU**: 2 cores limit, 1 core reservation
- **Health Check**: Every 30 seconds

## 🔍 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
lsof -i :3000

# Kill the process or use a different port
docker run -p 8080:3000 food-analyzer-backend:latest
```

#### 2. Out of Memory
```bash
# Increase Docker memory limit in Docker Desktop
# Or run with more memory
docker run -m 4g -p 3000:3000 food-analyzer-backend:latest
```

#### 3. Build Failures
```bash
# Clean build
./build.sh --clean

# Or manually clean
docker system prune -a
docker build --no-cache --target production -t food-analyzer-backend:latest .
```

#### 4. Python Model Issues
```bash
# Check Python dependencies
docker run --rm food-analyzer-backend:latest python3 -c "import torch; print(torch.__version__)"

# Test model loading
docker run --rm food-analyzer-backend:latest python3 -c "from ultralytics import YOLO; print('YOLO available')"
```

### Debug Mode

For debugging, you can run the container interactively:

```bash
# Run with bash shell
docker run -it --rm -p 3000:3000 food-analyzer-backend:latest /bin/bash

# Inside container, you can:
# - Check Python packages: pip list
# - Test models: python3 python_models/detect_food.py
# - Check Node.js: node --version
```

## 🚀 Production Deployment

### Using Docker Compose with Nginx

```bash
# Start with nginx reverse proxy
docker-compose --profile production up -d

# This will start:
# - food-analyzer-backend on port 3000 (internal)
# - nginx on port 80 (external)
```

### Kubernetes Deployment

Create a `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: food-analyzer-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: food-analyzer-backend
  template:
    metadata:
      labels:
        app: food-analyzer-backend
    spec:
      containers:
      - name: food-analyzer-backend
        image: food-analyzer-backend:latest
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "production"
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
          requests:
            memory: "1Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 📈 Performance Optimization

### Image Size Optimization

The multi-stage Dockerfile is optimized for size:
- Uses `debian:bookworm-slim` base image
- Installs only production dependencies
- Removes build tools and caches
- Uses CPU-only PyTorch

### Runtime Optimization

- **Memory**: 2GB limit prevents OOM issues
- **CPU**: 2 cores for parallel processing
- **Health Checks**: Ensure service availability
- **Non-root User**: Security best practice

## 🔒 Security

### Security Features

- Runs as non-root user (`appuser`)
- Uses slim base images
- Includes security headers via nginx
- Rate limiting on API endpoints
- Input validation and sanitization

### Security Best Practices

1. **Never run as root** in production
2. **Use specific image tags** instead of `latest`
3. **Scan images** for vulnerabilities
4. **Keep base images updated**
5. **Use secrets management** for sensitive data

## 📝 API Testing

Once the container is running, test the API:

```bash
# Health check
curl http://localhost:3000/health

# Test food detection (requires image data)
curl -X POST http://localhost:3000/api/food/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'
```

## 🧹 Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi food-analyzer-backend:latest

# Clean up all unused resources
docker system prune -a
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Node.js Docker Best Practices](https://nodejs.org/en/docs/guides/nodejs-docker-webapp/)
- [Python Docker Best Practices](https://docs.docker.com/language/python/)
