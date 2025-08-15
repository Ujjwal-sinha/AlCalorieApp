# Food Analyzer Backend - Docker Production Setup

## 🎯 Overview

This repository contains a production-ready Docker setup for the Food Analyzer Backend, which combines:
- **Node.js/TypeScript Express Backend**: Handles HTTP requests and spawns Python processes
- **Python AI Models**: YOLO, Vision Transformers, Swin, BLIP, and CLIP for food detection
- **YOLO Model**: `yolo11m.pt` for object detection

## 📁 Files Created

### Core Docker Files
- `Dockerfile` - Main multi-stage Dockerfile (2.36GB image)
- `Dockerfile.optimized` - Optimized version for smaller image size
- `.dockerignore` - Excludes unnecessary files from build context
- `docker-compose.yml` - For easy local development and production deployment
- `nginx.conf` - Reverse proxy configuration for production

### Build and Documentation
- `build.sh` - Automated build and run script
- `DOCKER_README.md` - Comprehensive usage guide
- `DOCKER_SUMMARY.md` - This summary document

## 🚀 Quick Start

### Option 1: Using the Build Script
```bash
cd food-analyzer-backend
./build.sh --clean  # Clean build
./build.sh          # Regular build
```

### Option 2: Using Docker Compose
```bash
# Development
docker-compose up --build

# Production with nginx
docker-compose --profile production up -d
```

### Option 3: Manual Docker Commands
```bash
# Build
docker build --target production -t food-analyzer-backend:latest .

# Run
docker run -p 3000:3000 food-analyzer-backend:latest
```

## 📊 Performance Metrics

### Current Image Size
- **Main Dockerfile**: 2.36GB
- **Target**: <1.5GB (requires optimization)

### Resource Requirements
- **Memory**: 2GB limit, 1GB reservation
- **CPU**: 2 cores limit, 1 core reservation
- **Storage**: ~2.4GB for image + runtime data

## 🔧 Architecture

### Multi-Stage Build Process
1. **Builder Stage**: Compiles TypeScript to JavaScript
2. **Python Dependencies Stage**: Installs AI/ML libraries
3. **Production Stage**: Creates final slim image

### Key Components
- **Base Image**: `debian:bookworm-slim`
- **Node.js**: 18.x with npm 10.8.2
- **Python**: 3.11 with CPU-only PyTorch
- **Security**: Non-root user (`appuser`)
- **Health Checks**: Built-in endpoint monitoring

## 🛠️ Technical Details

### Dependencies Installed
**Node.js (Production)**:
- express, cors, helmet, morgan
- sharp (rebuilt for ARM64)
- multer, uuid, joi
- axios, compression

**Python (AI/ML)**:
- torch, torchvision (CPU-only)
- transformers, ultralytics
- Pillow, numpy, opencv-python
- requests, tqdm

### System Dependencies
- gcc, g++ (for native modules)
- libgl1, libglib2.0-0 (for OpenCV)
- libsm6, libxext6, libxrender1
- libgomp1 (OpenMP support)

## 🔒 Security Features

### Built-in Security
- Non-root user execution
- Slim base images
- Minimal attack surface
- Security headers via nginx
- Rate limiting
- Input validation

### Best Practices
- Multi-stage builds
- Layer caching optimization
- Minimal dependencies
- Regular security updates

## 📈 Optimization Opportunities

### Image Size Reduction
1. **Use Alpine Linux**: Could reduce base image size
2. **Remove Development Tools**: Clean up after builds
3. **Optimize Python Dependencies**: Install only required packages
4. **Use .dockerignore**: Exclude unnecessary files
5. **Multi-arch Support**: Build for specific architectures

### Performance Improvements
1. **Layer Caching**: Optimize Docker layer order
2. **Dependency Pinning**: Use specific versions
3. **Build Context**: Minimize files copied
4. **Runtime Optimization**: Tune Node.js and Python settings

## 🧪 Testing

### Health Check
```bash
curl http://localhost:3000/health
# Expected: {"status":"healthy","timestamp":"...","uptime":...,"environment":"production"}
```

### Container Testing
```bash
# Test container startup
docker run --rm -p 3000:3000 food-analyzer-backend:latest

# Test Python models
docker run --rm food-analyzer-backend:latest python3 -c "import torch; print(torch.__version__)"

# Test Node.js
docker run --rm food-analyzer-backend:latest node --version
```

## 🚀 Deployment Options

### Local Development
```bash
docker-compose up --build
```

### Production with Nginx
```bash
docker-compose --profile production up -d
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: food-analyzer-backend
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: food-analyzer-backend
        image: food-analyzer-backend:latest
        ports:
        - containerPort: 3000
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
          requests:
            memory: "1Gi"
            cpu: "1"
```

### Cloud Platforms
- **AWS ECS/Fargate**: Use provided Dockerfile
- **Google Cloud Run**: Optimize for serverless
- **Azure Container Instances**: Direct deployment
- **Heroku**: Use container registry

## 🔍 Troubleshooting

### Common Issues
1. **Sharp Module Error**: Fixed with ARM64 rebuild
2. **Port Conflicts**: Use different port mapping
3. **Memory Issues**: Increase Docker memory limit
4. **Build Failures**: Clean build with `--no-cache`

### Debug Commands
```bash
# Check container logs
docker logs <container-name>

# Interactive shell
docker run -it --rm food-analyzer-backend:latest /bin/bash

# Check image layers
docker history food-analyzer-backend:latest

# Analyze image size
docker images food-analyzer-backend:latest
```

## 📚 Next Steps

### Immediate Actions
1. ✅ Docker setup complete
2. ✅ Multi-stage build working
3. ✅ Health checks implemented
4. ✅ Security best practices applied
5. 🔄 Image size optimization needed

### Future Improvements
1. **Reduce Image Size**: Target <1.5GB
2. **Add Monitoring**: Prometheus/Grafana
3. **CI/CD Pipeline**: Automated builds
4. **Multi-arch Support**: ARM64/x86_64
5. **Security Scanning**: Vulnerability checks

## 📞 Support

For issues or questions:
1. Check the `DOCKER_README.md` for detailed instructions
2. Review container logs for error messages
3. Test individual components separately
4. Verify system requirements and dependencies

---

**Status**: ✅ Production-ready Docker setup complete
**Image Size**: 2.36GB (needs optimization for <1.5GB target)
**Functionality**: ✅ All components working correctly
**Security**: ✅ Best practices implemented
