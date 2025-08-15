#!/bin/bash

# Food Analyzer Backend Docker Build Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="food-analyzer-backend"
TAG="latest"
CONTAINER_NAME="food-analyzer-backend-container"
PORT=${PORT:-3000}

echo -e "${BLUE}🚀 Food Analyzer Backend Docker Build Script${NC}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Clean up existing containers and images (optional)
if [ "$1" = "--clean" ]; then
    print_warning "Cleaning up existing containers and images..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    docker rmi $IMAGE_NAME:$TAG 2>/dev/null || true
fi

# Build the Docker image
print_status "Building Docker image..."
docker build \
    --target production \
    --tag $IMAGE_NAME:$TAG \
    --file Dockerfile \
    .

if [ $? -eq 0 ]; then
    print_status "Docker image built successfully!"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Show image size
IMAGE_SIZE=$(docker images $IMAGE_NAME:$TAG --format "table {{.Size}}" | tail -n 1)
print_status "Image size: $IMAGE_SIZE"

# Run the container
print_status "Starting container on port $PORT..."
docker run \
    --name $CONTAINER_NAME \
    --rm \
    --publish $PORT:3000 \
    --env NODE_ENV=production \
    --env PORT=3000 \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --memory=2g \
    --cpus=2.0 \
    $IMAGE_NAME:$TAG

print_status "Container started successfully!"
print_status "API available at: http://localhost:$PORT"
print_status "Health check at: http://localhost:$PORT/health"
print_status "Press Ctrl+C to stop the container"
