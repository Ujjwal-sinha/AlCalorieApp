#!/usr/bin/env node

/**
 * Test script for Python Models Service
 * Tests both local and remote Python models service
 */

const fs = require('fs');
const path = require('path');

// Configuration
const config = {
  localBackendUrl: 'http://localhost:8000',
  remoteBackendUrl: 'https://food-analyzer-backend.onrender.com',
  testImagePath: path.join(__dirname, 'test-image.jpg'),
  timeout: 60000
};

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}[${new Date().toISOString()}]${colors.reset} ${message}`);
}

function logSuccess(message) {
  log(`✅ ${message}`, 'green');
}

function logError(message) {
  log(`❌ ${message}`, 'red');
}

function logWarning(message) {
  log(`⚠️  ${message}`, 'yellow');
}

function logInfo(message) {
  log(`ℹ️  ${message}`, 'blue');
}

// Test functions
async function testHealthCheck(url, serviceName) {
  try {
    logInfo(`Testing ${serviceName} health check...`);
    const response = await fetch(`${url}/api/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(config.timeout)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    logSuccess(`${serviceName} health check passed`);
    logInfo(`Status: ${data.status}`);
    logInfo(`Models loaded: ${data.models ? Object.keys(data.models).length : 'N/A'}`);
    return true;
  } catch (error) {
    logError(`${serviceName} health check failed: ${error.message}`);
    return false;
  }
}

async function testFoodAnalysis(url, serviceName) {
  try {
    logInfo(`Testing ${serviceName} food analysis...`);
    
    // Check if test image exists
    if (!fs.existsSync(config.testImagePath)) {
      logWarning(`Test image not found at ${config.testImagePath}`);
      logInfo('Skipping food analysis test');
      return true;
    }
    
    // Read and encode test image
    const imageBuffer = fs.readFileSync(config.testImagePath);
    const base64Image = imageBuffer.toString('base64');
    
    const formData = new FormData();
    formData.append('image', new Blob([imageBuffer], { type: 'image/jpeg' }), 'test-image.jpg');
    
    const response = await fetch(`${url}/api/analyze`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(config.timeout)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    logSuccess(`${serviceName} food analysis completed`);
    logInfo(`Processing time: ${data.processingTime}ms`);
    logInfo(`Detected foods: ${data.detectedFoods?.length || 0}`);
    logInfo(`Confidence: ${data.confidence?.toFixed(3) || 'N/A'}`);
    
    if (data.detectedFoods && data.detectedFoods.length > 0) {
      logInfo(`Food items: ${data.detectedFoods.map(f => f.name).join(', ')}`);
    }
    
    return true;
  } catch (error) {
    logError(`${serviceName} food analysis failed: ${error.message}`);
    return false;
  }
}

async function testPythonModelsService(pythonServiceUrl) {
  try {
    logInfo(`Testing Python Models Service at ${pythonServiceUrl}...`);
    
    // Test health check
    const healthResponse = await fetch(`${pythonServiceUrl}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(config.timeout)
    });
    
    if (!healthResponse.ok) {
      throw new Error(`Health check failed: HTTP ${healthResponse.status}`);
    }
    
    const healthData = await healthResponse.json();
    logSuccess(`Python Models Service health check passed`);
    logInfo(`Service: ${healthData.service}`);
    logInfo(`Models loaded: ${healthData.models_loaded?.length || 0}`);
    
    // Test models endpoint
    const modelsResponse = await fetch(`${pythonServiceUrl}/models`, {
      method: 'GET',
      signal: AbortSignal.timeout(config.timeout)
    });
    
    if (modelsResponse.ok) {
      const modelsData = await modelsResponse.json();
      logSuccess(`Models endpoint working`);
      logInfo(`Available models: ${modelsData.available_models?.join(', ') || 'N/A'}`);
      logInfo(`Loaded models: ${modelsData.loaded_models?.join(', ') || 'N/A'}`);
    }
    
    return true;
  } catch (error) {
    logError(`Python Models Service test failed: ${error.message}`);
    return false;
  }
}

async function runTests() {
  log('🧪 Starting Python Models Service Tests', 'cyan');
  log('=====================================', 'cyan');
  
  const results = {
    localBackend: false,
    remoteBackend: false,
    pythonService: false
  };
  
  // Test local backend
  log('\n📋 Testing Local Backend', 'magenta');
  results.localBackend = await testHealthCheck(config.localBackendUrl, 'Local Backend');
  if (results.localBackend) {
    await testFoodAnalysis(config.localBackendUrl, 'Local Backend');
  }
  
  // Test remote backend
  log('\n📋 Testing Remote Backend', 'magenta');
  results.remoteBackend = await testHealthCheck(config.remoteBackendUrl, 'Remote Backend');
  if (results.remoteBackend) {
    await testFoodAnalysis(config.remoteBackendUrl, 'Remote Backend');
  }
  
  // Test Python Models Service (if URL is provided)
  const pythonServiceUrl = process.env.PYTHON_MODELS_URL;
  if (pythonServiceUrl) {
    log('\n📋 Testing Python Models Service', 'magenta');
    results.pythonService = await testPythonModelsService(pythonServiceUrl);
  } else {
    logWarning('PYTHON_MODELS_URL not set, skipping Python service test');
  }
  
  // Summary
  log('\n📊 Test Summary', 'cyan');
  log('==============', 'cyan');
  logInfo(`Local Backend: ${results.localBackend ? '✅ PASS' : '❌ FAIL'}`);
  logInfo(`Remote Backend: ${results.remoteBackend ? '✅ PASS' : '❌ FAIL'}`);
  logInfo(`Python Service: ${pythonServiceUrl ? (results.pythonService ? '✅ PASS' : '❌ FAIL') : '⚠️  SKIP'}`);
  
  const allPassed = Object.values(results).every(result => result === true);
  
  if (allPassed) {
    logSuccess('All tests passed! 🎉');
    process.exit(0);
  } else {
    logError('Some tests failed! 🔧');
    process.exit(1);
  }
}

// Handle errors
process.on('unhandledRejection', (reason, promise) => {
  logError(`Unhandled Rejection at: ${promise}, reason: ${reason}`);
  process.exit(1);
});

process.on('uncaughtException', (error) => {
  logError(`Uncaught Exception: ${error.message}`);
  process.exit(1);
});

// Run tests
runTests().catch(error => {
  logError(`Test runner failed: ${error.message}`);
  process.exit(1);
});
