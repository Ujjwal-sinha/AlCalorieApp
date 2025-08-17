#!/usr/bin/env node

/**
 * Test Integration Script
 * Tests the connection between backend and Python models service
 */

const fs = require('fs');
const path = require('path');

async function testIntegration() {
  console.log('🧪 Testing Integration Between Backend and Python Models Service\n');

  // Test 1: Check Python Models Service
  console.log('1️⃣ Testing Python Models Service...');
  try {
    const response = await fetch('https://food-detection-models.onrender.com/health');
    const data = await response.json();
    
    if (data.status === 'healthy') {
      console.log('✅ Python Models Service: HEALTHY');
      console.log(`   - Models Available: ${Object.keys(data.model_availability || {}).length}`);
      console.log(`   - Port: ${data.port || 'N/A'}`);
    } else {
      console.log('❌ Python Models Service: UNHEALTHY');
    }
  } catch (error) {
    console.log('❌ Python Models Service: CONNECTION FAILED');
    console.log(`   Error: ${error.message}`);
  }

  // Test 2: Check Backend Service
  console.log('\n2️⃣ Testing Backend Service...');
  try {
    const response = await fetch('https://food-analyzer-backend.onrender.com/health');
    const data = await response.json();
    
    if (response.ok) {
      console.log('✅ Backend Service: HEALTHY');
      console.log(`   - Status: ${data.status || 'OK'}`);
    } else {
      console.log('❌ Backend Service: UNHEALTHY');
    }
  } catch (error) {
    console.log('❌ Backend Service: CONNECTION FAILED');
    console.log(`   Error: ${error.message}`);
  }

  // Test 3: Test Detection Endpoint
  console.log('\n3️⃣ Testing Detection Endpoint...');
  try {
    // Create a simple test image (1x1 pixel)
    const testImageBuffer = Buffer.from([
      0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
      0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
      0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
      0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
      0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
      0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
      0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
      0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x01,
      0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01,
      0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0xFF, 0xC4,
      0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x0C,
      0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00, 0x8A, 0x00,
      0xFF, 0xD9
    ]);

    const testData = {
      model_type: 'yolo',
      image_data: testImageBuffer.toString('base64'),
      width: 1,
      height: 1
    };

    const response = await fetch('https://food-detection-models.onrender.com/detect', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData)
    });

    if (response.ok) {
      const result = await response.json();
      console.log('✅ Detection Endpoint: WORKING');
      console.log(`   - Success: ${result.success}`);
      console.log(`   - Processing Time: ${result.processing_time || 'N/A'}ms`);
    } else {
      console.log('❌ Detection Endpoint: FAILED');
      console.log(`   Status: ${response.status}`);
    }
  } catch (error) {
    console.log('❌ Detection Endpoint: CONNECTION FAILED');
    console.log(`   Error: ${error.message}`);
  }

  console.log('\n🎉 Integration Test Complete!');
  console.log('\n📋 Next Steps:');
  console.log('1. Update your Render backend environment variables');
  console.log('2. Add PYTHON_MODELS_URL=https://food-detection-models.onrender.com');
  console.log('3. Redeploy your backend service');
  console.log('4. Test the full integration from your frontend');
}

// Run the test
testIntegration().catch(console.error);

