// Test script to verify real Python detection functionality
// This tests the actual BLIP + YOLO detection through the Python API

const fs = require('fs');
const path = require('path');

async function testRealDetection() {
  console.log('🧪 Testing Real Python Detection');
  console.log('================================');

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // Test 1: Health check
  console.log('\n1. Testing Python API health...');
  try {
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();
    
    if (response.ok) {
      console.log('✅ Python API is healthy');
      console.log(`   Models available: ${data.models_available}`);
      console.log(`   Model status:`, data.model_status || 'Not provided');
    } else {
      console.log('❌ Python API health check failed');
      return;
    }
  } catch (error) {
    console.log(`❌ Cannot connect to Python API: ${error.message}`);
    console.log('   Make sure to start the Python API with: python start_api.py');
    return;
  }

  // Test 2: Create a test image (simple colored square)
  console.log('\n2. Creating test image...');
  
  // Create a simple test image data (1x1 red pixel as base64)
  const testImageBase64 = '/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A8A';
  
  console.log('✅ Test image created (base64 encoded)');

  // Test 3: Test enhanced image description
  console.log('\n3. Testing enhanced image description...');
  try {
    const response = await fetch(`${API_BASE}/api/describe-image-enhanced`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: testImageBase64,
        format: 'image/jpeg'
      })
    });

    const result = await response.json();
    
    if (response.ok && result.success) {
      console.log('✅ Enhanced image description successful');
      console.log(`   Method used: ${result.method}`);
      console.log(`   Description: ${result.description}`);
      console.log(`   Items found: ${result.items_found || 'Not specified'}`);
      console.log(`   Strategies used: ${result.strategies_used ? result.strategies_used.join(', ') : 'Not specified'}`);
    } else {
      console.log('❌ Enhanced image description failed');
      console.log(`   Error: ${result.error || 'Unknown error'}`);
    }
  } catch (error) {
    console.log(`❌ Enhanced image description request failed: ${error.message}`);
  }

  // Test 4: Test Groq LLM
  console.log('\n4. Testing Groq LLM...');
  try {
    const testPrompt = `You are an expert nutritionist. Analyze this food description:

FOOD DESCRIPTION: grilled chicken, broccoli, rice
ADDITIONAL CONTEXT: healthy lunch meal

Provide a brief nutritional analysis with calories and macronutrients.`;

    const response = await fetch(`${API_BASE}/api/groq-llm`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: testPrompt,
        temperature: 0.1,
        max_tokens: 500
      })
    });

    const result = await response.json();
    
    if (response.ok && result.success) {
      console.log('✅ Groq LLM query successful');
      console.log(`   Model used: ${result.model}`);
      console.log(`   Response length: ${result.content.length} characters`);
      console.log(`   Response preview: ${result.content.substring(0, 200)}...`);
    } else {
      console.log('❌ Groq LLM query failed');
      console.log(`   Error: ${result.error || 'Unknown error'}`);
    }
  } catch (error) {
    console.log(`❌ Groq LLM request failed: ${error.message}`);
  }

  // Test 5: Test complete food analysis
  console.log('\n5. Testing complete food analysis...');
  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: testImageBase64,
        context: 'healthy lunch meal',
        format: 'image/jpeg'
      })
    });

    const result = await response.json();
    
    if (response.ok && result.success) {
      console.log('✅ Complete food analysis successful');
      console.log(`   Food items found: ${result.food_items ? result.food_items.length : 0}`);
      console.log(`   Total calories: ${result.nutritional_data ? result.nutritional_data.total_calories : 'Not available'}`);
      console.log(`   Analysis length: ${result.analysis ? result.analysis.length : 0} characters`);
      
      if (result.food_items && result.food_items.length > 0) {
        console.log('   Food items:');
        result.food_items.slice(0, 3).forEach((item, index) => {
          console.log(`     ${index + 1}. ${item.item} - ${item.calories || 0} calories`);
        });
      }
    } else {
      console.log('❌ Complete food analysis failed');
      console.log(`   Error: ${result.error || 'Unknown error'}`);
    }
  } catch (error) {
    console.log(`❌ Complete food analysis request failed: ${error.message}`);
  }

  console.log('\n🎉 Real Detection Testing Complete!');
  console.log('\nNext Steps:');
  console.log('1. If models are not available, install Python dependencies:');
  console.log('   pip install -r requirements.txt');
  console.log('2. Make sure GROQ_API_KEY is set in your .env file');
  console.log('3. Start the Python API: python start_api.py');
  console.log('4. Test with real food images for better results');
}

// Run the test
testRealDetection().catch(console.error);