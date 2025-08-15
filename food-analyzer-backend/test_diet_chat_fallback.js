const axios = require('axios');
const fs = require('fs');
const path = require('path');

const BACKEND_URL = 'http://localhost:3001';

async function testDietChatFallback() {
  console.log('🧪 Testing Diet Chat Fallback Response...\n');

  try {
    // Check if backend is running
    console.log('1. Checking backend status...');
    try {
      await axios.get(`${BACKEND_URL}/api/health`);
      console.log('✅ Backend is running');
    } catch (error) {
      console.log('❌ Backend is not running. Please start the backend server first.');
      console.log('   Run: npm start');
      return;
    }

    // Check if we have a test image
    const testImagePath = path.join(__dirname, 'test_image.jpg');
    if (!fs.existsSync(testImagePath)) {
      console.log('❌ Test image not found. Please add a test_image.jpg file to test with.');
      console.log('   You can use any food image for testing.');
      return;
    }

    console.log('2. Uploading test image for analysis (without GROQ API key)...');
    
    const formData = new FormData();
    formData.append('image', fs.createReadStream(testImagePath));

    const response = await axios.post(`${BACKEND_URL}/api/analysis/expert`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 60 seconds timeout
    });

    const result = response.data;
    console.log('✅ Image analysis completed successfully');

    // Check for diet chat response
    console.log('\n📊 Diet Chat Response Check:');
    if (result.diet_chat_response) {
      console.log('✅ diet_chat_response found in API response');
      console.log(`- Answer: ${result.diet_chat_response.answer?.substring(0, 100)}...`);
      console.log(`- Suggestions: ${result.diet_chat_response.suggestions?.length || 0}`);
      console.log(`- Related Topics: ${result.diet_chat_response.relatedTopics?.length || 0}`);
      console.log(`- Confidence: ${result.diet_chat_response.confidence || 0}`);
      
      console.log('\n📝 Full Diet Chat Response:');
      console.log(JSON.stringify(result.diet_chat_response, null, 2));
    } else {
      console.log('❌ diet_chat_response NOT found in API response');
      console.log('Available keys in result:', Object.keys(result));
    }

    // Check for other analysis components
    console.log('\n📈 Other Analysis Components:');
    console.log(`- Success: ${result.success}`);
    console.log(`- Detected Foods: ${result.detectedFoods?.length || 0}`);
    console.log(`- Processing Time: ${result.processing_time}ms`);
    console.log(`- Model Used: ${result.model_used}`);
    console.log(`- Has GROQ Analysis: ${!!result.groq_analysis}`);
    console.log(`- Has Model Info: ${!!result.model_info}`);

    console.log('\n✅ Diet Chat Fallback Test Completed!');
    console.log('\n🎯 Next Steps:');
    console.log('1. Open the frontend application');
    console.log('2. Upload a food image');
    console.log('3. Check if the diet chat response appears (should show fallback message)');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
  }
}

// Run the test (no API key required for fallback test)
testDietChatFallback().catch(console.error);
