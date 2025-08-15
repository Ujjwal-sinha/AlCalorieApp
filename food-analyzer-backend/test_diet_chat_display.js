const axios = require('axios');
const fs = require('fs');
const path = require('path');

const BACKEND_URL = 'http://localhost:3001';

async function testDietChatDisplay() {
  console.log('🧪 Testing Diet Chat Display Integration...\n');

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

    console.log('2. Uploading test image for analysis...');
    
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
      console.log(`- Answer length: ${result.diet_chat_response.answer?.length || 0} characters`);
      console.log(`- Suggestions: ${result.diet_chat_response.suggestions?.length || 0}`);
      console.log(`- Related Topics: ${result.diet_chat_response.relatedTopics?.length || 0}`);
      console.log(`- Confidence: ${result.diet_chat_response.confidence || 0}`);
      
      console.log('\n📝 Diet Chat Answer Preview:');
      console.log(result.diet_chat_response.answer?.substring(0, 200) + '...');
      
      if (result.diet_chat_response.suggestions && result.diet_chat_response.suggestions.length > 0) {
        console.log('\n💡 Suggestions:');
        result.diet_chat_response.suggestions.forEach((suggestion, index) => {
          console.log(`  ${index + 1}. ${suggestion}`);
        });
      }
      
      if (result.diet_chat_response.relatedTopics && result.diet_chat_response.relatedTopics.length > 0) {
        console.log('\n🔗 Related Topics:');
        result.diet_chat_response.relatedTopics.forEach((topic, index) => {
          console.log(`  ${index + 1}. ${topic}`);
        });
      }
    } else {
      console.log('❌ diet_chat_response NOT found in API response');
      console.log('Available keys in result:', Object.keys(result));
    }

    // Check for GROQ analysis
    console.log('\n📈 GROQ Analysis Check:');
    if (result.groq_analysis) {
      console.log('✅ groq_analysis found in API response');
      console.log(`- Summary length: ${result.groq_analysis.summary?.length || 0} characters`);
      console.log(`- Health Score: ${result.groq_analysis.healthScore || 'N/A'}`);
      console.log(`- Recommendations: ${result.groq_analysis.recommendations?.length || 0}`);
    } else {
      console.log('❌ groq_analysis NOT found in API response');
    }

    // Check for model info
    console.log('\n🤖 Model Info Check:');
    if (result.model_info) {
      console.log('✅ model_info found in API response');
      console.log(`- Detection count: ${result.model_info.detection_count || 0}`);
      console.log(`- Total confidence: ${result.model_info.total_confidence || 0}`);
      console.log(`- Model performance: ${Object.keys(result.model_info.model_performance || {}).length} models`);
    } else {
      console.log('❌ model_info NOT found in API response');
    }

    console.log('\n✅ Diet Chat Display Test Completed!');
    console.log('\n🎯 Next Steps:');
    console.log('1. Open the frontend application');
    console.log('2. Upload a food image');
    console.log('3. Check if the diet chat response appears in the AI Model Detection Report');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
  }
}

// Check if GROQ API key is set
if (!process.env.GROQ_API_KEY) {
  console.log('⚠️  GROQ_API_KEY not found in environment variables.');
  console.log('   Please set your GROQ API key to test the diet chat feature.');
  console.log('   export GROQ_API_KEY="your-api-key-here"');
  process.exit(1);
}

// Run the test
testDietChatDisplay().catch(console.error);
