const axios = require('axios');
const fs = require('fs');
const path = require('path');

const BACKEND_URL = 'http://localhost:3001';

async function testAutoDietChat() {
  console.log('🧪 Testing Automatic Diet Chat Integration...\n');

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

    console.log('2. Uploading test image for automatic diet chat analysis...');
    
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

    // Display results
    console.log('\n📊 Analysis Results:');
    console.log(`- Success: ${result.success}`);
    console.log(`- Detected Foods: ${result.detectedFoods?.length || 0}`);
    console.log(`- Processing Time: ${result.processing_time}ms`);
    console.log(`- Model Used: ${result.model_used}`);

    if (result.detectedFoods && result.detectedFoods.length > 0) {
      console.log('\n🍽️ Detected Food Items:');
      result.detectedFoods.forEach((food, index) => {
        console.log(`  ${index + 1}. ${food.name} (${Math.round(food.confidence * 100)}% confidence)`);
      });
    }

    // Check for automatic diet chat response
    if (result.diet_chat_response) {
      console.log('\n🤖 Automatic Diet Chat Response:');
      console.log(`- Answer: ${result.diet_chat_response.answer.substring(0, 200)}...`);
      console.log(`- Confidence: ${Math.round(result.diet_chat_response.confidence * 100)}%`);
      console.log(`- Suggestions: ${result.diet_chat_response.suggestions?.length || 0}`);
      console.log(`- Related Topics: ${result.diet_chat_response.relatedTopics?.length || 0}`);
      
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
      console.log('\n❌ No automatic diet chat response generated');
    }

    // Check for GROQ analysis
    if (result.groq_analysis) {
      console.log('\n📈 GROQ Analysis:');
      console.log(`- Summary: ${result.groq_analysis.summary?.substring(0, 150)}...`);
      console.log(`- Health Score: ${result.groq_analysis.healthScore || 'N/A'}`);
      console.log(`- Recommendations: ${result.groq_analysis.recommendations?.length || 0}`);
    }

    // Check for nutrition data
    if (result.totalNutrition) {
      console.log('\n🍎 Nutrition Summary:');
      console.log(`- Total Calories: ${result.totalNutrition.total_calories}`);
      console.log(`- Protein: ${result.totalNutrition.total_protein}g`);
      console.log(`- Carbs: ${result.totalNutrition.total_carbs}g`);
      console.log(`- Fats: ${result.totalNutrition.total_fats}g`);
    }

    console.log('\n✅ Automatic Diet Chat Integration Test Completed Successfully!');
    console.log('\n🎯 Next Steps:');
    console.log('1. Open the frontend application');
    console.log('2. Upload a food image');
    console.log('3. View the automatic diet chat response in the analysis results');

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
  console.log('   Please set your GROQ API key to test the automatic diet chat feature.');
  console.log('   export GROQ_API_KEY="your-api-key-here"');
  process.exit(1);
}

// Run the test
testAutoDietChat().catch(console.error);
