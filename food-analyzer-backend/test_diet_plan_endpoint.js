const axios = require('axios');

const BACKEND_URL = 'http://localhost:3001';

async function testDietPlanEndpoint() {
  console.log('🧪 Testing Diet Plan Generation Endpoint...\n');

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

    console.log('2. Testing diet plan generation...');
    
    const testData = {
      detectedFoods: ['apple', 'chicken', 'rice', 'broccoli', 'carrot'],
      nutritionalData: {
        total_calories: 450,
        total_protein: 25,
        total_carbs: 60,
        total_fats: 15
      },
      userPreferences: {}
    };

    const response = await axios.post(`${BACKEND_URL}/api/analysis/generate-diet-plan`, testData, {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 seconds timeout
    });

    const result = response.data;
    console.log('✅ Diet plan generation completed successfully');

    // Check the response structure
    console.log('\n📊 Response Structure:');
    console.log(`- Success: ${result.success}`);
    console.log(`- Has dietPlan: ${!!result.dietPlan}`);
    
    if (result.dietPlan) {
      console.log('\n📝 Diet Plan Structure:');
      console.log(`- Breakfast: ${result.dietPlan.breakfast?.length || 0} items`);
      console.log(`- Lunch: ${result.dietPlan.lunch?.length || 0} items`);
      console.log(`- Dinner: ${result.dietPlan.dinner?.length || 0} items`);
      console.log(`- Snacks: ${result.dietPlan.snacks?.length || 0} items`);
      console.log(`- Hydration: ${result.dietPlan.hydration?.length || 0} items`);
      console.log(`- Total Calories: ${result.dietPlan.totalCalories || 'N/A'}`);
      console.log(`- Notes: ${result.dietPlan.notes ? 'Present' : 'Missing'}`);
      
      console.log('\n🍽️ Sample Diet Plan Content:');
      if (result.dietPlan.breakfast && result.dietPlan.breakfast.length > 0) {
        console.log('Breakfast:', result.dietPlan.breakfast[0]);
      }
      if (result.dietPlan.lunch && result.dietPlan.lunch.length > 0) {
        console.log('Lunch:', result.dietPlan.lunch[0]);
      }
    } else {
      console.log('❌ No diet plan in response');
      console.log('Full response:', JSON.stringify(result, null, 2));
    }

    console.log('\n✅ Diet Plan Endpoint Test Completed!');

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
  console.log('   Please set your GROQ API key to test the diet plan generation.');
  console.log('   export GROQ_API_KEY="your-api-key-here"');
  process.exit(1);
}

// Run the test
testDietPlanEndpoint().catch(console.error);
