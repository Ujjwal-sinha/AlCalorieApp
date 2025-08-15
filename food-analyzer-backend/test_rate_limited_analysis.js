const { GroqAnalysisService } = require('./dist/services/GroqAnalysisService');

async function testRateLimitedAnalysis() {
  console.log('🧪 Testing Rate-Limited Food Analysis...\n');

  const groqService = GroqAnalysisService.getInstance();

  // Test health check first
  console.log('1. Testing GROQ Service Health Check...');
  const healthCheck = await groqService.healthCheck();
  console.log('Health Check Result:', healthCheck);

  if (!healthCheck.available) {
    console.log('❌ GROQ service not available. Please check your API key.');
    return;
  }

  // Test with fewer foods to avoid rate limits
  console.log('\n2. Testing Rate-Limited Analysis...');
  
  const testRequest = {
    detectedFoods: ['apple', 'chicken breast'], // Only 2 foods
    nutritionalData: {
      total_calories: 300,
      total_protein: 25,
      total_carbs: 30,
      total_fats: 8,
      items: [
        { name: 'apple', calories: 95, protein: 0.5, carbs: 25, fats: 0.3 },
        { name: 'chicken breast', calories: 165, protein: 31, carbs: 0, fats: 3.6 }
      ]
    },
    foodItems: [
      { name: 'apple', calories: 95, protein: 0.5, carbs: 25, fats: 0.3 },
      { name: 'chicken breast', calories: 165, protein: 31, carbs: 0, fats: 3.6 }
    ],
    imageDescription: 'A healthy meal with grilled chicken and fresh apple',
    mealContext: 'Lunch'
  };

  try {
    console.log('Starting analysis...');
    const startTime = Date.now();
    
    const result = await groqService.generateComprehensiveAnalysis(testRequest);
    
    const endTime = Date.now();
    const duration = (endTime - startTime) / 1000;
    
    console.log(`\n✅ Analysis completed in ${duration.toFixed(1)} seconds!`);
    console.log('\n📊 Analysis Summary:');
    console.log('- Success:', result.success);
    console.log('- Summary Length:', result.summary?.length || 0, 'characters');
    console.log('- Health Score:', result.healthScore);
    console.log('- Recommendations Count:', result.recommendations?.length || 0);
    
    console.log('\n🍎 Food Item Reports:');
    if (result.foodItemReports) {
      console.log('- Number of food items analyzed:', Object.keys(result.foodItemReports).length);
      
      Object.entries(result.foodItemReports).forEach(([foodName, report]) => {
        console.log(`\n  📋 ${foodName}:`);
        console.log(`    - Nutrition Profile: ${report.nutritionProfile.substring(0, 80)}...`);
        console.log(`    - Health Benefits: ${report.healthBenefits.substring(0, 80)}...`);
      });
    } else {
      console.log('- No food item reports generated');
    }

    console.log('\n🍽️ Daily Meal Plan:');
    if (result.dailyMealPlan) {
      console.log('- Breakfast options:', result.dailyMealPlan.breakfast?.length || 0);
      console.log('- Lunch options:', result.dailyMealPlan.lunch?.length || 0);
      console.log('- Dinner options:', result.dailyMealPlan.dinner?.length || 0);
      console.log('- Total calories:', result.dailyMealPlan.totalCalories);
    } else {
      console.log('- No meal plan generated');
    }

    console.log('\n📝 Sample Summary:');
    console.log(result.summary?.substring(0, 150) + '...');

  } catch (error) {
    console.error('❌ Analysis failed:', error.message);
    if (error.message.includes('rate_limit')) {
      console.log('💡 Rate limit hit - this is expected with the free tier.');
      console.log('   The system will automatically retry with delays.');
    }
  }
}

// Run the test
testRateLimitedAnalysis().catch(console.error);
