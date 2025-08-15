const { GroqAnalysisService } = require('./dist/services/GroqAnalysisService');

async function testComprehensiveAnalysis() {
  console.log('🧪 Testing Comprehensive Food Analysis...\n');

  const groqService = GroqAnalysisService.getInstance();

  // Test health check first
  console.log('1. Testing GROQ Service Health Check...');
  const healthCheck = await groqService.healthCheck();
  console.log('Health Check Result:', healthCheck);

  if (!healthCheck.available) {
    console.log('❌ GROQ service not available. Please check your API key.');
    return;
  }

  // Test comprehensive analysis
  console.log('\n2. Testing Comprehensive Food Analysis...');
  
  const testRequest = {
    detectedFoods: ['apple', 'chicken breast', 'brown rice'],
    nutritionalData: {
      total_calories: 450,
      total_protein: 35,
      total_carbs: 45,
      total_fats: 12,
      items: [
        { name: 'apple', calories: 95, protein: 0.5, carbs: 25, fats: 0.3 },
        { name: 'chicken breast', calories: 165, protein: 31, carbs: 0, fats: 3.6 },
        { name: 'brown rice', calories: 190, protein: 4, carbs: 40, fats: 1.8 }
      ]
    },
    foodItems: [
      { name: 'apple', calories: 95, protein: 0.5, carbs: 25, fats: 0.3 },
      { name: 'chicken breast', calories: 165, protein: 31, carbs: 0, fats: 3.6 },
      { name: 'brown rice', calories: 190, protein: 4, carbs: 40, fats: 1.8 }
    ],
    imageDescription: 'A healthy meal with grilled chicken, brown rice, and a fresh apple',
    mealContext: 'Lunch'
  };

  try {
    const result = await groqService.generateComprehensiveAnalysis(testRequest);
    
    console.log('\n✅ Analysis completed successfully!');
    console.log('\n📊 Analysis Summary:');
    console.log('- Success:', result.success);
    console.log('- Summary Length:', result.summary?.length || 0, 'characters');
    console.log('- Health Score:', result.healthScore);
    console.log('- Recommendations Count:', result.recommendations?.length || 0);
    console.log('- Dietary Considerations Count:', result.dietaryConsiderations?.length || 0);
    
    console.log('\n🍎 Food Item Reports:');
    if (result.foodItemReports) {
      console.log('- Number of food items analyzed:', Object.keys(result.foodItemReports).length);
      
      Object.entries(result.foodItemReports).forEach(([foodName, report]) => {
        console.log(`\n  📋 ${foodName}:`);
        console.log(`    - Nutrition Profile: ${report.nutritionProfile.substring(0, 100)}...`);
        console.log(`    - Health Benefits: ${report.healthBenefits.substring(0, 100)}...`);
        console.log(`    - Cooking Methods: ${report.cookingMethods.substring(0, 100)}...`);
        console.log(`    - Serving Suggestions: ${report.servingSuggestions.substring(0, 100)}...`);
      });
    } else {
      console.log('- No food item reports generated');
    }

    console.log('\n🍽️ Daily Meal Plan:');
    if (result.dailyMealPlan) {
      console.log('- Breakfast options:', result.dailyMealPlan.breakfast?.length || 0);
      console.log('- Lunch options:', result.dailyMealPlan.lunch?.length || 0);
      console.log('- Dinner options:', result.dailyMealPlan.dinner?.length || 0);
      console.log('- Snack options:', result.dailyMealPlan.snacks?.length || 0);
      console.log('- Total calories:', result.dailyMealPlan.totalCalories);
    } else {
      console.log('- No meal plan generated');
    }

    console.log('\n📝 Sample Summary:');
    console.log(result.summary?.substring(0, 200) + '...');

    console.log('\n💡 Sample Recommendations:');
    if (result.recommendations && result.recommendations.length > 0) {
      result.recommendations.slice(0, 3).forEach((rec, index) => {
        console.log(`  ${index + 1}. ${rec}`);
      });
    }

  } catch (error) {
    console.error('❌ Analysis failed:', error.message);
  }
}

// Run the test
testComprehensiveAnalysis().catch(console.error);
