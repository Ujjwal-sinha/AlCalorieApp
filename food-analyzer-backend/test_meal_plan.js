const { GroqAnalysisService } = require('./dist/services/GroqAnalysisService');

async function testMealPlan() {
  console.log('Testing Daily Meal Plan Generation...');
  
  const groqService = GroqAnalysisService.getInstance();
  
  // Test meal plan generation
  const testRequest = {
    detectedFoods: ['chicken breast', 'brown rice', 'broccoli'],
    nutritionalData: {
      total_calories: 450,
      total_protein: 35,
      total_carbs: 45,
      total_fats: 12
    },
    foodItems: [
      {
        name: 'chicken breast',
        calories: 200,
        protein: 25,
        carbs: 0,
        fats: 5
      },
      {
        name: 'brown rice',
        calories: 150,
        protein: 3,
        carbs: 30,
        fats: 2
      },
      {
        name: 'broccoli',
        calories: 100,
        protein: 7,
        carbs: 15,
        fats: 5
      }
    ],
    imageDescription: 'A healthy lunch with grilled chicken, brown rice, and steamed broccoli',
    mealContext: 'Lunch meal analysis - need recommendations for rest of day'
  };
  
  try {
    const analysis = await groqService.generateComprehensiveAnalysis(testRequest);
    console.log('Analysis result:', {
      success: analysis.success,
      healthScore: analysis.healthScore,
      summaryLength: analysis.summary?.length || 0,
      recommendationsCount: analysis.recommendations?.length || 0,
      hasMealPlan: !!analysis.dailyMealPlan
    });
    
    if (analysis.success && analysis.dailyMealPlan) {
      console.log('\n=== DAILY MEAL PLAN ===');
      console.log('Total Calories:', analysis.dailyMealPlan.totalCalories);
      console.log('\nBreakfast:');
      analysis.dailyMealPlan.breakfast.forEach(item => console.log('  -', item));
      console.log('\nLunch:');
      analysis.dailyMealPlan.lunch.forEach(item => console.log('  -', item));
      console.log('\nDinner:');
      analysis.dailyMealPlan.dinner.forEach(item => console.log('  -', item));
      console.log('\nSnacks:');
      analysis.dailyMealPlan.snacks.forEach(item => console.log('  -', item));
      console.log('\nHydration:');
      analysis.dailyMealPlan.hydration.forEach(item => console.log('  -', item));
      console.log('\nNotes:', analysis.dailyMealPlan.notes);
    } else {
      console.log('\nError:', analysis.error);
    }
  } catch (error) {
    console.error('Analysis failed:', error);
  }
}

// Run the test
testMealPlan().catch(console.error);
