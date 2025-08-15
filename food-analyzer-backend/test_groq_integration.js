const { GroqAnalysisService } = require('./dist/services/GroqAnalysisService');

async function testGroqIntegration() {
  console.log('Testing GROQ Integration...');
  
  const groqService = GroqAnalysisService.getInstance();
  
  // Test health check
  console.log('\n1. Testing health check...');
  const healthCheck = await groqService.healthCheck();
  console.log('Health check result:', healthCheck);
  
  // Test analysis generation
  console.log('\n2. Testing analysis generation...');
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
    imageDescription: 'A healthy meal with grilled chicken, brown rice, and steamed broccoli',
    mealContext: 'Lunch meal analysis'
  };
  
  try {
    const analysis = await groqService.generateComprehensiveAnalysis(testRequest);
    console.log('Analysis result:', {
      success: analysis.success,
      healthScore: analysis.healthScore,
      summaryLength: analysis.summary?.length || 0,
      recommendationsCount: analysis.recommendations?.length || 0
    });
    
    if (analysis.success) {
      console.log('\nSummary:', analysis.summary);
      console.log('\nHealth Score:', analysis.healthScore);
      console.log('\nRecommendations:', analysis.recommendations);
    } else {
      console.log('\nError:', analysis.error);
    }
  } catch (error) {
    console.error('Analysis failed:', error);
  }
}

// Run the test
testGroqIntegration().catch(console.error);
