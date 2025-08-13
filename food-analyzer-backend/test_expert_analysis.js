const { FoodDetectionService } = require('./dist/services/FoodDetectionService');

async function testExpertAnalysis() {
  console.log('🧪 Testing Expert Analysis System...\n');

  try {
    const foodDetectionService = FoodDetectionService.getInstance();
    
    // Test health check
    console.log('📊 Checking system health...');
    const health = await foodDetectionService.healthCheck();
    console.log('Health Status:', health.status);
    console.log('Python Available:', health.pythonAvailable);
    console.log('Models:', health.models);
    console.log('');

    // Create a test image buffer (simple test)
    const testImageBuffer = Buffer.from('test-image-data');
    
    // Test expert analysis
    console.log('🔍 Testing Expert Analysis...');
    const result = await foodDetectionService.performExpertAnalysis(
      { buffer: testImageBuffer, width: 800, height: 600, format: 'jpeg' },
      'Test meal context'
    );

    console.log('✅ Expert Analysis Result:');
    console.log('- Success:', result.success);
    console.log('- Description:', result.description);
    console.log('- Detected Foods:', result.detected_foods);
    console.log('- Confidence:', result.confidence);
    console.log('- Processing Time:', result.processing_time + 'ms');
    console.log('- Model Used:', result.model_used);
    console.log('- Session ID:', result.sessionId);
    console.log('- Insights:', result.insights);
    console.log('');

    if (result.nutritional_data) {
      console.log('📈 Nutritional Data:');
      console.log('- Total Calories:', result.nutritional_data.total_calories);
      console.log('- Total Protein:', result.nutritional_data.total_protein + 'g');
      console.log('- Total Carbs:', result.nutritional_data.total_carbs + 'g');
      console.log('- Total Fats:', result.nutritional_data.total_fats + 'g');
      console.log('- Items Count:', result.nutritional_data.items.length);
    }

    console.log('\n🎉 Expert Analysis Test Completed Successfully!');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error('Stack trace:', error.stack);
  }
}

// Run the test
testExpertAnalysis();
