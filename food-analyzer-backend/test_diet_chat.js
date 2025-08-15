const { DietChatService } = require('./dist/services/DietChatService');

async function testDietChat() {
  console.log('🧪 Testing Diet Chat Service...\n');

  const dietChatService = DietChatService.getInstance();

  // Test health check first
  console.log('1. Testing Diet Chat Service Health Check...');
  const healthCheck = await dietChatService.healthCheck();
  console.log('Health Check Result:', healthCheck);

  if (!healthCheck.available) {
    console.log('❌ Diet chat service not available. Please check your API key.');
    return;
  }

  // Test sample questions
  console.log('\n2. Testing Sample Questions...');
  const sampleQuestions = dietChatService.getSampleQuestions();
  console.log('Sample Questions:', sampleQuestions.slice(0, 3));

  // Test diet query
  console.log('\n3. Testing Diet Query...');
  
  const testQuery = {
    question: "What should I eat to lose weight healthily?",
    context: "I'm a 30-year-old looking to lose 10 pounds",
    userHistory: ["How much protein do I need?"]
  };

  try {
    console.log('Sending query:', testQuery.question);
    const startTime = Date.now();
    
    const result = await dietChatService.answerDietQuery(testQuery);
    
    const endTime = Date.now();
    const duration = (endTime - startTime) / 1000;
    
    console.log(`\n✅ Diet chat completed in ${duration.toFixed(1)} seconds!`);
    console.log('\n📊 Response Summary:');
    console.log('- Answer Length:', result.answer?.length || 0, 'characters');
    console.log('- Confidence:', (result.confidence * 100).toFixed(1) + '%');
    console.log('- Suggestions Count:', result.suggestions?.length || 0);
    console.log('- Related Topics Count:', result.relatedTopics?.length || 0);
    
    console.log('\n💬 Answer Preview:');
    console.log(result.answer?.substring(0, 200) + '...');
    
    console.log('\n💡 Suggestions:');
    if (result.suggestions && result.suggestions.length > 0) {
      result.suggestions.forEach((suggestion, index) => {
        console.log(`  ${index + 1}. ${suggestion}`);
      });
    }
    
    console.log('\n🔗 Related Topics:');
    if (result.relatedTopics && result.relatedTopics.length > 0) {
      result.relatedTopics.forEach((topic, index) => {
        console.log(`  ${index + 1}. ${topic}`);
      });
    }

  } catch (error) {
    console.error('❌ Diet chat failed:', error.message);
    if (error.message.includes('rate_limit')) {
      console.log('💡 Rate limit hit - this is expected with the free tier.');
      console.log('   The system will automatically retry with delays.');
    }
  }
}

// Run the test
testDietChat().catch(console.error);
