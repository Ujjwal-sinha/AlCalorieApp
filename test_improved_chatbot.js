const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

async function testImprovedDietChatbot() {
  console.log('🧪 Testing Improved Diet Chatbot...\n');

  const testQuestions = [
    "What should I eat to lose weight healthily?",
    "How much protein do I need daily?",
    "What are the best sources of vitamin D?",
    "How can I improve my gut health?",
    "What's a good breakfast for energy?"
  ];

  try {
    // Test health endpoint
    console.log('1. Testing health endpoint...');
    const healthResponse = await axios.get(`${BASE_URL}/api/diet/health`);
    console.log('✅ Health check:', healthResponse.data);

    // Test sample questions endpoint
    console.log('\n2. Testing sample questions endpoint...');
    const questionsResponse = await axios.get(`${BASE_URL}/api/diet/sample-questions`);
    console.log('✅ Sample questions loaded:', questionsResponse.data.questions.length, 'questions');

    // Test diet queries
    console.log('\n3. Testing diet queries...');
    
    for (let i = 0; i < Math.min(3, testQuestions.length); i++) {
      const question = testQuestions[i];
      console.log(`\n   Testing: "${question}"`);
      
      try {
        const queryResponse = await axios.post(`${BASE_URL}/api/diet/query`, {
          question: question,
          context: 'I am a 30-year-old looking to improve my nutrition',
          userHistory: []
        });
        
        const data = queryResponse.data;
        console.log(`   ✅ Response length: ${data.answer.length} characters`);
        console.log(`   ✅ Confidence: ${(data.confidence * 100).toFixed(1)}%`);
        console.log(`   ✅ Suggestions: ${data.suggestions.length}`);
        console.log(`   ✅ Related topics: ${data.relatedTopics.length}`);
        console.log(`   📝 Answer preview: ${data.answer.substring(0, 100)}...`);
        
        // Wait between requests to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 2000));
        
      } catch (error) {
        console.log(`   ❌ Failed: ${error.message}`);
      }
    }

    console.log('\n🎉 Improved chatbot test completed!');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
  }
}

testImprovedDietChatbot();
