#!/usr/bin/env node

/**
 * Simple test script to verify the diet chatbot functionality
 * Run with: node test_diet_chatbot.js
 */

const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

async function testDietChatbot() {
  console.log('🧪 Testing Diet Chatbot Backend...\n');

  try {
    // Test health endpoint
    console.log('1. Testing health endpoint...');
    const healthResponse = await axios.get(`${BASE_URL}/api/diet/health`);
    console.log('✅ Health check:', healthResponse.data);

    // Test sample questions endpoint
    console.log('\n2. Testing sample questions endpoint...');
    const questionsResponse = await axios.get(`${BASE_URL}/api/diet/sample-questions`);
    console.log('✅ Sample questions:', questionsResponse.data);

    // Test diet query endpoint
    console.log('\n3. Testing diet query endpoint...');
    const queryResponse = await axios.post(`${BASE_URL}/api/diet/query`, {
      question: 'What should I eat to lose weight healthily?',
      context: 'I am a 30-year-old looking to lose 10 pounds',
      userHistory: []
    });
    console.log('✅ Diet query response:', queryResponse.data);

    console.log('\n🎉 All tests passed! The diet chatbot is working correctly.');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
  }
}

testDietChatbot();