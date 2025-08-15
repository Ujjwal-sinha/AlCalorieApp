const http = require('http');

function checkBackendStatus() {
  console.log('🔍 Checking Backend Status...\n');

  const options = {
    hostname: 'localhost',
    port: 3001,
    path: '/api/health',
    method: 'GET',
    timeout: 5000
  };

  const req = http.request(options, (res) => {
    console.log(`✅ Backend is running!`);
    console.log(`📊 Status Code: ${res.statusCode}`);
    console.log(`🌐 URL: http://localhost:${options.port}`);
    
    let data = '';
    res.on('data', (chunk) => {
      data += chunk;
    });
    
    res.on('end', () => {
      try {
        const response = JSON.parse(data);
        console.log(`📋 Response:`, response);
      } catch (e) {
        console.log(`📋 Raw Response:`, data);
      }
    });
  });

  req.on('error', (error) => {
    console.log('❌ Backend is not running or not accessible');
    console.log(`🔧 Error: ${error.message}`);
    console.log('\n💡 To start the backend:');
    console.log('   1. Make sure you have GROQ_API_KEY set');
    console.log('   2. Run: npm start');
    console.log('   3. Or run: node dist/server.js');
  });

  req.on('timeout', () => {
    console.log('⏰ Request timed out - backend might be slow to respond');
    req.destroy();
  });

  req.end();
}

// Also check if the server file exists
const fs = require('fs');
const path = require('path');

const serverPath = path.join(__dirname, 'dist', 'server.js');
if (fs.existsSync(serverPath)) {
  console.log('✅ Server file exists at:', serverPath);
} else {
  console.log('❌ Server file not found. Run: npm run build');
}

checkBackendStatus();
