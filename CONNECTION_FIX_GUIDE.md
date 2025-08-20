# 🔗 Fix Connection Issues - AI Nutrition Assistant

## ❌ Current Problem:
- "Disconnected" status in the app
- "Connection issues detected" warning
- 401 Invalid API Key errors in backend logs

## ✅ Solution Steps:

### Step 1: Get a Groq API Key
1. Go to https://console.groq.com/
2. Sign up for a free account (if you don't have one)
3. Go to "API Keys" section
4. Click "Create API Key"
5. Copy the API key (starts with `gsk_...`)

### Step 2: Update Local Environment
Edit `food-analyzer-backend/env.development`:
```bash
# Change this line:
GROQ_API_KEY=your_groq_api_key_here

# To this (replace with your actual API key):
GROQ_API_KEY=gsk_your_actual_api_key_here
```

### Step 3: Update Render Environment (if deployed)
1. Go to https://render.com
2. Find your `food-analyzer-backend` service
3. Go to **Environment** tab
4. Add/Update:
   - **Key**: `GROQ_API_KEY`
   - **Value**: `gsk_your_actual_api_key_here`
5. Redeploy the service

### Step 4: Restart Your Services
```bash
# Stop current backend (Ctrl+C)
# Then restart:
npm run dev
```

### Step 5: Test Connection
1. Refresh your frontend app
2. Check if "Disconnected" changes to "Connected"
3. Try asking a question

## 🎯 Expected Result:
- ✅ "Connected" status instead of "Disconnected"
- ✅ No more "Connection issues detected" warning
- ✅ AI assistant responds to questions
- ✅ No more 401 errors in backend logs

## 💡 Alternative: Use a Different API Key
If you have an existing Groq API key from before, just update the environment file with that key.

## 🚨 Important:
- Groq API keys start with `gsk_`
- Keep your API key secure and don't share it
- Free tier includes generous limits for testing
