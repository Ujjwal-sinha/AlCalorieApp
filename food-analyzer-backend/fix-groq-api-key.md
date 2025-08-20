# 🔑 Fix Groq API Key Error

## ❌ Current Problem:
```
Health check failed: AuthenticationError: 401 {"error":{"message":"Invalid API Key","type":"invalid_request_error","code":"invalid_api_key"}}
```

## ✅ Solution:

### Step 1: Get a Valid Groq API Key
1. Go to https://console.groq.com/
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the API key (starts with `gsk_...`)

### Step 2: Update Environment Variables

#### Option A: Update Local Development Environment
Edit `food-analyzer-backend/env.development`:
```bash
# Change this line:
GROQ_API_KEY=your_groq_api_key_here

# To this (replace with your actual API key):
GROQ_API_KEY=gsk_your_actual_api_key_here
```

#### Option B: Update Render Environment Variables
1. Go to https://render.com
2. Find your `food-analyzer-backend` service
3. Go to **Environment** tab
4. Add/Update environment variable:
   - **Key**: `GROQ_API_KEY`
   - **Value**: `gsk_your_actual_api_key_here`
5. Redeploy the service

### Step 3: Test the Fix
```bash
# Test locally
curl http://localhost:8000/api/diet/health

# Test deployed service
curl https://food-analyzer-backend.onrender.com/api/diet/health
```

## 💡 Alternative: Disable Groq Health Checks
If you don't want to use Groq right now, you can disable the health checks:

Edit `food-analyzer-backend/src/services/GroqAnalysisService.ts`:
```typescript
async healthCheck(): Promise<{ status: string; available: boolean; error?: string }> {
  // Temporarily disable health checks
  return {
    status: 'disabled',
    available: false,
    error: 'Groq health checks disabled'
  };
}
```

## 🎯 Expected Result:
- ✅ No more 401 errors
- ✅ Health checks pass
- ✅ Groq integration works properly
