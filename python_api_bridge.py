#!/usr/bin/env python3
"""
FastAPI bridge to integrate existing Streamlit food detection with Next.js frontend
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import io
from PIL import Image
import logging
import sys
import os
from typing import Dict, Any, Optional

# Add the calarieapp directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'calarieapp'))

# Import your existing functions
try:
    from app import (
        load_models,
        describe_image_enhanced,
        analyze_food_with_enhanced_prompt,
        models
    )
    MODELS_AVAILABLE = True
    print("✅ Successfully imported existing food detection models")
except ImportError as e:
    print(f"⚠️  Could not import existing models: {e}")
    print("Running in mock mode for development")
    MODELS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AI Calorie App API",
    description="FastAPI bridge for AI-powered food detection and nutrition analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AnalyzeRequest(BaseModel):
    image: str  # base64 encoded image
    context: Optional[str] = ""
    format: Optional[str] = "image/jpeg"

class NutritionData(BaseModel):
    total_calories: int
    total_protein: float
    total_carbs: float
    total_fats: float
    items: list

class AnalyzeResponse(BaseModel):
    success: bool
    analysis: str
    food_items: list
    nutritional_data: NutritionData
    improved_description: str
    error: Optional[str] = None

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize AI models on startup"""
    if MODELS_AVAILABLE:
        try:
            logger.info("Loading AI models...")
            # Models are already loaded when importing from app.py
            logger.info("✅ AI models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
    else:
        logger.info("Running in mock mode - models not available")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Calorie App API is running",
        "models_available": MODELS_AVAILABLE,
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_status = {}
    
    if MODELS_AVAILABLE:
        try:
            model_status = {
                'BLIP': models['blip_model'] is not None,
                'YOLO': models['yolo_model'] is not None,
                'CNN': models['cnn_model'] is not None,
                'LLM': models['llm'] is not None
            }
        except:
            model_status = {"error": "Could not check model status"}
    
    return {
        "status": "healthy",
        "models_available": MODELS_AVAILABLE,
        "model_status": model_status,
        "api_version": "1.0.0"
    }

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_food(request: AnalyzeRequest):
    """
    Analyze food image and return nutritional information
    """
    try:
        logger.info("Received food analysis request")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"Image loaded: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        if MODELS_AVAILABLE:
            # Use existing food detection system
            logger.info("Starting food detection with existing models...")
            
            # Step 1: Detect food items
            food_description = describe_image_enhanced(image)
            logger.info(f"Food description: {food_description}")
            
            # Step 2: Analyze nutrition
            analysis_result = analyze_food_with_enhanced_prompt(food_description, request.context)
            logger.info("Analysis completed successfully")
            
            if analysis_result["success"]:
                return AnalyzeResponse(
                    success=True,
                    analysis=analysis_result["analysis"],
                    food_items=analysis_result["food_items"],
                    nutritional_data=NutritionData(**analysis_result["nutritional_data"]),
                    improved_description=analysis_result["improved_description"]
                )
            else:
                raise HTTPException(status_code=500, detail="Analysis failed")
                
        else:
            # Mock response for development
            logger.info("Using mock response (models not available)")
            
            mock_response = {
                "success": True,
                "analysis": """## COMPREHENSIVE FOOD ANALYSIS

### IDENTIFIED FOOD ITEMS:
- Item: Mixed salad with grilled chicken (200g), Calories: 350, Protein: 35g, Carbs: 15g, Fats: 18g
- Item: Whole grain bread slice (30g), Calories: 80, Protein: 3g, Carbs: 15g, Fats: 1g
- Item: Olive oil dressing (10ml), Calories: 90, Protein: 0g, Carbs: 0g, Fats: 10g

### NUTRITIONAL TOTALS:
- Total Calories: 520 kcal
- Total Protein: 38g (29% of calories)
- Total Carbohydrates: 30g (23% of calories)
- Total Fats: 29g (48% of calories)

### MEAL COMPOSITION ANALYSIS:
- **Meal Type**: Lunch
- **Cuisine Style**: Mediterranean/Healthy
- **Portion Size**: Medium
- **Main Macronutrient**: Balanced with higher fat content

### NUTRITIONAL QUALITY ASSESSMENT:
- **Strengths**: High protein, good fiber content, healthy fats from olive oil
- **Areas for Improvement**: Well-balanced meal
- **Missing Nutrients**: Could add more colorful vegetables for vitamins

### HEALTH RECOMMENDATIONS:
1. **Excellent protein source** - Great for satiety and muscle maintenance
2. **Healthy fats** - Olive oil provides beneficial monounsaturated fats
3. **Balanced meal** - Good combination of macronutrients

### DIETARY CONSIDERATIONS:
- **Allergen Information**: Contains gluten (bread)
- **Dietary Restrictions**: Not suitable for gluten-free diets
- **Blood Sugar Impact**: Low to moderate glycemic impact""",
                "food_items": [
                    {"item": "Mixed salad with grilled chicken", "description": "Mixed salad with grilled chicken - 350 calories", "calories": 350},
                    {"item": "Whole grain bread slice", "description": "Whole grain bread slice - 80 calories", "calories": 80},
                    {"item": "Olive oil dressing", "description": "Olive oil dressing - 90 calories", "calories": 90}
                ],
                "nutritional_data": {
                    "total_calories": 520,
                    "total_protein": 38.0,
                    "total_carbs": 30.0,
                    "total_fats": 29.0,
                    "items": [
                        {"item": "Mixed salad with grilled chicken", "calories": 350, "protein": 35, "carbs": 15, "fats": 18},
                        {"item": "Whole grain bread slice", "calories": 80, "protein": 3, "carbs": 15, "fats": 1},
                        {"item": "Olive oil dressing", "calories": 90, "protein": 0, "carbs": 0, "fats": 10}
                    ]
                },
                "improved_description": "mixed salad with grilled chicken, whole grain bread slice, olive oil dressing"
            }
            
            return AnalyzeResponse(**mock_response)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in food analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-file")
async def analyze_food_file(
    file: UploadFile = File(...),
    context: str = Form("")
):
    """
    Alternative endpoint that accepts file upload directly
    """
    try:
        # Read and validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        image_data = await file.read()
        
        # Convert to base64 and use main analyze function
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        request = AnalyzeRequest(
            image=base64_image,
            context=context,
            format=file.content_type
        )
        
        return await analyze_food(request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI Calorie App API Bridge...")
    print("📡 This bridge connects your Next.js frontend with the existing Python food detection system")
    print("🔗 Frontend: http://localhost:3000")
    print("🔗 API: http://localhost:8000")
    print("🔗 Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "python_api_bridge:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )