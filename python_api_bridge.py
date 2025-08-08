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
        query_langchain,
        extract_items_and_nutrients,
        models
    )
    MODELS_AVAILABLE = True
    print("✅ Successfully imported existing food detection models")
except ImportError as e:
    print(f"⚠️  Could not import existing models: {e}")
    print("Running in mock mode for development")
    MODELS_AVAILABLE = False

def analyze_food_with_enhanced_prompt(food_description: str, context: str = "") -> Dict[str, Any]:
    """
    Enhanced food analysis function that combines food detection with nutritional analysis
    """
    try:
        # Create comprehensive prompt for LLM
        prompt = f"""
        Analyze the following food items and provide detailed nutritional information:
        
        Food Description: {food_description}
        Additional Context: {context}
        
        Please provide a comprehensive analysis in the following format:
        
        ## COMPREHENSIVE FOOD ANALYSIS
        
        ### IDENTIFIED FOOD ITEMS:
        - Item: [food name] ([portion size]), Calories: [number], Protein: [number]g, Carbs: [number]g, Fats: [number]g
        
        ### NUTRITIONAL TOTALS:
        - Total Calories: [number] kcal
        - Total Protein: [number]g ([percentage]% of calories)
        - Total Carbohydrates: [number]g ([percentage]% of calories)
        - Total Fats: [number]g ([percentage]% of calories)
        
        ### MEAL COMPOSITION ANALYSIS:
        - **Meal Type**: [breakfast/lunch/dinner/snack]
        - **Cuisine Style**: [cuisine type]
        - **Portion Size**: [small/medium/large]
        - **Main Macronutrient**: [protein/carbs/fats]
        
        ### NUTRITIONAL QUALITY ASSESSMENT:
        - **Strengths**: [nutritional benefits]
        - **Areas for Improvement**: [suggestions]
        - **Missing Nutrients**: [what could be added]
        
        ### HEALTH RECOMMENDATIONS:
        1. [recommendation 1]
        2. [recommendation 2]
        3. [recommendation 3]
        
        Be specific with calorie and macronutrient estimates based on typical portion sizes.
        """
        
        # Get analysis from LLM
        if MODELS_AVAILABLE:
            analysis_text = query_langchain(prompt)
        else:
            # Mock analysis for development
            analysis_text = f"""## COMPREHENSIVE FOOD ANALYSIS

### IDENTIFIED FOOD ITEMS:
- Item: {food_description} (estimated portion), Calories: 400, Protein: 25g, Carbs: 35g, Fats: 15g

### NUTRITIONAL TOTALS:
- Total Calories: 400 kcal
- Total Protein: 25g (25% of calories)
- Total Carbohydrates: 35g (35% of calories)
- Total Fats: 15g (34% of calories)

### MEAL COMPOSITION ANALYSIS:
- **Meal Type**: Mixed meal
- **Cuisine Style**: Various
- **Portion Size**: Medium
- **Main Macronutrient**: Balanced

### NUTRITIONAL QUALITY ASSESSMENT:
- **Strengths**: Balanced macronutrients
- **Areas for Improvement**: Add more vegetables
- **Missing Nutrients**: Fiber, vitamins

### HEALTH RECOMMENDATIONS:
1. Well-balanced meal with good protein content
2. Consider adding more vegetables for fiber
3. Good portion size for most adults"""
        
        # Extract structured data
        food_items, nutritional_totals = extract_items_and_nutrients(analysis_text)
        
        # If extraction failed, create fallback data
        if not food_items:
            estimated_calories = 400
            food_items = [{
                "item": food_description,
                "calories": estimated_calories,
                "protein": estimated_calories * 0.15 / 4,
                "carbs": estimated_calories * 0.50 / 4,
                "fats": estimated_calories * 0.35 / 9,
                "fiber": 5
            }]
            nutritional_totals = {
                "calories": estimated_calories,
                "protein": estimated_calories * 0.15 / 4,
                "carbs": estimated_calories * 0.50 / 4,
                "fats": estimated_calories * 0.35 / 9
            }
        
        return {
            "success": True,
            "analysis": analysis_text,
            "food_items": [{"item": item["item"], "description": f"{item['item']} - {item['calories']} calories", "calories": item["calories"]} for item in food_items],
            "nutritional_data": {
                "total_calories": int(nutritional_totals["calories"]),
                "total_protein": round(nutritional_totals["protein"], 1),
                "total_carbs": round(nutritional_totals["carbs"], 1),
                "total_fats": round(nutritional_totals["fats"], 1),
                "items": food_items
            },
            "improved_description": food_description.lower()
        }
        
    except Exception as e:
        logger.error(f"Food analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "analysis": "Analysis failed",
            "food_items": [],
            "nutritional_data": {"total_calories": 0, "total_protein": 0, "total_carbs": 0, "total_fats": 0, "items": []},
            "improved_description": food_description
        }

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