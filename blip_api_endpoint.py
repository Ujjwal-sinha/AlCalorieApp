#!/usr/bin/env python3
"""
Enhanced BLIP API endpoint for accurate food detection
This module provides a dedicated endpoint for BLIP model food analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
from PIL import Image
import logging
import time
from typing import Optional
import sys
import os

# Add the calarieapp directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'calarieapp'))

# Import BLIP components
try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
    import torch
    BLIP_AVAILABLE = True
    print("✅ BLIP model imports successful")
except ImportError as e:
    print(f"⚠️  BLIP model imports failed: {e}")
    BLIP_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app for BLIP analysis
app = FastAPI(
    title="BLIP Food Detection API",
    description="Enhanced BLIP-based food detection and analysis",
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
class BLIPAnalyzeRequest(BaseModel):
    image: str  # base64 encoded image
    prompt: Optional[str] = "Describe the food items in this image in detail:"
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.3

class BLIPAnalyzeResponse(BaseModel):
    success: bool
    description: str
    confidence: float
    processing_time: float
    model_used: str
    error: Optional[str] = None

# Global BLIP model variables
blip_model = None
blip_processor = None
device = None

def initialize_blip_model():
    """Initialize BLIP model for food detection"""
    global blip_model, blip_processor, device
    
    if not BLIP_AVAILABLE:
        logger.warning("BLIP model not available - using mock responses")
        return False
    
    try:
        logger.info("🔄 Loading BLIP model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load BLIP model optimized for food detection
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(device).eval()
        
        logger.info(f"✅ BLIP model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load BLIP model: {e}")
        return False

def enhance_food_detection_prompt(base_prompt: str) -> str:
    """Enhance the prompt for better food detection"""
    food_specific_additions = [
        "Include specific food names, ingredients, cooking methods, and portion details.",
        "Identify all visible foods including main dishes, sides, garnishes, and beverages.",
        "Mention cooking techniques like grilled, fried, baked, steamed, or raw preparations.",
        "Note any visible seasonings, sauces, or condiments.",
        "Describe the presentation style and any cultural cuisine indicators."
    ]
    
    enhanced_prompt = f"{base_prompt} {' '.join(food_specific_additions)}"
    return enhanced_prompt

def analyze_image_with_blip(image: Image.Image, prompt: str, max_tokens: int, temperature: float) -> dict:
    """Analyze image using BLIP model with enhanced food detection"""
    global blip_model, blip_processor, device
    
    if not blip_model or not blip_processor:
        raise Exception("BLIP model not initialized")
    
    start_time = time.time()
    
    try:
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Enhance the prompt for food detection
        enhanced_prompt = enhance_food_detection_prompt(prompt)
        
        # Process image and prompt
        inputs = blip_processor(image, text=enhanced_prompt, return_tensors="pt").to(device)
        
        # Generate description with optimized parameters for food detection
        with torch.no_grad():
            outputs = blip_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=8,  # Higher beam search for better quality
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True
            )
        
        # Decode the generated text
        description = blip_processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the description (remove the prompt if it's included)
        if enhanced_prompt.lower() in description.lower():
            description = description.replace(enhanced_prompt, "").strip()
        
        # Calculate confidence based on generation quality
        confidence = min(0.95, max(0.7, len(description.split()) / 20))
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ BLIP analysis completed in {processing_time:.2f}s")
        
        return {
            "success": True,
            "description": description,
            "confidence": confidence,
            "processing_time": processing_time,
            "model_used": "BLIP-Large"
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ BLIP analysis failed: {e}")
        
        return {
            "success": False,
            "description": "",
            "confidence": 0.0,
            "processing_time": processing_time,
            "model_used": "BLIP-Large",
            "error": str(e)
        }

@app.on_event("startup")
async def startup_event():
    """Initialize BLIP model on startup"""
    logger.info("🚀 Starting BLIP Food Detection API...")
    success = initialize_blip_model()
    if success:
        logger.info("✅ BLIP API ready for food detection")
    else:
        logger.warning("⚠️  BLIP API running in mock mode")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "BLIP Food Detection API",
        "status": "active",
        "blip_available": blip_model is not None,
        "device": str(device) if device else "unknown"
    }

@app.post("/api/blip-analyze", response_model=BLIPAnalyzeResponse)
async def analyze_food_with_blip(request: BLIPAnalyzeRequest):
    """
    Analyze food image using BLIP model
    """
    try:
        logger.info("🔍 Received BLIP food analysis request")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"📸 Image loaded: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        if blip_model and blip_processor:
            # Use real BLIP model
            result = analyze_image_with_blip(
                image, 
                request.prompt, 
                request.max_tokens, 
                request.temperature
            )
            
            if result["success"]:
                return BLIPAnalyzeResponse(
                    success=True,
                    description=result["description"],
                    confidence=result["confidence"],
                    processing_time=result["processing_time"],
                    model_used=result["model_used"]
                )
            else:
                raise HTTPException(status_code=500, detail=result.get("error", "BLIP analysis failed"))
        
        else:
            # Mock response when BLIP is not available
            logger.info("Using mock BLIP response")
            
            mock_descriptions = [
                "This image shows a delicious meal with grilled chicken breast, steamed broccoli, and brown rice. The chicken appears to be seasoned with herbs and has grill marks. The broccoli is bright green and properly cooked. The brown rice looks fluffy and well-prepared.",
                "The image contains a fresh salad with mixed greens, cherry tomatoes, cucumber slices, and what appears to be a vinaigrette dressing. There are also some croutons and possibly some cheese crumbles visible.",
                "This appears to be a pasta dish with marinara sauce, possibly spaghetti or linguine. There are visible herbs like basil, and what looks like parmesan cheese on top. The sauce appears rich and tomato-based.",
                "The image shows a breakfast plate with scrambled eggs, toast, and what appears to be some fruit. The eggs look fluffy and well-cooked, the toast is golden brown, and there might be some berries or sliced fruit as a side."
            ]
            
            import random
            mock_description = random.choice(mock_descriptions)
            
            return BLIPAnalyzeResponse(
                success=True,
                description=mock_description,
                confidence=0.85,
                processing_time=0.5,
                model_used="Mock-BLIP"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in BLIP analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Enhanced BLIP Food Detection API...")
    print("📡 This API provides accurate food detection using BLIP model")
    print("🔗 API: http://localhost:8001")
    print("🔗 Docs: http://localhost:8001/docs")
    
    uvicorn.run(
        "blip_api_endpoint:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )