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
import time

# Add the calarieapp directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'calarieapp'))

# Initialize models independently for FastAPI
models = {}
MODELS_AVAILABLE = False

def initialize_models():
    """Initialize all AI models for food detection"""
    global models, MODELS_AVAILABLE
    
    print("🔄 Initializing AI models for food detection...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("⚠️  GROQ_API_KEY not found - LLM will not be available")
    
    # Device configuration
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load LLM
    try:
        if groq_api_key:
            from langchain_groq import ChatGroq
            models['llm'] = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)
            print("✅ LLM (Groq) loaded successfully")
        else:
            models['llm'] = None
            print("❌ LLM not loaded - missing API key")
    except Exception as e:
        print(f"❌ Failed to load LLM: {e}")
        models['llm'] = None
    
    # Load BLIP
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        print("📥 Loading BLIP model (this may take a while)...")
        models['processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        models['blip_model'] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device).eval()
        print("✅ BLIP model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load BLIP: {e}")
        models['processor'] = None
        models['blip_model'] = None
    
    # Load YOLO
    try:
        from ultralytics import YOLO
        print("📥 Loading YOLO model...")
        models['yolo_model'] = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load YOLO: {e}")
        models['yolo_model'] = None
    
    # Check if any models are available
    blip_available = models.get('blip_model') is not None and models.get('processor') is not None
    yolo_available = models.get('yolo_model') is not None
    llm_available = models.get('llm') is not None
    
    MODELS_AVAILABLE = blip_available or yolo_available or llm_available
    
    print(f"\n📊 Model Status:")
    print(f"   BLIP: {'✅ Available' if blip_available else '❌ Not available'}")
    print(f"   YOLO: {'✅ Available' if yolo_available else '❌ Not available'}")
    print(f"   LLM:  {'✅ Available' if llm_available else '❌ Not available'}")
    print(f"   Overall: {'✅ Models ready' if MODELS_AVAILABLE else '❌ No models available'}")
    
    return MODELS_AVAILABLE

# Import utility functions
try:
    # Import additional dependencies
    import torch
    from PIL import ImageEnhance
    import numpy as np
    import re
    from langchain.schema import HumanMessage
    
    print("✅ Successfully imported dependencies")
    
except ImportError as e:
    print(f"⚠️  Could not import some dependencies: {e}")
    print("Some features may not work correctly")

def extract_food_items_from_text(text: str) -> set:
    """Extract individual food items from descriptive text - exact Python equivalent"""
    items = set()
    text = text.lower().strip()
    
    # Remove common prefixes and phrases
    prefixes_to_remove = [
        "a photo of", "an image of", "this image shows", "i can see", "there is", "there are",
        "the image contains", "visible in the image", "in this image", "this appears to be",
        "looking at this", "from what i can see", "it looks like", "this seems to be"
    ]
    
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text.replace(prefix, "").strip()
    
    # Enhanced separators for better splitting
    separators = [
        ',', ';', ' and ', ' with ', ' including ', ' plus ', ' also ', ' as well as ',
        ' along with ', ' together with ', ' accompanied by ', ' served with ', ' topped with ',
        ' garnished with ', ' mixed with ', ' combined with ', ' containing ', ' featuring '
    ]
    
    # Split text by separators
    parts = [text]
    for sep in separators:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(sep))
        parts = new_parts
    
    # Clean and filter parts
    skip_words = {
        'the', 'and', 'with', 'on', 'in', 'of', 'a', 'an', 'is', 'are', 'was', 'were',
        'this', 'that', 'these', 'those', 'some', 'many', 'few', 'several', 'various',
        'different', 'other', 'another', 'each', 'every', 'all', 'both', 'either',
        'neither', 'one', 'two', 'three', 'first', 'second', 'third', 'next', 'last',
        'here', 'there', 'where', 'when', 'how', 'what', 'which', 'who', 'why',
        'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might', 'must',
        'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been', 'being', 'am',
        'very', 'quite', 'rather', 'pretty', 'really', 'truly', 'actually', 'certainly',
        'probably', 'possibly', 'maybe', 'perhaps', 'likely', 'unlikely'
    }
    
    for part in parts:
        # Clean the part
        part = part.strip().rstrip('.,!?:;')
        part = re.sub(r'\s+', ' ', part)  # Remove extra whitespace
        
        # Skip if too short or is a skip word
        if len(part) <= 2 or part in skip_words:
            continue
        
        # Remove quantity descriptors but keep the food item
        quantity_patterns = [
            r'^(a|an|some|many|few|several|various|different|fresh|cooked|raw|fried|grilled|baked|roasted|steamed|boiled)\s+',
            r'^(small|medium|large|big|huge|tiny|little|sliced|diced|chopped|minced|whole|half|quarter)\s+',
            r'^(hot|cold|warm|cool|spicy|mild|sweet|sour|salty|bitter|savory|delicious|tasty)\s+',
            r'^\d+\s*(pieces?|slices?|cups?|tablespoons?|teaspoons?|ounces?|grams?|pounds?|lbs?|oz|g|kg)\s+(of\s+)?'
        ]
        
        for pattern in quantity_patterns:
            part = re.sub(pattern, '', part, flags=re.IGNORECASE).strip()
        
        # Skip if became too short after cleaning
        if len(part) <= 2:
            continue
        
        # Add the cleaned food item
        items.add(part)
    
    return items

def describe_image_enhanced_api(image: Image.Image) -> str:
    """Fast, optimized food detection with smart strategies for speed - EXACT Python equivalent"""
    global models
    
    if not models.get('processor') or not models.get('blip_model'):
        return "Image analysis unavailable. Please check model loading."
    
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        device = next(models['blip_model'].parameters()).device
        all_food_items = set()
        
        # Strategy 1: Fast BLIP with optimized prompts (reduced from 14 to 3 most effective)
        fast_prompts = [
            "List every food item, ingredient, dish, sauce, and beverage visible in this image:",
            "What are all the foods, vegetables, fruits, meats, grains, and drinks you can see?",
            "Identify each food component including main dishes, sides, garnishes, and condiments:"
        ]
        
        for prompt in fast_prompts:
            try:
                inputs = models['processor'](image, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = models['blip_model'].generate(
                        **inputs, 
                        max_new_tokens=300,  # Reduced from 400
                        num_beams=6,         # Reduced from 10
                        do_sample=True,
                        temperature=0.3,     # Slightly higher for faster generation
                        top_p=0.9,          # Reduced from 0.95
                        repetition_penalty=1.1  # Reduced from 1.2
                    )
                caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                
                if caption.startswith(prompt):
                    caption = caption.replace(prompt, "").strip()
                
                items = extract_food_items_from_text(caption)
                all_food_items.update(items)
                print(f"BLIP found: {len(items)} items")
                
            except Exception as e:
                print(f"BLIP prompt failed: {e}")
        
        # Strategy 2: Single-pass YOLO (optimized)
        if models.get('yolo_model'):
            try:
                img_np = np.array(image)
                # Single confidence level for speed
                results = models['yolo_model'](img_np, conf=0.1, iou=0.4)
                
                # Streamlined food terms (most common ones)
                food_terms = {
                    'apple', 'banana', 'orange', 'tomato', 'potato', 'carrot', 'onion', 'broccoli',
                    'chicken', 'fish', 'beef', 'pork', 'egg', 'cheese', 'bread', 'rice', 'pasta',
                    'pizza', 'burger', 'sandwich', 'salad', 'soup', 'cake', 'coffee', 'tea'
                }
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = models['yolo_model'].names[cls].lower()
                            
                            if conf > 0.1 and any(term in class_name for term in food_terms):
                                all_food_items.add(class_name)
                                
            except Exception as e:
                print(f"YOLO detection failed: {e}")
        
        # Strategy 3: Single enhanced image (fastest enhancement)
        try:
            # Use only the best enhancement technique
            enhancer = ImageEnhance.Contrast(image)
            enhanced_img = enhancer.enhance(1.3)
            
            inputs = models['processor'](enhanced_img, text="What food items are in this image?", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = models['blip_model'].generate(
                    **inputs, 
                    max_new_tokens=200,
                    num_beams=4,  # Reduced for speed
                    temperature=0.4
                )
            caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
            
            if "what food items are in this image?" in caption.lower():
                caption = caption.lower().replace("what food items are in this image?", "").strip()
            
            items = extract_food_items_from_text(caption)
            all_food_items.update(items)
            
        except Exception as e:
            print(f"Enhanced image analysis failed: {e}")
        
        # Fast filtering with essential food keywords
        if all_food_items:
            essential_food_keywords = {
                # Core foods
                'apple', 'banana', 'orange', 'tomato', 'potato', 'carrot', 'onion', 'garlic',
                'chicken', 'beef', 'pork', 'fish', 'egg', 'cheese', 'milk', 'bread', 'rice',
                'pasta', 'pizza', 'burger', 'sandwich', 'salad', 'soup', 'cake', 'cookie',
                'coffee', 'tea', 'juice', 'water', 'sauce', 'oil', 'butter', 'salt', 'pepper',
                'lettuce', 'spinach', 'broccoli', 'corn', 'beans', 'meat', 'vegetable', 'fruit'
            }
            
            final_items = []
            for item in all_food_items:
                item_clean = item.strip().lower()
                if (any(keyword in item_clean for keyword in essential_food_keywords) or 
                    (len(item_clean) > 3 and 
                     not any(non_food in item_clean for non_food in ['plate', 'bowl', 'cup', 'glass', 'table']))):
                    final_items.append(item_clean)
            
            if final_items:
                unique_items = sorted(set(final_items))
                result = ', '.join(unique_items)
                print(f"Fast detection found {len(unique_items)} items: {result}")
                return result
        
        # Quick fallback
        try:
            inputs = models['processor'](image, text="What food is in this image?", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = models['blip_model'].generate(**inputs, max_new_tokens=100, num_beams=3)
            caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
            
            if "what food is in this image?" in caption.lower():
                caption = caption.lower().replace("what food is in this image?", "").strip()
            
            if len(caption.split()) >= 2:
                return caption
                
        except Exception as e:
            print(f"Fallback failed: {e}")
        
        return "Food items detected. Add context for better identification."
            
    except Exception as e:
        print(f"Food detection error: {e}")
        return "Detection failed. Please try again."

def extract_items_and_nutrients(text):
    """Extract food items and nutritional data with enhanced parsing for detailed analysis - EXACT Python equivalent"""
    items = []
    
    try:
        # Enhanced patterns to capture more detailed nutritional information
        patterns = [
            # Standard format with fiber
            r'Item:\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fiber:\s*(\d+\.?\d*)\s*g)?',
            
            # Bullet point format with enhanced nutrients
            r'-\s*Item:\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fiber:\s*(\d+\.?\d*)\s*g)?',
            
            # Simple bullet format
            r'-\s*([^:,]+):\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?',
            
            # Alternative format without "Item:" prefix
            r'-\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                if len(match) >= 3:  # Ensure we have at least item, calories
                    # Handle different match group structures
                    if len(match) == 7 and match[0] and match[1]:  # Pattern with item description
                        item = f"{match[0].strip()}: {match[1].strip()}"
                        calories = int(match[2]) if match[2] else 0
                        protein = float(match[3]) if len(match) > 3 and match[3] else 0
                        carbs = float(match[4]) if len(match) > 4 and match[4] else 0
                        fats = float(match[5]) if len(match) > 5 and match[5] else 0
                        fiber = float(match[6]) if len(match) > 6 and match[6] else 0
                    else:  # Standard patterns
                        item = match[0].strip()
                        calories = int(match[1]) if match[1] else 0
                        protein = float(match[2]) if len(match) > 2 and match[2] else 0
                        carbs = float(match[3]) if len(match) > 3 and match[3] else 0
                        fats = float(match[4]) if len(match) > 4 and match[4] else 0
                        fiber = float(match[5]) if len(match) > 5 and match[5] else 0
                    
                    # Avoid duplicates
                    if not any(existing_item["item"].lower() == item.lower() for existing_item in items):
                        items.append({
                            "item": item,
                            "calories": calories,
                            "protein": protein,
                            "carbs": carbs,
                            "fats": fats,
                            "fiber": fiber
                        })
        
        # Enhanced total extraction from summary sections
        totals = {
            "calories": sum(item["calories"] for item in items),
            "protein": sum(item["protein"] for item in items if item["protein"]),
            "carbs": sum(item["carbs"] for item in items if item["carbs"]),
            "fats": sum(item["fats"] for item in items if item["fats"])
        }
        
        # Try to extract totals from summary sections if individual items weren't found
        if not items:
            total_patterns = [
                r'Total Calories?:\s*(\d{1,4})\s*(?:kcal|cal|calories)?',
                r'Total Protein:\s*(\d+\.?\d*)\s*g',
                r'Total Carbohydrates?:\s*(\d+\.?\d*)\s*g',
                r'Total Fats?:\s*(\d+\.?\d*)\s*g'
            ]
            
            calorie_match = re.search(total_patterns[0], text, re.IGNORECASE)
            protein_match = re.search(total_patterns[1], text, re.IGNORECASE)
            carbs_match = re.search(total_patterns[2], text, re.IGNORECASE)
            fats_match = re.search(total_patterns[3], text, re.IGNORECASE)
            
            if calorie_match:
                total_calories = int(calorie_match.group(1))
                total_protein = float(protein_match.group(1)) if protein_match else total_calories * 0.15 / 4
                total_carbs = float(carbs_match.group(1)) if carbs_match else total_calories * 0.50 / 4
                total_fats = float(fats_match.group(1)) if fats_match else total_calories * 0.35 / 9
                
                items.append({
                    "item": "Complete meal (from totals)",
                    "calories": total_calories,
                    "protein": total_protein,
                    "carbs": total_carbs,
                    "fats": total_fats,
                    "fiber": 5  # Estimated
                })
                
                totals = {
                    "calories": total_calories,
                    "protein": total_protein,
                    "carbs": total_carbs,
                    "fats": total_fats
                }
        
        # Final fallback: extract any calorie numbers
        if not items and len(text.strip()) > 10:
            calorie_matches = re.findall(r'(\d{2,4})\s*(?:cal|kcal|calories)', text, re.IGNORECASE)
            if calorie_matches:
                estimated_calories = max(int(cal) for cal in calorie_matches)
                items.append({
                    "item": "Meal items (detected but not fully parsed)",
                    "calories": estimated_calories,
                    "protein": estimated_calories * 0.15 / 4,
                    "carbs": estimated_calories * 0.50 / 4,
                    "fats": estimated_calories * 0.35 / 9,
                    "fiber": 3
                })
                totals = {
                    "calories": estimated_calories,
                    "protein": estimated_calories * 0.15 / 4,
                    "carbs": estimated_calories * 0.50 / 4,
                    "fats": estimated_calories * 0.35 / 9
                }
        
        print(f"Extracted {len(items)} food items with {totals['calories']} total calories")
        return items, totals
        
    except Exception as e:
        print(f"Error extracting items and nutrients: {e}")
        return [], {"calories": 0, "protein": 0, "carbs": 0, "fats": 0}

def query_langchain(prompt):
    """Query LLM using Groq - EXACT Python equivalent"""
    global models
    
    if not models.get('llm'):
        return "LLM service unavailable."
    try:
        from langchain.schema import HumanMessage
        response = models['llm']([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return f"Error querying LLM: {str(e)}"

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
        if MODELS_AVAILABLE and models.get('llm'):
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

class BLIPAnalyzeRequest(BaseModel):
    image: str  # base64 encoded image
    prompt: Optional[str] = "Describe the food items in this image in detail:"
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.3

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
    logger.info("🚀 Starting AI Calorie App API...")
    
    # Initialize models
    models_ready = initialize_models()
    
    if models_ready:
        logger.info("✅ AI models initialized successfully")
    else:
        logger.info("⚠️  Running in mock mode - no models available")

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
            
            # Step 1: Detect food items using the EXACT Python function
            food_description = describe_image_enhanced_api(image)
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

@app.post("/api/describe-image-enhanced")
async def describe_image_enhanced_endpoint(request: AnalyzeRequest):
    """
    Enhanced image description using BLIP + YOLO - EXACT Python implementation
    """
    try:
        logger.info("🔍 Starting EXACT Python describe_image_enhanced")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"📸 Image loaded: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        if MODELS_AVAILABLE and (models.get('processor') and models.get('blip_model')):
            # Use the enhanced detection function
            try:
                description = describe_image_enhanced_api(image)
                
                if description and description != "Detection failed. Please try again.":
                    # Count items for reporting
                    items_found = len(description.split(',')) if ',' in description else 1
                    
                    return {
                        "success": True,
                        "description": description,
                        "method": "python_blip_yolo_exact",
                        "items_found": items_found,
                        "strategies_used": ["blip_prompts", "yolo_detection", "enhanced_image"]
                    }
                else:
                    # If detection failed, fall through to mock
                    logger.warning("Detection returned empty or failed result")
                
            except Exception as e:
                logger.error(f"Python BLIP/YOLO analysis failed: {e}")
                # Fall through to mock
        
        # Enhanced mock response when models not available
        logger.info("Models not available - using enhanced mock")
        
        # Generate realistic mock based on common food combinations
        import random
        mock_combinations = [
            "grilled chicken breast, steamed broccoli, brown rice, olive oil",
            "mixed green salad, cherry tomatoes, cucumber, feta cheese, olive oil dressing",
            "salmon fillet, quinoa, roasted vegetables, lemon, herbs",
            "pasta, marinara sauce, basil, parmesan cheese, garlic",
            "stir-fried vegetables, tofu, soy sauce, sesame oil, rice",
            "turkey sandwich, lettuce, tomato, whole grain bread, mustard",
            "greek yogurt, mixed berries, granola, honey",
            "oatmeal, banana, walnuts, cinnamon, milk"
        ]
        
        mock_description = random.choice(mock_combinations)
        
        return {
            "success": True,
            "description": mock_description,
            "method": "enhanced_mock",
            "note": "Models not available - using realistic mock data"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced image description failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced description failed: {str(e)}")

@app.post("/api/blip-analyze")
async def blip_analyze_food(request: BLIPAnalyzeRequest):
    """
    Enhanced BLIP-based food analysis endpoint
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
        
        start_time = time.time()
        
        if MODELS_AVAILABLE and models['blip_model'] and models['processor']:
            # Use real BLIP model for enhanced food detection
            try:
                # Ensure image is in RGB format
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                device = next(models['blip_model'].parameters()).device
                
                # Enhanced prompt for food detection
                enhanced_prompt = f"{request.prompt} Include specific food names, ingredients, cooking methods, and portion details."
                
                # Process image and prompt
                inputs = models['processor'](image, text=enhanced_prompt, return_tensors="pt").to(device)
                
                # Generate description with optimized parameters
                with torch.no_grad():
                    outputs = models['blip_model'].generate(
                        **inputs,
                        max_new_tokens=request.max_tokens,
                        num_beams=8,  # Higher beam search for better quality
                        do_sample=True,
                        temperature=request.temperature,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        length_penalty=1.0,
                        early_stopping=True
                    )
                
                # Decode the generated text
                description = models['processor'].decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the description
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
                    "model_used": "BLIP-Base"
                }
                
            except Exception as e:
                logger.error(f"BLIP model analysis failed: {e}")
                # Fall through to mock response
        
        # Mock response when BLIP is not available
        logger.info("Using enhanced mock BLIP response")
        
        processing_time = time.time() - start_time
        
        # Enhanced mock descriptions based on common food scenarios
        mock_descriptions = [
            "This image shows a well-prepared meal featuring grilled chicken breast with visible grill marks, seasoned with herbs like rosemary and thyme. Alongside is steamed broccoli that appears bright green and properly cooked, maintaining its nutritional value. The brown rice looks fluffy and well-prepared, providing a good source of complex carbohydrates. The portion sizes appear appropriate for a balanced meal.",
            
            "The image contains a fresh and colorful salad with mixed greens including lettuce, spinach, and arugula. There are cherry tomatoes that look ripe and fresh, cucumber slices that appear crisp, and what seems to be a light vinaigrette dressing. Some croutons are visible, adding texture, and there might be some cheese crumbles, possibly feta or goat cheese, scattered throughout.",
            
            "This appears to be a pasta dish, likely spaghetti or linguine, with a rich marinara sauce made from tomatoes. Fresh basil leaves are visible as garnish, and there's what looks like freshly grated parmesan cheese on top. The sauce appears to have a good consistency and rich red color, suggesting it's made with quality tomatoes and herbs.",
            
            "The image shows a breakfast plate with fluffy scrambled eggs that appear to be cooked to perfection. There's golden-brown toast, likely whole grain bread, and what appears to be fresh fruit - possibly berries or sliced fruit as a healthy side. The eggs look creamy and well-seasoned, and the overall presentation suggests a nutritious breakfast meal.",
            
            "This appears to be a stir-fry dish with various colorful vegetables including bell peppers, broccoli, carrots, and what looks like snap peas. There might be some protein like chicken or tofu mixed in. The vegetables appear to be cooked just right, maintaining their vibrant colors and likely their crunch. The dish seems to be served over rice or noodles."
        ]
        
        import random
        mock_description = random.choice(mock_descriptions)
        
        return {
            "success": True,
            "description": mock_description,
            "confidence": 0.88,
            "processing_time": processing_time,
            "model_used": "Enhanced-Mock-BLIP"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in BLIP analysis: {e}")
        raise HTTPException(status_code=500, detail=f"BLIP analysis failed: {str(e)}")

class GroqLLMRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 1000

@app.post("/api/groq-llm")
async def query_groq_llm(request: GroqLLMRequest):
    """
    Query Groq LLM - exact equivalent to Python agents
    """
    try:
        logger.info("🤖 Querying Groq LLM")
        
        if MODELS_AVAILABLE and models.get('llm'):
            # Use the exact same LLM as Python
            try:
                from langchain.schema import HumanMessage
                response = models['llm']([HumanMessage(content=request.prompt)])
                
                return {
                    "success": True,
                    "content": response.content,
                    "model": "llama3-8b-8192"
                }
                
            except Exception as e:
                logger.error(f"Groq LLM query failed: {e}")
                # Fall through to mock
        
        # Enhanced mock LLM response
        logger.info("Using enhanced mock LLM response")
        
        # Generate contextual mock response based on prompt content
        if "FOOD DESCRIPTION:" in request.prompt:
            mock_response = """## IDENTIFIED FOOD ITEMS:
- Main dish: Grilled chicken breast (150g)
- Side dish: Steamed broccoli (100g)
- Grain: Brown rice (80g cooked)
- Condiment: Olive oil drizzle (5ml)
- Seasoning: Mixed herbs and spices

## DETAILED NUTRITIONAL BREAKDOWN:
- Item: Grilled chicken breast (150g), Calories: 231, Protein: 43.5g, Carbohydrates: 0g, Fats: 5g, Key nutrients: High protein, B vitamins, selenium
- Item: Steamed broccoli (100g), Calories: 34, Protein: 2.8g, Carbohydrates: 7g, Fats: 0.4g, Key nutrients: Vitamin C, Vitamin K, folate, fiber
- Item: Brown rice (80g cooked), Calories: 216, Protein: 5g, Carbohydrates: 45g, Fats: 1.8g, Key nutrients: Complex carbs, B vitamins, magnesium
- Item: Olive oil drizzle (5ml), Calories: 45, Protein: 0g, Carbohydrates: 0g, Fats: 5g, Key nutrients: Monounsaturated fats, Vitamin E

## MEAL TOTALS:
- Total Calories: 526 kcal
- Total Protein: 51.3g (39% of calories)
- Total Carbohydrates: 52g (40% of calories)
- Total Fats: 12.2g (21% of calories)

## MEAL ASSESSMENT:
- Meal type: Lunch/Dinner
- Cuisine style: Healthy Mediterranean-style
- Nutritional balance: Well-balanced with high protein content
- Portion size: Medium

## HEALTH INSIGHTS:
- Positive aspects: High protein content supports muscle maintenance, good fiber from vegetables, healthy cooking methods
- Areas for improvement: Well-balanced meal with good macronutrient distribution
- Missing nutrients: Could benefit from additional colorful vegetables for more antioxidants

## RECOMMENDATIONS:
- Healthier alternatives: This is already a very healthy meal choice
- Portion adjustments: Portions appear appropriate for most adults
- Complementary foods: Could add a small portion of colorful vegetables like bell peppers or carrots"""
        else:
            mock_response = "This appears to be a well-balanced meal with good nutritional content. The combination of protein, vegetables, and complex carbohydrates provides a solid foundation for a healthy diet."
        
        return {
            "success": True,
            "content": mock_response,
            "model": "mock-llm"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Groq LLM query failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM query failed: {str(e)}")

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