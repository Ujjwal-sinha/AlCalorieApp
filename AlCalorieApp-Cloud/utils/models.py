import logging
import os
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Try to import AI/ML libraries with error handling
try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import AIMessage
    GROQ_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain Groq library not available: {e}")
    GROQ_AVAILABLE = False

try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Transformers library not available: {e}")
    TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyTorch library not available: {e}")
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NumPy library not available: {e}")
    NUMPY_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from captum.attr import GradientShap
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

try:
    from lime.lime_image import LimeImageExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

def load_yolo_model_with_retry(max_retries=3, timeout=30):
    """Load YOLO model with retry logic and timeout"""
    if not YOLO_AVAILABLE:
        return None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to load YOLO model (attempt {attempt + 1}/{max_retries})")
            
            # Set a timeout for the operation
            start_time = time.time()
            
            # Try loading with different strategies
            yolo_model = None
            
            # Strategy 1: Local file
            if os.path.exists("yolov8n.pt"):
                try:
                    yolo_model = YOLO("yolov8n.pt")
                    logger.info("YOLO loaded successfully from local file")
                    return yolo_model
                except Exception as e:
                    logger.warning(f"Local file loading failed: {e}")
            
            # Strategy 2: Download from Ultralytics
            if yolo_model is None:
                try:
                    yolo_model = YOLO('yolov8n.pt')
                    logger.info("YOLO downloaded and loaded successfully")
                    return yolo_model
                except Exception as e:
                    logger.warning(f"Download loading failed: {e}")
            
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"YOLO loading timed out after {timeout} seconds")
                break
                
        except Exception as e:
            logger.warning(f"YOLO loading attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
    
    logger.error("All YOLO loading attempts failed")
    return None

def load_models() -> Dict[str, Any]:
    """Load AI models with comprehensive error handling"""
    models = {}
    
    # Get API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Load LLM
    if GROQ_AVAILABLE and groq_api_key:
        try:
            models['llm'] = ChatGroq(
                model_name="llama3-8b-8192", 
                api_key=groq_api_key,
                temperature=0.3,
                max_tokens=2000
            )
            logger.info("LLM loaded successfully with LangChain GROQ integration")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            models['llm'] = None
    else:
        logger.warning("LangChain Groq not available or API key missing - LLM features disabled")
        models['llm'] = None
    
    # Load BLIP
    if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            models['processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            models['blip_model'] = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(device).eval()
            models['device'] = device
            logger.info("BLIP loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BLIP: {e}")
            models['processor'] = None
            models['blip_model'] = None
            models['device'] = None
    else:
        logger.warning("Transformers or PyTorch not available - BLIP features disabled")
        models['processor'] = None
        models['blip_model'] = None
        models['device'] = None
    
    # Load Vision Transformer models with robust error handling
    models['vit_processor'] = None
    models['vit_model'] = None
    models['swin_processor'] = None
    models['swin_model'] = None
    
    if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        device = models.get('device', torch.device("cpu"))
        
        # Load ViT-B/16 with multiple strategies
        logger.info("Attempting to load Vision Transformer (ViT-B/16)...")
        
        # Strategy 1: Try loading ViT-B/16
        try:
            from transformers import ViTImageProcessor, ViTForImageClassification
            
            # Load processor first
            models['vit_processor'] = ViTImageProcessor.from_pretrained(
                'google/vit-base-patch16-224',
                cache_dir=None,
                force_download=False
            )
            logger.info("ViT processor loaded successfully")
            
            # Load model
            models['vit_model'] = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                cache_dir=None,
                force_download=False,
                torch_dtype=torch.float32  # Ensure compatibility
            )
            
            # Move to device and set to eval mode
            models['vit_model'] = models['vit_model'].to(device)
            models['vit_model'].eval()
            
            logger.info("✅ Vision Transformer (ViT-B/16) loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load ViT-B/16 model: {e}")
            models['vit_processor'] = None
            models['vit_model'] = None
            
            # Strategy 2: Try smaller ViT model
            try:
                logger.info("Trying smaller ViT model...")
                from transformers import ViTImageProcessor, ViTForImageClassification
                
                models['vit_processor'] = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
                models['vit_model'] = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
                models['vit_model'] = models['vit_model'].to(device).eval()
                
                logger.info("✅ Alternative ViT model loaded successfully")
                
            except Exception as e2:
                logger.warning(f"Failed to load alternative ViT model: {e2}")
                models['vit_processor'] = None
                models['vit_model'] = None
        
        # Load Swin Transformer with memory optimization
        logger.info("Attempting to load Swin Transformer...")
        try:
            from transformers import AutoImageProcessor, SwinForImageClassification
            
            # Load Swin-T (smaller version for memory efficiency)
            models['swin_processor'] = AutoImageProcessor.from_pretrained(
                'microsoft/swin-tiny-patch4-window7-224',
                cache_dir=None,
                force_download=False
            )
            logger.info("Swin processor loaded successfully")
            
            # Load model with memory optimization
            models['swin_model'] = SwinForImageClassification.from_pretrained(
                'microsoft/swin-tiny-patch4-window7-224',
                cache_dir=None,
                force_download=False,
                torch_dtype=torch.float32
            )
            
            # Move to device and set to eval mode
            models['swin_model'] = models['swin_model'].to(device)
            models['swin_model'].eval()
            
            logger.info("✅ Swin Transformer (Swin-T) loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load Swin Transformer: {e}")
            models['swin_processor'] = None
            models['swin_model'] = None
    else:
        logger.warning("Transformers or PyTorch not available - Vision Transformer features disabled")
    
    # Load YOLO (optional) - Enhanced for Streamlit Cloud
    models['yolo_model'] = load_yolo_model_with_retry()
    
    # Load CNN model for visualizations (optional)
    if TORCH_AVAILABLE:
        try:
            import torchvision.models as torchvision_models
            device = models.get('device', torch.device("cpu"))
            models['cnn_model'] = torchvision_models.densenet121(weights="IMAGENET1K_V1")
            models['cnn_model'] = models['cnn_model'].to(device).eval()
            logger.info("CNN model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CNN model: {e}")
            models['cnn_model'] = None
    else:
        models['cnn_model'] = None
    
    # Load CLIP model for similarity scoring
    if CLIP_AVAILABLE and TORCH_AVAILABLE:
        try:
            device = models.get('device', torch.device("cpu"))
            models['clip_processor'] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            models['clip_model'] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            models['clip_model'] = models['clip_model'].to(device).eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}")
            models['clip_processor'] = None
            models['clip_model'] = None
    else:
        logger.warning("CLIP not available - similarity scoring features disabled")
        models['clip_processor'] = None
        models['clip_model'] = None
    
    # Set availability flags
    models['GROQ_AVAILABLE'] = GROQ_AVAILABLE
    models['TRANSFORMERS_AVAILABLE'] = TRANSFORMERS_AVAILABLE
    models['TORCH_AVAILABLE'] = TORCH_AVAILABLE
    models['NUMPY_AVAILABLE'] = NUMPY_AVAILABLE
    models['YOLO_AVAILABLE'] = YOLO_AVAILABLE
    models['CV2_AVAILABLE'] = CV2_AVAILABLE
    models['CAPTUM_AVAILABLE'] = CAPTUM_AVAILABLE
    models['LIME_AVAILABLE'] = LIME_AVAILABLE
    models['CLIP_AVAILABLE'] = CLIP_AVAILABLE
    
    return models

def get_model_status(models: Dict[str, Any]) -> Dict[str, bool]:
    """Get status of all models"""
    return {
        'BLIP (Image Analysis)': models.get('blip_model') is not None,
        'ViT-B/16 (Vision Transformer)': models.get('vit_model') is not None,
        'Swin Transformer': models.get('swin_model') is not None,
        'CLIP (Similarity Scoring)': models.get('clip_model') is not None,
        'LLM (Nutrition Analysis)': models.get('llm') is not None,
        'YOLO (Object Detection)': models.get('yolo_model') is not None,
        'CNN (Visualizations)': models.get('cnn_model') is not None
    }

def check_model_availability(models: Dict[str, Any], required_models: list = None) -> Dict[str, Any]:
    """Check if required models are available"""
    if required_models is None:
        required_models = ['blip_model', 'llm']
    
    status = {}
    for model in required_models:
        status[model] = models.get(model) is not None
    
    return status

def generate_comprehensive_food_analysis(detected_foods: list, nutritional_data: dict, models: Dict[str, Any]) -> dict:
    """Generate comprehensive food analysis using GROQ LLM with LangChain"""
    if not models.get('llm'):
        return {
            'success': False,
            'error': 'LLM not available',
            'summary': 'AI analysis not available',
            'health_score': 5,
            'recommendations': ['Enable GROQ API for detailed analysis']
        }
    
    try:
        # Create a comprehensive prompt template
        prompt_template = ChatPromptTemplate.from_template("""
        You are an expert nutritionist and food analyst. Provide a comprehensive analysis for the following meal:

        DETECTED FOODS: {detected_foods}
        NUTRITIONAL DATA: {nutritional_data}

        Please provide your analysis in this exact format:

        ## COMPREHENSIVE FOOD ANALYSIS

        ### EXECUTIVE SUMMARY:
        [2-3 sentence summary of nutritional profile and health implications]

        ### DETAILED NUTRITIONAL ANALYSIS:
        [Break down each food item with portion estimates and nutritional impact]

        ### MEAL COMPOSITION ASSESSMENT:
        - **Meal Type**: [Breakfast/Lunch/Dinner/Snack]
        - **Cuisine Style**: [If identifiable]
        - **Portion Size**: [Small/Medium/Large/Extra Large]
        - **Cooking Methods**: [Grilled, fried, baked, etc.]
        - **Main Macronutrient**: [Carb-heavy/Protein-rich/Fat-dense/Balanced]

        ### NUTRITIONAL QUALITY SCORE: [1-10]
        **Score**: [X]/10
        **Justification**: [Explain the score based on nutritional balance, variety, and health factors]

        ### STRENGTHS:
        [What's nutritionally good about this meal - 2-3 points]

        ### AREAS FOR IMPROVEMENT:
        [What could be better - 2-3 specific suggestions]

        ### HEALTH RECOMMENDATIONS:
        1. **Immediate Suggestions**: [2-3 specific tips for this meal]
        2. **Portion Adjustments**: [If needed]
        3. **Complementary Foods**: [What to add for better nutrition]
        4. **Timing Considerations**: [Best time to eat this meal]

        ### DIETARY CONSIDERATIONS:
        - **Allergen Information**: [Common allergens present]
        - **Dietary Restrictions**: [Vegan/Vegetarian/Gluten-free compatibility]
        - **Blood Sugar Impact**: [High/Medium/Low glycemic impact]
        - **Special Considerations**: [Any other important dietary notes]

        Be specific, practical, and evidence-based. Focus on actionable insights.
        """)

        # Format the prompt with actual data
        formatted_prompt = prompt_template.format_messages({
            'detected_foods': ', '.join(detected_foods) if isinstance(detected_foods, list) else str(detected_foods),
            'nutritional_data': str(nutritional_data)
        })

        # Generate analysis using GROQ
        response = models['llm'].invoke(formatted_prompt)
        analysis_text = response.content

        # Parse the response to extract key components
        import re
        
        # Extract health score
        score_match = re.search(r'NUTRITIONAL QUALITY SCORE:.*?\*\*Score\*\*:\s*(\d+)/10', analysis_text, re.DOTALL)
        health_score = int(score_match.group(1)) if score_match else 5

        # Extract summary
        summary_match = re.search(r'### EXECUTIVE SUMMARY:\s*\n([\s\S]*?)(?=\n###|$)', analysis_text)
        summary = summary_match.group(1).strip() if summary_match else 'Analysis completed successfully.'

        # Extract recommendations
        recommendations_match = re.search(r'### HEALTH RECOMMENDATIONS:\s*\n([\s\S]*?)(?=\n###|$)', analysis_text)
        recommendations = []
        if recommendations_match:
            rec_text = recommendations_match.group(1)
            recommendations = [line.strip() for line in rec_text.split('\n') if line.strip() and not line.startswith('#')]

        return {
            'success': True,
            'summary': summary,
            'detailed_analysis': analysis_text,
            'health_score': health_score,
            'recommendations': recommendations[:5],  # Limit to 5 recommendations
            'dietary_considerations': ['Review detailed analysis for specific dietary information'],
            'model_used': 'GROQ Llama3-8b-8192'
        }

    except Exception as e:
        logger.error(f"Error generating comprehensive analysis: {e}")
        return {
            'success': False,
            'error': str(e),
            'summary': 'Analysis failed due to technical error',
            'health_score': 5,
            'recommendations': ['Try again later', 'Check GROQ API configuration']
        }
