import logging
import os
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Try to import AI/ML libraries with error handling
try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage
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
            models['llm'] = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)
            logger.info("LLM loaded successfully")
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
    
    # Set availability flags
    models['GROQ_AVAILABLE'] = GROQ_AVAILABLE
    models['TRANSFORMERS_AVAILABLE'] = TRANSFORMERS_AVAILABLE
    models['TORCH_AVAILABLE'] = TORCH_AVAILABLE
    models['NUMPY_AVAILABLE'] = NUMPY_AVAILABLE
    models['YOLO_AVAILABLE'] = YOLO_AVAILABLE
    models['CV2_AVAILABLE'] = CV2_AVAILABLE
    models['CAPTUM_AVAILABLE'] = CAPTUM_AVAILABLE
    models['LIME_AVAILABLE'] = LIME_AVAILABLE
    
    return models

def get_model_status(models: Dict[str, Any]) -> Dict[str, bool]:
    """Get status of all models"""
    return {
        'BLIP (Image Analysis)': models.get('blip_model') is not None,
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
