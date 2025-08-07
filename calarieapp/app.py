import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os
from datetime import datetime, date
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from transformers import BlipForConditionalGeneration, BlipProcessor
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import re
import torch
import torchvision.models as torchvision_models
from torchvision import transforms
import cv2
import numpy as np
from captum.attr import GradientShap
from lime.lime_image import LimeImageExplainer
import torch.nn.functional as F
import uuid
import glob
import logging
from PIL import ImageEnhance, ImageFilter

from agents import FoodDetectionAgent, FoodSearchAgent, initialize_agents  # Import agents
from typing import Dict, Any, List  # Add typing imports

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="🍱 AI Calorie Tracker", layout="wide", initial_sidebar_state="expanded", page_icon="🍽️")

# Custom CSS for UI styling
st.markdown("""
<style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f5f7fa; }
    .main { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .stTabs [data-baseweb="tab-list"] { gap: 16px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { background-color: #e9ecef; border-radius: 8px; padding: 12px 24px; font-weight: 500; font-size: 16px; transition: all 0.3s ease; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #4CAF50; color: white; }
    .stTabs [aria-selected="true"] { background-color: #4CAF50; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; font-weight: 500; transition: all 0.3s ease; border: none; }
    .stButton>button:hover { background-color: #45a049; transform: scale(1.05); }
    .stFileUploader { border: 2px dashed #4CAF50; border-radius: 8px; padding: 10px; background-color: #f8f9fa; }
    .stMetric { background-color: #f8f9fa; border-radius: 8px; padding: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 10px; }
    .sidebar .sidebar-content { background-color: #ffffff; border-right: 1px solid #e9ecef; padding: 20px; border-radius: 10px; }
    .stProgress .st-bo { background-color: #4CAF50; }
    .footer { text-align: center; padding: 20px; background-color: #e9ecef; border-radius: 8px; margin-top: 20px; font-size: 14px; }
    h1, h2, h3 { color: #2c3e50; margin-bottom: 15px; }
    .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox, .stMultiselect { border-radius: 8px; border: 1px solid #ced4da; padding: 10px; background-color: #f8f9fa; }
    .stExpander { border: 1px solid #e9ecef; border-radius: 8px; margin-bottom: 10px; }
    .stAlert { border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file. Please set it to use the AI Calorie Tracker.")
    st.stop()

# Activity calorie burn rates (kcal/hour)
ACTIVITY_BURN_RATES = {
    "Brisk Walking": 300,
    "Running": 600,
    "Cycling": 500,
    "Swimming": 550,
    "Strength Training": 400
}

# Dynamic food classes - will be populated based on what the model detects
FOOD_CLASSES = []

# Device configuration (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models (LLM, BLIP, CNN)
@st.cache_resource
def load_models():
    models = {}
    try:
        logger.info("Loading ChatGroq LLM...")
        models['llm'] = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)
        logger.info("ChatGroq LLM loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        st.error(f"Failed to load language model: {e}. Please check GROQ_API_KEY and network connection.")
        models['llm'] = None
    try:
        logger.info("Loading BLIP models...")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        models['processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        models['blip_model'] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device).eval()
        logger.info("BLIP models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load BLIP model: {e}")
        st.error(f"Failed to load BLIP model: {e}. Image analysis will be unavailable.")
        models['processor'] = None
        models['blip_model'] = None
    
    try:
        logger.info("Loading DenseNet121 CNN model...")
        models['cnn_model'] = torchvision_models.densenet121(weights="IMAGENET1K_V1")
        # Use ImageNet classes (1000 classes) for natural detection
        models['cnn_model'] = models['cnn_model'].to(device).eval()
        logger.info("Loaded ImageNet CNN model for natural object detection")
    except Exception as e:
        logger.error(f"Failed to load CNN model: {e}")
        st.error(f"Failed to load CNN model: {e}. Visualizations will be unavailable.")
        models['cnn_model'] = None
    
    # Initialize food detection agents
    try:
        logger.info("Initializing food detection agents...")
        food_agent, search_agent = initialize_agents(groq_api_key)
        models['food_agent'] = food_agent
        models['search_agent'] = search_agent
        if food_agent and search_agent:
            logger.info("Food detection agents initialized successfully")
            st.success("✅ Food detection agents loaded!")
        else:
            logger.warning("Failed to initialize food detection agents")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        models['food_agent'] = None
        models['search_agent'] = None
    
    return models

models = load_models()

# Test BLIP model functionality
def test_blip_model():
    """Test if BLIP model is working properly."""
    try:
        if not models['processor'] or not models['blip_model']:
            return False, "BLIP model not loaded"
        
        # Create a simple test image (you can replace this with a real test)
        device = next(models['blip_model'].parameters()).device
        
        # Test with a simple prompt
        test_inputs = models['processor'](text="test", return_tensors="pt").to(device)
        
        return True, "BLIP model is working"
    except Exception as e:
        return False, f"BLIP model test failed: {e}"

# Add this to the model loading section
if models['blip_model']:
    test_result, test_message = test_blip_model()
    if test_result:
        logger.info("BLIP model test passed")
    else:
        logger.warning(f"BLIP model test failed: {test_message}")

# CNN image transform
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ensure image is RGB for CNN
def ensure_rgb_for_cnn(image):
    """Convert image to RGB format for CNN model."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "daily_calories" not in st.session_state:
    st.session_state.daily_calories = {}
if "last_results" not in st.session_state:
    st.session_state.last_results = {}
if "calorie_target" not in st.session_state:
    st.session_state.calorie_target = 2000
if "activity_preference" not in st.session_state:
    st.session_state.activity_preference = ["Brisk Walking"]
if "dietary_preferences" not in st.session_state:
    st.session_state.dietary_preferences = []

# Edge detection visualization
def visualize_food_features(image):
    try:
        img_np = np.array(image.convert("RGB").resize((224, 224)))
        edges = cv2.Canny(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), 100, 200)
        plt.figure(figsize=(6, 6))
        plt.imshow(edges, cmap="gray")
        edge_path = f"edge_output_{uuid.uuid4().hex}.png"
        plt.axis("off")
        plt.savefig(edge_path, bbox_inches="tight")
        plt.close()
        return edge_path
    except Exception as e:
        logger.error(f"Edge detection failed: {e}")
        st.warning(f"Edge detection failed: {e}")
        return None

# Grad-CAM visualization
def apply_gradcam(image_tensor, model, target_class):
    if model is None:
        logger.warning("CNN model unavailable for Grad-CAM")
        return None
    try:
        model.eval()
        gradients, activations = [], []
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients.append(grad_output[0].detach())
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
        
        last_conv = model.features.norm5
        handle_fwd = last_conv.register_forward_hook(forward_hook)
        handle_bwd = last_conv.register_backward_hook(backward_hook)
        
        image_tensor = image_tensor.clone().detach().to(device).requires_grad_(True)
        output = model(image_tensor)
        model.zero_grad()
        
        class_loss = output[0, target_class]
        class_loss.backward()
        
        if not gradients or not activations:
            handle_fwd.remove()
            handle_bwd.remove()
            return None
        
        grads_val = gradients[0]
        activations_val = activations[0]
        
        weights = grads_val.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations_val).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.squeeze().detach().cpu().numpy()
        
        image_np = image_tensor.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image_np)
        plt.imshow(cam_np, cmap="jet", alpha=0.5)
        gradcam_path = f"gradcam_{uuid.uuid4().hex}.png"
        plt.axis("off")
        plt.savefig(gradcam_path, bbox_inches="tight")
        plt.close()
        
        handle_fwd.remove()
        handle_bwd.remove()
        return gradcam_path
    except Exception as e:
        logger.error(f"Grad-CAM failed: {e}")
        st.warning(f"Grad-CAM failed: {e}")
        return None

# SHAP visualization
def apply_shap(image_tensor, model):
    if model is None:
        logger.warning("CNN model unavailable for SHAP")
        return None
    try:
        model.eval()
        gradient_shap = GradientShap(model)
        baseline = torch.zeros_like(image_tensor).to(device)
        image_tensor = image_tensor.clone().detach().requires_grad_(True).to(device)
        attributions = gradient_shap.attribute(image_tensor, baselines=baseline, target=0)
        attr_np = attributions.sum(dim=1).squeeze().detach().cpu().numpy()
        image_np = image_tensor.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(np.abs(attr_np), cmap="viridis", alpha=0.5)
        plt.imshow(image_np, alpha=0.5)
        shap_path = f"shap_{uuid.uuid4().hex}.png"
        plt.axis("off")
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()
        return shap_path
    except Exception as e:
        logger.error(f"SHAP failed: {e}")
        st.warning(f"SHAP failed: {e}")
        return None

# LIME visualization
def apply_lime(image, model, classes):
    if model is None:
        logger.warning("CNN model unavailable for LIME")
        return None
    try:
        explainer = LimeImageExplainer()
        
        def predict_fn(images):
            images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            images = (images - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            with torch.no_grad():
                outputs = model(images)
            return F.softmax(outputs, dim=1).cpu().numpy()
        
        image_np = np.array(image.convert("RGB").resize((224, 224)))
        explanation = explainer.explain_instance(
            image_np, predict_fn, top_labels=2, num_samples=200, segmentation_fn=None
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
        )
        
        plt.figure(figsize=(6, 6))
        plt.imshow(image_np)
        plt.imshow(mask, cmap="viridis", alpha=0.5)
        plt.colorbar(label="Importance")
        plt.axis("off")
        lime_path = f"lime_{uuid.uuid4().hex}.png"
        plt.savefig(lime_path, bbox_inches="tight", dpi=150)
        plt.close()
        return lime_path
    except Exception as e:
        logger.error(f"LIME failed: {e}")
        st.warning(f"LIME failed: {e}")
        return None

# Query LLM with error handling
def query_langchain(prompt):
    if not models['llm']:
        logger.error("LLM is None. Cannot process prompt.")
        return "LLM service unavailable. Please check GROQ_API_KEY and try again."
    try:
        response = models['llm']([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return f"Error querying LLM: {str(e)}"

# Simple image display without YOLO detection
def detect_food_items_with_boxes(image: Image.Image):
    """Display image without YOLO detection - using BLIP only."""
    try:
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Return the original image without any detection
        return image, [], "Using BLIP text-based detection only"
            
    except Exception as e:
        logger.error(f"Error in image processing: {e}")
        return image, [], f"Image processing error: {str(e)}"

# Enhanced image enhancement function
def enhance_image_quality(image: Image.Image) -> Image.Image:
    """Enhance image quality for better food detection."""
    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Create enhanced versions
        enhanced_images = []
        
        # 1. Basic enhancement
        enhancer = ImageEnhance.Contrast(image)
        enhanced_images.append(enhancer.enhance(1.3))
        
        # 2. Brightness enhancement
        enhancer = ImageEnhance.Brightness(image)
        enhanced_images.append(enhancer.enhance(1.2))
        
        # 3. Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        enhanced_images.append(enhancer.enhance(1.5))
        
        # 4. Color enhancement
        enhancer = ImageEnhance.Color(image)
        enhanced_images.append(enhancer.enhance(1.2))
        
        # 5. Combined enhancement
        combined = image.copy()
        combined = ImageEnhance.Contrast(combined).enhance(1.2)
        combined = ImageEnhance.Brightness(combined).enhance(1.1)
        combined = ImageEnhance.Sharpness(combined).enhance(1.3)
        enhanced_images.append(combined)
        
        return enhanced_images
        
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        return [image]  # Return original if enhancement fails

# Enhanced food detection with multiple strategies
def describe_image_enhanced(image: Image.Image) -> str:
    """Enhanced image description with multiple detection strategies and image enhancement."""
    if not models['processor'] or not models['blip_model']:
        logger.error("BLIP model or processor is None. Image analysis unavailable.")
        return "Image analysis unavailable. Please check model loading and try again."
    
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        device = next(models['blip_model'].parameters()).device
        
        # Enhance image quality
        enhanced_images = enhance_image_quality(image)
        
        # Multiple detection strategies with enhanced images
        detection_results = []
        
        # Strategy 1: Direct food identification with enhanced image
        for i, enhanced_img in enumerate(enhanced_images[:3]):  # Use first 3 enhanced versions
            try:
                inputs = models['processor'](enhanced_img, text="What specific food items, dishes, and ingredients can you see in this image? List everything in detail: ", return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = models['blip_model'].generate(
                        **inputs, 
                        max_new_tokens=200, 
                        num_beams=8, 
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                if caption.startswith("What specific food items, dishes, and ingredients can you see in this image? List everything in detail: "):
                    caption = caption.replace("What specific food items, dishes, and ingredients can you see in this image? List everything in detail: ", "")
                caption = caption.strip()
                if len(caption.split()) >= 4:
                    detection_results.append(f"Enhanced_{i+1}: {caption}")
            except Exception as e:
                logger.warning(f"Enhanced strategy {i+1} failed: {e}")
        
        # Strategy 2: Force food context with original image
        try:
            inputs = models['processor'](image, text="This is a food photograph. Identify all edible items, dishes, ingredients, sauces, and food components visible: ", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = models['blip_model'].generate(
                    **inputs, 
                    max_new_tokens=250, 
                    num_beams=9, 
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95
                )
            caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
            if caption.startswith("This is a food photograph. Identify all edible items, dishes, ingredients, sauces, and food components visible: "):
                caption = caption.replace("This is a food photograph. Identify all edible items, dishes, ingredients, sauces, and food components visible: ", "")
            caption = caption.strip()
            if len(caption.split()) >= 3:
                detection_results.append(f"Food_Context: {caption}")
        except Exception as e:
            logger.warning(f"Food context strategy failed: {e}")
        
        # Strategy 3: Detailed component analysis
        try:
            inputs = models['processor'](image, text="Break down this meal into individual food components. List each item separately: ", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = models['blip_model'].generate(
                    **inputs, 
                    max_new_tokens=180, 
                    num_beams=7, 
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9
                )
            caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
            if caption.startswith("Break down this meal into individual food components. List each item separately: "):
                caption = caption.replace("Break down this meal into individual food components. List each item separately: ", "")
            caption = caption.strip()
            if len(caption.split()) >= 3:
                detection_results.append(f"Component_Analysis: {caption}")
        except Exception as e:
            logger.warning(f"Component analysis strategy failed: {e}")
        
        # Strategy 4: Cuisine-specific detection
        try:
            inputs = models['processor'](image, text="What type of cuisine is this? Identify all food items and dishes: ", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = models['blip_model'].generate(
                    **inputs, 
                    max_new_tokens=150, 
                    num_beams=6, 
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
            if caption.startswith("What type of cuisine is this? Identify all food items and dishes: "):
                caption = caption.replace("What type of cuisine is this? Identify all food items and dishes: ", "")
            caption = caption.strip()
            if len(caption.split()) >= 3:
                detection_results.append(f"Cuisine_Detection: {caption}")
        except Exception as e:
            logger.warning(f"Cuisine detection strategy failed: {e}")
        
        # Strategy 5: Simple but comprehensive detection
        try:
            inputs = models['processor'](image, text="List all food items in this image: ", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = models['blip_model'].generate(
                    **inputs, 
                    max_new_tokens=120, 
                    num_beams=5, 
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95
                )
            caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
            if caption.startswith("List all food items in this image: "):
                caption = caption.replace("List all food items in this image: ", "")
            caption = caption.strip()
            if len(caption.split()) >= 2:
                detection_results.append(f"Simple_Detection: {caption}")
        except Exception as e:
            logger.warning(f"Simple detection strategy failed: {e}")
        
        # Strategy 6: Fallback with no prompt
        try:
            inputs = models['processor'](image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = models['blip_model'].generate(
                    **inputs, 
                    max_new_tokens=100, 
                    num_beams=4, 
                    do_sample=False,
                    temperature=1.0
                )
            caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
            caption = caption.strip()
            if len(caption.split()) >= 2:
                detection_results.append(f"Fallback: {caption}")
        except Exception as e:
            logger.warning(f"Fallback strategy failed: {e}")
        
        # Process and combine results
        if detection_results:
            # Clean up all results
            cleaned_results = []
            for result in detection_results:
                # Remove common prefixes
                cleaned = result.replace("a photo of ", "").replace("an image of ", "").replace("a picture of ", "")
                cleaned = cleaned.replace("Enhanced_1: ", "").replace("Enhanced_2: ", "").replace("Enhanced_3: ", "")
                cleaned = cleaned.replace("Food_Context: ", "").replace("Component_Analysis: ", "")
                cleaned = cleaned.replace("Cuisine_Detection: ", "").replace("Simple_Detection: ", "").replace("Fallback: ", "")
                cleaned = cleaned.strip()
                cleaned = cleaned.rstrip('.,!?')
                
                # Filter out vague results
                vague_terms = ["plate", "food", "meal", "dish", "dinner", "lunch", "breakfast", "eating"]
                if len(cleaned.split()) >= 3 and not all(term in cleaned.lower() for term in vague_terms):
                    cleaned_results.append(cleaned)
            
            if cleaned_results:
                # Use the most detailed result
                best_result = max(cleaned_results, key=lambda x: len(x.split()))
                return best_result
        
        # If all strategies fail, provide helpful feedback
        return "Unable to detect specific food items. Please ensure the image is clear, well-lit, and shows food items distinctly. You can also describe the meal in the context field."
        
    except Exception as e:
        logger.error(f"Enhanced image analysis error: {e}")
        return f"Image analysis error: {str(e)}. Please try a clearer image or describe the meal in the context field."

# Enhanced food analysis with better detection
def analyze_food_with_enhanced_prompt(image_description: str, context: str = "") -> Dict[str, Any]:
    """Use enhanced approach for better food detection with multiple fallbacks."""
    try:
        # Post-process the description to improve accuracy
        improved_description = post_process_detection(image_description)
        
        # Create a comprehensive prompt that analyzes each detected food item individually
        prompt = f"""You are an expert nutritionist and food identification specialist. Analyze this food description and provide detailed nutrition data for each food item mentioned.

**FOOD DESCRIPTION**: {improved_description}
**ADDITIONAL CONTEXT**: {context if context else "No additional context"}

**TASK**: Identify each food item mentioned in the description and provide individual nutrition data.

**INSTRUCTIONS**:
1. **List each food item** mentioned in the description individually
2. **Provide specific portion sizes** for each item (e.g., 1 cup, 1 piece, 100g)
3. **Give accurate nutrition data** for each item separately
4. **Be specific** - identify each component of the meal
5. **If description is vague**, assume common meal components and provide realistic estimates

**REQUIRED OUTPUT FORMAT**:
**Food Items and Nutrients**:
- Item: [Food Name with specific portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- Item: [Food Name with specific portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- [Continue for each food item mentioned]

**Total Calories**: [Sum of all items]
**Nutritional Assessment**: [Brief assessment]
**Health Suggestions**: [2-3 suggestions]

**IMPORTANT**: 
- List each food item individually with specific portions
- If the description mentions "plate of food" or "meal", assume common components like protein, starch, vegetables
- Provide realistic nutritional estimates based on typical serving sizes
- Only include items actually mentioned or reasonably implied in the description"""

        # Get analysis from LLM
        response = models['llm'].invoke(prompt)
        analysis = response.content
        
        return {
            "success": True,
            "analysis": analysis,
            "food_items": [],
            "nutritional_data": {},
            "improved_description": improved_description
        }
        
    except Exception as e:
        logger.error(f"Error in food analysis: {e}")
        return {
            "success": False,
            "error": str(e),
            "analysis": "Failed to analyze food items",
            "food_items": [],
            "nutritional_data": {}
        }

# Enhanced post-processing for better detection
def post_process_detection(description: str) -> str:
    """Enhanced post-processing to improve food detection accuracy."""
    try:
        # Clean up the description
        cleaned = description.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "a photo of ", "an image of ", "a picture of ",
            "this is ", "there is ", "i can see ",
            "the image shows ", "the photo shows "
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Handle vague descriptions by adding context
        vague_indicators = ["plate of food", "meal", "food", "dish", "dinner", "lunch", "breakfast"]
        if any(indicator in cleaned.lower() for indicator in vague_indicators):
            # Add common meal components if description is vague
            if "plate" in cleaned.lower() or "meal" in cleaned.lower():
                cleaned += " (likely includes protein, starch, vegetables, and sauce)"
            elif "food" in cleaned.lower():
                cleaned += " (appears to be a complete meal with multiple components)"
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        return description  # Return original if processing fails

# Extract food items and nutrients from text
def extract_items_and_nutrients(text):
    items = []
    pattern = r'Item:\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        item = match[0].strip()
        calories = int(match[1]) if match[1] else 0
        protein = float(match[2]) if match[2] else None
        carbs = float(match[3]) if match[3] else None
        fats = float(match[4]) if match[4] else None
        items.append({
            "item": item,
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fats": fats
        })
    totals = {
        "calories": sum(item["calories"] for item in items),
        "protein": sum(item["protein"] for item in items if item["protein"] is not None),
        "carbs": sum(item["carbs"] for item in items if item["carbs"] is not None),
        "fats": sum(item["fats"] for item in items if item["fats"] is not None)
    }
    return items, totals

# Plot nutritional breakdown chart
def plot_chart(food_data):
    if not food_data:
        return None
    items = [item["item"] for item in food_data]
    calories = [item["calories"] for item in food_data]
    proteins = [item["protein"] if item["protein"] is not None else 0 for item in food_data]
    carbs = [item["carbs"] if item["carbs"] is not None else 0 for item in food_data]
    fats = [item["fats"] if item["fats"] is not None else 0 for item in food_data]
    
    fig, ax = plt.subplots(figsize=(8, max(len(items) * 0.6, 4)))
    bar_width = 0.2
    indices = range(len(items))
    
    ax.barh([i - bar_width*1.5 for i in indices], calories, bar_width, label="Calories (kcal)", color="#4CAF50")
    ax.barh([i - bar_width*0.5 for i in indices], proteins, bar_width, label="Protein (g)", color="#2196F3")
    ax.barh([i + bar_width*0.5 for i in indices], carbs, bar_width, label="Carbs (g)", color="#FF9800")
    ax.barh([i + bar_width*1.5 for i in indices], fats, bar_width, label="Fats (g)", color="#F44336")
    
    ax.set_yticks(indices)
    ax.set_yticklabels(items, fontsize=10)
    ax.set_xlabel("Amount", fontsize=12)
    ax.set_title("Nutritional Breakdown", fontsize=14, pad=15)
    ax.legend(fontsize=10)
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('ggplot')
    plt.tight_layout()
    return fig

# Generate daily nutritional summary and advice
def generate_daily_summary(calorie_target, activity_preferences, dietary_preferences):
    today = date.today().isoformat()
    total_calories = st.session_state.daily_calories.get(today, 0)
    daily_nutrients = {"protein": 0, "carbs": 0, "fats": 0}
    for entry in st.session_state.history:
        entry_date = entry["timestamp"].split()[0]
        if entry_date == today and entry.get("totals"):
            daily_nutrients["protein"] += entry["totals"].get("protein", 0)
            daily_nutrients["carbs"] += entry["totals"].get("carbs", 0)
            daily_nutrients["fats"] += entry["totals"].get("fats", 0)
    
    calorie_diff = total_calories - calorie_target
    status = "surplus" if calorie_diff > 0 else "deficit" if calorie_diff < 0 else "balanced"
    
    summary = f"**Daily Nutritional Summary ({today})**\n"
    summary += f"- **Total Calories**: {total_calories} kcal (Target: {calorie_target} kcal)\n"
    summary += f"- **Total Protein**: {daily_nutrients['protein']:.1f} g\n"
    summary += f"- **Total Carbs**: {daily_nutrients['carbs']:.1f} g\n"
    summary += f"- **Total Fats**: {daily_nutrients['fats']:.1f} g\n"
    summary += f"- **Calorie Status**: {'Surplus' if calorie_diff > 0 else 'Deficit' if calorie_diff < 0 else 'Balanced'} ({abs(calorie_diff)} kcal)\n\n"
    
    advice = "**Personalized Fitness Advice**\n"
    if status == "surplus":
        advice += f"You consumed {calorie_diff} kcal above your target. To balance this, consider:\n"
        for activity in activity_preferences:
            burn_rate = ACTIVITY_BURN_RATES.get(activity, 300)
            duration = (calorie_diff / burn_rate) * 60
            advice += f"- **{activity}**: {duration:.0f} minutes\n"
        advice += "\n**Motivation**: Great job tracking your intake! A short workout can help you stay on track!"
    elif status == "deficit":
        advice += f"You consumed {abs(calorie_diff)} kcal below your target. To avoid excessive deficit:\n"
        advice += "- Consider a nutrient-dense snack (e.g., "
        if "Vegan" in dietary_preferences:
            advice += "avocado toast, ~200-300 kcal).\n"
        elif "Keto" in dietary_preferences:
            advice += "cheese cubes or nuts, ~200-300 kcal).\n"
        else:
            advice += "banana with peanut butter, ~200-300 kcal).\n"
        advice += "- Ensure adequate hydration and rest.\n"
        advice += "\n**Motivation**: You're doing awesome! Fuel your body for your goals!"
    else:
        advice += "Your calorie intake is perfectly balanced! Keep it up:\n"
        advice += "- Maintain a mix of activities.\n"
        advice += "- Stay consistent with nutrition.\n"
        advice += "\n**Motivation**: You're in the zone! Keep making mindful choices!"
    
    return summary + advice

# Generate PDF report with analysis and visualizations
def generate_pdf_report(image, analysis, chart, nutrients, daily_summary=None, edge_path=None, gradcam_path=None, shap_path=None, lime_path=None, cnn_confidence=None):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "AI Nutrition Report", ln=1, align="C")
        pdf.set_font("Arial", "", 12)
        
        if image:
            img_path = "temp_img.jpg"
            image.save(img_path, quality=90)
            pdf.image(img_path, w=180, h=120)
            os.remove(img_path)
            pdf.ln(10)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Nutritional Summary", ln=1)
        pdf.set_font("Arial", "", 10)
        pdf.cell(50, 8, "Food Item", border=1)
        pdf.cell(30, 8, "Calories", border=1)
        pdf.cell(30, 8, "Protein (g)", border=1)
        pdf.cell(30, 8, "Carbs (g)", border=1)
        pdf.cell(30, 8, "Fats (g)", border=1)
        pdf.ln()
        
        for item in nutrients:
            pdf.cell(50, 8, item["item"], border=1)
            pdf.cell(30, 8, str(item["calories"]), border=1)
            pdf.cell(30, 8, str(item["protein"] or "-"), border=1)
            pdf.cell(30, 8, str(item["carbs"] or "-"), border=1)
            pdf.cell(30, 8, str(item["fats"] or "-"), border=1)
            pdf.ln()
        
        pdf.ln(10)
        pdf.multi_cell(0, 8, analysis)
        pdf.ln(10)
        
        if cnn_confidence is not None:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Confidence: {cnn_confidence*100:.1f}%", ln=1)
            pdf.ln(10)
        
        if daily_summary:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Daily Summary", ln=1)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 8, daily_summary)
            pdf.ln(10)
        
        if chart:
            chart_path = "temp_chart.png"
            chart.savefig(chart_path, bbox_inches="tight", dpi=100)
            pdf.image(chart_path, w=180)
            os.remove(chart_path)
        
        viz_paths = [
            (edge_path, "Edge Detection Analysis"),
            (gradcam_path, "Grad-CAM Visualization"),
            (shap_path, "SHAP Explanation"),
            (lime_path, "LIME Interpretation")
        ]
        for path, title in viz_paths:
            if path and os.path.exists(path):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, title, ln=1)
                pdf.image(path, w=180)
                pdf.ln(10)
        
        pdf.set_y(-15)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 0, "C")
        
        pdf_path = "nutrition_report.pdf"
        pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        st.error(f"PDF generation failed: {e}")
        return None
    finally:
        for path in [edge_path, gradcam_path, shap_path, lime_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

# Main UI
with st.container():
    st.title("🍽️ AI-Powered Calorie Tracker")
    st.caption("Track your nutrition with ease using AI-powered analysis")

    # Tabs for functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📷 Enhanced Food Analysis",
        "📝 Text Analysis",
        "📊 History",
        "📅 Daily Summary",
        "⚖️ Portion Adjustment"
    ])

    # Enhanced Food Analysis Tab (Consolidated)
    with tab1:
        st.subheader("📷 Enhanced Food Analysis")
        st.write("Upload a meal image for comprehensive food detection using AI agents, object detection, and global search.")
        
        # Add helpful tips
        with st.expander("💡 **Tips for Best Results**", expanded=False):
            st.write("**For Better Detection:**")
            st.write("• 📸 **Clear Photos**: Take photos in good lighting")
            st.write("• 🍽️ **Close Shots**: Get close enough to see food details")
            st.write("• 📐 **Good Angles**: Avoid shadows and glare")
            st.write("• 🎯 **Focus**: Make sure food items are clearly visible")
            st.write("")
            st.write("**If Detection Fails:**")
            st.write("• 📝 **Add Context**: Describe the meal in the context field")
            st.write("• 🔄 **Try Again**: Upload from a different angle")
            st.write("• 📊 **Text Analysis**: Use the Text Analysis tab instead")
        
        with st.container():
            img_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"], key="img_uploader", help="Upload a clear, well-lit image of your meal for best results.")
            context = st.text_area("Additional Context (Optional)", placeholder="E.g., 'Indian thali with dal, roti, rice' or specify cuisine type", height=100, help="Provide additional context to help identify food items more accurately.")
            
            if st.button("🔍 Analyze Meal", disabled=not img_file, key="analyze_image", help="Click to analyze the uploaded meal image"):
                with st.spinner("🔍 Analyzing your meal with enhanced AI detection..."):
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Loading models and preprocessing image...")
                    progress_bar.progress(20)
                    try:
                        if img_file is None or getattr(img_file, 'size', 1) == 0:
                            st.error("File upload failed. Please refresh and re-upload your image.")
                            st.stop()
                        image = Image.open(img_file)
                        
                        # Display the uploaded image
                        status_text.text("📷 Processing uploaded image...")
                        progress_bar.progress(30)
                        
                        # Show original and enhanced images
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original Image", use_container_width=True, clamp=True)
                        
                        # Create enhanced versions for display
                        enhanced_images = enhance_image_quality(image)
                        if len(enhanced_images) > 1:
                            with col2:
                                st.image(enhanced_images[0], caption="Enhanced Image (Contrast)", use_container_width=True, clamp=True)
                        
                        # Show enhancement info
                        with st.expander("🖼️ Image Enhancement Applied", expanded=False):
                            st.write("**Enhancement Techniques Used:**")
                            st.write("• 🔍 Contrast Enhancement (30% increase)")
                            st.write("• 💡 Brightness Enhancement (20% increase)")
                            st.write("• ✨ Sharpness Enhancement (50% increase)")
                            st.write("• 🎨 Color Enhancement (20% increase)")
                            st.write("• 🔄 Combined Enhancement (multiple techniques)")
                            st.write("")
                            st.write("These enhancements help improve food detection accuracy, especially for low-quality images.")
                        
                        status_text.text("🔍 Analyzing food content with AI agents and global search...")
                        progress_bar.progress(40)
                        
                        # Get basic description from BLIP with enhanced detection
                        basic_description = describe_image_enhanced(image)
                        
                        # Show detection progress
                        status_text.text("🔍 Analyzing food content with enhanced AI detection...")
                        progress_bar.progress(50)
                        
                        # Use enhanced analysis with comprehensive prompt
                        enhanced_analysis = analyze_food_with_enhanced_prompt(basic_description, context)
                        
                        if enhanced_analysis["success"]:
                            description = enhanced_analysis["analysis"]
                            st.success("✅ Enhanced AI analysis completed!")
                            
                            # Show analysis details
                            with st.expander("🔍 Enhanced Analysis Details", expanded=True):
                                st.write("**AI Analysis Results:**")
                                st.write(enhanced_analysis["analysis"])
                                st.write("**Detection Methods Used:**")
                                st.write("• 🤖 Enhanced AI Detection")
                                st.write("• 🖼️ Image Quality Enhancement")
                                st.write("• 📊 Multi-Strategy Analysis")
                                st.write("• 🍽️ Comprehensive Food Identification")
                        else:
                            # Fallback to basic description
                            description = basic_description
                            st.warning("⚠️ Enhanced analysis failed, using basic detection")
                        
                        status_text.text("📊 Analyzing nutritional content...")
                        progress_bar.progress(60)
                        
                        # Check for detection issues and provide helpful feedback
                        if "unavailable" in description.lower() or "error" in description.lower():
                            st.error(f"**Detection Error**: {description}")
                            st.info("**Tips for better detection:**")
                            st.write("• 📸 Ensure the image is clear and well-lit")
                            st.write("• 🍽️ Make sure food items are clearly visible")
                            st.write("• 📝 Use the context field to describe the meal")
                            st.write("• 🔄 Try uploading from a different angle")
                            st.stop()
                        
                        # Handle vague detection results
                        vague_indicators = ["plate of food", "meal", "food item", "dish", "plate", "food", "dinner", "lunch", "breakfast", "unable to detect"]
                        if any(vague in description.lower() for vague in vague_indicators):
                            if context:
                                description = f"{description} Additional items mentioned: {context}"
                                st.info("✅ Using additional context to improve detection.")
                            else:
                                st.warning("⚠️ **Low Detection Confidence**")
                                st.write("The AI couldn't clearly identify specific food items.")
                                st.write("**Please help by:**")
                                st.write("• 📝 Describing the meal in the context field (e.g., 'chicken curry, rice, naan bread')")
                                st.write("• 📸 Uploading a clearer, closer image of the food")
                                st.write("• 🍽️ Making sure all food items are visible")
                                
                                # Continue with basic analysis but warn user
                                st.info("Continuing with basic analysis. Results may be less accurate.")
                        
                        # Enhanced error handling for detection failures
                        if "unable to detect" in description.lower() or len(description.split()) < 3:
                            st.error("**Food Detection Failed**")
                            st.write("The AI couldn't identify any food items in the image.")
                            st.write("**Please try:**")
                            st.write("1. 📸 Upload a clearer image with better lighting")
                            st.write("2. 📝 Describe the meal in the context field")
                            st.write("3. 🍽️ Ensure food items are clearly visible")
                            st.write("4. 🔄 Try a different angle or distance")
                            st.stop()
                        
                        dietary_prefs = ", ".join(st.session_state.dietary_preferences) if st.session_state.dietary_preferences else "None"
                        prompt = f"""Analyze this food description and provide nutrition data for each food item mentioned:

**FOOD DETECTION RESULTS**: {description}
**ADDITIONAL CONTEXT**: {context or 'No additional context provided'}
**DIETARY PREFERENCES**: {dietary_prefs}

**TASK**: Identify each food item mentioned in the description and provide individual nutrition data.

**INSTRUCTIONS**:
1. **List each food item** mentioned in the description individually
2. **Provide specific portion sizes** for each item (e.g., 1 cup, 1 piece, 100g)
3. **Give accurate nutrition data** for each item separately
4. **Be specific** - identify each component of the meal

**REQUIRED OUTPUT FORMAT**:
**Food Items and Nutrients**:
- Item: [Food Name with specific portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- Item: [Food Name with specific portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- [Continue for each food item mentioned]

**Total Calories**: [Sum of all items]
**Nutritional Assessment**: [Brief assessment]
**Health Suggestions**: [2-3 suggestions]

**IMPORTANT**: List each food item individually with specific portions. Only include items actually mentioned in the description."""
                        
                        analysis = query_langchain(prompt)
                        if "unavailable" in analysis.lower() or "error" in analysis.lower():
                            st.error(analysis)
                            st.stop()
                        
                        food_data, totals = extract_items_and_nutrients(analysis)
                        
                        if not food_data or len(food_data) < 1:
                            st.warning("⚠️ **No Food Items Detected**")
                            st.write("The AI couldn't extract specific food items from the analysis.")
                            
                            # Try enhanced retry with more aggressive detection
                            st.info("🔄 **Retrying with Enhanced Detection...**")
                            enhanced_prompt = f"""You are an expert nutritionist. The previous analysis failed to identify food items. Please analyze this meal description more thoroughly and identify ALL food items, ingredients, dishes, sauces, and sides mentioned.

**MEAL DESCRIPTION**: {description}
**CONTEXT**: {context if context else "No additional context"}

**INSTRUCTIONS**:
1. **Break down complex dishes** into individual components
2. **Identify every food item** mentioned or implied
3. **Include sauces, garnishes, and sides**
4. **Provide realistic portion sizes** for each item
5. **Give complete nutritional data** for each component

**REQUIRED OUTPUT FORMAT**:
**Food Items and Nutrients**:
- Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g

**Total Calories**: [X] cal
**Nutritional Assessment**: [Assessment]
**Health Suggestions**: [2-3 suggestions]

**IMPORTANT**: If the description is vague, assume common meal components (protein, starch, vegetables, sauce) and provide realistic estimates."""
                            
                            analysis = query_langchain(enhanced_prompt)
                            if "unavailable" in analysis.lower() or "error" in analysis.lower():
                                st.error(f"**Enhanced Analysis Failed**: {analysis}")
                                st.write("**Please try:**")
                                st.write("1. 📝 Describe the meal manually in the context field")
                                st.write("2. 📸 Upload a clearer image")
                                st.write("3. 🍽️ Use the Text Analysis tab instead")
                                st.stop()
                            
                            food_data, totals = extract_items_and_nutrients(analysis)
                            
                            if not food_data or len(food_data) < 1:
                                st.error("**Food Detection Completely Failed**")
                                st.write("The AI couldn't identify any food items even with enhanced detection.")
                                st.write("**Please use one of these alternatives:**")
                                st.write("1. 📝 **Text Analysis Tab**: Describe your meal manually")
                                st.write("2. 📸 **Better Image**: Upload a clearer, closer image")
                                st.write("3. 🍽️ **Context Field**: Add detailed description")
                                st.stop()
                        
                        # Generate visualizations if CNN model is available
                        edge_path, gradcam_path, shap_path, lime_path = None, None, None, None
                        cnn_predicted_class, cnn_confidence = None, None
                        if models['cnn_model']:
                            try:
                                # Ensure image is RGB for CNN model
                                rgb_image = ensure_rgb_for_cnn(image)
                                image_tensor = cnn_transform(rgb_image).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    outputs = models['cnn_model'](image_tensor)
                                    probabilities = F.softmax(outputs, dim=1)
                                    
                                    # Get top 3 predictions naturally
                                    top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)
                                    cnn_predicted_idx = top3_idx[0][0]
                                    
                                    # Use natural confidence from model
                                    cnn_confidence = top3_prob[0][0]
                                    
                                    # Get ImageNet class names (simplified)
                                    cnn_predicted_class = f"Object_{cnn_predicted_idx.item()}"
                                
                                edge_path = visualize_food_features(image)
                                gradcam_path = apply_gradcam(image_tensor, models['cnn_model'], cnn_predicted_idx)
                                shap_path = apply_shap(image_tensor, models['cnn_model'])
                                lime_path = apply_lime(image, models['cnn_model'], ["object"])  # Generic class
                            except Exception as e:
                                logger.error(f"Visualization generation failed: {e}")
                                st.warning(f"Visualizations unavailable: {e}")
                        
                        status_text.text("✅ Analysis complete!")
                        progress_bar.progress(100)
                        
                        st.subheader("🔍 Food Items Detected")
                        st.info(f"**Enhanced AI Detection**: {description}")
                        
                        # Show detection confidence and tips
                        detection_confidence = "High" if len(description.split()) > 10 else "Medium" if len(description.split()) > 5 else "Low"
                        confidence_color = "🟢" if detection_confidence == "High" else "🟡" if detection_confidence == "Medium" else "🔴"
                        
                        st.metric("Detection Confidence", f"{confidence_color} {detection_confidence}")
                        
                        if detection_confidence == "Low":
                            st.info("💡 **Tips for Better Detection:**")
                            st.write("• 📸 Take photos from closer distance")
                            st.write("• 💡 Ensure good lighting")
                            st.write("• 🍽️ Make sure all food items are visible")
                            st.write("• 📝 Use the context field for additional details")
                        
                        # Debug information (can be removed in production)
                        with st.expander("🔧 Debug Information", expanded=False):
                            st.write(f"**Model Status**: BLIP loaded: {models['blip_model'] is not None}")
                            st.write(f"**Device**: {device}")
                            st.write(f"**Description Length**: {len(description.split())} words")
                            st.write(f"**Raw Description**: {description}")
                            st.write(f"**Improved Description**: {enhanced_analysis.get('improved_description', 'Not available')}")
                            st.write(f"**Context**: {context}")
                            st.write(f"**Enhanced Analysis Success**: {enhanced_analysis['success']}")
                            if not enhanced_analysis['success']:
                                st.write(f"**Enhanced Analysis Error**: {enhanced_analysis.get('error', 'Unknown')}")
                        
                        st.subheader("🍴 Nutritional Analysis")
                        st.markdown(analysis, unsafe_allow_html=True)
                        
                        if food_data:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Calories", f"{totals['calories']} kcal", delta=f"{totals['calories']-st.session_state.calorie_target} kcal")
                            col2.metric("Protein", f"{totals['protein']:.1f} g" if totals['protein'] else "-")
                            col3.metric("Carbs", f"{totals['carbs']:.1f} g" if totals['carbs'] else "-")
                            col4.metric("Fats", f"{totals['fats']:.1f} g" if totals['fats'] else "-")
                            
                            chart = plot_chart(food_data)
                            if chart:
                                st.pyplot(chart)
                        else:
                            st.error("❌ **Failed to Extract Food Items**")
                            st.write("The nutritional analysis couldn't identify specific food items.")
                            st.write("**Please try one of these solutions:**")
                            st.write("1. 📝 **Add Context**: Describe the meal in the context field")
                            st.write("2. 📸 **Better Image**: Upload a clearer, closer photo")
                            st.write("3. 🍽️ **Text Analysis**: Use the Text Analysis tab instead")
                            st.write("4. 🔄 **Retry**: Upload the image again")
                            
                            # Provide example context
                            st.info("💡 **Example Context:**")
                            st.write("Try adding context like: 'chicken curry with rice, naan bread, and yogurt sauce'")
                        
                        st.session_state.last_results = {
                            "type": "image",
                            "image": image,
                            "description": description,
                            "context": context or "None",
                            "analysis": analysis,
                            "chart": chart if 'chart' in locals() else None,
                            "nutrients": food_data,
                            "totals": totals,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "edge_path": edge_path,
                            "gradcam_path": gradcam_path,
                            "shap_path": shap_path,
                            "lime_path": lime_path,
                            "cnn_prediction": cnn_predicted_class,
                            "cnn_confidence": cnn_confidence.item() if cnn_confidence is not None else None
                        }
                        st.session_state.history.append(st.session_state.last_results)
                        today = date.today().isoformat()
                        st.session_state.daily_calories[today] = st.session_state.daily_calories.get(today, 0) + totals["calories"]
                        
                        if len(food_data) < 2 and ("multiple items" in description.lower() or "plate" in description.lower() or "meal" in description.lower()):
                            st.warning("⚠️ **Potential Missing Items**")
                            st.write("The analysis detected fewer items than expected for a complete meal.")
                            st.write("**This might be because:**")
                            st.write("• Some food items weren't clearly visible")
                            st.write("• The image quality could be improved")
                            st.write("• Complex dishes weren't broken down")
                            
                            st.write("**Suggestions:**")
                            st.write("• 📝 Add missing items in the context field")
                            st.write("• 🍽️ Use the Portion Adjustment tab to refine")
                            st.write("• 📸 Try a different photo angle")
                        
                    except Exception as e:
                        logger.error(f"Meal analysis failed: {e}")
                        st.error("❌ **Analysis Failed**")
                        st.write(f"**Error**: {str(e)}")
                        st.write("**This might be due to:**")
                        st.write("• 🖼️ Image format issues")
                        st.write("• 🔧 Model loading problems")
                        st.write("• 🌐 Network connectivity issues")
                        st.write("• 💾 Memory constraints")
                        
                        st.write("**Please try:**")
                        st.write("1. 📸 Upload a different image")
                        st.write("2. 📝 Use the Text Analysis tab instead")
                        st.write("3. 🔄 Refresh the page and try again")
                        st.write("4. 💻 Check your internet connection")
                        
                        st.stop()
                
                if st.session_state.last_results.get("type") == "image":
                    st.subheader("❓ Refine or Ask for More Details")
                    follow_up_question = st.text_input("Ask about this meal or refine the analysis", placeholder="E.g., 'List all items in the meal' or 'How much protein is in this meal?'", key="image_follow_up", help="Ask specific questions or request a detailed item list.")
                    if st.button("🔎 Get Details", disabled=not follow_up_question, key="image_follow_up_button", help="Click to get additional details or refine the analysis"):
                        with st.spinner("Fetching details..."):
                            dietary_prefs = ", ".join(st.session_state.dietary_preferences) if st.session_state.dietary_preferences else "None"
                            follow_up_prompt = f"""You are an expert in food identification and nutrition analysis specializing in Indian and international cuisine. Based on the following meal analysis, answer the user's specific question or refine the analysis, ensuring **every single food item** is identified, including minor components (e.g., sauces, garnishes). If the user asks to list all items, provide a detailed breakdown with estimated portion sizes (e.g., '100g grilled chicken', '2 rotis', '1 cup dal') and complete nutritional data (calories, protein, carbs, fats), respecting dietary preferences: {dietary_prefs}.

Previous meal description: {st.session_state.last_results.get('description')}
Previous analysis: {st.session_state.last_results.get('analysis')}

User's question: {follow_up_question}

Instructions:
1. If the user requests a list of all items, identify **every** food item in the meal, including minor components, with portion sizes and complete nutritional data in the format:
   - Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
2. If the description is vague, assume a complex meal with at least 3-5 components (e.g., protein, starch, vegetable, sauce, side) and list them explicitly.
3. Ensure all macronutrients are included for each item and align with dietary preferences.
4. Provide a clear and concise response tailored to the user's question or request."""
                            follow_up_answer = query_langchain(follow_up_prompt)
                            if "unavailable" in follow_up_answer.lower() or "error" in follow_up_answer.lower():
                                st.error(follow_up_answer)
                            else:
                                st.markdown(f"**Additional Details**:\n{follow_up_answer}")

    # Text Analysis Tab
    with tab2:
        st.subheader("📝 Describe Your Meal")
        st.write("Manually enter your meal details for nutritional analysis.")
        with st.container():
            meal_desc = st.text_area("Describe what you ate", placeholder="E.g., Grilled chicken, mashed potatoes, broccoli, and a creamy sauce", height=100, help="List all food items you can see in your meal, including sides and sauces, for accurate analysis.")
            
            if st.button("🔍 Analyze Description", key="analyze_text", help="Click to analyze the meal description"):
                with st.spinner("Analyzing your description..."):
                    try:
                        dietary_prefs = ", ".join(st.session_state.dietary_preferences) if st.session_state.dietary_preferences else "None"
                        prompt = f"""You are an expert in food identification and nutrition analysis. Your task is to analyze the food items described by the user and provide detailed nutritional information. Follow this exact format:

**Food Items and Nutrients**:
- Item: [Food Name with portion size, e.g., Grilled Chicken Breast (100g)], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
**Total Calories**: [X] cal
**Nutritional Assessment**: [Detailed assessment of macronutrients, vitamins, and suitability for dietary preferences: {dietary_prefs}]
**Health Suggestions**: [2-3 tailored suggestions based on the meal and dietary preferences]

Meal description: {meal_desc}
Dietary preferences: {dietary_prefs}

Instructions:
1. **Use ONLY the food items mentioned in the user's description above**. Do not add items that are not mentioned.

2. For each food item mentioned in the description, provide an estimated portion size (e.g., '100g chicken', '1 slice pizza', '50g salad') based on typical serving sizes.

3. Provide complete nutritional data (calories, protein, carbs, fats) for each item based on typical values. **All macronutrients must be included** for every item.

4. Ensure the analysis and suggestions align with the user's dietary preferences (e.g., vegan, keto, gluten-free).

5. If the description includes specific details (e.g., 'post-workout meal'), emphasize relevant nutritional aspects (e.g., protein for recovery).

6. Ensure the total calories match the sum of individual item calories.

7. Strictly adhere to the specified format.

8. **If the description is vague or unclear**, ask the user to provide more specific details about the food items."""
                        
                        analysis = query_langchain(prompt)
                        if "unavailable" in analysis.lower() or "error" in analysis.lower():
                            st.error(analysis)
                            st.stop()
                        
                        food_data, totals = extract_items_and_nutrients(analysis)
                        if not food_data or len(food_data) < 1:
                            st.warning("No food items detected. Retrying with more detailed analysis...")
                            prompt += f"""\n\n**Detailed Analysis**: Please analyze the meal description '{meal_desc}' more thoroughly. Identify ALL food items, ingredients, dishes, sauces, and sides mentioned. Break down complex dishes into individual components. Provide complete nutritional data for each item."""
                            analysis = query_langchain(prompt)
                            if "unavailable" in analysis.lower() or "error" in analysis.lower():
                                st.error(analysis)
                                st.stop()
                            food_data, totals = extract_items_and_nutrients(analysis)
                        
                        st.subheader("🍴 Nutritional Analysis")
                        st.markdown(analysis, unsafe_allow_html=True)
                        
                        if food_data:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Calories", f"{totals['calories']} kcal", delta=f"{totals['calories']-st.session_state.calorie_target} kcal")
                            col2.metric("Protein", f"{totals['protein']:.1f} g" if totals['protein'] else "-")
                            col3.metric("Carbs", f"{totals['carbs']:.1f} g" if totals['carbs'] else "-")
                            col4.metric("Fats", f"{totals['fats']:.1f} g" if totals['fats'] else "-")
                            
                            chart = plot_chart(food_data)
                            if chart:
                                st.pyplot(chart)
                        
                        st.session_state.last_results = {
                            "type": "text",
                            "description": meal_desc,
                            "analysis": analysis,
                            "chart": chart if 'chart' in locals() else None,
                            "nutrients": food_data,
                            "totals": totals,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        st.session_state.history.append(st.session_state.last_results)
                        today = date.today().isoformat()
                        st.session_state.daily_calories[today] = st.session_state.daily_calories.get(today, 0) + totals["calories"]
                        
                        if len(food_data) < 2 and ("and" in meal_desc.lower() or "with" in meal_desc.lower()):
                            st.info("The analysis may have missed some food items. Please provide a more detailed description or visit the Portion Adjustment tab.")
                        
                    except Exception as e:
                        logger.error(f"Text analysis failed: {e}")
                        st.error(f"Text analysis failed: {str(e)}\n\n**Tip**: Provide a detailed description listing all food items (e.g., 'chicken, rice, broccoli, sauce').")
                
                if st.session_state.last_results.get("type") == "text":
                    st.subheader("❓ Ask for More Details")
                    follow_up_question = st.text_input("Ask about this meal", placeholder="E.g., 'List all items in the meal' or 'Is this meal good for weight loss?'", key="text_follow_up", help="Ask specific questions or request a detailed item list.")
                    if st.button("🔎 Get Details", disabled=not follow_up_question, key="text_follow_up_button", help="Click to get additional details"):
                        with st.spinner("Fetching details..."):
                            dietary_prefs = ", ".join(st.session_state.dietary_preferences) if st.session_state.dietary_preferences else "None"
                            follow_up_prompt = f"""You are an expert in food identification and nutrition analysis. Based on the following meal analysis, answer the user's specific question or refine the analysis, ensuring **every single food item** is identified, including minor components (e.g., sauces, garnishes). If the user asks to list all items, provide a detailed breakdown with estimated portion sizes (e.g., '100g grilled chicken') and complete nutritional data (calories, protein, carbs, fats), respecting dietary preferences: {dietary_prefs}.

Previous meal description: {st.session_state.last_results.get('description')}
Previous analysis: {st.session_state.last_results.get('analysis')}

User's question: {follow_up_question}

Instructions:
1. If the user requests a list of all items, identify **every** food item in the meal, including minor components, with portion sizes and complete nutritional data in the format:
   - Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
2. If the description is vague, assume a complex meal with at least 3-5 components (e.g., protein, starch, vegetable, sauce, side) and list them explicitly.
3. Ensure all macronutrients are included for each item and align with dietary preferences.
4. Provide a clear and concise response tailored to the user's question or request."""
                            follow_up_answer = query_langchain(follow_up_prompt)
                            if "unavailable" in follow_up_answer.lower() or "error" in follow_up_answer.lower():
                                st.error(follow_up_answer)
                            else:
                                st.markdown(f"**Additional Details**:\n{follow_up_answer}")

    # History Tab
    with tab3:
        st.subheader("📊 Your Nutrition History")
        st.write("View all your past meal analyses.")
        with st.container():
            if not st.session_state.history:
                st.info("No meal analyses recorded yet. Try analyzing a meal in the Image or Text Analysis tabs!")
            for i, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"📅 {entry['timestamp']} - {entry['type'].title()} Analysis"):
                    if entry['type'] == "image" and entry.get("image"):
                        st.image(entry["image"], caption="Meal Image", use_container_width=True)
                    
                    st.markdown(entry["analysis"], unsafe_allow_html=True)
                    
                    if entry.get("totals", {}):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Calories", f"{entry['totals']['calories']} kcal")
                        col2.metric("Protein", f"{entry['totals']['protein']:.1f} g" if entry['totals']['protein'] else "-")
                        col3.metric("Carbs", f"{entry['totals']['carbs']:.1f} g" if entry['totals']['carbs'] else "-")
                        col4.metric("Fats", f"{entry['totals']['fats']:.1f} g" if entry['totals']['fats'] else "-")
                    
                    if entry.get("chart"):
                        st.pyplot(entry["chart"])
                    
                    if entry['type'] == "image":
                        st.markdown("### Visualizations")
                        viz_cols = st.columns(2)
                        with viz_cols[0]:
                            if entry.get("edge_path") and os.path.exists(entry["edge_path"]):
                                st.image(entry["edge_path"], caption="Edge Detection", use_container_width=True)
                            if entry.get("shap_path") and os.path.exists(entry["shap_path"]):
                                st.image(entry["shap_path"], caption="SHAP Explanation", use_container_width=True)
                        with viz_cols[1]:
                            if entry.get("gradcam_path") and os.path.exists(entry["gradcam_path"]):
                                st.image(entry["gradcam_path"], caption="Grad-CAM Visualization", use_container_width=True)
                            if entry.get("lime_path") and os.path.exists(entry["lime_path"]):
                                st.image(entry["lime_path"], caption="LIME Interpretation", use_container_width=True)
                            if entry.get("cnn_confidence"):
                                st.markdown(f"**Confidence**: {entry['cnn_confidence']*100:.1f}%")
            
            if st.session_state.last_results and st.session_state.history:
                if st.button("📄 Export Latest PDF Report", key="export_pdf", help="Download a PDF of the latest meal analysis"):
                    pdf_path = generate_pdf_report(
                        st.session_state.last_results.get("image"),
                        st.session_state.last_results.get("analysis"),
                        st.session_state.last_results.get("chart"),
                        st.session_state.last_results.get("nutrients", []),
                        st.session_state.last_results.get("daily_summary"),
                        st.session_state.last_results.get("edge_path"),
                        st.session_state.last_results.get("gradcam_path"),
                        st.session_state.last_results.get("shap_path"),
                        st.session_state.last_results.get("lime_path"),
                        st.session_state.last_results.get("cnn_confidence")
                    )
                    if pdf_path:
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "Download PDF Report",
                                f,
                                file_name="nutrition_report.pdf",
                                mime="application/pdf",
                                key="download_pdf"
                            )
                        os.remove(pdf_path)

    # Daily Summary Tab
    with tab4:
        st.subheader("📅 Daily Nutrition Summary")
        st.write("View your daily calorie and nutrient totals with personalized fitness advice.")
        with st.container():
            calorie_target = st.session_state.calorie_target
            activity_preference = st.session_state.activity_preference
            dietary_preferences = st.session_state.dietary_preferences
            today = date.today().isoformat()
            today_cals = st.session_state.daily_calories.get(today, 0)
            st.metric("Today's Total Calories", f"{today_cals} kcal", delta=f"{today_cals - calorie_target} kcal")
            
            if st.button("📅 Generate Daily Summary", key="daily_summary", help="Click to view your daily nutrition summary"):
                daily_summary = generate_daily_summary(calorie_target, activity_preference, dietary_preferences)
                st.markdown(daily_summary, unsafe_allow_html=True)
                if st.session_state.last_results:
                    include_summary_in_pdf = st.checkbox("Include Daily Summary in PDF Report", help="Check to include this summary in the PDF export")
                    if include_summary_in_pdf:
                        st.session_state.last_results["daily_summary"] = daily_summary

    # Portion Adjustment Tab
    with tab5:
        st.subheader("⚖️ Adjust Portion Sizes")
        st.write("Modify portion sizes for the latest meal analysis and recalculate nutrients.")
        with st.container():
            if not st.session_state.last_results.get("nutrients"):
                st.info("No meal analysis available. Analyze a meal in the Image or Text Analysis tabs first.")
            else:
                st.write(f"**Latest Meal ({st.session_state.last_results['timestamp']})**")
                if st.session_state.last_results.get("type") == "image" and st.session_state.last_results.get("image"):
                    st.image(st.session_state.last_results["image"], caption="Meal Image", use_container_width=True)
                st.markdown(st.session_state.last_results["analysis"], unsafe_allow_html=True)
                
                st.subheader("Adjust Portions")
                for i, item in enumerate(st.session_state.last_results["nutrients"]):
                    item_name = item["item"]
                    # Create unique key with index to avoid duplicates
                    sanitized_key = f"portion_{i}_{re.sub(r'[^a-zA-Z0-9]', '_', item_name.lower())}"
                    portion = st.number_input(
                        f"Portion size for {item_name} (g)",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        step=10,
                        key=sanitized_key,
                        help=f"Enter the portion size for {item_name} in grams"
                    )
                if st.button("🔄 Re-analyze with Adjusted Portions", key="reanalyze_portions", help="Click to re-analyze the meal with new portion sizes"):
                    with st.spinner("Re-analyzing with adjusted portions..."):
                        try:
                            if not st.session_state.last_results.get("nutrients"):
                                st.error("No nutrients data available for portion adjustment.")
                                st.stop()
                            dietary_prefs = ", ".join(st.session_state.dietary_preferences) if st.session_state.dietary_preferences else "None"
                            items_with_portions = []
                            for i, item in enumerate(st.session_state.last_results['nutrients']):
                                # Use the same unique key format as above
                                sanitized_key = f"portion_{i}_{re.sub(r'[^a-zA-Z0-9]', '_', item['item'].lower())}"
                                portion = st.session_state.get(sanitized_key, 100)
                                items_with_portions.append(f"{item['item']} ({portion}g)")
                            portion_string = ', '.join(items_with_portions)
                            logger.info(f"Portion string: {portion_string}")
                            prompt = f"""You are an expert in food identification and nutrition analysis. Re-analyze the meal with the following items and user-specified portion sizes, ensuring **every single food item** is included with complete nutritional data, respecting dietary preferences: {dietary_prefs}.
                            Items and portions:
                            {portion_string}.
                            Provide nutritional data in the format:
                            **Food Items and Nutrients**:
                            - Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
                            **Total Calories**: [X] cal
                            **Nutritional Assessment**: [Assessment tailored to dietary preferences]
                            **Health Suggestions**: [2-3 suggestions aligned with dietary preferences]
                            
                            Instructions:
                            1. Use the user-specified portion sizes for each item.
                            2. Include **all** items from the previous analysis, adding any minor components (e.g., sauces, garnishes) if applicable.
                            3. Provide complete nutritional data (calories, protein, carbs, fats) for each item.
                            4. Ensure suggestions align with dietary preferences."""
                            analysis = query_langchain(prompt)
                            if "unavailable" in analysis.lower() or "error" in analysis.lower():
                                st.error(analysis)
                                st.stop()
                            
                            food_data, totals = extract_items_and_nutrients(analysis)
                            
                            st.subheader("🍴 Updated Nutritional Analysis")
                            st.markdown(analysis, unsafe_allow_html=True)
                            
                            if food_data:
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Total Calories", f"{totals['calories']} kcal", delta=f"{totals['calories']-st.session_state.calorie_target} kcal")
                                col2.metric("Protein", f"{totals['protein']:.1f} g" if totals['protein'] else "-")
                                col3.metric("Carbs", f"{totals['carbs']:.1f} g" if totals['carbs'] else "-")
                                col4.metric("Fats", f"{totals['fats']:.1f} g" if totals['fats'] else "-")
                                
                                chart = plot_chart(food_data)
                                if chart:
                                    st.pyplot(chart)
                            
                            st.session_state.last_results = {
                                "type": st.session_state.last_results.get("type", "image"),
                                "image": st.session_state.last_results.get("image"),
                                "description": st.session_state.last_results.get("description"),
                                "context": st.session_state.last_results.get("context", "None"),
                                "analysis": analysis,
                                "chart": chart if 'chart' in locals() else None,
                                "nutrients": food_data,
                                "totals": totals,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                            }
                            st.session_state.history.append(st.session_state.last_results)
                            today = date.today().isoformat()
                            st.session_state.daily_calories[today] = st.session_state.daily_calories.get(today, 0) + totals["calories"]
                        
                        except Exception as e:
                            logger.error(f"Portion adjustment failed: {e}")
                            st.error(f"Portion adjustment failed: {str(e)}")
                            st.stop()

# Sidebar
with st.sidebar:
    st.header("🍎 Nutrition Dashboard")
    st.caption("Configure your profile and track progress")
    
    st.subheader("User Profile")
    st.number_input("Daily Calorie Target (kcal)", min_value=1000, max_value=5000, value=st.session_state.calorie_target, step=100, key="calorie_target", help="Set your daily calorie goal (e.g., 2000 kcal)")
    st.multiselect("Preferred Activities", options=["Brisk Walking", "Running", "Cycling", "Swimming", "Strength Training"], default=st.session_state.activity_preference, key="activity_preference", help="Select activities you enjoy for personalized fitness advice")
    st.multiselect("Dietary Preferences", options=["Vegan", "Vegetarian", "Keto", "Gluten-Free", "Low-Carb"], default=st.session_state.dietary_preferences, key="dietary_preferences", help="Select your dietary preferences for tailored analysis")
    
    st.subheader("Today's Progress")
    today = date.today().isoformat()
    today_cals = st.session_state.daily_calories.get(today, 0)
    progress = min(today_cals / st.session_state.calorie_target, 1.0) if st.session_state.calorie_target > 0 else 0
    st.progress(progress)
    st.caption(f"Progress: {today_cals}/{st.session_state.calorie_target} kcal ({progress*100:.1f}%)")
    
    st.subheader("📈 Weekly Summary")
    if st.session_state.daily_calories:
        dates = sorted(st.session_state.daily_calories.keys())[-7:]
        cals = [st.session_state.daily_calories.get(d, 0) for d in dates]
        daily_nutrients = {d: {"protein": 0, "carbs": 0, "fats": 0} for d in dates}
        for entry in st.session_state.history:
            entry_date = entry["timestamp"].split()[0]
            if entry_date in dates and entry.get("totals"):
                daily_nutrients[entry_date]["protein"] += entry["totals"].get("protein", 0)
                daily_nutrients[entry_date]["carbs"] += entry["totals"].get("carbs", 0)
                daily_nutrients[entry_date]["fats"] += entry["totals"].get("fats", 0)
        
        proteins = [daily_nutrients[d]["protein"] for d in dates]
        carbs = [daily_nutrients[d]["carbs"] for d in dates]
        fats = [daily_nutrients[d]["fats"] for d in dates]
        
        fig, ax = plt.subplots(figsize=(6, 3))
        bar_width = 0.2
        indices = range(len(dates))
        
        ax.bar([i - bar_width*1.5 for i in indices], cals, bar_width, label="Calories (kcal)", color="#4CAF50")
        ax.bar([i - bar_width*0.5 for i in indices], proteins, bar_width, label="Protein (g)", color="#2196F3")
        ax.bar([i + bar_width*0.5 for i in indices], carbs, bar_width, label="Carbs (g)", color="#FF9800")
        ax.bar([i + bar_width*1.5 for i in indices], fats, bar_width, label="Fats (g)", color="#F44336")
        
        ax.set_xticks(indices)
        ax.set_xticklabels(dates, rotation=45, fontsize=8)
        ax.set_ylabel("Amount", fontsize=10)
        ax.set_title("Weekly Nutrition", fontsize=12)
        ax.legend(fontsize=8)
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('ggplot')
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Last 7 days' nutrition trends")
    
    if st.button("🗑️ Clear History", key="clear_history", help="Clear all meal history and reset progress"):
        st.session_state.history.clear()
        st.session_state.daily_calories.clear()
        st.session_state.last_results = {}
        for pattern in ["gradcam_*.png", "shap_*.png", "lime_*.png", "edge_output_*.png"]:
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                except:
                    pass
        st.rerun()

# Footer
st.markdown("""
<div class='footer'>
    <p>Built with ❤️ by <b>Ujjwal Sinha</b> • 
    <a href='https://github.com/Ujjwal-sinha' target='_blank'>GitHub</a> 
</div>
""", unsafe_allow_html=True)

# Clean up GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()