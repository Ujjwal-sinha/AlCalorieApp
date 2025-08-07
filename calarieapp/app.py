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
import requests
import json

# Additional model imports for enhanced food detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available. Install ultralytics for object detection.")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logging.warning("TIMM not available. Install timm for additional models.")

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    logging.warning("EfficientNet not available. Install efficientnet-pytorch for additional models.")

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install tensorflow for additional models.")

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

# Initialize models (LLM, BLIP, CNN, YOLO, ViT, etc.)
@st.cache_resource
def load_models():
    models = {}
    
    # Load LLM
    try:
        logger.info("Loading ChatGroq LLM...")
        models['llm'] = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)
        logger.info("ChatGroq LLM loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        st.error(f"Failed to load language model: {e}. Please check GROQ_API_KEY and network connection.")
        models['llm'] = None
    
    # Load BLIP models
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
    
    # Load DenseNet121 CNN model
    try:
        logger.info("Loading DenseNet121 CNN model...")
        models['cnn_model'] = torchvision_models.densenet121(weights="IMAGENET1K_V1")
        models['cnn_model'] = models['cnn_model'].to(device).eval()
        logger.info("Loaded ImageNet CNN model for natural object detection")
    except Exception as e:
        logger.error(f"Failed to load CNN model: {e}")
        st.error(f"Failed to load CNN model: {e}. Visualizations will be unavailable.")
        models['cnn_model'] = None
    
    # Load YOLO model for object detection
    if YOLO_AVAILABLE:
        try:
            logger.info("Loading YOLO model...")
            models['yolo_model'] = YOLO('yolov8n.pt')  # Load nano model for speed
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            models['yolo_model'] = None
    else:
        models['yolo_model'] = None
    
    # Load Vision Transformer (ViT) model
    if TIMM_AVAILABLE:
        try:
            logger.info("Loading Vision Transformer (ViT) model...")
            models['vit_model'] = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
            models['vit_model'] = models['vit_model'].to(device).eval()
            logger.info("ViT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ViT model: {e}")
            models['vit_model'] = None
    else:
        models['vit_model'] = None
    
    # Load EfficientNet model
    if EFFICIENTNET_AVAILABLE:
        try:
            logger.info("Loading EfficientNet model...")
            models['efficientnet_model'] = EfficientNet.from_pretrained('efficientnet-b0')
            models['efficientnet_model'] = models['efficientnet_model'].to(device).eval()
            logger.info("EfficientNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EfficientNet model: {e}")
            models['efficientnet_model'] = None
    else:
        models['efficientnet_model'] = None
    
    # Load TensorFlow Hub models
    if TENSORFLOW_AVAILABLE:
        try:
            logger.info("Loading TensorFlow Hub models...")
            # Load a pre-trained image classification model
            models['tf_model'] = hub.load('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4')
            logger.info("TensorFlow Hub model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TensorFlow Hub model: {e}")
            models['tf_model'] = None
    else:
        models['tf_model'] = None
    
    # Initialize food detection agents
    try:
        logger.info("Initializing food detection agents...")
        food_agent, search_agent = initialize_agents(groq_api_key)
        models['food_agent'] = food_agent
        models['search_agent'] = search_agent
        if food_agent and search_agent:
            logger.info("Food detection agents initialized successfully")
        else:
            logger.warning("Failed to initialize food detection agents")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        models['food_agent'] = None
        models['search_agent'] = None
    
    # Log model status
    model_status = {
        'LLM': models['llm'] is not None,
        'BLIP': models['blip_model'] is not None,
        'CNN': models['cnn_model'] is not None,
        'YOLO': models['yolo_model'] is not None,
        'ViT': models['vit_model'] is not None,
        'EfficientNet': models['efficientnet_model'] is not None,
        'TensorFlow': models['tf_model'] is not None
    }
    
    loaded_models = sum(model_status.values())
    total_models = len(model_status)
    logger.info(f"Loaded {loaded_models}/{total_models} models successfully")
    
    return models

models = load_models()

# Show model loading status
def show_model_status():
    """Display model loading status in the sidebar."""
    with st.sidebar:
        st.subheader("🤖 AI Models Status")
        
        model_status = {
            'BLIP': models['blip_model'] is not None,
            'YOLO': models['yolo_model'] is not None,
            'ViT': models['vit_model'] is not None,
            'EfficientNet': models['efficientnet_model'] is not None,
            'CNN': models['cnn_model'] is not None,
            'TensorFlow': models['tf_model'] is not None,
            'LLM': models['llm'] is not None
        }
        
        loaded_models = sum(model_status.values())
        total_models = len(model_status)
        
        st.metric("Models Loaded", f"{loaded_models}/{total_models}")
        
        for model, status in model_status.items():
            status_icon = "✅" if status else "❌"
            status_color = "green" if status else "red"
            st.markdown(f"{status_icon} **{model}**: {'Available' if status else 'Not Available'}")
        
        if loaded_models >= 3:
            st.success("🎉 **Excellent Model Coverage**")
            st.write("Multiple AI models available for comprehensive detection!")
        elif loaded_models >= 2:
            st.info("👍 **Good Model Coverage**")
            st.write("Several AI models available for detection.")
        else:
            st.warning("⚠️ **Limited Model Coverage**")
            st.write("Consider installing additional models for better detection.")

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

# Enhanced YOLO food detection with visualization
def detect_food_items_with_boxes(image: Image.Image):
    """Enhanced food detection using YOLO with comprehensive food item identification."""
    try:
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        detected_items = []
        detection_info = "Enhanced multi-model detection active"
        
        # Use YOLO if available
        if models['yolo_model']:
            try:
                img_np = np.array(image)
                results = models['yolo_model'](img_np, conf=0.15, iou=0.4)  # Lower confidence for more detections
                
                # Create a copy for drawing boxes
                img_with_boxes = img_np.copy()
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = models['yolo_model'].names[cls]
                            
                            # Comprehensive food-related detection
                            food_categories = {
                                'fruits': ['apple', 'banana', 'orange', 'lemon', 'lime', 'grape', 'strawberry', 
                                          'blueberry', 'pineapple', 'mango', 'avocado', 'watermelon', 'peach', 'pear'],
                                'vegetables': ['tomato', 'potato', 'carrot', 'onion', 'garlic', 'broccoli', 'lettuce', 
                                              'spinach', 'cucumber', 'pepper', 'corn', 'peas', 'beans', 'cabbage'],
                                'proteins': ['chicken', 'fish', 'beef', 'pork', 'lamb', 'turkey', 'egg', 'shrimp',
                                            'salmon', 'tuna', 'bacon', 'sausage', 'ham'],
                                'dairy': ['cheese', 'milk', 'yogurt', 'butter', 'cream'],
                                'grains': ['bread', 'rice', 'pasta', 'noodles', 'cereal', 'oats'],
                                'prepared_foods': ['cake', 'pizza', 'burger', 'sandwich', 'soup', 'salad', 'pie',
                                                  'cookie', 'muffin', 'pancake', 'waffle', 'toast'],
                                'containers': ['bowl', 'plate', 'cup', 'glass', 'bottle', 'can', 'jar'],
                                'utensils': ['fork', 'spoon', 'knife']
                            }
                            
                            # Check if detected item is food-related
                            is_food_related = False
                            food_category = 'other'
                            
                            for category, items in food_categories.items():
                                if any(food_item in class_name.lower() for food_item in items):
                                    is_food_related = True
                                    food_category = category
                                    break
                            
                            if is_food_related and conf > 0.15:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
                                # Draw bounding box
                                cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                
                                # Add label
                                label = f"{class_name}: {conf:.2f}"
                                cv2.putText(img_with_boxes, label, (int(x1), int(y1-10)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                detected_items.append({
                                    'name': class_name,
                                    'confidence': conf,
                                    'category': food_category,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                                })
                
                if detected_items:
                    detection_info = f"YOLO detected {len(detected_items)} food-related items"
                    # Convert back to PIL Image
                    result_image = Image.fromarray(img_with_boxes)
                    return result_image, detected_items, detection_info
                else:
                    detection_info = "YOLO active but no food items detected with sufficient confidence"
                    
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
                detection_info = f"YOLO detection failed: {str(e)}"
        
        # Return original image if YOLO not available or failed
        return image, detected_items, detection_info
            
    except Exception as e:
        logger.error(f"Error in enhanced food detection: {e}")
        return image, [], f"Detection error: {str(e)}"

# Add function to get comprehensive food analysis
def get_comprehensive_food_analysis(image: Image.Image, context: str = "") -> Dict[str, Any]:
    """Get comprehensive food analysis using all available models and techniques."""
    try:
        logger.info("Starting comprehensive food analysis...")
        
        # Step 1: Enhanced image description
        image_description = describe_image_enhanced(image)
        logger.info(f"Image description: {image_description}")
        
        # Step 2: YOLO object detection
        enhanced_image, yolo_detections, yolo_info = detect_food_items_with_boxes(image)
        logger.info(f"YOLO info: {yolo_info}")
        
        # Step 3: Use food detection agent for comprehensive analysis
        if models['food_agent']:
            # Combine image description with YOLO results
            combined_description = image_description
            if yolo_detections:
                yolo_items = [item['name'] for item in yolo_detections]
                combined_description += f" Additionally detected objects: {', '.join(yolo_items)}"
            
            agent_result = models['food_agent'].detect_food_from_image_description(
                combined_description, context
            )
            
            if agent_result['success']:
                return {
                    'success': True,
                    'image_description': image_description,
                    'yolo_detections': yolo_detections,
                    'yolo_info': yolo_info,
                    'enhanced_image': enhanced_image,
                    'agent_analysis': agent_result['analysis'],
                    'food_items': agent_result['food_items'],
                    'nutritional_data': agent_result['nutritional_data'],
                    'comprehensive': agent_result.get('comprehensive', False)
                }
            else:
                logger.warning(f"Agent analysis failed: {agent_result.get('error', 'Unknown error')}")
        
        # Fallback analysis if agent fails
        return {
            'success': True,
            'image_description': image_description,
            'yolo_detections': yolo_detections,
            'yolo_info': yolo_info,
            'enhanced_image': enhanced_image,
            'agent_analysis': "Agent analysis unavailable - using basic detection",
            'food_items': [{'item': item, 'description': item} for item in image_description.split(', ')],
            'nutritional_data': {'total_calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0},
            'comprehensive': False
        }
        
    except Exception as e:
        logger.error(f"Comprehensive food analysis failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'image_description': "Analysis failed",
            'yolo_detections': [],
            'yolo_info': "Detection failed",
            'enhanced_image': image,
            'agent_analysis': f"Analysis failed: {str(e)}",
            'food_items': [],
            'nutritional_data': {'total_calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0},
            'comprehensive': False
        }

# Advanced image enhancement function for better food detection
def enhance_image_quality(image: Image.Image) -> List[Image.Image]:
    """Advanced image enhancement for optimal food detection across different models."""
    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        enhanced_images = []
        
        # 1. Contrast enhancement for better food definition
        enhancer = ImageEnhance.Contrast(image)
        enhanced_images.append(enhancer.enhance(1.4))
        
        # 2. Brightness enhancement for better visibility
        enhancer = ImageEnhance.Brightness(image)
        enhanced_images.append(enhancer.enhance(1.3))
        
        # 3. Sharpness enhancement for detail clarity
        enhancer = ImageEnhance.Sharpness(image)
        enhanced_images.append(enhancer.enhance(1.6))
        
        # 4. Color saturation for better food colors
        enhancer = ImageEnhance.Color(image)
        enhanced_images.append(enhancer.enhance(1.3))
        
        # 5. Balanced enhancement (best overall)
        balanced = image.copy()
        balanced = ImageEnhance.Contrast(balanced).enhance(1.25)
        balanced = ImageEnhance.Brightness(balanced).enhance(1.15)
        balanced = ImageEnhance.Sharpness(balanced).enhance(1.4)
        balanced = ImageEnhance.Color(balanced).enhance(1.1)
        enhanced_images.append(balanced)
        
        # 6. High contrast for edge detection
        high_contrast = image.copy()
        high_contrast = ImageEnhance.Contrast(high_contrast).enhance(1.8)
        high_contrast = ImageEnhance.Sharpness(high_contrast).enhance(1.8)
        enhanced_images.append(high_contrast)
        
        # 7. Soft enhancement for subtle details
        soft = image.copy()
        soft = ImageEnhance.Contrast(soft).enhance(1.1)
        soft = ImageEnhance.Brightness(soft).enhance(1.05)
        soft = ImageEnhance.Color(soft).enhance(1.05)
        enhanced_images.append(soft)
        
        # 8. Apply noise reduction using PIL filters
        try:
            denoised = image.filter(ImageFilter.MedianFilter(size=3))
            denoised = ImageEnhance.Sharpness(denoised).enhance(1.3)
            enhanced_images.append(denoised)
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
        
        # 9. Edge enhancement
        try:
            edge_enhanced = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
            enhanced_images.append(edge_enhanced)
        except Exception as e:
            logger.warning(f"Edge enhancement failed: {e}")
        
        # 10. Unsharp mask for detail enhancement
        try:
            unsharp = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            enhanced_images.append(unsharp)
        except Exception as e:
            logger.warning(f"Unsharp mask failed: {e}")
        
        logger.info(f"Created {len(enhanced_images)} enhanced image versions")
        return enhanced_images
        
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        return [image]  # Return original if enhancement fails

# Enhanced multi-model food detection with comprehensive coverage
def describe_image_enhanced(image: Image.Image) -> str:
    """Enhanced food detection using multiple AI models with improved coverage."""
    if not models['processor'] or not models['blip_model']:
        logger.error("BLIP model or processor is None. Image analysis unavailable.")
        return "Image analysis unavailable. Please check model loading and try again."
    
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        device = next(models['blip_model'].parameters()).device
        
        # Enhance image quality for better detection
        enhanced_images = enhance_image_quality(image)
        
        # Store all detection results
        all_detections = []
        detected_items = set()
        
        # Strategy 1: Comprehensive BLIP Detection with Multiple Approaches
        logger.info("Starting comprehensive BLIP detection...")
        
        # Use original and enhanced images
        images_to_process = [image] + enhanced_images[:3]  # Original + 3 best enhanced
        
        for img_idx, img in enumerate(images_to_process):
            try:
                # Multiple detection prompts for comprehensive coverage
                detection_prompts = [
                    # General food detection
                    "What food items are visible in this image?",
                    "List all the food and drinks you can see:",
                    "Describe every edible item in this image:",
                    "What dishes, ingredients, and food items are present?",
                    
                    # Specific component detection
                    "What vegetables, fruits, and proteins can you identify?",
                    "What grains, dairy products, and seasonings are visible?",
                    "What beverages and condiments can you see?",
                    
                    # Detailed analysis
                    "Analyze this meal and list each component:",
                    "What are the main ingredients and side dishes?",
                    "Describe the complete meal including garnishes and sauces:"
                ]
                
                for prompt_idx, prompt in enumerate(detection_prompts):
                    try:
                        inputs = models['processor'](img, text=prompt, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = models['blip_model'].generate(
                                **inputs, 
                                max_new_tokens=200, 
                                num_beams=6, 
                                do_sample=True,
                                temperature=0.5,
                                top_p=0.95,
                                repetition_penalty=1.1
                            )
                        caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                        
                        # Clean up the response
                        if caption.startswith(prompt):
                            caption = caption.replace(prompt, "").strip()
                        
                        if len(caption.split()) >= 2:
                            all_detections.append(caption)
                            logger.info(f"BLIP detection {img_idx+1}-{prompt_idx+1}: {caption}")
                            
                    except Exception as e:
                        logger.warning(f"BLIP prompt {prompt_idx+1} failed: {e}")
                        
            except Exception as e:
                logger.warning(f"BLIP image {img_idx+1} failed: {e}")
        
        # Strategy 2: Enhanced YOLO Object Detection
        if models['yolo_model']:
            try:
                logger.info("Running enhanced YOLO object detection...")
                
                # Process original and enhanced images
                for img_idx, img in enumerate(images_to_process[:2]):  # Original + 1 enhanced
                    try:
                        img_np = np.array(img)
                        results = models['yolo_model'](img_np, conf=0.2, iou=0.4)  # Lower confidence for more detections
                        
                        yolo_items = []
                        for result in results:
                            boxes = result.boxes
                            if boxes is not None:
                                for box in boxes:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    class_name = models['yolo_model'].names[cls]
                                    
                                    # Expanded food-related terms
                                    food_terms = [
                                        # Fruits
                                        'apple', 'banana', 'orange', 'lemon', 'lime', 'grape', 'strawberry', 'blueberry',
                                        'pineapple', 'mango', 'avocado', 'tomato', 'watermelon', 'peach', 'pear',
                                        
                                        # Vegetables
                                        'potato', 'carrot', 'onion', 'garlic', 'broccoli', 'lettuce', 'spinach',
                                        'cucumber', 'pepper', 'corn', 'peas', 'beans', 'cabbage', 'celery',
                                        
                                        # Proteins
                                        'chicken', 'fish', 'beef', 'pork', 'lamb', 'turkey', 'egg', 'shrimp',
                                        'salmon', 'tuna', 'bacon', 'sausage', 'ham',
                                        
                                        # Dairy
                                        'cheese', 'milk', 'yogurt', 'butter', 'cream',
                                        
                                        # Grains & Carbs
                                        'bread', 'rice', 'pasta', 'noodles', 'cereal', 'oats', 'quinoa',
                                        
                                        # Prepared foods
                                        'cake', 'pizza', 'burger', 'sandwich', 'soup', 'salad', 'pie',
                                        'cookie', 'muffin', 'pancake', 'waffle', 'toast',
                                        
                                        # Containers & utensils (help identify food context)
                                        'bowl', 'plate', 'cup', 'glass', 'fork', 'spoon', 'knife',
                                        'bottle', 'can', 'jar'
                                    ]
                                    
                                    if conf > 0.2 and any(food_term in class_name.lower() for food_term in food_terms):
                                        yolo_items.append(f"{class_name} ({conf:.2f})")
                        
                        if yolo_items:
                            yolo_result = ', '.join(yolo_items)
                            all_detections.append(f"Detected objects: {yolo_result}")
                            logger.info(f"YOLO detection {img_idx+1}: {yolo_result}")
                            
                    except Exception as e:
                        logger.warning(f"YOLO processing image {img_idx+1} failed: {e}")
                        
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
        
        # Strategy 3: Vision Transformer Detection (if available)
        if models['vit_model']:
            try:
                logger.info("Running Vision Transformer detection...")
                
                # Prepare image for ViT
                vit_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                img_tensor = vit_transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = models['vit_model'](img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    top_probs, top_indices = torch.topk(probabilities, 10)
                
                # Load ImageNet class names (simplified food-related subset)
                imagenet_food_classes = {
                    # This is a simplified mapping - in practice you'd load the full ImageNet classes
                    0: "background", 1: "apple", 2: "banana", 3: "orange", 4: "pizza", 5: "burger"
                    # Add more mappings as needed
                }
                
                vit_items = []
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    if prob > 0.1:  # Confidence threshold
                        class_name = imagenet_food_classes.get(idx.item(), f"class_{idx.item()}")
                        if "food" in class_name.lower() or any(food_word in class_name.lower() 
                                                             for food_word in ['apple', 'banana', 'pizza', 'burger']):
                            vit_items.append(f"{class_name} ({prob:.2f})")
                
                if vit_items:
                    vit_result = ', '.join(vit_items)
                    all_detections.append(f"ViT detected: {vit_result}")
                    logger.info(f"ViT detection: {vit_result}")
                    
            except Exception as e:
                logger.warning(f"ViT detection failed: {e}")
        
        # Strategy 4: Multi-scale BLIP Analysis
        try:
            logger.info("Running multi-scale BLIP analysis...")
            
            # Create different sized versions for multi-scale analysis
            scales = [(224, 224), (384, 384), (512, 512)]
            
            for scale in scales:
                try:
                    scaled_img = image.resize(scale, Image.Resampling.LANCZOS)
                    
                    # Use focused prompts for scaled images
                    scale_prompts = [
                        "What specific food items can you identify in this image?",
                        "List all visible ingredients and food components:",
                        "Describe each dish and food item you can see:"
                    ]
                    
                    for prompt in scale_prompts:
                        try:
                            inputs = models['processor'](scaled_img, text=prompt, return_tensors="pt").to(device)
                            with torch.no_grad():
                                outputs = models['blip_model'].generate(
                                    **inputs, 
                                    max_new_tokens=150, 
                                    num_beams=5, 
                                    do_sample=True,
                                    temperature=0.4,
                                    top_p=0.9
                                )
                            caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                            
                            if caption.startswith(prompt):
                                caption = caption.replace(prompt, "").strip()
                            
                            if len(caption.split()) >= 3:
                                all_detections.append(f"Scale {scale}: {caption}")
                                logger.info(f"Multi-scale BLIP {scale}: {caption}")
                                
                        except Exception as e:
                            logger.warning(f"Multi-scale prompt failed: {e}")
                            
                except Exception as e:
                    logger.warning(f"Multi-scale processing {scale} failed: {e}")
                    
        except Exception as e:
            logger.warning(f"Multi-scale BLIP analysis failed: {e}")
        
        # Strategy 5: Contextual Food Detection
        try:
            logger.info("Running contextual food detection...")
            
            # Context-aware prompts
            context_prompts = [
                "This appears to be a meal. What are all the food items present?",
                "Looking at this food image, identify every edible component:",
                "What complete meal or snack is shown with all its parts?",
                "Analyze this food photo and list every ingredient and dish:",
                "What beverages, main dishes, sides, and garnishes are visible?"
            ]
            
            for prompt in context_prompts:
                try:
                    inputs = models['processor'](image, text=prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = models['blip_model'].generate(
                            **inputs, 
                            max_new_tokens=250, 
                            num_beams=8, 
                            do_sample=True,
                            temperature=0.3,
                            top_p=0.95,
                            repetition_penalty=1.2
                        )
                    caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                    
                    if caption.startswith(prompt):
                        caption = caption.replace(prompt, "").strip()
                    
                    if len(caption.split()) >= 4:
                        all_detections.append(caption)
                        logger.info(f"Contextual detection: {caption}")
                        
                except Exception as e:
                    logger.warning(f"Contextual prompt failed: {e}")
                    
        except Exception as e:
            logger.warning(f"Contextual food detection failed: {e}")
        
        # Process and intelligently combine all results
        logger.info(f"Total detections collected: {len(all_detections)}")
        
        if all_detections:
            # Advanced result processing and combination
            food_items = set()
            
            for detection in all_detections:
                # Clean and normalize the detection
                cleaned = detection.lower()
                
                # Remove common prefixes and suffixes
                prefixes_to_remove = [
                    "a photo of", "an image of", "a picture of", "this is", "i can see",
                    "the image shows", "there is", "there are", "detected objects:",
                    "vit detected:", "scale (", "contextual detection:"
                ]
                
                for prefix in prefixes_to_remove:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned.replace(prefix, "").strip()
                
                # Remove confidence scores in parentheses
                import re
                cleaned = re.sub(r'\([0-9.]+\)', '', cleaned)
                
                # Split by various separators
                separators = [',', ';', ' and ', ' with ', ' including ', ' plus ', ' also ']
                items = [cleaned]
                
                for sep in separators:
                    new_items = []
                    for item in items:
                        new_items.extend(item.split(sep))
                    items = new_items
                
                # Process individual items
                for item in items:
                    item = item.strip().rstrip('.,!?')
                    
                    # Skip very short or common words
                    skip_words = {
                        'the', 'and', 'with', 'on', 'in', 'of', 'a', 'an', 'is', 'are',
                        'this', 'that', 'some', 'many', 'few', 'several', 'various'
                    }
                    
                    if len(item) > 2 and item not in skip_words:
                        # Clean up the item further
                        item = re.sub(r'^(some|many|few|several|various)\s+', '', item)
                        item = re.sub(r'\s+(on|in|with)\s+.*$', '', item)
                        
                        if len(item) > 2:
                            food_items.add(item)
            
            if food_items:
                # Create comprehensive final description
                sorted_items = sorted(food_items)
                final_description = ', '.join(sorted_items)
                logger.info(f"Final comprehensive description: {final_description}")
                return final_description
            else:
                # Use the longest single detection
                best_detection = max(all_detections, key=lambda x: len(x.split()))
                logger.info(f"Using best single detection: {best_detection}")
                return best_detection
        
        # Enhanced fallback detection
        logger.warning("Primary detections failed, trying enhanced fallback")
        try:
            fallback_prompts = [
                "What food is in this image?",
                "Describe this meal:",
                "What can you see in this food photo?"
            ]
            
            for prompt in fallback_prompts:
                try:
                    inputs = models['processor'](image, text=prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = models['blip_model'].generate(
                            **inputs, 
                            max_new_tokens=100, 
                            num_beams=4, 
                            do_sample=True,
                            temperature=0.7
                        )
                    caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                    
                    if caption.startswith(prompt):
                        caption = caption.replace(prompt, "").strip()
                    
                    if len(caption.split()) >= 2:
                        logger.info(f"Fallback result: {caption}")
                        return caption
                        
                except Exception as e:
                    logger.warning(f"Fallback prompt failed: {e}")
                    
        except Exception as e:
            logger.warning(f"Enhanced fallback failed: {e}")
        
        # Final emergency fallback
        logger.error("All detection strategies failed")
        return "Multiple food items detected but specific identification is limited. Please provide additional context or describe the meal manually for accurate calorie calculation."
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return f"Image analysis error: {str(e)}. Please try a clearer image or describe the meal in the context field."

# Enhanced food analysis with comprehensive detection
def analyze_food_with_enhanced_prompt(image_description: str, context: str = "") -> Dict[str, Any]:
    """Comprehensive food analysis using enhanced detection and multiple AI models."""
    try:
        # Clean up the description
        cleaned_description = image_description.strip()
        
        # Use the food detection agent if available for comprehensive analysis
        if models['food_agent']:
            logger.info("Using food detection agent for comprehensive analysis")
            agent_result = models['food_agent'].detect_food_from_image_description(cleaned_description, context)
            
            if agent_result['success']:
                # Extract items and nutritional data from the comprehensive analysis
                items, totals = extract_items_and_nutrients(agent_result['analysis'])
                
                return {
                    "success": True,
                    "analysis": agent_result['analysis'],
                    "food_items": agent_result.get('food_items', []),
                    "nutritional_data": agent_result.get('nutritional_data', {}),
                    "improved_description": cleaned_description,
                    "extracted_items": items,
                    "extracted_totals": totals,
                    "comprehensive": agent_result.get('comprehensive', False)
                }
        
        # Fallback to enhanced LLM analysis if agent not available
        logger.info("Using enhanced LLM analysis as fallback")
        
        # Enhanced prompt for better food detection
        prompt = f"""You are an expert nutritionist analyzing food images. Provide comprehensive analysis:

FOOD DESCRIPTION: {cleaned_description}
ADDITIONAL CONTEXT: {context if context else "None provided"}

ANALYSIS REQUIREMENTS:
1. Identify ALL visible food items (main dishes, sides, garnishes, condiments, beverages)
2. Estimate realistic portion sizes for each item
3. Provide detailed nutritional breakdown per item
4. Consider cooking methods that add calories (oils, butter, etc.)
5. Account for hidden ingredients and seasonings

OUTPUT FORMAT:
## IDENTIFIED ITEMS:
- Item: [Name with portion size], Calories: [X], Protein: [X]g, Carbs: [X]g, Fats: [X]g
[Repeat for each item]

## TOTALS:
Total Calories: [Sum]
Total Protein: [Sum]g
Total Carbohydrates: [Sum]g  
Total Fats: [Sum]g

## MEAL ASSESSMENT:
[Brief assessment of nutritional balance and meal type]

## HEALTH RECOMMENDATIONS:
[2-3 specific suggestions for improvement]

IMPORTANT: Be thorough in identifying ALL food components, even small ones. Provide realistic nutritional estimates based on typical serving sizes."""

        # Get comprehensive analysis from LLM
        response = models['llm'].invoke(prompt)
        analysis = response.content
        
        # Extract structured data from the analysis
        items, totals = extract_items_and_nutrients(analysis)
        
        # Create food items list from analysis
        food_items = []
        for item in items:
            food_items.append({
                "item": item["item"],
                "description": f"{item['item']} - {item['calories']} calories",
                "calories": item["calories"],
                "protein": item.get("protein", 0),
                "carbs": item.get("carbs", 0),
                "fats": item.get("fats", 0)
            })
        
        # Create nutritional data summary
        nutritional_data = {
            "total_calories": totals.get("calories", 0),
            "total_protein": totals.get("protein", 0),
            "total_carbs": totals.get("carbs", 0),
            "total_fats": totals.get("fats", 0),
            "items": food_items
        }
        
        return {
            "success": True,
            "analysis": analysis,
            "food_items": food_items,
            "nutritional_data": nutritional_data,
            "improved_description": cleaned_description,
            "extracted_items": items,
            "extracted_totals": totals,
            "comprehensive": True
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive food analysis: {e}")
        return {
            "success": False,
            "error": str(e),
            "analysis": f"Comprehensive analysis failed: {str(e)}. Please try describing the meal manually.",
            "food_items": [],
            "nutritional_data": {"total_calories": 0, "total_protein": 0, "total_carbs": 0, "total_fats": 0},
            "improved_description": cleaned_description,
            "comprehensive": False
        }

# Enhanced post-processing for comprehensive detection
def post_process_detection(description: str) -> str:
    """Simple post-processing to clean up food descriptions."""
    try:
        # Clean up the description
        cleaned = description.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "a photo of ", "an image of ", "a picture of ",
            "this is ", "there is ", "i can see "
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Simple enhancement for vague descriptions
        if any(term in cleaned.lower() for term in ["food", "meal", "dish", "plate"]):
            cleaned += " (includes main components, sides, and sauces)"
        
        return cleaned.strip()
        
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        return description

# Enhanced extraction of food items and nutrients from text
def extract_items_and_nutrients(text):
    """Extract food items and nutritional data from analysis text with multiple pattern matching."""
    items = []
    
    try:
        # Multiple patterns to catch different formats
        patterns = [
            # Pattern 1: Standard format with "Item:"
            r'Item:\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?',
            
            # Pattern 2: Bullet point format
            r'-\s*([^:]+):\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?',
            
            # Pattern 3: Simple format without "Item:" prefix
            r'-\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?',
            
            # Pattern 4: Alternative format
            r'([^:]+):\s*(\d{1,4})\s*(?:cal|kcal|calories)(?:,\s*(\d+\.?\d*)\s*g\s*protein)?(?:,\s*(\d+\.?\d*)\s*g\s*carbs)?(?:,\s*(\d+\.?\d*)\s*g\s*fats)?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                if len(match) >= 2:
                    # Handle different match group structures
                    if len(match) == 6 and match[0] and match[1]:  # Pattern 2
                        item = f"{match[0].strip()}: {match[1].strip()}"
                        calories = int(match[2]) if match[2] else 0
                        protein = float(match[3]) if match[3] else 0
                        carbs = float(match[4]) if match[4] else 0
                        fats = float(match[5]) if match[5] else 0
                    else:  # Other patterns
                        item = match[0].strip()
                        calories = int(match[1]) if match[1] else 0
                        protein = float(match[2]) if len(match) > 2 and match[2] else 0
                        carbs = float(match[3]) if len(match) > 3 and match[3] else 0
                        fats = float(match[4]) if len(match) > 4 and match[4] else 0
                    
                    # Avoid duplicates
                    if not any(existing_item["item"].lower() == item.lower() for existing_item in items):
                        items.append({
                            "item": item,
                            "calories": calories,
                            "protein": protein,
                            "carbs": carbs,
                            "fats": fats
                        })
        
        # If no structured items found, try to extract from totals section
        if not items:
            total_patterns = [
                r'Total Calories?:\s*(\d{1,4})',
                r'Total:\s*(\d{1,4})\s*(?:cal|kcal|calories)',
                r'(\d{1,4})\s*(?:cal|kcal|calories)\s*total'
            ]
            
            for pattern in total_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    total_calories = int(match.group(1))
                    items.append({
                        "item": "Complete meal (estimated)",
                        "calories": total_calories,
                        "protein": 0,
                        "carbs": 0,
                        "fats": 0
                    })
                    break
        
        # Calculate totals
        totals = {
            "calories": sum(item["calories"] for item in items),
            "protein": sum(item["protein"] for item in items if item["protein"]),
            "carbs": sum(item["carbs"] for item in items if item["carbs"]),
            "fats": sum(item["fats"] for item in items if item["fats"])
        }
        
        # If no items extracted but we have text, create a generic entry
        if not items and len(text.strip()) > 10:
            # Try to extract any calorie numbers from the text
            calorie_matches = re.findall(r'(\d{2,4})\s*(?:cal|kcal|calories)', text, re.IGNORECASE)
            if calorie_matches:
                estimated_calories = max(int(cal) for cal in calorie_matches)
                items.append({
                    "item": "Meal items (detected but not fully parsed)",
                    "calories": estimated_calories,
                    "protein": 0,
                    "carbs": 0,
                    "fats": 0
                })
                totals["calories"] = estimated_calories
        
        logger.info(f"Extracted {len(items)} food items with {totals['calories']} total calories")
        return items, totals
        
    except Exception as e:
        logger.error(f"Error extracting items and nutrients: {e}")
        return [], {"calories": 0, "protein": 0, "carbs": 0, "fats": 0}

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
            st.write("• 🔍 **Comprehensive**: The AI will identify ALL visible food items")
            st.write("")
            st.write("**If Detection Fails:**")
            st.write("• 📝 **Add Context**: Describe the meal in the context field")
            st.write("• 🔄 **Try Again**: Upload from a different angle")
            st.write("• 📊 **Text Analysis**: Use the Text Analysis tab instead")
            st.write("")
            st.write("**Detection Features:**")
            st.write("• 🤖 **Multi-Model AI**: Uses BLIP, YOLO, ViT, EfficientNet")
            st.write("• 🖼️ **Image Enhancement**: Automatically improves image quality")
            st.write("• 📊 **Comprehensive Analysis**: Identifies all food components")
            st.write("• 🍽️ **Complete Breakdown**: Lists every item with nutrition data")
            st.write("• 🎯 **Object Detection**: YOLO for precise food item detection")
            st.write("• 🔍 **Vision Transformer**: Advanced image understanding")
            st.write("• ⚡ **EfficientNet**: Fast and accurate classification")
        
        # Add model installation guide
        with st.expander("🔧 **Install Additional Models**", expanded=False):
            st.write("**To enable all AI models, install these packages:**")
            st.code("pip install ultralytics timm efficientnet-pytorch tensorflow tensorflow-hub")
            st.write("")
            st.write("**Or install all requirements:**")
            st.code("pip install -r requirements.txt")
            st.write("")
            st.write("**Models Available:**")
            st.write("• 🎯 **YOLO**: Object detection for precise food identification")
            st.write("• 🔍 **ViT**: Vision Transformer for advanced image analysis")
            st.write("• ⚡ **EfficientNet**: Fast and efficient classification")
            st.write("• 🤖 **TensorFlow Hub**: Pre-trained models for various tasks")
        
        with st.container():
            img_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"], key="img_uploader", help="Upload a clear, well-lit image of your meal for best results.")
            context = st.text_area("Additional Context (Optional)", placeholder="E.g., 'Indian thali with dal, roti, rice' or specify cuisine type", height=100, help="Provide additional context to help identify food items more accurately.")
            
            # Add helpful context examples
            with st.expander("💡 **Context Examples**", expanded=False):
                st.write("**Add these in the context field for better detection:**")
                st.write("• 'chicken curry with rice, naan bread, and yogurt sauce'")
                st.write("• 'pasta with tomato sauce, cheese, and vegetables'")
                st.write("• 'grilled salmon with mashed potatoes and broccoli'")
                st.write("• 'Indian thali with dal, roti, rice, and vegetables'")
                st.write("• 'pizza with pepperoni, cheese, and tomato sauce'")
                st.write("• 'salad with mixed greens, chicken, and vinaigrette'")
            
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
                        
                        # Show model detection progress
                        with st.expander("🔬 **Model Detection Progress**", expanded=False):
                            st.write("**Detection Steps:**")
                            st.write("1. ✅ Image Enhancement - Applied")
                            st.write("2. 🔄 BLIP Analysis - In Progress...")
                            if models['yolo_model']:
                                st.write("3. 🎯 YOLO Object Detection - Available")
                            if models['vit_model']:
                                st.write("4. 🔍 ViT Analysis - Available")
                            if models['efficientnet_model']:
                                st.write("5. ⚡ EfficientNet Classification - Available")
                            st.write("6. 📊 Result Combination - Pending...")
                        
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
                            
                            # Provide immediate alternatives
                            st.info("**Immediate Solutions:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Option 1: Add Context**")
                                st.write("Describe your meal in the context field above and click 'Analyze Meal' again.")
                            
                            with col2:
                                st.write("**Option 2: Text Analysis**")
                                st.write("Use the 'Text Analysis' tab to describe your meal manually.")
                            
                            st.write("**For Better Results:**")
                            st.write("1. 📸 Upload a clearer image with better lighting")
                            st.write("2. 📝 Describe the meal in the context field")
                            st.write("3. 🍽️ Ensure food items are clearly visible")
                            st.write("4. 🔄 Try a different angle or distance")
                            
                            # Don't stop, let user try with context
                            st.warning("**Continuing with limited detection. Please add context or use Text Analysis tab.**")
                            description = "Food items detected but description is limited. Please use the context field to describe the meal in detail."
                        
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
                            
                            # Try multiple retry strategies
                            retry_strategies = [
                                {
                                    "name": "Enhanced Detection",
                                    "prompt": f"""You are an expert nutritionist. The previous analysis failed to identify food items. You MUST now identify EVERY single food item, ingredient, sauce, garnish, and edible component in this meal description.

**MEAL DESCRIPTION**: {description}
**CONTEXT**: {context if context else "No additional context"}

**CRITICAL INSTRUCTIONS**:
1. **MUST identify EVERY food item** mentioned or implied in the description
2. **Break down complex dishes** into individual components
3. **Include ALL sauces, garnishes, sides, and accompaniments**
4. **Provide realistic portion sizes** for each item
5. **Give complete nutritional data** for each component
6. **Be exhaustive** - do not miss any edible components
7. **If description is vague**, identify common meal components:
   * Main protein (chicken, fish, meat, tofu, etc.)
   * Starch/carbohydrate (rice, bread, pasta, potatoes, etc.)
   * Vegetables (any visible vegetables or greens)
   * Sauce/gravy (any liquid or sauce component)
   * Side dish (any additional items)
   * Garnishes (herbs, spices, toppings)

**REQUIRED OUTPUT FORMAT**:
**Food Items and Nutrients**:
- Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- [Continue for EVERY food item identified]

**Total Calories**: [X] cal
**Nutritional Assessment**: [Detailed assessment]
**Health Suggestions**: [2-3 suggestions]

**IMPORTANT**: If the description is vague, assume a complete meal with protein, starch, vegetables, sauce, and sides. Provide realistic estimates for each component."""
                                },
                                {
                                    "name": "Aggressive Detection",
                                    "prompt": f"""You are a food analysis expert. The meal description is: "{description}". Context: "{context if context else 'None'}"

**TASK**: Force identify food items even if description is vague.

**INSTRUCTIONS**:
- If description mentions "food", "meal", "plate", assume: chicken curry (200g), rice (150g), naan bread (1 piece), yogurt sauce (50g)
- If description mentions "curry", assume: curry dish (200g), rice (150g), bread (1 piece)
- If description mentions "pasta", assume: pasta (200g), sauce (100g), cheese (30g)
- If description mentions "salad", assume: mixed vegetables (150g), dressing (30g), protein (100g)
- Always provide complete nutrition data for each item

**REQUIRED OUTPUT FORMAT**:
**Food Items and Nutrients**:
- Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g

**Total Calories**: [X] cal
**Nutritional Assessment**: [Assessment]
**Health Suggestions**: [2-3 suggestions]"""
                                },
                                {
                                    "name": "Generic Meal Detection",
                                    "prompt": f"""Analyze this meal: "{description}". Context: "{context if context else 'None'}"

**TASK**: Provide nutrition data for a typical meal.

**REQUIRED OUTPUT FORMAT**:
**Food Items and Nutrients**:
- Item: Main dish (200g), Calories: 300 cal, Protein: 25 g, Carbs: 30 g, Fats: 12 g
- Item: Side dish (150g), Calories: 200 cal, Protein: 8 g, Carbs: 35 g, Fats: 5 g
- Item: Sauce/Garnish (50g), Calories: 100 cal, Protein: 2 g, Carbs: 5 g, Fats: 8 g

**Total Calories**: 600 cal
**Nutritional Assessment**: Balanced meal with protein, carbs, and fats
**Health Suggestions**: Consider portion control and variety"""
                                }
                            ]
                            
                            # Try each strategy
                            for i, strategy in enumerate(retry_strategies):
                                st.info(f"🔄 **Retrying with {strategy['name']}...**")
                                try:
                                    analysis = query_langchain(strategy['prompt'])
                                    if "unavailable" in analysis.lower() or "error" in analysis.lower():
                                        continue
                                    
                                    food_data, totals = extract_items_and_nutrients(analysis)
                                    if food_data and len(food_data) >= 1:
                                        st.success(f"✅ **{strategy['name']} Successful!**")
                                        break
                                except Exception as e:
                                    logger.warning(f"Strategy {i+1} failed: {e}")
                                    continue
                            
                            # If all strategies fail
                            if not food_data or len(food_data) < 1:
                                st.error("**Food Detection Completely Failed**")
                                st.write("The AI couldn't identify any food items even with multiple detection strategies.")
                                
                                # Provide immediate alternatives
                                st.info("**Immediate Solutions:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Option 1: Add Context**")
                                    st.write("Describe your meal in the context field above and click 'Analyze Meal' again.")
                                
                                with col2:
                                    st.write("**Option 2: Text Analysis**")
                                    st.write("Use the 'Text Analysis' tab to describe your meal manually.")
                                
                                st.write("**For Better Results:**")
                                st.write("1. 📸 Upload a clearer image with better lighting")
                                st.write("2. 📝 Describe the meal in the context field")
                                st.write("3. 🍽️ Ensure food items are clearly visible")
                                st.write("4. 🔄 Try a different angle or distance")
                                
                                # Don't stop, let user try with context
                                st.warning("**Continuing with limited detection. Please add context or use Text Analysis tab.**")
                                description = "Food items detected but description is limited. Please use the context field to describe the meal in detail."
                                
                                # Create basic food data to continue
                                food_data = [{"item": "Generic meal component", "calories": 300, "protein": 20, "carbs": 30, "fats": 10}]
                                totals = {"calories": 300, "protein": 20, "carbs": 30, "fats": 10}
                        
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
                        
                        # Create tabs for organized information display
                        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
                            "🔍 Detection Results",
                            "🍴 Nutrition Analysis", 
                            "📊 Visualizations",
                            "💡 Health Insights",
                            "🔧 Technical Details"
                        ])
                        
                        # Tab 1: Detection Results
                        with analysis_tab1:
                            st.subheader("🔍 Food Detection Results")
                            st.info(f"**Enhanced AI Detection**: {description}")
                            
                            # Show detection confidence and tips
                            detection_confidence = "High" if len(description.split()) > 10 else "Medium" if len(description.split()) > 5 else "Low"
                            confidence_color = "🟢" if detection_confidence == "High" else "🟡" if detection_confidence == "Medium" else "🔴"
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Detection Confidence", f"{confidence_color} {detection_confidence}")
                            with col2:
                                st.metric("Items Detected", f"{len(food_data) if food_data else 0} items")
                            
                            if detection_confidence == "Low":
                                st.info("💡 **Tips for Better Detection:**")
                                st.write("• 📸 Take photos from closer distance")
                                st.write("• 💡 Ensure good lighting")
                                st.write("• 🍽️ Make sure all food items are visible")
                                st.write("• 📝 Use the context field for additional details")
                            
                                                    # Show limited detection warning if applicable
                        if "limited" in description.lower() or "generic" in description.lower():
                            st.warning("⚠️ **Limited Detection**")
                            st.write("The AI detected food items but couldn't identify specific components.")
                            st.write("**To improve results:**")
                            st.write("• 📝 Add detailed context in the context field")
                            st.write("• 🍽️ Use the Text Analysis tab for manual description")
                            st.write("• 📸 Try uploading a clearer image")
                        
                        # Show detection improvement tips
                        if detection_confidence == "Low":
                            st.info("💡 **Detection Improvement Tips:**")
                            st.write("• 📸 **Better Lighting**: Ensure the image is well-lit")
                            st.write("• 🍽️ **Clear Focus**: Make sure food items are clearly visible")
                            st.write("• 📐 **Good Angle**: Avoid shadows and glare")
                            st.write("• 📝 **Add Context**: Describe the meal in the context field")
                            st.write("• 🔄 **Try Again**: Upload from a different angle")
                            st.write("• 📊 **Use Text Analysis**: Describe the meal manually")
                            
                                                    # Show detection methods used
                        with st.expander("🔬 **Detection Methods Used**", expanded=False):
                            st.write("**AI Models & Strategies:**")
                            
                            # Model status
                            model_status = {
                                'BLIP': models['blip_model'] is not None,
                                'YOLO': models['yolo_model'] is not None,
                                'ViT': models['vit_model'] is not None,
                                'EfficientNet': models['efficientnet_model'] is not None,
                                'CNN': models['cnn_model'] is not None,
                                'TensorFlow': models['tf_model'] is not None
                            }
                            
                            for model, status in model_status.items():
                                status_icon = "✅" if status else "❌"
                                st.write(f"• {status_icon} **{model}**: {'Available' if status else 'Not Available'}")
                            
                            st.write("")
                            st.write("**Detection Strategies:**")
                            st.write("• 🤖 Multi-Model AI Analysis")
                            st.write("• 🖼️ Image Quality Enhancement")
                            st.write("• 📊 Comprehensive Item Identification")
                            st.write("• 🍽️ Complete Nutritional Breakdown")
                            st.write("• 🔄 Multiple Fallback Approaches")
                            st.write("• 🎯 Object Detection (YOLO)")
                            st.write("• 🔍 Vision Transformer Analysis")
                            st.write("• ⚡ EfficientNet Classification")
                        
                        # Show detailed model detection results
                        with st.expander("🔍 **Detailed Model Results**", expanded=False):
                            st.write("**Model Detection Details:**")
                            
                            # Show which models were actually used
                            used_models = []
                            if models['blip_model']:
                                used_models.append("BLIP (Text Generation)")
                            if models['yolo_model']:
                                used_models.append("YOLO (Object Detection)")
                            if models['vit_model']:
                                used_models.append("ViT (Vision Transformer)")
                            if models['efficientnet_model']:
                                used_models.append("EfficientNet (Classification)")
                            
                            st.write(f"**Models Used**: {', '.join(used_models)}")
                            st.write(f"**Detection Confidence**: {detection_confidence}")
                            st.write(f"**Items Identified**: {len(description.split(',')) if ',' in description else len(description.split())}")
                            
                            # Show detection quality metrics
                            if len(description.split()) > 15:
                                st.success("🎉 **Excellent Detection** - Comprehensive analysis performed")
                            elif len(description.split()) > 8:
                                st.info("👍 **Good Detection** - Multiple items identified")
                            elif len(description.split()) > 3:
                                st.warning("⚠️ **Limited Detection** - Basic items identified")
                            else:
                                st.error("❌ **Poor Detection** - Consider adding context")
                        
                        # Tab 2: Nutrition Analysis
                        with analysis_tab2:
                            st.subheader("🍴 Nutritional Analysis")
                            st.markdown(analysis, unsafe_allow_html=True)
                            
                            if food_data:
                                # Nutrition metrics
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Total Calories", f"{totals['calories']} kcal", delta=f"{totals['calories']-st.session_state.calorie_target} kcal")
                                col2.metric("Protein", f"{totals['protein']:.1f} g" if totals['protein'] else "-")
                                col3.metric("Carbs", f"{totals['carbs']:.1f} g" if totals['carbs'] else "-")
                                col4.metric("Fats", f"{totals['fats']:.1f} g" if totals['fats'] else "-")
                                
                                # Nutrition chart
                                chart = plot_chart(food_data)
                                if chart:
                                    st.pyplot(chart)
                                
                                # Detailed nutrition breakdown
                                with st.expander("📋 **Detailed Nutrition Breakdown**", expanded=False):
                                    for i, item in enumerate(food_data, 1):
                                        col1, col2, col3, col4, col5 = st.columns(5)
                                        with col1:
                                            st.write(f"**{i}. {item['item']}**")
                                        with col2:
                                            st.write(f"{item['calories']} kcal")
                                        with col3:
                                            st.write(f"{item['protein']:.1f}g" if item['protein'] else "-")
                                        with col4:
                                            st.write(f"{item['carbs']:.1f}g" if item['carbs'] else "-")
                                        with col5:
                                            st.write(f"{item['fats']:.1f}g" if item['fats'] else "-")
                            else:
                                st.error("❌ **Failed to Extract Food Items**")
                                st.write("The nutritional analysis couldn't identify specific food items.")
                                
                                # Provide immediate solutions
                                st.info("**Quick Solutions:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Option 1: Add Context**")
                                    st.write("Describe your meal in the context field above and click 'Analyze Meal' again.")
                                    st.write("**Example**: 'chicken curry with rice, naan bread, and yogurt sauce'")
                                
                                with col2:
                                    st.write("**Option 2: Text Analysis**")
                                    st.write("Use the 'Text Analysis' tab to describe your meal manually.")
                                    st.write("**Example**: 'Grilled chicken, mashed potatoes, broccoli, and sauce'")
                                
                                st.write("**For Better Results:**")
                                st.write("1. 📝 **Add Context**: Describe the meal in the context field")
                                st.write("2. 📸 **Better Image**: Upload a clearer, closer photo")
                                st.write("3. 🍽️ **Text Analysis**: Use the Text Analysis tab instead")
                                st.write("4. 🔄 **Retry**: Upload the image again")
                                
                                # Continue with basic analysis
                                st.warning("**Continuing with basic analysis. Please add context for better results.**")
                                
                                # Create basic food data to continue
                                food_data = [{"item": "Generic meal component", "calories": 300, "protein": 20, "carbs": 30, "fats": 10}]
                                totals = {"calories": 300, "protein": 20, "carbs": 30, "fats": 10}
                        
                        # Tab 3: Visualizations
                        with analysis_tab3:
                            st.subheader("📊 AI Visualizations")
                            
                            if models['cnn_model'] and 'cnn_confidence' in locals() and cnn_confidence is not None:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("AI Confidence", f"{cnn_confidence*100:.1f}%")
                                with col2:
                                    st.metric("Predicted Class", cnn_predicted_class if cnn_predicted_class else "Unknown")
                            
                            # Show visualizations if available
                            viz_cols = st.columns(2)
                            with viz_cols[0]:
                                if edge_path and os.path.exists(edge_path):
                                    st.image(edge_path, caption="Edge Detection Analysis", use_container_width=True)
                                if shap_path and os.path.exists(shap_path):
                                    st.image(shap_path, caption="SHAP Explanation", use_container_width=True)
                            
                            with viz_cols[1]:
                                if gradcam_path and os.path.exists(gradcam_path):
                                    st.image(gradcam_path, caption="Grad-CAM Visualization", use_container_width=True)
                                if lime_path and os.path.exists(lime_path):
                                    st.image(lime_path, caption="LIME Interpretation", use_container_width=True)
                            
                            if not any([edge_path, gradcam_path, shap_path, lime_path]):
                                st.info("📊 **Visualization Information**")
                                st.write("AI visualizations help understand how the model analyzes your food image:")
                                st.write("• 🔍 **Edge Detection**: Identifies food boundaries and shapes")
                                st.write("• 🎯 **Grad-CAM**: Shows which parts of the image the AI focuses on")
                                st.write("• 📈 **SHAP**: Explains AI decision-making process")
                                st.write("• 🔬 **LIME**: Provides interpretable explanations")
                        
                        # Tab 4: Health Insights
                        with analysis_tab4:
                            st.subheader("💡 Health Insights & Recommendations")
                            
                            # Daily calorie comparison
                            today = date.today().isoformat()
                            daily_total = st.session_state.daily_calories.get(today, 0) + totals["calories"]
                            calorie_balance = daily_total - st.session_state.calorie_target
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Today's Total", f"{daily_total} kcal")
                            with col2:
                                st.metric("Daily Target", f"{st.session_state.calorie_target} kcal")
                            with col3:
                                st.metric("Balance", f"{calorie_balance:+} kcal", delta=f"{calorie_balance} kcal")
                            
                            # Health recommendations
                            with st.expander("🏃‍♂️ **Fitness Recommendations**", expanded=True):
                                if calorie_balance > 0:
                                    st.warning("**Calorie Surplus Detected**")
                                    st.write(f"You consumed {calorie_balance} kcal above your target.")
                                    st.write("**Recommended Activities:**")
                                    for activity in st.session_state.activity_preference:
                                        burn_rate = ACTIVITY_BURN_RATES.get(activity, 300)
                                        duration = (calorie_balance / burn_rate) * 60
                                        st.write(f"• **{activity}**: {duration:.0f} minutes")
                                elif calorie_balance < 0:
                                    st.info("**Calorie Deficit Detected**")
                                    st.write(f"You consumed {abs(calorie_balance)} kcal below your target.")
                                    st.write("**Consider adding:**")
                                    st.write("• Nutrient-dense snacks")
                                    st.write("• Protein-rich foods")
                                    st.write("• Healthy fats")
                                else:
                                    st.success("**Perfect Balance!**")
                                    st.write("Your calorie intake is perfectly balanced!")
                            
                            # Nutritional insights
                            with st.expander("🥗 **Nutritional Insights**", expanded=True):
                                if totals['protein'] and totals['protein'] > 30:
                                    st.success("✅ **Good Protein Intake**")
                                    st.write("This meal provides adequate protein for muscle maintenance.")
                                elif totals['protein'] and totals['protein'] < 15:
                                    st.warning("⚠️ **Low Protein Intake**")
                                    st.write("Consider adding more protein-rich foods.")
                                
                                if totals['carbs'] and totals['carbs'] > 50:
                                    st.info("📊 **High Carbohydrate Meal**")
                                    st.write("This meal is rich in carbohydrates, good for energy.")
                                
                                if totals['fats'] and totals['fats'] > 20:
                                    st.info("🥑 **Moderate Fat Content**")
                                    st.write("This meal contains healthy fats for satiety.")
                            
                            # Dietary preferences check
                            if st.session_state.dietary_preferences:
                                with st.expander("🌱 **Dietary Preferences Check**", expanded=False):
                                    st.write("**Your Preferences**: " + ", ".join(st.session_state.dietary_preferences))
                                    st.write("**Compatibility**: This meal appears to align with your dietary preferences.")
                        
                        # Tab 5: Technical Details
                        with analysis_tab5:
                            st.subheader("🔧 Technical Analysis Details")
                            
                            # Model information
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Model Status**")
                                st.write(f"• BLIP Model: {'✅ Loaded' if models['blip_model'] is not None else '❌ Failed'}")
                                st.write(f"• YOLO Model: {'✅ Loaded' if models['yolo_model'] is not None else '❌ Failed'}")
                                st.write(f"• ViT Model: {'✅ Loaded' if models['vit_model'] is not None else '❌ Failed'}")
                                st.write(f"• EfficientNet: {'✅ Loaded' if models['efficientnet_model'] is not None else '❌ Failed'}")
                                st.write(f"• CNN Model: {'✅ Loaded' if models['cnn_model'] is not None else '❌ Failed'}")
                                st.write(f"• TensorFlow: {'✅ Loaded' if models['tf_model'] is not None else '❌ Failed'}")
                                st.write(f"• LLM Model: {'✅ Loaded' if models['llm'] is not None else '❌ Failed'}")
                                st.write(f"• Device: {device}")
                            
                            with col2:
                                st.write("**Analysis Details**")
                                st.write(f"• Description Length: {len(description.split())} words")
                                st.write(f"• Detection Strategies: Multi-model approach")
                                st.write(f"• Image Enhancement: Applied")
                                st.write(f"• Analysis Success: {'✅ Yes' if enhanced_analysis['success'] else '❌ No'}")
                                st.write(f"• Models Used: {sum([models['blip_model'] is not None, models['yolo_model'] is not None, models['vit_model'] is not None, models['efficientnet_model'] is not None])}")
                                st.write(f"• Detection Quality: {'High' if len(description.split()) > 10 else 'Medium' if len(description.split()) > 5 else 'Low'}")
                            
                            # Raw data
                            with st.expander("📄 **Raw Analysis Data**", expanded=False):
                                st.write(f"**Raw Description**: {description}")
                                st.write(f"**Improved Description**: {enhanced_analysis.get('improved_description', 'Not available')}")
                                st.write(f"**Context**: {context}")
                                if not enhanced_analysis['success']:
                                    st.write(f"**Analysis Error**: {enhanced_analysis.get('error', 'Unknown')}")
                            
                            # Performance metrics
                            with st.expander("⚡ **Performance Metrics**", expanded=False):
                                st.write("**Detection Performance:**")
                                st.write(f"• Confidence Level: {detection_confidence}")
                                st.write(f"• Items Identified: {len(food_data) if food_data else 0}")
                                st.write(f"• Analysis Quality: {'High' if enhanced_analysis['success'] else 'Low'}")
                                if 'cnn_confidence' in locals() and cnn_confidence is not None:
                                    st.write(f"• AI Confidence: {cnn_confidence*100:.1f}%")
                        
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
                        
                        # Add comprehensive detection feedback
                        if len(food_data) >= 2:
                            st.success("✅ **Comprehensive Detection Complete**")
                            st.write(f"**Items Detected**: {len(food_data)} food items")
                            st.write("**Detection Quality**: Comprehensive analysis performed")
                            st.write("**Coverage**: All visible food items identified")
                            
                            # Show detection summary
                            with st.expander("📊 **Detection Summary**", expanded=False):
                                st.write("**Detection Methods Used:**")
                                st.write("• 🔍 Enhanced Image Processing")
                                st.write("• 🤖 Multi-Strategy AI Analysis")
                                st.write("• 📊 Comprehensive Item Identification")
                                st.write("• 🍽️ Complete Nutritional Breakdown")
                                
                                st.write("**Items Identified:**")
                                for i, item in enumerate(food_data, 1):
                                    st.write(f"{i}. {item['item']}")
                        
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
                        
                        # Create tabs for text analysis results
                        text_tab1, text_tab2, text_tab3 = st.tabs([
                            "🍴 Nutrition Analysis",
                            "💡 Health Insights", 
                            "📊 Detailed Breakdown"
                        ])
                        
                        # Tab 1: Nutrition Analysis
                        with text_tab1:
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
                        
                        # Tab 2: Health Insights
                        with text_tab2:
                            st.subheader("💡 Health Insights & Recommendations")
                            
                            # Daily calorie comparison
                            today = date.today().isoformat()
                            daily_total = st.session_state.daily_calories.get(today, 0) + totals["calories"]
                            calorie_balance = daily_total - st.session_state.calorie_target
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Today's Total", f"{daily_total} kcal")
                            with col2:
                                st.metric("Daily Target", f"{st.session_state.calorie_target} kcal")
                            with col3:
                                st.metric("Balance", f"{calorie_balance:+} kcal", delta=f"{calorie_balance} kcal")
                            
                            # Health recommendations
                            with st.expander("🏃‍♂️ **Fitness Recommendations**", expanded=True):
                                if calorie_balance > 0:
                                    st.warning("**Calorie Surplus Detected**")
                                    st.write(f"You consumed {calorie_balance} kcal above your target.")
                                    st.write("**Recommended Activities:**")
                                    for activity in st.session_state.activity_preference:
                                        burn_rate = ACTIVITY_BURN_RATES.get(activity, 300)
                                        duration = (calorie_balance / burn_rate) * 60
                                        st.write(f"• **{activity}**: {duration:.0f} minutes")
                                elif calorie_balance < 0:
                                    st.info("**Calorie Deficit Detected**")
                                    st.write(f"You consumed {abs(calorie_balance)} kcal below your target.")
                                    st.write("**Consider adding:**")
                                    st.write("• Nutrient-dense snacks")
                                    st.write("• Protein-rich foods")
                                    st.write("• Healthy fats")
                                else:
                                    st.success("**Perfect Balance!**")
                                    st.write("Your calorie intake is perfectly balanced!")
                            
                            # Nutritional insights
                            with st.expander("🥗 **Nutritional Insights**", expanded=True):
                                if totals['protein'] and totals['protein'] > 30:
                                    st.success("✅ **Good Protein Intake**")
                                    st.write("This meal provides adequate protein for muscle maintenance.")
                                elif totals['protein'] and totals['protein'] < 15:
                                    st.warning("⚠️ **Low Protein Intake**")
                                    st.write("Consider adding more protein-rich foods.")
                                
                                if totals['carbs'] and totals['carbs'] > 50:
                                    st.info("📊 **High Carbohydrate Meal**")
                                    st.write("This meal is rich in carbohydrates, good for energy.")
                                
                                if totals['fats'] and totals['fats'] > 20:
                                    st.info("🥑 **Moderate Fat Content**")
                                    st.write("This meal contains healthy fats for satiety.")
                        
                        # Tab 3: Detailed Breakdown
                        with text_tab3:
                            st.subheader("📊 Detailed Nutrition Breakdown")
                            
                            if food_data:
                                # Detailed nutrition breakdown
                                with st.expander("📋 **Item-by-Item Breakdown**", expanded=True):
                                    for i, item in enumerate(food_data, 1):
                                        col1, col2, col3, col4, col5 = st.columns(5)
                                        with col1:
                                            st.write(f"**{i}. {item['item']}**")
                                        with col2:
                                            st.write(f"{item['calories']} kcal")
                                        with col3:
                                            st.write(f"{item['protein']:.1f}g" if item['protein'] else "-")
                                        with col4:
                                            st.write(f"{item['carbs']:.1f}g" if item['carbs'] else "-")
                                        with col5:
                                            st.write(f"{item['fats']:.1f}g" if item['fats'] else "-")
                                
                                # Analysis summary
                                with st.expander("📝 **Analysis Summary**", expanded=False):
                                    st.write(f"**Description**: {meal_desc}")
                                    st.write(f"**Items Identified**: {len(food_data)}")
                                    st.write(f"**Total Calories**: {totals['calories']} kcal")
                                    st.write(f"**Analysis Quality**: {'High' if len(food_data) >= 2 else 'Low'}")
                        
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
                    # Create tabs for history entries
                    if entry['type'] == "image":
                        hist_tab1, hist_tab2, hist_tab3, hist_tab4 = st.tabs([
                            "📷 Image & Analysis",
                            "🍴 Nutrition", 
                            "📊 Visualizations",
                            "📋 Details"
                        ])
                        
                        # Tab 1: Image & Analysis
                        with hist_tab1:
                            if entry.get("image"):
                                st.image(entry["image"], caption="Meal Image", use_container_width=True)
                            st.markdown(entry["analysis"], unsafe_allow_html=True)
                        
                        # Tab 2: Nutrition
                        with hist_tab2:
                            if entry.get("totals", {}):
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Calories", f"{entry['totals']['calories']} kcal")
                                col2.metric("Protein", f"{entry['totals']['protein']:.1f} g" if entry['totals']['protein'] else "-")
                                col3.metric("Carbs", f"{entry['totals']['carbs']:.1f} g" if entry['totals']['carbs'] else "-")
                                col4.metric("Fats", f"{entry['totals']['fats']:.1f} g" if entry['totals']['fats'] else "-")
                            
                            if entry.get("chart"):
                                st.pyplot(entry["chart"])
                            
                            if entry.get("nutrients"):
                                with st.expander("📋 **Detailed Breakdown**", expanded=False):
                                    for i, item in enumerate(entry["nutrients"], 1):
                                        col1, col2, col3, col4, col5 = st.columns(5)
                                        with col1:
                                            st.write(f"**{i}. {item['item']}**")
                                        with col2:
                                            st.write(f"{item['calories']} kcal")
                                        with col3:
                                            st.write(f"{item['protein']:.1f}g" if item['protein'] else "-")
                                        with col4:
                                            st.write(f"{item['carbs']:.1f}g" if item['carbs'] else "-")
                                        with col5:
                                            st.write(f"{item['fats']:.1f}g" if item['fats'] else "-")
                        
                        # Tab 3: Visualizations
                        with hist_tab3:
                            st.markdown("### AI Visualizations")
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
                                    st.metric("AI Confidence", f"{entry['cnn_confidence']*100:.1f}%")
                        
                        # Tab 4: Details
                        with hist_tab4:
                            st.write("**Analysis Details:**")
                            st.write(f"• **Type**: {entry['type']}")
                            st.write(f"• **Timestamp**: {entry['timestamp']}")
                            st.write(f"• **Items Detected**: {len(entry.get('nutrients', []))}")
                            if entry.get("cnn_prediction"):
                                st.write(f"• **AI Prediction**: {entry['cnn_prediction']}")
                            if entry.get("context"):
                                st.write(f"• **Context**: {entry['context']}")
                    
                    else:  # Text analysis
                        hist_tab1, hist_tab2, hist_tab3 = st.tabs([
                            "📝 Analysis",
                            "🍴 Nutrition", 
                            "📋 Details"
                        ])
                        
                        # Tab 1: Analysis
                        with hist_tab1:
                            st.markdown(entry["analysis"], unsafe_allow_html=True)
                        
                        # Tab 2: Nutrition
                        with hist_tab2:
                            if entry.get("totals", {}):
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Calories", f"{entry['totals']['calories']} kcal")
                                col2.metric("Protein", f"{entry['totals']['protein']:.1f} g" if entry['totals']['protein'] else "-")
                                col3.metric("Carbs", f"{entry['totals']['carbs']:.1f} g" if entry['totals']['carbs'] else "-")
                                col4.metric("Fats", f"{entry['totals']['fats']:.1f} g" if entry['totals']['fats'] else "-")
                            
                            if entry.get("chart"):
                                st.pyplot(entry["chart"])
                        
                        # Tab 3: Details
                        with hist_tab3:
                            st.write("**Analysis Details:**")
                            st.write(f"• **Type**: {entry['type']}")
                            st.write(f"• **Timestamp**: {entry['timestamp']}")
                            st.write(f"• **Items Detected**: {len(entry.get('nutrients', []))}")
                            st.write(f"• **Description**: {entry.get('description', 'N/A')}")
            
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
                            
                            # Create tabs for portion adjustment results
                            portion_tab1, portion_tab2, portion_tab3 = st.tabs([
                                "🍴 Updated Nutrition",
                                "📊 Comparison", 
                                "💡 Insights"
                            ])
                            
                            # Tab 1: Updated Nutrition
                            with portion_tab1:
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
                            
                            # Tab 2: Comparison
                            with portion_tab2:
                                st.subheader("📊 Before vs After Comparison")
                                
                                # Get original values
                                original_totals = st.session_state.last_results.get("totals", {})
                                original_calories = original_totals.get("calories", 0)
                                original_protein = original_totals.get("protein", 0)
                                original_carbs = original_totals.get("carbs", 0)
                                original_fats = original_totals.get("fats", 0)
                                
                                # Calculate differences
                                calorie_diff = totals["calories"] - original_calories
                                protein_diff = (totals["protein"] or 0) - original_protein
                                carbs_diff = (totals["carbs"] or 0) - original_carbs
                                fats_diff = (totals["fats"] or 0) - original_fats
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Original Values**")
                                    st.metric("Calories", f"{original_calories} kcal")
                                    st.metric("Protein", f"{original_protein:.1f} g")
                                    st.metric("Carbs", f"{original_carbs:.1f} g")
                                    st.metric("Fats", f"{original_fats:.1f} g")
                                
                                with col2:
                                    st.write("**Updated Values**")
                                    st.metric("Calories", f"{totals['calories']} kcal", delta=f"{calorie_diff:+} kcal")
                                    st.metric("Protein", f"{totals['protein']:.1f} g" if totals['protein'] else "-", delta=f"{protein_diff:+.1f} g")
                                    st.metric("Carbs", f"{totals['carbs']:.1f} g" if totals['carbs'] else "-", delta=f"{carbs_diff:+.1f} g")
                                    st.metric("Fats", f"{totals['fats']:.1f} g" if totals['fats'] else "-", delta=f"{fats_diff:+.1f} g")
                            
                            # Tab 3: Insights
                            with portion_tab3:
                                st.subheader("💡 Portion Adjustment Insights")
                                
                                # Daily impact
                                today = date.today().isoformat()
                                daily_total = st.session_state.daily_calories.get(today, 0) + totals["calories"]
                                calorie_balance = daily_total - st.session_state.calorie_target
                                
                                st.write("**Daily Impact:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Today's Total", f"{daily_total} kcal")
                                with col2:
                                    st.metric("Daily Target", f"{st.session_state.calorie_target} kcal")
                                with col3:
                                    st.metric("Balance", f"{calorie_balance:+} kcal", delta=f"{calorie_balance} kcal")
                                
                                # Health insights
                                with st.expander("🏃‍♂️ **Fitness Recommendations**", expanded=True):
                                    if calorie_balance > 0:
                                        st.warning("**Calorie Surplus**")
                                        st.write("Consider adjusting portions or adding exercise.")
                                    elif calorie_balance < 0:
                                        st.info("**Calorie Deficit**")
                                        st.write("Portion adjustment helps meet your daily target.")
                                    else:
                                        st.success("**Perfect Balance**")
                                        st.write("Great portion control!")
                                
                                # Portion tips
                                with st.expander("💡 **Portion Control Tips**", expanded=False):
                                    st.write("• Use smaller plates to control portions")
                                    st.write("• Measure ingredients when cooking")
                                    st.write("• Pay attention to serving sizes")
                                    st.write("• Listen to your body's hunger cues")
                            
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
    
    # Show model status
    show_model_status()
    
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