import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os
from datetime import datetime, date
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch
import torchvision.models as torchvision_models
from torchvision import transforms
import numpy as np
import logging
from PIL import ImageEnhance, ImageFilter
import re
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from captum.attr import GradientShap
from lime.lime_image import LimeImageExplainer
import uuid
import os

# Additional model imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="🍱 AI Calorie Tracker", layout="wide", page_icon="🍽️")

# Simple CSS
st.markdown("""
<style>
    .main { background-color: #ffffff; padding: 20px; border-radius: 10px; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stMetric { background-color: #f8f9fa; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
@st.cache_resource
def load_models():
    models = {}
    
    # Load LLM
    try:
        models['llm'] = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)
        logger.info("LLM loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        models['llm'] = None
    
    # Load BLIP
    try:
        models['processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        models['blip_model'] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device).eval()
        logger.info("BLIP loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load BLIP: {e}")
        models['processor'] = None
        models['blip_model'] = None
    
    # Load YOLO
    if YOLO_AVAILABLE:
        try:
            models['yolo_model'] = YOLO('yolov8n.pt')
            logger.info("YOLO loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            models['yolo_model'] = None
    else:
        models['yolo_model'] = None
    
    # Load CNN model for visualizations
    try:
        models['cnn_model'] = torchvision_models.densenet121(weights="IMAGENET1K_V1")
        models['cnn_model'] = models['cnn_model'].to(device).eval()
        logger.info("CNN model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load CNN model: {e}")
        models['cnn_model'] = None
    
    return models

models = load_models()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "daily_calories" not in st.session_state:
    st.session_state.daily_calories = {}
if "calorie_target" not in st.session_state:
    st.session_state.calorie_target = 2000

# Fast image processing (optimized for speed)
def enhance_image_quality(image: Image.Image) -> List[Image.Image]:
    """Fast image enhancement - only essential enhancements for speed."""
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        enhanced_images = []
        
        # Only the most effective enhancement (contrast)
        enhancer = ImageEnhance.Contrast(image)
        enhanced_images.append(enhancer.enhance(1.3))
        
        # One combined enhancement for backup
        balanced = image.copy()
        balanced = ImageEnhance.Contrast(balanced).enhance(1.2)
        balanced = ImageEnhance.Sharpness(balanced).enhance(1.3)
        enhanced_images.append(balanced)
        
        return enhanced_images
        
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        return [image]

# CNN image transform
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Visualization functions
def visualize_food_features(image):
    """Edge detection visualization with enhanced error handling."""
    try:
        logger.info("Starting edge detection visualization...")
        
        # Ensure image is in correct format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize and convert to numpy array
        img_np = np.array(image.resize((224, 224)))
        logger.info(f"Image shape: {img_np.shape}")
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Create matplotlib figure
        plt.figure(figsize=(8, 8))
        plt.imshow(edges, cmap="gray")
        plt.title("Edge Detection - Food Boundaries")
        plt.axis("off")
        
        # Save to temporary file
        edge_path = f"temp_edge_{uuid.uuid4().hex}.png"
        plt.savefig(edge_path, bbox_inches="tight", dpi=150, facecolor='white')
        plt.close()
        
        logger.info(f"Edge detection completed, saved to: {edge_path}")
        return edge_path
        
    except Exception as e:
        logger.error(f"Edge detection failed: {e}")
        plt.close()  # Ensure plot is closed even on error
        return None

def apply_gradcam(image_tensor, model, target_class):
    """Grad-CAM visualization with enhanced error handling."""
    if model is None:
        logger.warning("No model provided for Grad-CAM")
        return None
        
    try:
        logger.info("Starting Grad-CAM visualization...")
        model.eval()
        gradients, activations = [], []
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients.append(grad_output[0].detach())
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
        
        # Register hooks
        last_conv = model.features.norm5
        handle_fwd = last_conv.register_forward_hook(forward_hook)
        handle_bwd = last_conv.register_backward_hook(backward_hook)
        
        # Forward pass
        image_tensor = image_tensor.clone().detach().to(device).requires_grad_(True)
        output = model(image_tensor)
        model.zero_grad()
        
        # Use the top predicted class if target_class is 0
        if target_class == 0:
            target_class = output.argmax(dim=1).item()
        
        class_loss = output[0, target_class]
        class_loss.backward()
        
        if not gradients or not activations:
            handle_fwd.remove()
            handle_bwd.remove()
            logger.warning("No gradients or activations captured")
            return None
        
        # Generate CAM
        grads_val = gradients[0]
        activations_val = activations[0]
        
        weights = grads_val.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations_val).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.squeeze().detach().cpu().numpy()
        
        # Denormalize image
        image_np = image_tensor.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        # Create visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(image_np)
        plt.imshow(cam_np, cmap="jet", alpha=0.6)
        plt.title("Grad-CAM - AI Model Focus Areas")
        plt.axis("off")
        
        gradcam_path = f"temp_gradcam_{uuid.uuid4().hex}.png"
        plt.savefig(gradcam_path, bbox_inches="tight", dpi=150, facecolor='white')
        plt.close()
        
        # Clean up hooks
        handle_fwd.remove()
        handle_bwd.remove()
        
        logger.info(f"Grad-CAM completed, saved to: {gradcam_path}")
        return gradcam_path
        
    except Exception as e:
        logger.error(f"Grad-CAM failed: {e}")
        plt.close()  # Ensure plot is closed
        try:
            handle_fwd.remove()
            handle_bwd.remove()
        except:
            pass
        return None

def apply_shap(image_tensor, model):
    """SHAP visualization with enhanced error handling."""
    if model is None:
        logger.warning("No model provided for SHAP")
        return None
        
    try:
        logger.info("Starting SHAP analysis...")
        model.eval()
        
        # Create SHAP explainer
        gradient_shap = GradientShap(model)
        baseline = torch.zeros_like(image_tensor).to(device)
        image_tensor = image_tensor.clone().detach().requires_grad_(True).to(device)
        
        # Generate attributions
        attributions = gradient_shap.attribute(image_tensor, baselines=baseline, target=0, n_samples=50)
        attr_np = attributions.sum(dim=1).squeeze().detach().cpu().numpy()
        
        # Denormalize image
        image_np = image_tensor.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        # Create visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(image_np, alpha=0.7)
        plt.imshow(np.abs(attr_np), cmap="viridis", alpha=0.6)
        plt.title("SHAP Analysis - Feature Importance")
        plt.colorbar(label="Attribution Magnitude")
        plt.axis("off")
        
        shap_path = f"temp_shap_{uuid.uuid4().hex}.png"
        plt.savefig(shap_path, bbox_inches="tight", dpi=150, facecolor='white')
        plt.close()
        
        logger.info(f"SHAP analysis completed, saved to: {shap_path}")
        return shap_path
        
    except Exception as e:
        logger.error(f"SHAP failed: {e}")
        plt.close()  # Ensure plot is closed
        return None

def apply_lime(image, model, classes):
    """LIME visualization with enhanced error handling."""
    if model is None:
        logger.warning("No model provided for LIME")
        return None
        
    try:
        logger.info("Starting LIME explanation...")
        explainer = LimeImageExplainer()
        
        def predict_fn(images):
            # Convert images to tensor format
            images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            # Normalize
            images = (images - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            
            with torch.no_grad():
                outputs = model(images)
            return F.softmax(outputs, dim=1).cpu().numpy()
        
        # Prepare image
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image.resize((224, 224)))
        
        # Generate explanation
        explanation = explainer.explain_instance(
            image_np, 
            predict_fn, 
            top_labels=2, 
            num_samples=100,  # Reduced for speed
            segmentation_fn=None
        )
        
        # Get explanation visualization
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=8, 
            hide_rest=False
        )
        
        # Create visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(temp)
        plt.imshow(mask, cmap="viridis", alpha=0.4)
        plt.title("LIME Explanation - Local Interpretability")
        plt.colorbar(label="Feature Importance")
        plt.axis("off")
        
        lime_path = f"temp_lime_{uuid.uuid4().hex}.png"
        plt.savefig(lime_path, bbox_inches="tight", dpi=150, facecolor='white')
        plt.close()
        
        logger.info(f"LIME explanation completed, saved to: {lime_path}")
        return lime_path
        
    except Exception as e:
        logger.error(f"LIME failed: {e}")
        plt.close()  # Ensure plot is closed
        return None

# Extract food items from text - Enhanced version
def extract_food_items_from_text(text: str) -> set:
    """Extract individual food items from descriptive text with comprehensive parsing."""
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

# Optimized fast food detection
def describe_image_enhanced(image: Image.Image) -> str:
    """Fast, optimized food detection with smart strategies for speed."""
    if not models['processor'] or not models['blip_model']:
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
                logger.info(f"BLIP found: {len(items)} items")
                
            except Exception as e:
                logger.warning(f"BLIP prompt failed: {e}")
        
        # Strategy 2: Single-pass YOLO (optimized)
        if models['yolo_model']:
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
                logger.warning(f"YOLO detection failed: {e}")
        
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
            logger.warning(f"Enhanced image analysis failed: {e}")
        
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
                logger.info(f"Fast detection found {len(unique_items)} items: {result}")
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
            logger.warning(f"Fallback failed: {e}")
        
        return "Food items detected. Add context for better identification."
            
    except Exception as e:
        logger.error(f"Food detection error: {e}")
        return "Detection failed. Please try again."

# Query LLM
def query_langchain(prompt):
    if not models['llm']:
        return "LLM service unavailable."
    try:
        response = models['llm']([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return f"Error querying LLM: {str(e)}"

# Enhanced nutritional data extraction
def extract_items_and_nutrients(text):
    """Extract food items and nutritional data with enhanced parsing for detailed analysis."""
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
        
        logger.info(f"Extracted {len(items)} food items with {totals['calories']} total calories")
        return items, totals
        
    except Exception as e:
        logger.error(f"Error extracting items and nutrients: {e}")
        return [], {"calories": 0, "protein": 0, "carbs": 0, "fats": 0}

# Estimate calories from description
def estimate_calories_from_description(description: str) -> int:
    """Estimate calories based on food description."""
    description = description.lower()
    
    calorie_map = {
        'pizza': 300, 'burger': 500, 'sandwich': 350, 'salad': 150,
        'rice': 200, 'pasta': 250, 'chicken': 200, 'fish': 180,
        'beef': 250, 'pork': 220, 'egg': 70, 'cheese': 100,
        'bread': 80, 'potato': 160, 'apple': 80, 'banana': 100,
        'cake': 350, 'cookie': 150, 'soup': 120, 'coffee': 5
    }
    
    total_calories = 0
    for food, calories in calorie_map.items():
        if food in description:
            total_calories += calories
    
    if total_calories == 0:
        if len(description.split()) > 10:
            total_calories = 600
        elif len(description.split()) > 5:
            total_calories = 400
        else:
            total_calories = 250
    
    return max(total_calories, 100)

# Enhanced food analysis with detailed formatting
def analyze_food_with_enhanced_prompt(image_description: str, context: str = "") -> Dict[str, Any]:
    """Comprehensive food analysis with detailed, well-formatted responses."""
    try:
        # Enhanced prompt for detailed analysis
        prompt = f"""You are an expert nutritionist analyzing food images. Provide a comprehensive, detailed analysis:

DETECTED FOODS: {image_description}
MEAL CONTEXT: {context if context else "General meal analysis"}

Please provide a thorough analysis following this EXACT format:

## COMPREHENSIVE FOOD ANALYSIS

### IDENTIFIED FOOD ITEMS:
For each food item detected, provide detailed breakdown:
- Item: [Food name with estimated portion size], Calories: [X], Protein: [X]g, Carbs: [X]g, Fats: [X]g, Fiber: [X]g
- Item: [Food name with estimated portion size], Calories: [X], Protein: [X]g, Carbs: [X]g, Fats: [X]g, Fiber: [X]g
[Continue for ALL items - include main dishes, sides, sauces, garnishes, beverages]

### NUTRITIONAL TOTALS:
- Total Calories: [X] kcal
- Total Protein: [X]g ([X]% of calories)
- Total Carbohydrates: [X]g ([X]% of calories)
- Total Fats: [X]g ([X]% of calories)
- Total Fiber: [X]g
- Estimated Sodium: [X]mg

### MEAL COMPOSITION ANALYSIS:
- **Meal Type**: [Breakfast/Lunch/Dinner/Snack]
- **Cuisine Style**: [If identifiable]
- **Portion Size**: [Small/Medium/Large/Extra Large]
- **Cooking Methods**: [Grilled, fried, baked, etc.]
- **Main Macronutrient**: [Carb-heavy/Protein-rich/Fat-dense/Balanced]

### NUTRITIONAL QUALITY ASSESSMENT:
- **Strengths**: [What's nutritionally good about this meal]
- **Areas for Improvement**: [What could be better]
- **Missing Nutrients**: [What important nutrients might be lacking]
- **Calorie Density**: [High/Medium/Low - calories per volume]

### HEALTH RECOMMENDATIONS:
1. **Immediate Suggestions**: [2-3 specific tips for this meal]
2. **Portion Adjustments**: [If needed]
3. **Complementary Foods**: [What to add for better nutrition]
4. **Timing Considerations**: [Best time to eat this meal]

### DIETARY CONSIDERATIONS:
- **Allergen Information**: [Common allergens present]
- **Dietary Restrictions**: [Vegan/Vegetarian/Gluten-free compatibility]
- **Blood Sugar Impact**: [High/Medium/Low glycemic impact]

CRITICAL REQUIREMENTS:
- Identify EVERY visible food component, no matter how small
- Include cooking oils, seasonings, and hidden ingredients in calorie counts
- Provide realistic portion estimates based on visual cues
- Be thorough with nutritional breakdowns
- Consider preparation methods that add calories"""

        # Get comprehensive analysis
        response = models['llm'].invoke(prompt)
        analysis = response.content
        
        # Extract structured data with enhanced parsing
        items, totals = extract_items_and_nutrients(analysis)
        
        # Create detailed food items list
        food_items = []
        for item in items:
            food_items.append({
                "item": item["item"],
                "description": f"{item['item']} - {item['calories']} calories",
                "calories": item["calories"],
                "protein": item.get("protein", 0),
                "carbs": item.get("carbs", 0),
                "fats": item.get("fats", 0),
                "fiber": item.get("fiber", 0)
            })
        
        # Enhanced fallback with detailed estimation
        if not food_items and image_description:
            estimated_calories = estimate_calories_from_description(image_description)
            
            # Create detailed fallback analysis
            fallback_analysis = f"""## COMPREHENSIVE FOOD ANALYSIS

### IDENTIFIED FOOD ITEMS:
- Item: Complete meal ({image_description}), Calories: {estimated_calories}, Protein: {estimated_calories * 0.15 / 4:.1f}g, Carbs: {estimated_calories * 0.50 / 4:.1f}g, Fats: {estimated_calories * 0.35 / 9:.1f}g

### NUTRITIONAL TOTALS:
- Total Calories: {estimated_calories} kcal
- Total Protein: {estimated_calories * 0.15 / 4:.1f}g (15% of calories)
- Total Carbohydrates: {estimated_calories * 0.50 / 4:.1f}g (50% of calories)
- Total Fats: {estimated_calories * 0.35 / 9:.1f}g (35% of calories)

### MEAL COMPOSITION ANALYSIS:
- **Meal Type**: Mixed meal
- **Portion Size**: Medium (estimated)
- **Main Macronutrient**: Balanced

### NUTRITIONAL QUALITY ASSESSMENT:
- **Note**: Limited analysis due to detection constraints
- **Recommendation**: Provide more specific food descriptions for detailed analysis

### HEALTH RECOMMENDATIONS:
1. **Balance**: Ensure adequate protein, vegetables, and whole grains
2. **Portion Control**: Monitor serving sizes for calorie management
3. **Hydration**: Include water with your meal

### DIETARY CONSIDERATIONS:
- **Analysis Limitation**: Detailed allergen and dietary information requires clearer food identification"""

            food_items = [{
                "item": "Complete meal",
                "description": f"Meal with: {image_description}",
                "calories": estimated_calories,
                "protein": estimated_calories * 0.15 / 4,
                "carbs": estimated_calories * 0.50 / 4,
                "fats": estimated_calories * 0.35 / 9,
                "fiber": 5  # Estimated
            }]
            
            totals = {
                "calories": estimated_calories,
                "protein": estimated_calories * 0.15 / 4,
                "carbs": estimated_calories * 0.50 / 4,
                "fats": estimated_calories * 0.35 / 9
            }
            
            analysis = fallback_analysis
        
        # Ensure minimum detail level
        if len(analysis) < 500:  # If analysis is too brief, enhance it
            enhancement_prompt = f"""The previous analysis was too brief. Please provide a more detailed analysis of: {image_description}

Include:
- Detailed breakdown of each food component
- Comprehensive nutritional information
- Health assessment and recommendations
- Meal composition analysis

Make it thorough and informative."""
            
            try:
                enhanced_response = models['llm'].invoke(enhancement_prompt)
                analysis = enhanced_response.content
            except:
                pass  # Keep original if enhancement fails
        
        return {
            "success": True,
            "analysis": analysis,
            "food_items": food_items,
            "nutritional_data": {
                "total_calories": int(totals.get("calories", 0)),
                "total_protein": round(totals.get("protein", 0), 1),
                "total_carbs": round(totals.get("carbs", 0), 1),
                "total_fats": round(totals.get("fats", 0), 1),
                "items": food_items
            },
            "improved_description": image_description,
            "detailed": True
        }
        
    except Exception as e:
        logger.error(f"Food analysis error: {e}")
        
        # Enhanced error fallback
        estimated_calories = estimate_calories_from_description(image_description)
        
        error_analysis = f"""## FOOD ANALYSIS (Limited Mode)

### DETECTED ITEMS:
- Item: {image_description}, Calories: {estimated_calories} (estimated)

### ANALYSIS NOTE:
Due to technical limitations, a detailed analysis could not be completed. 

### BASIC NUTRITIONAL ESTIMATE:
- Estimated Calories: {estimated_calories} kcal
- This is a rough estimate based on typical meal components

### RECOMMENDATIONS:
1. For accurate analysis, try uploading a clearer image
2. Add specific food descriptions in the context field
3. Consider manual entry for precise tracking

### NEXT STEPS:
- Retry with better lighting or closer image
- Describe specific foods in the context field
- Use the text analysis feature for manual entry"""

        return {
            "success": True,
            "analysis": error_analysis,
            "food_items": [{
                "item": "Estimated meal",
                "description": image_description,
                "calories": estimated_calories,
                "protein": 0,
                "carbs": 0,
                "fats": 0,
                "fiber": 0
            }],
            "nutritional_data": {
                "total_calories": estimated_calories,
                "total_protein": 0,
                "total_carbs": 0,
                "total_fats": 0,
                "items": []
            },
            "improved_description": image_description,
            "detailed": False
        }

# Main UI
st.title("🍱 AI Calorie Tracker")
st.caption("Track your nutrition with AI-powered food analysis")

# Simplified tabs
tab1, tab2, tab3 = st.tabs([
    "📷 Food Analysis",
    "📊 History", 
    "📅 Daily Summary"
])

# Food Analysis Tab
with tab1:
    st.subheader("📷 Food Analysis")
    st.write("Upload a food image for AI-powered calorie and nutrition analysis.")
    
    with st.expander("💡 Tips for Better Results"):
        st.write("• Take clear photos in good lighting")
        st.write("• Include all food items in the frame") 
        st.write("• Add context description if needed")
        st.write("• Try different angles if detection is incomplete")
    
    # Simple upload interface
    img_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
    context = st.text_area("Additional Context (Optional)", 
                          placeholder="Describe the meal if needed (e.g., 'chicken curry with rice')", 
                          height=80)
    
    if st.button("🔍 Analyze Food", disabled=not img_file):
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📷 Loading image...")
            progress_bar.progress(10)
            
            image = Image.open(img_file)
            
            # Display image immediately
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            status_text.text("🔍 Detecting food items...")
            progress_bar.progress(30)
            
            # Get food description (optimized)
            food_description = describe_image_enhanced(image)
            
            status_text.text("📊 Analyzing nutrition...")
            progress_bar.progress(70)
            
            # Analyze the food (optimized)
            analysis_result = analyze_food_with_enhanced_prompt(food_description, context)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if analysis_result["success"]:
                st.success("✅ Food analysis completed!")
                
                # Show detected food items
                st.subheader("🍽️ Detected Food Items")
                st.write(f"**Foods found:** {food_description}")
                
                # Show nutrition summary
                nutrition = analysis_result["nutritional_data"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Calories", f"{nutrition['total_calories']}")
                with col2:
                    st.metric("Protein", f"{nutrition['total_protein']:.1f}g")
                with col3:
                    st.metric("Carbs", f"{nutrition['total_carbs']:.1f}g")
                with col4:
                    st.metric("Fats", f"{nutrition['total_fats']:.1f}g")
                
                # Show detailed analysis with better formatting
                with st.expander("📊 Comprehensive Nutritional Analysis", expanded=True):
                    # Check if we have detailed analysis
                    if analysis_result.get("detailed", False):
                        st.markdown("### 🔬 **Complete Analysis Report**")
                        st.markdown(analysis_result["analysis"])
                        
                        # Show individual food items if available
                        if analysis_result["food_items"] and len(analysis_result["food_items"]) > 1:
                            st.markdown("### 📋 **Individual Food Items Breakdown**")
                            
                            for i, item in enumerate(analysis_result["food_items"], 1):
                                with st.container():
                                    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                                    with col1:
                                        st.write(f"**{i}. {item['item']}**")
                                    with col2:
                                        st.write(f"{item['calories']} cal")
                                    with col3:
                                        st.write(f"{item['protein']:.1f}g protein")
                                    with col4:
                                        st.write(f"{item['carbs']:.1f}g carbs")
                                    with col5:
                                        st.write(f"{item['fats']:.1f}g fats")
                    else:
                        st.markdown("### 📊 **Basic Analysis**")
                        st.write(analysis_result["analysis"])
                        st.info("💡 **Tip**: For more detailed analysis, try uploading a clearer image or add specific food descriptions in the context field.")
                
                # AI Visualizations - Always Generated
                with st.expander("🔬 AI Visualizations", expanded=True):
                    st.write("**AI Model Interpretability Visualizations**")
                    
                    if models['cnn_model']:
                        with st.spinner("Generating AI visualizations..."):
                            try:
                                # Prepare image for CNN
                                image_rgb = image.convert("RGB")
                                image_tensor = cnn_transform(image_rgb).unsqueeze(0).to(device)
                                
                                # Generate all visualizations
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Edge Detection**")
                                    edge_path = visualize_food_features(image)
                                    if edge_path:
                                        edge_img = Image.open(edge_path)
                                        st.image(edge_img, caption="Edge Detection - Food Boundaries", use_container_width=True)
                                        os.remove(edge_path)
                                    else:
                                        st.warning("Edge detection failed")
                                    
                                    st.write("**SHAP Analysis**")
                                    shap_path = apply_shap(image_tensor, models['cnn_model'])
                                    if shap_path:
                                        shap_img = Image.open(shap_path)
                                        st.image(shap_img, caption="SHAP - Feature Importance", use_container_width=True)
                                        os.remove(shap_path)
                                    else:
                                        st.warning("SHAP analysis failed")
                                
                                with col2:
                                    st.write("**Grad-CAM**")
                                    gradcam_path = apply_gradcam(image_tensor, models['cnn_model'], 0)
                                    if gradcam_path:
                                        gradcam_img = Image.open(gradcam_path)
                                        st.image(gradcam_img, caption="Grad-CAM - Model Focus Areas", use_container_width=True)
                                        os.remove(gradcam_path)
                                    else:
                                        st.warning("Grad-CAM failed")
                                    
                                    st.write("**LIME Explanation**")
                                    lime_path = apply_lime(image, models['cnn_model'], ["food"])
                                    if lime_path:
                                        lime_img = Image.open(lime_path)
                                        st.image(lime_img, caption="LIME - Local Interpretability", use_container_width=True)
                                        os.remove(lime_path)
                                    else:
                                        st.warning("LIME explanation failed")
                                
                                st.info("💡 **Visualization Guide:**")
                                st.write("• **Edge Detection**: Shows food boundaries and textures")
                                st.write("• **Grad-CAM**: Highlights areas the AI focuses on")
                                st.write("• **SHAP**: Shows feature importance for predictions")
                                st.write("• **LIME**: Explains local decision-making regions")
                                
                            except Exception as e:
                                st.error(f"Visualization generation failed: {str(e)}")
                                st.info("This might be due to:")
                                st.write("• Missing dependencies (captum, lime)")
                                st.write("• GPU/CPU compatibility issues")
                                st.write("• Image processing errors")
                    else:
                        st.warning("CNN model not available for visualizations.")
                        st.info("The CNN model is required for generating AI visualizations.")
                
                # Save to history
                entry = {
                    "timestamp": datetime.now(),
                    "type": "image",
                    "description": food_description,
                    "analysis": analysis_result["analysis"],
                    "nutritional_data": analysis_result["nutritional_data"],
                    "context": context
                }
                st.session_state.history.append(entry)
                
                # Update daily calories
                today = date.today().isoformat()
                if today not in st.session_state.daily_calories:
                    st.session_state.daily_calories[today] = 0
                st.session_state.daily_calories[today] += analysis_result["nutritional_data"]["total_calories"]
                
            else:
                st.warning("Analysis had issues. Try adding more context or describing the meal manually.")
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.info("Try uploading a clearer image or add more context description.")

# History Tab  
with tab2:
    st.subheader("📊 Analysis History")
    st.write("View your previous food analyses and track your nutrition over time.")
    
    if not st.session_state.history:
        st.info("No meal analyses recorded yet. Try analyzing a meal in the Food Analysis tab!")
    else:
        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"📅 {entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - {entry.get('description', 'Meal Analysis')}"):
                
                # Show nutrition summary
                nutrition = entry.get('nutritional_data', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Calories", f"{nutrition.get('total_calories', 0)}")
                with col2:
                    st.metric("Protein", f"{nutrition.get('total_protein', 0):.1f}g")
                with col3:
                    st.metric("Carbs", f"{nutrition.get('total_carbs', 0):.1f}g")
                with col4:
                    st.metric("Fats", f"{nutrition.get('total_fats', 0):.1f}g")
                
                # Show analysis
                if entry.get('analysis'):
                    with st.expander("📝 Full Analysis"):
                        st.write(entry['analysis'])

# Daily Summary Tab
with tab3:
    st.subheader("📅 Daily Nutrition Summary")
    
    today = date.today().isoformat()
    today_cals = st.session_state.daily_calories.get(today, 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Today's Calories", f"{today_cals} kcal")
    with col2:
        st.metric("Target", f"{st.session_state.calorie_target} kcal")
    
    # Progress bar
    progress = min(today_cals / st.session_state.calorie_target, 1.0) if st.session_state.calorie_target > 0 else 0
    st.progress(progress)
    st.caption(f"Progress: {progress*100:.1f}%")
    
    # Weekly summary
    if st.session_state.daily_calories:
        st.subheader("📈 Weekly Summary")
        dates = sorted(st.session_state.daily_calories.keys())[-7:]
        cals = [st.session_state.daily_calories.get(d, 0) for d in dates]
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(dates, cals, color='#4CAF50')
        ax.set_ylabel('Calories')
        ax.set_title('Daily Calorie Intake (Last 7 Days)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Sidebar
with st.sidebar:
    st.header("🍎 Nutrition Dashboard")
    
    # Model status
    st.subheader("🤖 AI Models Status")
    model_status = {
        'BLIP': models['blip_model'] is not None,
        'YOLO': models['yolo_model'] is not None,
        'LLM': models['llm'] is not None
    }
    
    for model, status in model_status.items():
        status_icon = "✅" if status else "❌"
        st.write(f"{status_icon} **{model}**: {'Available' if status else 'Not Available'}")
    
    st.subheader("User Profile")
    st.number_input("Daily Calorie Target (kcal)", min_value=1000, max_value=5000, 
                   value=st.session_state.calorie_target, step=100, key="calorie_target")
    
    st.subheader("Today's Progress")
    today = date.today().isoformat()
    today_cals = st.session_state.daily_calories.get(today, 0)
    progress = min(today_cals / st.session_state.calorie_target, 1.0) if st.session_state.calorie_target > 0 else 0
    st.progress(progress)
    st.caption(f"{today_cals}/{st.session_state.calorie_target} kcal ({progress*100:.1f}%)")
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.history.clear()
        st.session_state.daily_calories.clear()
        st.success("History cleared!")
        st.rerun()

# Footer - Clean and Professional
st.markdown("---")

# Create footer using Streamlit components for better rendering
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 20px;">
        <h3 style="color: #2c3e50; margin-bottom: 15px;">🍱 AI Calorie Tracker</h3>
        <p style="color: #495057; font-size: 14px; margin-bottom: 15px;">
            🔬 AI Visualizations: Edge Detection • Grad-CAM • SHAP • LIME
        </p>
        <p style="color: #495057; font-size: 16px; font-weight: 600; margin-bottom: 15px;">
            Developed by <strong style="color: #4CAF50;">Ujjwal Sinha</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Social links using Streamlit columns
    social_col1, social_col2 = st.columns(2)
    
    with social_col1:
        st.markdown("""
        <a href="https://github.com/Ujjwal-sinha" target="_blank" style="text-decoration: none;">
            <div style="display: inline-flex; align-items: center; justify-content: center; width: 100%; padding: 10px 20px; background-color: #4CAF50; color: white; border-radius: 8px; font-weight: 500; text-align: center;">
                📱 GitHub
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    with social_col2:
        st.markdown("""
        <a href="https://www.linkedin.com/in/sinhaujjwal01/" target="_blank" style="text-decoration: none;">
            <div style="display: inline-flex; align-items: center; justify-content: center; width: 100%; padding: 10px 20px; background-color: #4CAF50; color: white; border-radius: 8px; font-weight: 500; text-align: center;">
                💼 LinkedIn
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 15px; padding-top: 15px; border-top: 1px solid #dee2e6;">
        <p style="color: #6c757d; font-size: 12px; margin: 0;">
            © 2024 Ujjwal Sinha • Built with ❤️ using Streamlit & Advanced AI
        </p>
        <p style="color: #6c757d; font-size: 11px; margin: 5px 0 0 0;">
            🚀 Enhanced Food Detection • 🔬 AI Interpretability • 📊 Nutrition Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)