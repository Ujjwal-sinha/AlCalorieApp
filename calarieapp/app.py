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

# Enhanced image processing
def enhance_image_quality(image: Image.Image) -> List[Image.Image]:
    """Enhance image quality for better food detection."""
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        enhanced_images = []
        
        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        enhanced_images.append(enhancer.enhance(1.4))
        
        # Brightness enhancement
        enhancer = ImageEnhance.Brightness(image)
        enhanced_images.append(enhancer.enhance(1.3))
        
        # Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        enhanced_images.append(enhancer.enhance(1.6))
        
        # Combined enhancement
        balanced = image.copy()
        balanced = ImageEnhance.Contrast(balanced).enhance(1.25)
        balanced = ImageEnhance.Brightness(balanced).enhance(1.15)
        balanced = ImageEnhance.Sharpness(balanced).enhance(1.4)
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
    """Edge detection visualization."""
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
        return None

def apply_gradcam(image_tensor, model, target_class):
    """Grad-CAM visualization."""
    if model is None:
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
        return None

def apply_shap(image_tensor, model):
    """SHAP visualization."""
    if model is None:
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
        return None

def apply_lime(image, model, classes):
    """LIME visualization."""
    if model is None:
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

# Enhanced comprehensive food detection
def describe_image_enhanced(image: Image.Image) -> str:
    """Comprehensive food detection using multiple strategies to catch ALL items."""
    if not models['processor'] or not models['blip_model']:
        return "Image analysis unavailable. Please check model loading."
    
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        device = next(models['blip_model'].parameters()).device
        all_food_items = set()
        
        # Strategy 1: Comprehensive BLIP with specialized prompts
        comprehensive_prompts = [
            # General detection
            "List every single food item, ingredient, dish, and beverage visible in this image:",
            "What are all the foods, drinks, vegetables, fruits, meats, and grains you can see?",
            "Identify each individual food component including main dishes, sides, sauces, and garnishes:",
            
            # Specific category detection
            "What vegetables, fruits, herbs, and plant-based foods are in this image?",
            "What proteins, meats, fish, eggs, and dairy products can you identify?",
            "What grains, breads, pasta, rice, and carbohydrates are visible?",
            "What sauces, dressings, condiments, and seasonings can you see?",
            "What beverages, drinks, and liquids are present?",
            
            # Detail-focused detection
            "Look closely and identify every small ingredient, garnish, and food component:",
            "What cooking ingredients, spices, and flavor enhancers are visible?",
            "Describe all the individual food elements that make up this meal:",
            
            # Context-aware detection
            "This appears to be a meal. What are ALL the food items and ingredients present?",
            "Analyze this food image and list every edible component you can identify:",
            "What complete meal components including appetizers, mains, sides, and desserts are shown?"
        ]
        
        for prompt in comprehensive_prompts:
            try:
                inputs = models['processor'](image, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = models['blip_model'].generate(
                        **inputs, 
                        max_new_tokens=400,
                        num_beams=10,
                        do_sample=True,
                        temperature=0.2,
                        top_p=0.95,
                        repetition_penalty=1.2
                    )
                caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                
                if caption.startswith(prompt):
                    caption = caption.replace(prompt, "").strip()
                
                items = extract_food_items_from_text(caption)
                all_food_items.update(items)
                logger.info(f"BLIP found: {items}")
                
            except Exception as e:
                logger.warning(f"BLIP prompt failed: {e}")
        
        # Strategy 2: Enhanced YOLO with expanded food detection
        if models['yolo_model']:
            try:
                img_np = np.array(image)
                # Multiple confidence levels to catch more items
                confidence_levels = [0.05, 0.1, 0.15, 0.2]
                
                for conf_level in confidence_levels:
                    results = models['yolo_model'](img_np, conf=conf_level, iou=0.3)
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = models['yolo_model'].names[cls]
                                
                                # Expanded food-related terms
                                food_related_terms = [
                                    # Fruits
                                    'apple', 'banana', 'orange', 'lemon', 'lime', 'grape', 'strawberry', 'blueberry',
                                    'pineapple', 'mango', 'avocado', 'watermelon', 'peach', 'pear', 'cherry', 'kiwi',
                                    
                                    # Vegetables
                                    'tomato', 'potato', 'carrot', 'onion', 'garlic', 'broccoli', 'lettuce', 'spinach',
                                    'cucumber', 'pepper', 'corn', 'peas', 'beans', 'cabbage', 'celery', 'mushroom',
                                    'eggplant', 'zucchini', 'asparagus', 'cauliflower',
                                    
                                    # Proteins
                                    'chicken', 'fish', 'beef', 'pork', 'lamb', 'turkey', 'egg', 'shrimp',
                                    'salmon', 'tuna', 'bacon', 'sausage', 'ham', 'steak', 'meat',
                                    
                                    # Dairy
                                    'cheese', 'milk', 'yogurt', 'butter', 'cream', 'ice cream',
                                    
                                    # Grains & Carbs
                                    'bread', 'rice', 'pasta', 'noodles', 'cereal', 'oats', 'quinoa', 'wheat',
                                    
                                    # Prepared foods
                                    'pizza', 'burger', 'sandwich', 'soup', 'salad', 'pie', 'cake', 'cookie',
                                    'muffin', 'pancake', 'waffle', 'toast', 'bagel', 'donut', 'croissant',
                                    
                                    # Beverages
                                    'coffee', 'tea', 'juice', 'water', 'soda', 'beer', 'wine', 'smoothie',
                                    
                                    # General food terms
                                    'food', 'meal', 'dish', 'snack', 'dessert', 'appetizer'
                                ]
                                
                                if conf > 0.05 and any(term in class_name.lower() for term in food_related_terms):
                                    all_food_items.add(class_name.lower())
                                    logger.info(f"YOLO found: {class_name} ({conf:.2f})")
                                
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
        
        # Strategy 3: Multi-scale enhanced image analysis
        try:
            enhanced_images = enhance_image_quality(image)
            scales = [(224, 224), (384, 384), (512, 512)]
            
            for scale in scales:
                try:
                    scaled_img = image.resize(scale, Image.Resampling.LANCZOS)
                    
                    scale_prompts = [
                        "What food items are in this image?",
                        "List all visible ingredients and food components:",
                        "What dishes and food elements can you identify?"
                    ]
                    
                    for prompt in scale_prompts:
                        try:
                            inputs = models['processor'](scaled_img, text=prompt, return_tensors="pt").to(device)
                            with torch.no_grad():
                                outputs = models['blip_model'].generate(
                                    **inputs, 
                                    max_new_tokens=250,
                                    num_beams=8,
                                    temperature=0.3
                                )
                            caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                            
                            if caption.startswith(prompt):
                                caption = caption.replace(prompt, "").strip()
                            
                            items = extract_food_items_from_text(caption)
                            all_food_items.update(items)
                            
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    continue
                    
            # Enhanced images analysis
            for enhanced_img in enhanced_images[:3]:
                try:
                    enhanced_prompts = [
                        "What food items are visible in this enhanced image?",
                        "List all the ingredients and food components you can see:"
                    ]
                    
                    for prompt in enhanced_prompts:
                        inputs = models['processor'](enhanced_img, text=prompt, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = models['blip_model'].generate(
                                **inputs, 
                                max_new_tokens=200,
                                num_beams=6,
                                temperature=0.4
                            )
                        caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                        
                        if caption.startswith(prompt):
                            caption = caption.replace(prompt, "").strip()
                        
                        items = extract_food_items_from_text(caption)
                        all_food_items.update(items)
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.warning(f"Multi-scale analysis failed: {e}")
        
        # Strategy 4: Contextual and semantic analysis
        try:
            contextual_prompts = [
                "This is a food photograph. Identify every single edible item:",
                "What ingredients went into making this meal?",
                "What are all the food components that make up this dish?",
                "List every food item including hidden ingredients and seasonings:",
                "What cooking ingredients, spices, and garnishes are present?"
            ]
            
            for prompt in contextual_prompts:
                try:
                    inputs = models['processor'](image, text=prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = models['blip_model'].generate(
                            **inputs, 
                            max_new_tokens=350,
                            num_beams=12,
                            do_sample=True,
                            temperature=0.1,
                            top_p=0.98
                        )
                    caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                    
                    if caption.startswith(prompt):
                        caption = caption.replace(prompt, "").strip()
                    
                    items = extract_food_items_from_text(caption)
                    all_food_items.update(items)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.warning(f"Contextual analysis failed: {e}")
        
        # Enhanced filtering and cleaning
        if all_food_items:
            # Comprehensive food keywords
            food_keywords = {
                # Fruits
                'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'raspberry', 'blackberry',
                'tomato', 'avocado', 'lemon', 'lime', 'pineapple', 'mango', 'peach', 'pear', 'cherry',
                'watermelon', 'cantaloupe', 'kiwi', 'papaya', 'coconut', 'fig', 'date', 'plum',
                
                # Vegetables
                'potato', 'carrot', 'onion', 'garlic', 'broccoli', 'lettuce', 'spinach', 'kale',
                'cucumber', 'pepper', 'bell pepper', 'chili', 'corn', 'peas', 'beans', 'cabbage',
                'celery', 'mushroom', 'eggplant', 'zucchini', 'squash', 'asparagus', 'cauliflower',
                'radish', 'beet', 'turnip', 'parsnip', 'leek', 'scallion', 'shallot',
                
                # Proteins
                'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'fish', 'salmon', 'tuna',
                'shrimp', 'crab', 'lobster', 'egg', 'tofu', 'tempeh', 'seitan', 'bacon',
                'sausage', 'ham', 'steak', 'ground beef', 'meat', 'protein',
                
                # Dairy
                'cheese', 'milk', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese',
                'mozzarella', 'cheddar', 'parmesan', 'feta', 'goat cheese', 'ice cream',
                
                # Grains & Carbs
                'bread', 'rice', 'pasta', 'noodles', 'spaghetti', 'linguine', 'penne', 'fusilli',
                'cereal', 'oats', 'quinoa', 'barley', 'wheat', 'flour', 'bagel', 'croissant',
                'muffin', 'biscuit', 'cracker', 'pretzel', 'tortilla', 'pita',
                
                # Prepared foods
                'pizza', 'burger', 'sandwich', 'wrap', 'burrito', 'taco', 'quesadilla',
                'soup', 'stew', 'chili', 'curry', 'stir fry', 'salad', 'coleslaw',
                'pie', 'cake', 'cookie', 'brownie', 'pancake', 'waffle', 'french toast',
                'omelet', 'scrambled eggs', 'fried eggs',
                
                # Seasonings & Condiments
                'salt', 'pepper', 'garlic powder', 'onion powder', 'paprika', 'cumin',
                'oregano', 'basil', 'thyme', 'rosemary', 'parsley', 'cilantro', 'dill',
                'sauce', 'ketchup', 'mustard', 'mayo', 'mayonnaise', 'ranch', 'vinaigrette',
                'soy sauce', 'hot sauce', 'barbecue sauce', 'marinara', 'pesto',
                'oil', 'olive oil', 'butter', 'margarine', 'dressing', 'marinade',
                
                # Beverages
                'coffee', 'tea', 'juice', 'water', 'soda', 'beer', 'wine', 'smoothie',
                'milkshake', 'latte', 'cappuccino', 'espresso', 'cocktail',
                
                # Nuts & Seeds
                'almond', 'walnut', 'pecan', 'cashew', 'peanut', 'pistachio', 'hazelnut',
                'sunflower seeds', 'pumpkin seeds', 'sesame seeds', 'chia seeds', 'flax seeds',
                
                # General terms
                'food', 'meal', 'dish', 'ingredient', 'snack', 'appetizer', 'entree', 'dessert',
                'side dish', 'garnish', 'topping', 'filling', 'seasoning', 'spice', 'herb'
            }
            
            final_items = []
            for item in all_food_items:
                item_clean = item.strip().lower()
                
                # More inclusive filtering
                if (any(keyword in item_clean for keyword in food_keywords) or 
                    (len(item_clean) > 2 and 
                     not any(non_food in item_clean for non_food in ['plate', 'bowl', 'cup', 'glass', 'fork', 'knife', 'spoon', 'table', 'napkin']) and
                     any(food_indicator in item_clean for food_indicator in ['fried', 'grilled', 'baked', 'roasted', 'steamed', 'boiled', 'fresh', 'cooked', 'raw', 'sliced', 'diced', 'chopped']))):
                    final_items.append(item_clean)
            
            if final_items:
                # Remove duplicates and sort
                unique_items = sorted(set(final_items))
                result = ', '.join(unique_items)
                logger.info(f"Final detected food items ({len(unique_items)}): {result}")
                return result
        
        # Enhanced fallback
        try:
            fallback_prompts = [
                "What food is in this image?",
                "Describe this meal:",
                "What can you see in this food photo?",
                "What ingredients are visible?"
            ]
            
            for prompt in fallback_prompts:
                inputs = models['processor'](image, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = models['blip_model'].generate(**inputs, max_new_tokens=150, num_beams=6)
                caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                
                if caption.startswith(prompt):
                    caption = caption.replace(prompt, "").strip()
                
                if len(caption.split()) >= 3:
                    return caption
                    
        except Exception as e:
            logger.warning(f"Fallback failed: {e}")
        
        return "Multiple food items detected but specific identification is challenging. Please add context description."
            
    except Exception as e:
        logger.error(f"Food detection error: {e}")
        return "Detection failed. Please try again or describe manually."

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

# Extract nutritional data
def extract_items_and_nutrients(text):
    """Extract food items and nutritional data from analysis text."""
    items = []
    
    try:
        patterns = [
            r'Item:\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?',
            r'-\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                item = match[0].strip()
                calories = int(match[1]) if match[1] else 0
                protein = float(match[2]) if len(match) > 2 and match[2] else 0
                carbs = float(match[3]) if len(match) > 3 and match[3] else 0
                fats = float(match[4]) if len(match) > 4 and match[4] else 0
                
                if not any(existing_item["item"].lower() == item.lower() for existing_item in items):
                    items.append({
                        "item": item,
                        "calories": calories,
                        "protein": protein,
                        "carbs": carbs,
                        "fats": fats
                    })
        
        # Calculate totals
        totals = {
            "calories": sum(item["calories"] for item in items),
            "protein": sum(item["protein"] for item in items if item["protein"]),
            "carbs": sum(item["carbs"] for item in items if item["carbs"]),
            "fats": sum(item["fats"] for item in items if item["fats"])
        }
        
        # If no items extracted, try to extract calories from text
        if not items and len(text.strip()) > 10:
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

# Analyze food
def analyze_food_with_enhanced_prompt(image_description: str, context: str = "") -> Dict[str, Any]:
    """Direct, effective food analysis."""
    try:
        prompt = f"""Analyze this food and provide detailed nutritional information:

FOODS DETECTED: {image_description}
CONTEXT: {context if context else "Standard meal"}

Please provide a comprehensive analysis in this format:

FOOD ITEMS IDENTIFIED:
- [Item 1]: [portion size], Calories: [X], Protein: [X]g, Carbs: [X]g, Fats: [X]g
- [Item 2]: [portion size], Calories: [X], Protein: [X]g, Carbs: [X]g, Fats: [X]g
[Continue for all items]

MEAL TOTALS:
Total Calories: [X]
Total Protein: [X]g
Total Carbohydrates: [X]g
Total Fats: [X]g

ASSESSMENT:
[Brief nutritional assessment and meal type]

RECOMMENDATIONS:
[2-3 health suggestions]

IMPORTANT: 
- Identify every food item mentioned, even small components
- Provide realistic portion estimates
- Include cooking oils, sauces, and seasonings in calorie counts
- Be thorough and accurate with nutritional values"""

        response = models['llm'].invoke(prompt)
        analysis = response.content
        
        items, totals = extract_items_and_nutrients(analysis)
        
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
        
        if not food_items and image_description:
            estimated_calories = estimate_calories_from_description(image_description)
            food_items = [{
                "item": "Complete meal",
                "description": f"Meal with: {image_description}",
                "calories": estimated_calories,
                "protein": estimated_calories * 0.15 / 4,
                "carbs": estimated_calories * 0.50 / 4,
                "fats": estimated_calories * 0.35 / 9
            }]
            totals = {
                "calories": estimated_calories,
                "protein": estimated_calories * 0.15 / 4,
                "carbs": estimated_calories * 0.50 / 4,
                "fats": estimated_calories * 0.35 / 9
            }
        
        return {
            "success": True,
            "analysis": analysis,
            "food_items": food_items,
            "nutritional_data": {
                "total_calories": totals.get("calories", 0),
                "total_protein": totals.get("protein", 0),
                "total_carbs": totals.get("carbs", 0),
                "total_fats": totals.get("fats", 0),
                "items": food_items
            },
            "improved_description": image_description
        }
        
    except Exception as e:
        logger.error(f"Food analysis error: {e}")
        estimated_calories = estimate_calories_from_description(image_description)
        return {
            "success": True,
            "analysis": f"Basic analysis: {image_description}\nEstimated calories: {estimated_calories}",
            "food_items": [{
                "item": "Detected meal",
                "description": image_description,
                "calories": estimated_calories,
                "protein": 0,
                "carbs": 0,
                "fats": 0
            }],
            "nutritional_data": {
                "total_calories": estimated_calories,
                "total_protein": 0,
                "total_carbs": 0,
                "total_fats": 0,
                "items": []
            },
            "improved_description": image_description
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
        with st.spinner("Analyzing your food..."):
            try:
                image = Image.open(img_file)
                
                # Display image
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Get food description
                food_description = describe_image_enhanced(image)
                
                # Analyze the food
                analysis_result = analyze_food_with_enhanced_prompt(food_description, context)
                
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
                    
                    # Show detailed analysis
                    with st.expander("📊 Detailed Analysis"):
                        st.write(analysis_result["analysis"])
                    
                    # Add visualizations
                    with st.expander("🔬 AI Visualizations"):
                        st.write("**AI Model Interpretability Visualizations**")
                        
                        if models['cnn_model']:
                            try:
                                # Prepare image for CNN
                                image_rgb = image.convert("RGB")
                                image_tensor = cnn_transform(image_rgb).unsqueeze(0).to(device)
                                
                                # Generate visualizations
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Edge Detection**")
                                    edge_path = visualize_food_features(image)
                                    if edge_path:
                                        edge_img = Image.open(edge_path)
                                        st.image(edge_img, caption="Edge Detection - Food Boundaries", use_container_width=True)
                                        os.remove(edge_path)
                                    
                                    st.write("**SHAP Analysis**")
                                    shap_path = apply_shap(image_tensor, models['cnn_model'])
                                    if shap_path:
                                        shap_img = Image.open(shap_path)
                                        st.image(shap_img, caption="SHAP - Feature Importance", use_container_width=True)
                                        os.remove(shap_path)
                                
                                with col2:
                                    st.write("**Grad-CAM**")
                                    gradcam_path = apply_gradcam(image_tensor, models['cnn_model'], 0)
                                    if gradcam_path:
                                        gradcam_img = Image.open(gradcam_path)
                                        st.image(gradcam_img, caption="Grad-CAM - Model Focus Areas", use_container_width=True)
                                        os.remove(gradcam_path)
                                    
                                    st.write("**LIME Explanation**")
                                    lime_path = apply_lime(image, models['cnn_model'], ["food"])
                                    if lime_path:
                                        lime_img = Image.open(lime_path)
                                        st.image(lime_img, caption="LIME - Local Interpretability", use_container_width=True)
                                        os.remove(lime_path)
                                
                                st.info("💡 **Visualization Guide:**")
                                st.write("• **Edge Detection**: Shows food boundaries and textures")
                                st.write("• **Grad-CAM**: Highlights areas the AI focuses on")
                                st.write("• **SHAP**: Shows feature importance for predictions")
                                st.write("• **LIME**: Explains local decision-making regions")
                                
                            except Exception as e:
                                st.warning(f"Visualization generation failed: {e}")
                        else:
                            st.info("CNN model not available for visualizations.")
                    
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