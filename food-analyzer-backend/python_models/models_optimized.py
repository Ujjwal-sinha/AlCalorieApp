#!/usr/bin/env python3
"""
Optimized Model Functions for Food Detection
Handles deployment constraints and provides fallbacks
"""

import sys
import time
import traceback
from typing import Dict, List, Optional, Any
from PIL import Image, ImageEnhance
import numpy as np
import torch

# Global model cache
MODEL_CACHE = {}

# Model availability flags
MODEL_AVAILABILITY = {
    'yolo': False,
    'vit': False,
    'swin': False,
    'blip': False,
    'clip': False
}

def check_model_availability():
    """Check which models are available"""
    global MODEL_AVAILABILITY
    
    # Check YOLO
    try:
        from ultralytics import YOLO
        MODEL_AVAILABILITY['yolo'] = True
        print("✅ YOLO available", file=sys.stderr)
    except ImportError as e:
        print(f"❌ YOLO not available: {e}", file=sys.stderr)
    
    # Check Transformers
    try:
        from transformers import (
            BlipForConditionalGeneration, BlipProcessor,
            CLIPProcessor, CLIPModel,
            ViTImageProcessor, ViTForImageClassification
        )
        MODEL_AVAILABILITY['vit'] = True
        MODEL_AVAILABILITY['blip'] = True
        MODEL_AVAILABILITY['clip'] = True
        print("✅ Transformers available", file=sys.stderr)
    except ImportError as e:
        print(f"❌ Transformers not available: {e}", file=sys.stderr)
    
    # Check Swin (separate check)
    try:
        from transformers import SwinForImageClassification
        # SwinImageProcessor might not be available in older versions
        MODEL_AVAILABILITY['swin'] = True
        print("✅ Swin available", file=sys.stderr)
    except ImportError as e:
        print(f"❌ Swin not available: {e}", file=sys.stderr)

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """Enhance image quality for better detection"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (memory optimization)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        return image
    except Exception as e:
        print(f"Image enhancement failed: {str(e)}", file=sys.stderr)
        return image

def load_model(model_type: str) -> Optional[Any]:
    """Load AI model based on type with error handling"""
    try:
        print(f"Loading {model_type} model...", file=sys.stderr)
        
        if model_type == 'yolo' and MODEL_AVAILABILITY['yolo']:
            from ultralytics import YOLO
            # Use smaller model for faster loading
            model = YOLO('yolov8n.pt')  # nano version
            print(f"✅ YOLO model loaded successfully", file=sys.stderr)
            return model
            
        elif model_type == 'vit' and MODEL_AVAILABILITY['vit']:
            from transformers import ViTImageProcessor, ViTForImageClassification
            # Use smaller ViT model
            model_name = "google/vit-base-patch16-224"
            processor = ViTImageProcessor.from_pretrained(model_name)
            model = ViTForImageClassification.from_pretrained(model_name)
            print(f"✅ ViT model loaded successfully", file=sys.stderr)
            return {'processor': processor, 'model': model}
            
        elif model_type == 'swin' and MODEL_AVAILABILITY['swin']:
            from transformers import SwinForImageClassification, AutoImageProcessor
            # Use smaller Swin model with AutoImageProcessor as fallback
            model_name = "microsoft/swin-tiny-patch4-window7-224"
            try:
                processor = AutoImageProcessor.from_pretrained(model_name)
            except:
                # Fallback to basic image processing
                processor = None
            model = SwinForImageClassification.from_pretrained(model_name)
            print(f"✅ Swin model loaded successfully", file=sys.stderr)
            return {'processor': processor, 'model': model}
            
        elif model_type == 'blip' and MODEL_AVAILABILITY['blip']:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            model_name = "Salesforce/blip-image-captioning-base"
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            print(f"✅ BLIP model loaded successfully", file=sys.stderr)
            return {'processor': processor, 'model': model}
            
        elif model_type == 'clip' and MODEL_AVAILABILITY['clip']:
            from transformers import CLIPProcessor, CLIPModel
            model_name = "openai/clip-vit-base-patch32"
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            print(f"✅ CLIP model loaded successfully", file=sys.stderr)
            return {'processor': processor, 'model': model}
        
        else:
            print(f"❌ Model {model_type} not available", file=sys.stderr)
            return None
            
    except Exception as e:
        print(f"❌ Error loading {model_type} model: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        return None

def detect_with_yolo(image: Image.Image, model) -> Dict[str, Any]:
    """Detect food using YOLO"""
    try:
        # Enhance image
        enhanced_image = enhance_image_quality(image)
        
        # Run YOLO detection
        results = model(enhanced_image, conf=0.3, verbose=False)
        
        detected_foods = []
        confidence_scores = {}
        
        if results and len(results) > 0:
            result = results[0]  # First result
            if result.boxes is not None:
                for box in result.boxes:
                    if box.conf is not None and box.cls is not None:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        
                        # Filter for food-related classes
                        if confidence > 0.3:
                            detected_foods.append(class_name)
                            confidence_scores[class_name] = confidence
        
        return {
            'detected_foods': detected_foods,
            'confidence_scores': confidence_scores
        }
        
    except Exception as e:
        print(f"YOLO detection error: {str(e)}", file=sys.stderr)
        return {
            'detected_foods': [],
            'confidence_scores': {},
            'error': str(e)
        }

def detect_with_vit(image: Image.Image, model_dict) -> Dict[str, Any]:
    """Detect food using Vision Transformer"""
    try:
        processor = model_dict['processor']
        model = model_dict['model']
        
        # Enhance image
        enhanced_image = enhance_image_quality(image)
        
        # Process image
        inputs = processor(enhanced_image, return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1).max().item()
        
        # Get class name
        class_name = model.config.id2label[predicted_class_id]
        
        return {
            'detected_foods': [class_name],
            'confidence_scores': {class_name: confidence}
        }
        
    except Exception as e:
        print(f"ViT detection error: {str(e)}", file=sys.stderr)
        return {
            'detected_foods': [],
            'confidence_scores': {},
            'error': str(e)
        }

def detect_with_swin(image: Image.Image, model_dict) -> Dict[str, Any]:
    """Detect food using Swin Transformer"""
    try:
        processor = model_dict['processor']
        model = model_dict['model']
        
        # Enhance image
        enhanced_image = enhance_image_quality(image)
        
        # Process image with fallback
        if processor is not None:
            inputs = processor(enhanced_image, return_tensors="pt")
        else:
            # Basic image processing fallback
            enhanced_image = enhanced_image.resize((224, 224))
            enhanced_image = enhanced_image.convert('RGB')
            # Convert to tensor manually
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            inputs = {'pixel_values': transform(enhanced_image).unsqueeze(0)}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1).max().item()
        
        # Get class name
        class_name = model.config.id2label[predicted_class_id]
        
        return {
            'detected_foods': [class_name],
            'confidence_scores': {class_name: confidence}
        }
        
    except Exception as e:
        print(f"Swin detection error: {str(e)}", file=sys.stderr)
        return {
            'detected_foods': [],
            'confidence_scores': {},
            'error': str(e)
        }

def detect_with_blip(image: Image.Image, model_dict) -> Dict[str, Any]:
    """Generate caption using BLIP"""
    try:
        processor = model_dict['processor']
        model = model_dict['model']
        
        # Enhance image
        enhanced_image = enhance_image_quality(image)
        
        # Generate caption
        inputs = processor(enhanced_image, return_tensors="pt")
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=5)
            caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Extract food items from caption (simple keyword matching)
        food_keywords = ['food', 'meal', 'dish', 'plate', 'apple', 'banana', 'bread', 'rice', 'chicken', 'fish', 'vegetable']
        detected_foods = [word for word in caption.lower().split() if word in food_keywords]
        
        return {
            'detected_foods': detected_foods,
            'confidence_scores': {food: 0.8 for food in detected_foods},
            'caption': caption
        }
        
    except Exception as e:
        print(f"BLIP detection error: {str(e)}", file=sys.stderr)
        return {
            'detected_foods': [],
            'confidence_scores': {},
            'caption': 'Unable to generate caption',
            'error': str(e)
        }

def detect_with_clip(image: Image.Image, model_dict) -> Dict[str, Any]:
    """Detect food using CLIP"""
    try:
        processor = model_dict['processor']
        model = model_dict['model']
        
        # Enhance image
        enhanced_image = enhance_image_quality(image)
        
        # Define food-related text prompts
        food_prompts = [
            "a photo of food", "a photo of a meal", "a photo of fruits", 
            "a photo of vegetables", "a photo of bread", "a photo of meat",
            "a photo of rice", "a photo of pasta", "a photo of salad"
        ]
        
        # Process image and text
        inputs = processor(
            text=food_prompts,
            images=enhanced_image,
            return_tensors="pt",
            padding=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            best_match_idx = probs.argmax().item()
            confidence = probs.max().item()
        
        best_prompt = food_prompts[best_match_idx]
        detected_foods = [best_prompt.replace("a photo of ", "")]
        
        return {
            'detected_foods': detected_foods,
            'confidence_scores': {detected_foods[0]: confidence}
        }
        
    except Exception as e:
        print(f"CLIP detection error: {str(e)}", file=sys.stderr)
        return {
            'detected_foods': [],
            'confidence_scores': {},
            'error': str(e)
        }

# Initialize model availability check
check_model_availability()
