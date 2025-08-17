#!/usr/bin/env python3
"""
Minimal Model Functions for Food Detection
Only YOLO and ViT models to fit within 512MB RAM
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

# Model availability flags (only YOLO and ViT)
MODEL_AVAILABILITY = {
    'yolo': False,
    'vit': False,
    'swin': False,  # Disabled
    'blip': False,  # Disabled
    'clip': False   # Disabled
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
    
    # Check Transformers (only ViT)
    try:
        from transformers import ViTImageProcessor, ViTForImageClassification
        MODEL_AVAILABILITY['vit'] = True
        print("✅ ViT available", file=sys.stderr)
    except ImportError as e:
        print(f"❌ ViT not available: {e}", file=sys.stderr)
    
    # Other models disabled for memory optimization
    print("ℹ️ Swin, BLIP, CLIP disabled for memory optimization", file=sys.stderr)

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """Enhance image quality for better detection"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (memory optimization)
        max_size = 512  # Reduced for memory
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
        
        else:
            print(f"❌ Model {model_type} not available or disabled", file=sys.stderr)
            return None
            
    except Exception as e:
        print(f"❌ Error loading {model_type} model: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        return None

def detect_with_yolo(image: Image.Image, model) -> Dict[str, Any]:
    """Detect food using YOLO"""
    try:
        enhanced_image = enhance_image_quality(image)
        results = model(enhanced_image, conf=0.3, verbose=False)
        
        detected_foods = []
        confidence_scores = {}
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    if box.conf is not None and box.cls is not None:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        
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
        
        enhanced_image = enhance_image_quality(image)
        inputs = processor(enhanced_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1).max().item()
        
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
    """Swin detection disabled for memory optimization"""
    return {
        'detected_foods': [],
        'confidence_scores': {},
        'error': 'Swin model disabled for memory optimization'
    }

def detect_with_blip(image: Image.Image, model_dict) -> Dict[str, Any]:
    """BLIP detection disabled for memory optimization"""
    return {
        'detected_foods': [],
        'confidence_scores': {},
        'caption': 'BLIP model disabled for memory optimization',
        'error': 'BLIP model not available in minimal version'
    }

def detect_with_clip(image: Image.Image, model_dict) -> Dict[str, Any]:
    """CLIP detection disabled for memory optimization"""
    return {
        'detected_foods': [],
        'confidence_scores': {},
        'error': 'CLIP model disabled for memory optimization'
    }

# Initialize model availability check
check_model_availability()
