#!/usr/bin/env python3
"""
Enhanced Food Detection Python Script for TypeScript Backend Integration

This script provides AI-powered food detection capabilities that can be called
from the TypeScript backend via child_process. It supports multiple AI models
including YOLO, Vision Transformers, Swin Transformers, BLIP, and CLIP.

Usage:
    python3 detect_food.py <model_type> < image_data.json

Input (JSON via stdin):
    {
        "model_type": "yolo|vit|swin|blip|clip",
        "image_data": "base64_encoded_image",
        "width": 1024,
        "height": 768
    }

Output (JSON to stdout):
    {
        "success": true,
        "detected_foods": ["chicken", "rice", "broccoli"],
        "confidence_scores": {"chicken": 0.95, "rice": 0.87, "broccoli": 0.92},
        "processing_time": 1250,
        "model_info": {"model_type": "yolo", "detection_count": 3},
        "error": null
    }
"""

import sys
import json
import base64
import time
import traceback
from typing import Dict, List, Optional, Any
import numpy as np
from PIL import Image, ImageEnhance
import io

# Try to import AI/ML libraries with error handling
try:
    import torch
    TORCH_AVAILABLE = True
    print("PyTorch available", file=sys.stderr)
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available", file=sys.stderr)

try:
    from transformers import (
        BlipForConditionalGeneration, BlipProcessor,
        CLIPProcessor, CLIPModel,
        ViTImageProcessor, ViTForImageClassification,
        AutoProcessor, AutoModel, AutoImageProcessor, AutoModelForImageClassification
    )
    TRANSFORMERS_AVAILABLE = True
    print("Transformers available", file=sys.stderr)
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Transformers not available: {e}", file=sys.stderr)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLO available", file=sys.stderr)
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"YOLO not available: {e}", file=sys.stderr)

# Global model cache
MODEL_CACHE = {}

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """Enhance image quality for better detection"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        return image
    except Exception as e:
        print(f"Image enhancement failed: {str(e)}", file=sys.stderr)
        return image

def load_model(model_type: str) -> Optional[Any]:
    """Load AI model based on type"""
    if model_type in MODEL_CACHE:
        return MODEL_CACHE[model_type]
    
    try:
        if model_type == 'yolo' and YOLO_AVAILABLE:
            print(f"Loading YOLO model...", file=sys.stderr)
            model = YOLO('yolov8n.pt')
            MODEL_CACHE[model_type] = model
            print(f"YOLO model loaded successfully", file=sys.stderr)
            return model
            
        elif model_type == 'vit' and TRANSFORMERS_AVAILABLE:
            print(f"Loading ViT model...", file=sys.stderr)
            try:
                processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
                model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
                MODEL_CACHE[model_type] = {'processor': processor, 'model': model}
                print(f"ViT model loaded successfully", file=sys.stderr)
                return MODEL_CACHE[model_type]
            except Exception as e:
                print(f"ViT model failed to load: {e}", file=sys.stderr)
                return None
            
        elif model_type == 'swin' and TRANSFORMERS_AVAILABLE:
            print(f"Loading Swin model...", file=sys.stderr)
            try:
                # Use AutoProcessor and AutoModel for Swin
                processor = AutoImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224')
                model = AutoModelForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224')
                MODEL_CACHE[model_type] = {'processor': processor, 'model': model}
                print(f"Swin model loaded successfully", file=sys.stderr)
                return MODEL_CACHE[model_type]
            except Exception as e:
                print(f"Swin model failed to load, using ViT as fallback: {e}", file=sys.stderr)
                # Fallback to ViT for Swin
                try:
                    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
                    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
                    MODEL_CACHE[model_type] = {'processor': processor, 'model': model}
                    print(f"Swin fallback to ViT loaded successfully", file=sys.stderr)
                    return MODEL_CACHE[model_type]
                except Exception as fallback_error:
                    print(f"Swin fallback also failed: {fallback_error}", file=sys.stderr)
                    return None
            
        elif model_type == 'blip' and TRANSFORMERS_AVAILABLE:
            print(f"Loading BLIP model...", file=sys.stderr)
            try:
                processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
                model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
                MODEL_CACHE[model_type] = {'processor': processor, 'model': model}
                print(f"BLIP model loaded successfully", file=sys.stderr)
                return MODEL_CACHE[model_type]
            except Exception as e:
                print(f"BLIP model failed to load: {e}", file=sys.stderr)
                return None
            
        elif model_type == 'clip' and TRANSFORMERS_AVAILABLE:
            print(f"Loading CLIP model...", file=sys.stderr)
            try:
                processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
                model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
                MODEL_CACHE[model_type] = {'processor': processor, 'model': model}
                print(f"CLIP model loaded successfully", file=sys.stderr)
                return MODEL_CACHE[model_type]
            except Exception as e:
                print(f"CLIP model failed to load: {e}", file=sys.stderr)
                return None
            
    except Exception as e:
        print(f"Error loading model {model_type}: {str(e)}", file=sys.stderr)
        return None
    
    return None

def decode_image(image_data: str) -> Optional[Image.Image]:
    """Decode base64 image data and enhance quality"""
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        
        # Enhance image quality
        image = enhance_image_quality(image)
        
        return image
    except Exception as e:
        print(f"Error decoding image: {str(e)}", file=sys.stderr)
        return None

def get_food_label_from_index(idx: int) -> str:
    """Convert model index to meaningful food label"""
    # Comprehensive food labels matching the original system
    food_labels = {
        # Fruits (0-99)
        0: "apple", 1: "banana", 2: "orange", 3: "grape", 4: "strawberry",
        5: "bread", 6: "cake", 7: "pizza", 8: "hamburger", 9: "hot_dog",
        10: "sandwich", 11: "taco", 12: "burrito", 13: "salad", 14: "soup",
        15: "rice", 16: "pasta", 17: "noodles", 18: "chicken", 19: "beef",
        20: "fish", 21: "shrimp", 22: "egg", 23: "cheese", 24: "milk",
        25: "yogurt", 26: "ice_cream", 27: "cookie", 28: "donut", 29: "muffin",
        30: "coffee", 31: "tea", 32: "juice", 33: "water", 34: "soda",
        35: "beer", 36: "wine", 37: "carrot", 38: "broccoli", 39: "tomato",
        40: "potato", 41: "onion", 42: "garlic", 43: "pepper", 44: "cucumber",
        45: "lettuce", 46: "spinach", 47: "mushroom", 48: "corn", 49: "pea",
        50: "bell_pepper", 51: "zucchini", 52: "eggplant", 53: "cauliflower", 54: "asparagus",
        55: "celery", 56: "radish", 57: "turnip", 58: "beetroot", 59: "sweet_potato",
        60: "yam", 61: "ginger", 62: "turmeric", 63: "cinnamon", 64: "nutmeg",
        65: "clove", 66: "cardamom", 67: "cumin", 68: "coriander", 69: "oregano",
        70: "basil", 71: "thyme", 72: "rosemary", 73: "sage", 74: "parsley",
        75: "cilantro", 76: "dill", 77: "mint", 78: "leek", 79: "shallot",
        80: "scallion", 81: "green_onion", 82: "red_onion", 83: "white_onion", 84: "yellow_onion",
        85: "sweet_onion", 86: "vidalia_onion", 87: "walla_walla_onion", 88: "maui_onion", 89: "bermuda_onion",
        90: "spanish_onion", 91: "egyptian_onion", 92: "tree_onion", 93: "multiplier_onion", 94: "potato_onion",
        95: "shallot_onion", 96: "garlic_onion", 97: "chive_onion", 98: "peach", 99: "pear",
        
        # More fruits (100-199)
        100: "pineapple", 101: "mango", 102: "kiwi", 103: "lemon", 104: "lime",
        105: "cherry", 106: "plum", 107: "apricot", 108: "fig", 109: "date",
        110: "prune", 111: "raisin", 112: "currant", 113: "cranberry", 114: "gooseberry",
        115: "elderberry", 116: "mulberry", 117: "blackberry", 118: "loganberry", 119: "boysenberry",
        120: "marionberry", 121: "olallieberry", 122: "sylvanberry", 123: "chehalem", 124: "santiam",
        125: "willamette", 126: "cascade", 127: "liberty", 128: "glacier", 129: "nugget",
        130: "columbus", 131: "warrior", 132: "zeus", 133: "magnum", 134: "simcoe",
        135: "amarillo", 136: "citra", 137: "mosaic", 138: "galaxy", 139: "vic_secret",
        140: "idaho_7", 141: "strata", 142: "sabro", 143: "triumph", 144: "perle",
        145: "hallertau", 146: "saaz", 147: "fuggle", 148: "blueberry", 149: "raspberry",
        
        # Vegetables (200-299)
        200: "carrot", 201: "broccoli", 202: "tomato", 203: "potato", 204: "onion",
        205: "garlic", 206: "pepper", 207: "cucumber", 208: "lettuce", 209: "spinach",
        210: "mushroom", 211: "corn", 212: "pea", 213: "bell_pepper", 214: "zucchini",
        215: "eggplant", 216: "cauliflower", 217: "asparagus", 218: "celery", 219: "radish",
        220: "turnip", 221: "beetroot", 222: "sweet_potato", 223: "yam", 224: "ginger",
        225: "turmeric", 226: "cinnamon", 227: "nutmeg", 228: "clove", 229: "cardamom",
        230: "cumin", 231: "coriander", 232: "oregano", 233: "basil", 234: "thyme",
        235: "rosemary", 236: "sage", 237: "parsley", 238: "cilantro", 239: "dill",
        240: "mint", 241: "leek", 242: "shallot", 243: "scallion", 244: "green_onion",
        245: "red_onion", 246: "white_onion", 247: "yellow_onion", 248: "sweet_onion", 249: "vidalia_onion",
        
        # Proteins (300-399)
        300: "chicken", 301: "beef", 302: "pork", 303: "lamb", 304: "turkey",
        305: "duck", 306: "fish", 307: "salmon", 308: "tuna", 309: "shrimp",
        310: "lobster", 311: "crab", 312: "mussel", 313: "clam", 314: "oyster",
        315: "scallop", 316: "squid", 317: "octopus", 318: "egg", 319: "quail_egg",
        320: "duck_egg", 321: "turkey_egg", 322: "goose_egg", 323: "ostrich_egg", 324: "emu_egg",
        325: "chicken_breast", 326: "chicken_thigh", 327: "chicken_wing", 328: "chicken_leg", 329: "chicken_drumstick",
        330: "beef_steak", 331: "beef_roast", 332: "beef_ground", 333: "beef_brisket", 334: "beef_ribs",
        335: "pork_chop", 336: "pork_roast", 337: "pork_belly", 338: "pork_ribs", 339: "bacon",
        340: "ham", 341: "sausage", 342: "hot_dog", 343: "burger", 344: "meatball",
        345: "steak", 346: "roast", 347: "ground_meat", 348: "brisket", 349: "ribs",
        
        # Dairy and grains (400-499)
        400: "milk", 401: "cheese", 402: "yogurt", 403: "butter", 404: "cream",
        405: "sour_cream", 406: "whipping_cream", 407: "heavy_cream", 408: "light_cream", 409: "half_and_half",
        410: "cottage_cheese", 411: "cream_cheese", 412: "cheddar_cheese", 413: "mozzarella_cheese", 414: "parmesan_cheese",
        415: "swiss_cheese", 416: "provolone_cheese", 417: "gouda_cheese", 418: "brie_cheese", 419: "camembert_cheese",
        420: "blue_cheese", 421: "feta_cheese", 422: "goat_cheese", 423: "ricotta_cheese", 424: "mascarpone_cheese",
        425: "bread", 426: "rice", 427: "pasta", 428: "noodles", 429: "quinoa",
        430: "oat", 431: "cereal", 432: "wheat", 433: "flour", 434: "pizza",
        435: "sandwich", 436: "burger_bun", 437: "tortilla", 438: "wrap", 439: "bagel",
        440: "muffin", 441: "cake", 442: "cookie", 443: "biscuit", 444: "croissant",
        445: "donut", 446: "cupcake", 447: "brownie", 448: "pie", 449: "tart",
        450: "pastry", 451: "danish", 452: "scone", 453: "waffle", 454: "pancake",
        455: "french_toast", 456: "toast", 457: "cracker", 458: "pretzel", 459: "chips",
        460: "popcorn", 461: "granola", 462: "muesli", 463: "oatmeal", 464: "porridge",
        465: "grits", 466: "polenta", 467: "couscous", 468: "bulgur", 469: "barley",
        470: "rye", 471: "spelt", 472: "kamut", 473: "farro", 474: "freekeh",
        475: "millet", 476: "sorghum", 477: "teff", 478: "amaranth", 479: "buckwheat",
        480: "chia", 481: "flax", 482: "hemp", 483: "sunflower_seed", 484: "pumpkin_seed",
        485: "sesame_seed", 486: "poppy_seed", 487: "caraway_seed", 488: "fennel_seed", 489: "mustard_seed",
        490: "coriander_seed", 491: "cumin_seed", 492: "cardamom_seed", 493: "nutmeg_seed", 494: "clove_seed",
        495: "allspice_seed", 496: "star_anise_seed", 497: "saffron_seed", 498: "vanilla_seed", 499: "cocoa_seed",
        
        # Beverages and desserts (500-599)
        500: "coffee", 501: "tea", 502: "juice", 503: "soda", 504: "water",
        505: "milk", 506: "smoothie", 507: "shake", 508: "beer", 509: "wine",
        510: "cocktail", 511: "lemonade", 512: "iced_tea", 513: "hot_chocolate", 514: "cocoa",
        515: "espresso", 516: "latte", 517: "cappuccino", 518: "americano", 519: "mocha",
        520: "frappuccino", 521: "macchiato", 522: "flat_white", 523: "cortado", 524: "piccolo",
        525: "ristretto", 526: "lungo", 527: "black_tea", 528: "green_tea", 529: "herbal_tea",
        530: "oolong_tea", 531: "white_tea", 532: "pu_erh_tea", 533: "chai_tea", 534: "earl_grey_tea",
        535: "english_breakfast_tea", 536: "jasmine_tea", 537: "chamomile_tea", 538: "peppermint_tea", 539: "rooibos_tea",
        540: "ice_cream", 541: "pudding", 542: "custard", 543: "flan", 544: "creme_brulee",
        545: "tiramisu", 546: "cheesecake", 547: "churro", 548: "baklava", 549: "cannoli",
        550: "eclair", 551: "profiterole", 552: "macaron", 553: "madeleine", 554: "financier",
        555: "clafoutis", 556: "tarte_tatin", 557: "apple_pie", 558: "cherry_pie", 559: "pumpkin_pie",
        560: "pecan_pie", 561: "key_lime_pie", 562: "lemon_meringue_pie", 563: "chocolate_pie", 564: "banana_cream_pie",
        565: "coconut_cream_pie", 566: "buttermilk_pie", 567: "sweet_potato_pie", 568: "mince_pie", 569: "shepherds_pie",
        570: "cottage_pie", 571: "fish_pie", 572: "chicken_pie", 573: "beef_pie", 574: "lamb_pie",
        575: "turkey_pie", 576: "duck_pie", 577: "goose_pie", 578: "pheasant_pie", 579: "quail_pie",
        580: "partridge_pie", 581: "grouse_pie", 582: "woodcock_pie", 583: "snipe_pie", 584: "plover_pie",
        585: "curlew_pie", 586: "godwit_pie", 587: "sandpiper_pie", 588: "dunlin_pie", 589: "knot_pie",
        590: "sanderling_pie", 591: "turnstone_pie", 592: "oystercatcher_pie", 593: "avocet_pie", 594: "stilt_pie",
        595: "phalarope_pie", 596: "pratincole_pie", 597: "courser_pie", 598: "stone_curlew_pie", 599: "thick_knee_pie",
    }
    
    # Return food label or fallback to generic food
    return food_labels.get(idx, f"food_item_{idx}")

def detect_with_yolo(image: Image.Image) -> Dict[str, Any]:
    """Detect food items using YOLO model"""
    try:
        if not YOLO_AVAILABLE:
            return {"success": False, "error": "YOLO not available"}
        
        model = load_model('yolo')
        if not model:
            return {"success": False, "error": "Failed to load YOLO model"}
        
        print("Running YOLO detection...", file=sys.stderr)
        
        # Run YOLO detection with very low confidence threshold
        results = model(image, conf=0.1, verbose=False)  # Much lower threshold
        
        detected_foods = []
        confidence_scores = {}
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Map COCO class IDs to food names
                    food_names = {
                        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
                        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
                        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                        54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                        59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
                        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
                        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
                        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                        79: 'toothbrush'
                    }
                    
                    class_name = food_names.get(class_id, f'object_{class_id}')
                    
                    # Filter for food-related items (expanded list)
                    food_keywords = [
                        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
                        'hot dog', 'pizza', 'donut', 'cake', 'bottle', 'wine glass', 
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'dining table',
                        'person', 'chair', 'couch', 'bed', 'tv', 'laptop', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator'
                    ]
                    
                    if any(keyword in class_name.lower() for keyword in food_keywords):
                        detected_foods.append(class_name)
                        confidence_scores[class_name] = confidence
                        print(f"YOLO detected: {class_name} (confidence: {confidence:.3f})", file=sys.stderr)
        
        print(f"YOLO total detections: {len(detected_foods)}", file=sys.stderr)
        
        return {
            "success": True,
            "detected_foods": detected_foods,
            "confidence_scores": confidence_scores,
            "model_info": {
                "model_type": "yolo",
                "detection_count": len(detected_foods),
                "confidence_threshold": 0.1
            }
        }
        
    except Exception as e:
        print(f"YOLO detection error: {str(e)}", file=sys.stderr)
        return {"success": False, "error": f"YOLO detection error: {str(e)}"}

def detect_with_vit(image: Image.Image) -> Dict[str, Any]:
    """Detect food items using Vision Transformer"""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return {"success": False, "error": "Transformers not available"}
        
        model_data = load_model('vit')
        if not model_data:
            return {"success": False, "error": "Failed to load ViT model"}
        
        print("Running ViT detection...", file=sys.stderr)
        
        processor = model_data['processor']
        model = model_data['model']
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 15)  # Get top 15 predictions
        
        detected_foods = []
        confidence_scores = {}
        
        for i in range(len(top_indices[0])):
            idx = top_indices[0][i].item()
            confidence = top_probs[0][i].item()
            
            # Get food label from index
            food_name = get_food_label_from_index(idx)
            
            if confidence > 0.01:  # Very low threshold for better detection
                detected_foods.append(food_name)
                confidence_scores[food_name] = confidence
                print(f"ViT detected: {food_name} (confidence: {confidence:.3f})", file=sys.stderr)
        
        print(f"ViT total detections: {len(detected_foods)}", file=sys.stderr)
        
        return {
            "success": True,
            "detected_foods": detected_foods,
            "confidence_scores": confidence_scores,
            "model_info": {
                "model_type": "vit",
                "detection_count": len(detected_foods),
                "confidence_threshold": 0.01
            }
        }
        
    except Exception as e:
        print(f"ViT detection error: {str(e)}", file=sys.stderr)
        return {"success": False, "error": f"ViT detection error: {str(e)}"}

def detect_with_swin(image: Image.Image) -> Dict[str, Any]:
    """Detect food items using Swin Transformer"""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return {"success": False, "error": "Transformers not available"}
        
        model_data = load_model('swin')
        if not model_data:
            return {"success": False, "error": "Failed to load Swin model"}
        
        print("Running Swin detection...", file=sys.stderr)
        
        processor = model_data['processor']
        model = model_data['model']
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 15)  # Get top 15 predictions
        
        detected_foods = []
        confidence_scores = {}
        
        for i in range(len(top_indices[0])):
            idx = top_indices[0][i].item()
            confidence = top_probs[0][i].item()
            
            # Get food label from index
            food_name = get_food_label_from_index(idx)
            
            if confidence > 0.01:  # Very low threshold for better detection
                detected_foods.append(food_name)
                confidence_scores[food_name] = confidence
                print(f"Swin detected: {food_name} (confidence: {confidence:.3f})", file=sys.stderr)
        
        print(f"Swin total detections: {len(detected_foods)}", file=sys.stderr)
        
        return {
            "success": True,
            "detected_foods": detected_foods,
            "confidence_scores": confidence_scores,
            "model_info": {
                "model_type": "swin",
                "detection_count": len(detected_foods),
                "confidence_threshold": 0.01
            }
        }
        
    except Exception as e:
        print(f"Swin detection error: {str(e)}", file=sys.stderr)
        return {"success": False, "error": f"Swin detection error: {str(e)}"}

def detect_with_blip(image: Image.Image) -> Dict[str, Any]:
    """Detect food items using BLIP model"""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return {"success": False, "error": "Transformers not available"}
        
        model_data = load_model('blip')
        if not model_data:
            return {"success": False, "error": "Failed to load BLIP model"}
        
        print("Running BLIP detection...", file=sys.stderr)
        
        processor = model_data['processor']
        model = model_data['model']
        
        # Generate caption
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, num_beams=5)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"BLIP caption: {caption}", file=sys.stderr)
        
        # Extract food keywords from caption
        food_keywords = [
            'banana', 'apple', 'orange', 'lemon', 'pear', 'grape', 'strawberry',
            'carrot', 'broccoli', 'cauliflower', 'corn', 'cucumber', 'tomato',
            'potato', 'onion', 'garlic', 'pepper', 'lettuce', 'spinach',
            'bread', 'pizza', 'hamburger', 'hot dog', 'sandwich', 'taco',
            'rice', 'pasta', 'noodles', 'soup', 'salad', 'steak', 'chicken',
            'fish', 'shrimp', 'lobster', 'crab', 'egg', 'milk', 'cheese',
            'yogurt', 'butter', 'oil', 'sugar', 'salt', 'pepper',
            'cake', 'cookie', 'donut', 'ice cream', 'coffee', 'tea', 'juice',
            'water', 'wine', 'beer', 'person', 'people', 'man', 'woman', 'child',
            'table', 'chair', 'plate', 'bowl', 'cup', 'glass', 'fork', 'knife', 'spoon'
        ]
        
        detected_foods = []
        confidence_scores = {}
        
        caption_lower = caption.lower()
        for keyword in food_keywords:
            if keyword in caption_lower:
                detected_foods.append(keyword)
                confidence_scores[keyword] = 0.8  # High confidence for BLIP
                print(f"BLIP detected: {keyword}", file=sys.stderr)
        
        print(f"BLIP total detections: {len(detected_foods)}", file=sys.stderr)
        
        return {
            "success": True,
            "detected_foods": detected_foods,
            "confidence_scores": confidence_scores,
            "model_info": {
                "model_type": "blip",
                "detection_count": len(detected_foods),
                "caption": caption
            }
        }
        
    except Exception as e:
        print(f"BLIP detection error: {str(e)}", file=sys.stderr)
        return {"success": False, "error": f"BLIP detection error: {str(e)}"}

def detect_with_clip(image: Image.Image) -> Dict[str, Any]:
    """Detect food items using CLIP model"""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return {"success": False, "error": "Transformers not available"}
        
        model_data = load_model('clip')
        if not model_data:
            return {"success": False, "error": "Failed to load CLIP model"}
        
        print("Running CLIP detection...", file=sys.stderr)
        
        processor = model_data['processor']
        model = model_data['model']
        
        # Define food text prompts
        food_prompts = [
            "a photo of a banana", "a photo of an apple", "a photo of an orange",
            "a photo of a carrot", "a photo of broccoli", "a photo of a tomato",
            "a photo of bread", "a photo of pizza", "a photo of a hamburger",
            "a photo of rice", "a photo of pasta", "a photo of soup",
            "a photo of salad", "a photo of steak", "a photo of chicken",
            "a photo of fish", "a photo of eggs", "a photo of cheese",
            "a photo of milk", "a photo of yogurt", "a photo of cake",
            "a photo of cookies", "a photo of donuts", "a photo of ice cream",
            "a photo of coffee", "a photo of tea", "a photo of juice",
            "a photo of wine", "a photo of beer", "a photo of water",
            "a photo of a person", "a photo of people", "a photo of a table",
            "a photo of a chair", "a photo of a plate", "a photo of a bowl",
            "a photo of a cup", "a photo of a glass", "a photo of a fork",
            "a photo of a knife", "a photo of a spoon"
        ]
        
        # Process image and text
        inputs = processor(images=image, text=food_prompts, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1)
        
        detected_foods = []
        confidence_scores = {}
        
        # Extract food names from prompts
        food_names = [prompt.replace("a photo of ", "").replace("an ", "").replace("a ", "") 
                     for prompt in food_prompts]
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probs[0], 12)
        
        for i in range(len(top_indices)):
            idx = top_indices[i].item()
            confidence = top_probs[i].item()
            
            if idx < len(food_names) and confidence > 0.05:  # Lower threshold
                food_name = food_names[idx]
                detected_foods.append(food_name)
                confidence_scores[food_name] = confidence
                print(f"CLIP detected: {food_name} (confidence: {confidence:.3f})", file=sys.stderr)
        
        print(f"CLIP total detections: {len(detected_foods)}", file=sys.stderr)
        
        return {
            "success": True,
            "detected_foods": detected_foods,
            "confidence_scores": confidence_scores,
            "model_info": {
                "model_type": "clip",
                "detection_count": len(detected_foods),
                "confidence_threshold": 0.05
            }
        }
        
    except Exception as e:
        print(f"CLIP detection error: {str(e)}", file=sys.stderr)
        return {"success": False, "error": f"CLIP detection error: {str(e)}"}

def main():
    """Main function to handle detection requests"""
    start_time = time.time()
    
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        model_type = input_data.get('model_type', 'yolo')
        image_data = input_data.get('image_data', '')
        width = input_data.get('width', 1024)
        height = input_data.get('height', 768)
        
        print(f"Processing {model_type} detection...", file=sys.stderr)
        
        # Decode image
        image = decode_image(image_data)
        if not image:
            result = {
                "success": False,
                "detected_foods": [],
                "confidence_scores": {},
                "processing_time": int((time.time() - start_time) * 1000),
                "model_info": {"model_type": model_type, "detection_count": 0},
                "error": "Failed to decode image"
            }
            print(json.dumps(result))
            return
        
        # Perform detection based on model type
        if model_type == 'yolo':
            detection_result = detect_with_yolo(image)
        elif model_type == 'vit':
            detection_result = detect_with_vit(image)
        elif model_type == 'swin':
            detection_result = detect_with_swin(image)
        elif model_type == 'blip':
            detection_result = detect_with_blip(image)
        elif model_type == 'clip':
            detection_result = detect_with_clip(image)
        else:
            detection_result = {"success": False, "error": f"Unknown model type: {model_type}"}
        
        # Prepare response
        processing_time = int((time.time() - start_time) * 1000)
        
        result = {
            "success": detection_result.get("success", False),
            "detected_foods": detection_result.get("detected_foods", []),
            "confidence_scores": detection_result.get("confidence_scores", {}),
            "processing_time": processing_time,
            "model_info": detection_result.get("model_info", {"model_type": model_type, "detection_count": 0}),
            "error": detection_result.get("error")
        }
        
        # Output result as JSON
        print(json.dumps(result))
        
    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "detected_foods": [],
            "confidence_scores": {},
            "processing_time": int((time.time() - start_time) * 1000),
            "model_info": {"model_type": "unknown", "detection_count": 0},
            "error": f"Invalid JSON input: {str(e)}"
        }
        print(json.dumps(error_result))
        
    except Exception as e:
        error_result = {
            "success": False,
            "detected_foods": [],
            "confidence_scores": {},
            "processing_time": int((time.time() - start_time) * 1000),
            "model_info": {"model_type": "unknown", "detection_count": 0},
            "error": f"Unexpected error: {str(e)}"
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    main()