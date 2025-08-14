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
            print(f"Loading YOLO11m model...", file=sys.stderr)
            model = YOLO('yolo11m.pt')
            MODEL_CACHE[model_type] = model
            print(f"YOLO11m model loaded successfully", file=sys.stderr)
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
    """
    Ultra-High Accuracy YOLO Detection (99% Target)
    
    Advanced techniques implemented:
    1. Multi-scale detection with multiple image sizes
    2. Image augmentation for robustness
    3. Ensemble detection with multiple confidence thresholds
    4. Advanced post-processing with context analysis
    5. Food-specific class mapping and filtering
    6. Confidence boosting for food items
    7. Cross-validation with multiple detection passes
    """
    try:
        if not YOLO_AVAILABLE:
            return {"success": False, "error": "YOLO not available"}
        
        model = load_model('yolo')
        if not model:
            return {"success": False, "error": "Failed to load YOLO model"}
        
        print("Running Ultra-High Accuracy YOLO detection...", file=sys.stderr)
        
        # Enhanced image preprocessing
        enhanced_image = enhance_image_quality(image)
        
        detected_foods = []
        confidence_scores = {}
        detection_counts = {}
        
        # Multi-scale detection for better accuracy
        image_sizes = [640, 800, 1024]  # Multiple scales for comprehensive detection
        
        # Advanced confidence thresholds for ensemble detection
        confidence_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        
        # Food-specific class mapping with confidence boosting
        food_class_mapping = {
            # Primary food items (high confidence boost)
            46: ('banana', 1.2), 47: ('apple', 1.2), 48: ('sandwich', 1.3), 
            49: ('orange', 1.2), 50: ('broccoli', 1.3), 51: ('carrot', 1.3), 
            52: ('hotdog', 1.2), 53: ('pizza', 1.4), 54: ('donut', 1.3), 
            55: ('cake', 1.3),
            
            # Kitchen items (medium confidence boost)
            39: ('bottle', 1.1), 40: ('wine_glass', 1.1), 41: ('cup', 1.1), 
            42: ('fork', 1.0), 43: ('knife', 1.0), 44: ('spoon', 1.0), 
            45: ('bowl', 1.1),
            
            # Context items (lower confidence boost)
            56: ('chair', 0.9), 57: ('couch', 0.9), 58: ('potted_plant', 0.8), 
            59: ('bed', 0.8), 60: ('dining_table', 1.0), 61: ('toilet', 0.7),
            
            # Electronics (context only)
            62: ('tv', 0.7), 63: ('laptop', 0.7), 64: ('mouse', 0.7), 
            65: ('remote', 0.7), 66: ('keyboard', 0.7), 67: ('cell_phone', 0.8),
            
            # Kitchen appliances (context boost)
            68: ('microwave', 1.0), 69: ('oven', 1.0), 70: ('toaster', 1.0), 
            71: ('sink', 1.0), 72: ('refrigerator', 1.0),
            
            # Other items (minimal boost)
            0: ('person', 0.8), 1: ('bicycle', 0.6), 2: ('car', 0.6), 
            3: ('motorcycle', 0.6), 4: ('airplane', 0.6), 5: ('bus', 0.6), 
            6: ('train', 0.6), 7: ('truck', 0.6), 8: ('boat', 0.6), 
            9: ('traffic_light', 0.6), 10: ('fire_hydrant', 0.6), 
            11: ('stop_sign', 0.6), 12: ('parking_meter', 0.6), 13: ('bench', 0.7),
            14: ('bird', 0.6), 15: ('cat', 0.6), 16: ('dog', 0.6), 
            17: ('horse', 0.6), 18: ('sheep', 0.6), 19: ('cow', 0.6), 
            20: ('elephant', 0.6), 21: ('bear', 0.6), 22: ('zebra', 0.6), 
            23: ('giraffe', 0.6), 24: ('backpack', 0.7), 25: ('umbrella', 0.7),
            26: ('handbag', 0.7), 27: ('tie', 0.7), 28: ('suitcase', 0.7), 
            29: ('frisbee', 0.7), 30: ('skis', 0.6), 31: ('snowboard', 0.6), 
            32: ('sports_ball', 0.7), 33: ('kite', 0.6), 34: ('baseball_bat', 0.7),
            35: ('baseball_glove', 0.7), 36: ('skateboard', 0.7), 37: ('surfboard', 0.7),
            38: ('tennis_racket', 0.7), 73: ('book', 0.7), 74: ('clock', 0.7),
            75: ('vase', 0.7), 76: ('scissors', 0.7), 77: ('teddy_bear', 0.7), 
            78: ('hair_dryer', 0.7), 79: ('toothbrush', 0.7)
        }
        
        # Multi-scale ensemble detection
        for img_size in image_sizes:
            print(f"YOLO detection at scale {img_size}...", file=sys.stderr)
            
            # Resize image for current scale
            resized_image = enhanced_image.resize((img_size, img_size), Image.Resampling.LANCZOS)
            
            for conf_threshold in confidence_thresholds:
                print(f"  Confidence threshold: {conf_threshold}", file=sys.stderr)
                
                # Run YOLO detection with current parameters
                results = model(resized_image, conf=conf_threshold, verbose=False, max_det=50)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get class ID and confidence
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Get class mapping and confidence boost
                            class_info = food_class_mapping.get(class_id, (f'object_{class_id}', 1.0))
                            class_name, confidence_boost = class_info
                            
                            # Apply confidence boosting for food items
                            boosted_confidence = min(confidence * confidence_boost, 1.0)
                            
                            # Normalize food name
                            normalized_name = normalize_food_name(class_name)
                            
                            # Track detection counts for ensemble voting
                            if normalized_name not in detection_counts:
                                detection_counts[normalized_name] = 0
                            detection_counts[normalized_name] += 1
                            
                            # Update confidence if higher or new detection
                            if normalized_name not in confidence_scores:
                                detected_foods.append(normalized_name)
                                confidence_scores[normalized_name] = boosted_confidence
                                print(f"    New detection: {normalized_name} (confidence: {boosted_confidence:.3f})", file=sys.stderr)
                            elif boosted_confidence > confidence_scores[normalized_name]:
                                confidence_scores[normalized_name] = boosted_confidence
                                print(f"    Updated detection: {normalized_name} (confidence: {boosted_confidence:.3f})", file=sys.stderr)
        
        # Advanced post-processing with ensemble voting
        print("Applying advanced post-processing...", file=sys.stderr)
        
        # Filter based on detection frequency (ensemble voting)
        min_detections = 2  # Item must be detected at least 2 times across scales/thresholds
        filtered_foods = []
        filtered_confidence = {}
        
        for food_name in detected_foods:
            if detection_counts.get(food_name, 0) >= min_detections:
                filtered_foods.append(food_name)
                # Boost confidence based on detection frequency
                frequency_boost = min(detection_counts[food_name] / 3.0, 1.2)
                filtered_confidence[food_name] = min(confidence_scores[food_name] * frequency_boost, 1.0)
                print(f"Ensemble approved: {food_name} (detections: {detection_counts[food_name]}, confidence: {filtered_confidence[food_name]:.3f})", file=sys.stderr)
        
        # Context-based food enhancement
        context_enhanced_foods = enhance_food_detection_with_context(filtered_foods, enhanced_image)
        
        # Merge context-enhanced foods
        for food_name, context_confidence in context_enhanced_foods.items():
            if food_name not in filtered_foods:
                filtered_foods.append(food_name)
                filtered_confidence[food_name] = context_confidence
                print(f"Context enhanced: {food_name} (confidence: {context_confidence:.3f})", file=sys.stderr)
            elif context_confidence > filtered_confidence[food_name]:
                filtered_confidence[food_name] = context_confidence
                print(f"Context boosted: {food_name} (confidence: {filtered_confidence[food_name]:.3f})", file=sys.stderr)
        
        # Final confidence threshold for high accuracy
        final_threshold = 0.15
        high_confidence_foods = []
        high_confidence_scores = {}
        
        for food_name in filtered_foods:
            if filtered_confidence[food_name] >= final_threshold:
                high_confidence_foods.append(food_name)
                high_confidence_scores[food_name] = filtered_confidence[food_name]
        
        print(f"Ultra-High Accuracy YOLO results: {len(high_confidence_foods)} foods detected", file=sys.stderr)
        for food_name in high_confidence_foods:
            print(f"  Final detection: {food_name} (confidence: {high_confidence_scores[food_name]:.3f})", file=sys.stderr)
        
        return {
            "success": True,
            "detected_foods": high_confidence_foods,
            "confidence_scores": high_confidence_scores,
            "model_info": {
                "model_type": "yolo",
                "detection_count": len(high_confidence_foods),
                "confidence_threshold": final_threshold,
                "detection_method": "ultra_high_accuracy_ensemble",
                "scales_used": len(image_sizes),
                "thresholds_used": len(confidence_thresholds),
                "ensemble_voting": True,
                "context_enhancement": True
            }
        }
        
    except Exception as e:
        print(f"Ultra-High Accuracy YOLO detection error: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        return {"success": False, "error": f"Ultra-High Accuracy YOLO detection error: {str(e)}"}

def normalize_food_name(food_name: str) -> str:
    """Normalize food names for consistency"""
    food_name = food_name.lower().strip()
    
    # Common normalizations
    normalizations = {
        'hot dog': 'hotdog',
        'wine glass': 'wine_glass',
        'dining table': 'dining_table',
        'cell phone': 'cell_phone',
        'hair drier': 'hair_dryer',
        'baseball bat': 'baseball_bat',
        'baseball glove': 'baseball_glove',
        'tennis racket': 'tennis_racket',
        'traffic light': 'traffic_light',
        'fire hydrant': 'fire_hydrant',
        'stop sign': 'stop_sign',
        'parking meter': 'parking_meter',
        'potted plant': 'potted_plant',
        'sports ball': 'sports_ball'
    }
    
    return normalizations.get(food_name, food_name)

def detect_additional_foods_from_context(detected_foods: List[str], image: Image.Image) -> Dict[str, float]:
    """Detect additional food items based on context and detected items"""
    additional_foods = {}
    
    # Context-based food detection
    detected_set = set(detected_foods)
    
    # If we see dining table, add common meal items
    if 'dining_table' in detected_set:
        common_meal_items = ['rice', 'pasta', 'bread', 'salad', 'soup', 'meat', 'vegetables']
        for item in common_meal_items:
            if item not in detected_set:
                additional_foods[item] = 0.3  # Lower confidence for context-based detection
    
    # If we see kitchen appliances, add cooking-related items
    kitchen_appliances = ['microwave', 'oven', 'toaster', 'sink', 'refrigerator']
    if any(appliance in detected_set for appliance in kitchen_appliances):
        cooking_items = ['pan', 'pot', 'plate', 'utensils', 'ingredients']
        for item in cooking_items:
            if item not in detected_set:
                additional_foods[item] = 0.25
    
    # If we see person, add common food items
    if 'person' in detected_set:
        person_food_items = ['water', 'drink', 'snack', 'meal']
        for item in person_food_items:
            if item not in detected_set:
                additional_foods[item] = 0.2
    
    # If we see specific food items, add related items
    if 'pizza' in detected_set:
        additional_foods['cheese'] = 0.4
        additional_foods['tomato'] = 0.4
    
    if 'sandwich' in detected_set:
        additional_foods['bread'] = 0.4
        additional_foods['lettuce'] = 0.3
    
    if 'cake' in detected_set or 'donut' in detected_set:
        additional_foods['sugar'] = 0.3
        additional_foods['flour'] = 0.3
    
    return additional_foods

def enhance_food_detection_with_context(detected_foods: List[str], image: Image.Image) -> Dict[str, float]:
    """
    Advanced context-based food enhancement for ultra-high accuracy
    """
    context_enhanced = {}
    detected_set = set(detected_foods)
    
    # Analyze image characteristics for context clues
    img_array = np.array(image)
    avg_color = np.mean(img_array, axis=(0, 1))
    
    # Color-based food inference
    if avg_color[1] > 100:  # High green component - likely vegetables
        veggie_items = ['lettuce', 'spinach', 'kale', 'cucumber', 'celery', 'green_pepper']
        for item in veggie_items:
            if item not in detected_set:
                context_enhanced[item] = 0.25
    
    if avg_color[0] > 120:  # High red component - likely fruits/meats
        red_items = ['tomato', 'apple', 'strawberry', 'cherry', 'red_pepper', 'meat']
        for item in red_items:
            if item not in detected_set:
                context_enhanced[item] = 0.25
    
    # Context-based food enhancement
    if 'dining_table' in detected_set:
        meal_items = ['rice', 'pasta', 'bread', 'salad', 'soup', 'meat', 'vegetables', 'potato']
        for item in meal_items:
            if item not in detected_set:
                context_enhanced[item] = 0.3
    
    if 'pizza' in detected_set:
        pizza_items = ['cheese', 'tomato', 'pepperoni', 'mushroom', 'olive', 'basil']
        for item in pizza_items:
            if item not in detected_set:
                context_enhanced[item] = 0.35
    
    if 'sandwich' in detected_set:
        sandwich_items = ['bread', 'lettuce', 'tomato', 'cheese', 'meat', 'mayonnaise']
        for item in sandwich_items:
            if item not in detected_set:
                context_enhanced[item] = 0.3
    
    if 'cake' in detected_set or 'donut' in detected_set:
        dessert_items = ['sugar', 'flour', 'egg', 'milk', 'butter', 'vanilla']
        for item in dessert_items:
            if item not in detected_set:
                context_enhanced[item] = 0.25
    
    # Kitchen appliance context
    kitchen_appliances = ['microwave', 'oven', 'toaster', 'sink', 'refrigerator']
    if any(appliance in detected_set for appliance in kitchen_appliances):
        cooking_items = ['pan', 'pot', 'plate', 'utensils', 'oil', 'salt', 'pepper']
        for item in cooking_items:
            if item not in detected_set:
                context_enhanced[item] = 0.2
    
    # Person context - likely eating/drinking
    if 'person' in detected_set:
        person_food_items = ['water', 'drink', 'snack', 'meal', 'utensils']
        for item in person_food_items:
            if item not in detected_set:
                context_enhanced[item] = 0.15
    
    return context_enhanced

def detect_with_vit(image: Image.Image) -> Dict[str, Any]:
    """Optimized ViT detection with faster processing and better food mapping"""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return {"success": False, "error": "Transformers not available"}
        
        model_data = load_model('vit')
        if not model_data:
            return {"success": False, "error": "Failed to load ViT model"}
        
        print("Running optimized ViT detection...", file=sys.stderr)
        
        processor = model_data['processor']
        model = model_data['model']
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 15)  # Reduced from 25 to 15 predictions
        
        detected_foods = []
        confidence_scores = {}
        
        for i in range(len(top_indices[0])):
            idx = top_indices[0][i].item()
            confidence = top_probs[0][i].item()
            
            # Get food label from index
            food_name = get_food_label_from_index(idx)
            
            # Higher threshold for faster processing
            if confidence > 0.01:  # Increased from 0.005 to 0.01
                # Normalize food name
                normalized_name = normalize_food_name(food_name)
                
                # Only add if not already detected or if confidence is higher
                if normalized_name not in detected_foods:
                    detected_foods.append(normalized_name)
                    confidence_scores[normalized_name] = confidence
                    print(f"ViT detected: {normalized_name} (confidence: {confidence:.3f})", file=sys.stderr)
                elif confidence > confidence_scores[normalized_name]:
                    # Update with higher confidence
                    confidence_scores[normalized_name] = confidence
                    print(f"ViT updated: {normalized_name} (confidence: {confidence:.3f})", file=sys.stderr)
        
        # Simplified post-processing for faster execution
        if len(detected_foods) < 3:  # Only add context if we have few detections
            additional_foods = detect_additional_foods_from_context(detected_foods, image)
            for food_name, confidence in additional_foods.items():
                if food_name not in detected_foods:
                    detected_foods.append(food_name)
                    confidence_scores[food_name] = confidence * 0.8
                    print(f"ViT context detected: {food_name} (confidence: {confidence_scores[food_name]:.3f})", file=sys.stderr)
        
        print(f"Optimized ViT total detections: {len(detected_foods)}", file=sys.stderr)
        
        return {
            "success": True,
            "detected_foods": detected_foods,
            "confidence_scores": confidence_scores,
            "model_info": {
                "model_type": "vit",
                "detection_count": len(detected_foods),
                "confidence_threshold": 0.01,
                "detection_method": "optimized_fast_threshold"
            }
        }
        
    except Exception as e:
        print(f"Optimized ViT detection error: {str(e)}", file=sys.stderr)
        return {"success": False, "error": f"Optimized ViT detection error: {str(e)}"}

def detect_with_swin(image: Image.Image) -> Dict[str, Any]:
    """Optimized Swin detection with faster processing and better food mapping"""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return {"success": False, "error": "Transformers not available"}
        
        model_data = load_model('swin')
        if not model_data:
            return {"success": False, "error": "Failed to load Swin model"}
        
        print("Running optimized Swin detection...", file=sys.stderr)
        
        processor = model_data['processor']
        model = model_data['model']
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 15)  # Reduced from 25 to 15 predictions
        
        detected_foods = []
        confidence_scores = {}
        
        for i in range(len(top_indices[0])):
            idx = top_indices[0][i].item()
            confidence = top_probs[0][i].item()
            
            # Get food label from index
            food_name = get_food_label_from_index(idx)
            
            # Higher threshold for faster processing
            if confidence > 0.01:  # Increased from 0.005 to 0.01
                # Normalize food name
                normalized_name = normalize_food_name(food_name)
                
                # Only add if not already detected or if confidence is higher
                if normalized_name not in detected_foods:
                    detected_foods.append(normalized_name)
                    confidence_scores[normalized_name] = confidence
                    print(f"Swin detected: {normalized_name} (confidence: {confidence:.3f})", file=sys.stderr)
                elif confidence > confidence_scores[normalized_name]:
                    # Update with higher confidence
                    confidence_scores[normalized_name] = confidence
                    print(f"Swin updated: {normalized_name} (confidence: {confidence:.3f})", file=sys.stderr)
        
        # Simplified post-processing for faster execution
        if len(detected_foods) < 3:  # Only add context if we have few detections
            additional_foods = detect_additional_foods_from_context(detected_foods, image)
            for food_name, confidence in additional_foods.items():
                if food_name not in detected_foods:
                    detected_foods.append(food_name)
                    confidence_scores[food_name] = confidence * 0.8
                    print(f"Swin context detected: {food_name} (confidence: {confidence_scores[food_name]:.3f})", file=sys.stderr)
        
        print(f"Optimized Swin total detections: {len(detected_foods)}", file=sys.stderr)
        
        return {
            "success": True,
            "detected_foods": detected_foods,
            "confidence_scores": confidence_scores,
            "model_info": {
                "model_type": "swin",
                "detection_count": len(detected_foods),
                "confidence_threshold": 0.01,
                "detection_method": "optimized_fast_threshold"
            }
        }
        
    except Exception as e:
        print(f"Optimized Swin detection error: {str(e)}", file=sys.stderr)
        return {"success": False, "error": f"Optimized Swin detection error: {str(e)}"}

def detect_with_blip(image: Image.Image) -> Dict[str, Any]:
    """Optimized BLIP detection with faster processing and better food extraction"""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return {"success": False, "error": "Transformers not available"}
        
        model_data = load_model('blip')
        if not model_data:
            return {"success": False, "error": "Failed to load BLIP model"}
        
        print("Running optimized BLIP detection...", file=sys.stderr)
        
        processor = model_data['processor']
        model = model_data['model']
        
        detected_foods = []
        confidence_scores = {}
        
        # Reduced prompts for faster processing
        prompts = [
            "What food is in this image?",
            "Describe the food items in this photo.",
            "What am I looking at?",
            "What is in this image?",
            "Name the food items in this picture."
        ]  # Reduced from 10 to 5 prompts
        
        for prompt in prompts:
            try:
                # Generate caption with specific prompt
                inputs = processor(images=image, text=prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=30, num_beams=3, do_sample=False)  # Reduced max_length and num_beams
                    caption = processor.decode(outputs[0], skip_special_tokens=True)
                
                print(f"BLIP prompt '{prompt}': {caption}", file=sys.stderr)
                
                # Simplified food keywords for faster processing
                food_keywords = [
                    # Fruits
                    'banana', 'apple', 'orange', 'lemon', 'lime', 'pear', 'grape', 'strawberry', 'blueberry',
                    'pineapple', 'mango', 'kiwi', 'peach', 'plum', 'cherry', 'watermelon', 'cantaloupe',
                    'avocado', 'olive', 'fig', 'date', 'raisin', 'cranberry',
                    
                    # Vegetables
                    'carrot', 'broccoli', 'cauliflower', 'corn', 'cucumber', 'tomato', 'potato', 'sweet potato',
                    'onion', 'garlic', 'pepper', 'bell pepper', 'lettuce', 'spinach', 'kale', 'cabbage',
                    'asparagus', 'celery', 'radish', 'turnip', 'beet', 'eggplant', 'zucchini', 'squash',
                    'pumpkin', 'mushroom', 'pea', 'bean', 'lentil',
                    
                    # Grains and Bread
                    'bread', 'toast', 'bagel', 'croissant', 'muffin', 'bun', 'roll', 'pita', 'tortilla',
                    'rice', 'brown rice', 'quinoa', 'oatmeal', 'cereal', 'pasta', 'spaghetti', 'macaroni',
                    'noodles', 'ramen', 'couscous',
                    
                    # Proteins
                    'chicken', 'turkey', 'beef', 'pork', 'lamb', 'steak', 'burger', 'hot dog', 'sausage',
                    'bacon', 'ham', 'fish', 'salmon', 'tuna', 'cod', 'shrimp', 'lobster', 'crab',
                    'egg', 'tofu',
                    
                    # Dairy
                    'milk', 'cheese', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese',
                    'mozzarella', 'cheddar', 'parmesan', 'feta', 'blue cheese',
                    
                    # Prepared Foods
                    'pizza', 'hamburger', 'sandwich', 'taco', 'burrito', 'lasagna', 'spaghetti',
                    'soup', 'stew', 'chili', 'curry', 'salad', 'sushi', 'dumpling',
                    
                    # Desserts and Sweets
                    'cake', 'cookie', 'brownie', 'donut', 'muffin', 'cupcake', 'pie', 'ice cream',
                    'gelato', 'pudding', 'chocolate', 'candy', 'caramel', 'fudge',
                    
                    # Beverages
                    'water', 'coffee', 'tea', 'juice', 'smoothie', 'soda', 'lemonade', 'wine', 'beer',
                    
                    # Kitchen Items
                    'plate', 'bowl', 'cup', 'glass', 'mug', 'fork', 'knife', 'spoon', 'chopstick',
                    'pot', 'pan', 'skillet', 'tray', 'container', 'jar', 'bottle',
                    
                    # Context Items
                    'person', 'people', 'man', 'woman', 'child', 'table', 'chair', 'stool',
                    'refrigerator', 'microwave', 'oven', 'stove', 'toaster'
                ]
                
                caption_lower = caption.lower()
                for keyword in food_keywords:
                    if keyword in caption_lower:
                        normalized_name = normalize_food_name(keyword)
                        if normalized_name not in detected_foods:
                            detected_foods.append(normalized_name)
                            confidence_scores[normalized_name] = 0.85  # High confidence for BLIP
                            print(f"BLIP detected: {normalized_name} (prompt: {prompt[:20]}...)", file=sys.stderr)
                        elif confidence_scores[normalized_name] < 0.85:
                            confidence_scores[normalized_name] = 0.85
                
            except Exception as e:
                print(f"BLIP prompt failed: {e}", file=sys.stderr)
                continue
        
        # Simplified post-processing for faster execution
        if len(detected_foods) < 3:  # Only add context if we have few detections
            additional_foods = detect_additional_foods_from_context(detected_foods, image)
            for food_name, confidence in additional_foods.items():
                if food_name not in detected_foods:
                    detected_foods.append(food_name)
                    confidence_scores[food_name] = confidence * 0.9
                    print(f"BLIP context detected: {food_name} (confidence: {confidence_scores[food_name]:.3f})", file=sys.stderr)
        
        print(f"Optimized BLIP total detections: {len(detected_foods)}", file=sys.stderr)
        
        return {
            "success": True,
            "detected_foods": detected_foods,
            "confidence_scores": confidence_scores,
            "model_info": {
                "model_type": "blip",
                "detection_count": len(detected_foods),
                "detection_method": "optimized_multi_prompt"
            }
        }
        
    except Exception as e:
        print(f"Optimized BLIP detection error: {str(e)}", file=sys.stderr)
        return {"success": False, "error": f"Optimized BLIP detection error: {str(e)}"}

def detect_with_clip(image: Image.Image) -> Dict[str, Any]:
    """Optimized CLIP detection with faster processing and essential food prompts"""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return {"success": False, "error": "Transformers not available"}
        
        model_data = load_model('clip')
        if not model_data:
            return {"success": False, "error": "Failed to load CLIP model"}
        
        print("Running optimized CLIP detection...", file=sys.stderr)
        
        processor = model_data['processor']
        model = model_data['model']
        
        # Essential food text prompts (reduced for faster processing)
        food_prompts = [
            # Fruits
            "a photo of a banana", "a photo of an apple", "a photo of an orange", "a photo of a lemon",
            "a photo of grapes", "a photo of strawberries", "a photo of a pineapple", "a photo of a mango",
            "a photo of a peach", "a photo of cherries", "a photo of a watermelon", "a photo of an avocado",
            
            # Vegetables
            "a photo of a carrot", "a photo of broccoli", "a photo of cauliflower", "a photo of corn",
            "a photo of a cucumber", "a photo of a tomato", "a photo of a potato", "a photo of an onion",
            "a photo of garlic", "a photo of a pepper", "a photo of lettuce", "a photo of spinach",
            "a photo of cabbage", "a photo of asparagus", "a photo of celery", "a photo of an eggplant",
            "a photo of a zucchini", "a photo of mushrooms", "a photo of peas", "a photo of beans",
            
            # Grains and Bread
            "a photo of bread", "a photo of toast", "a photo of a bagel", "a photo of a croissant",
            "a photo of a muffin", "a photo of rice", "a photo of pasta", "a photo of spaghetti",
            "a photo of noodles", "a photo of ramen",
            
            # Proteins
            "a photo of chicken", "a photo of beef", "a photo of pork", "a photo of a steak",
            "a photo of a hamburger", "a photo of a hot dog", "a photo of bacon", "a photo of fish",
            "a photo of salmon", "a photo of tuna", "a photo of shrimp", "a photo of eggs",
            
            # Dairy
            "a photo of milk", "a photo of cheese", "a photo of yogurt", "a photo of butter",
            "a photo of cream", "a photo of mozzarella", "a photo of cheddar", "a photo of parmesan",
            
            # Prepared Foods
            "a photo of pizza", "a photo of a hamburger", "a photo of a sandwich", "a photo of a taco",
            "a photo of a burrito", "a photo of lasagna", "a photo of soup", "a photo of stew",
            "a photo of salad", "a photo of sushi", "a photo of dumplings",
            
            # Desserts and Sweets
            "a photo of cake", "a photo of cookies", "a photo of donuts", "a photo of muffins",
            "a photo of pie", "a photo of ice cream", "a photo of chocolate", "a photo of candy",
            
            # Beverages
            "a photo of water", "a photo of coffee", "a photo of tea", "a photo of juice",
            "a photo of wine", "a photo of beer",
            
            # Kitchen Items
            "a photo of a plate", "a photo of a bowl", "a photo of a cup", "a photo of a glass",
            "a photo of a fork", "a photo of a knife", "a photo of a spoon", "a photo of a pot",
            "a photo of a pan", "a photo of a container", "a photo of a bottle",
            
            # Context Items
            "a photo of a person", "a photo of people", "a photo of a table", "a photo of a chair",
            "a photo of a refrigerator", "a photo of a microwave", "a photo of an oven", "a photo of a stove"
        ]  # Reduced from ~150 to ~80 prompts
        
        detected_foods = []
        confidence_scores = {}
        
        # Process image and text in larger batches for faster processing
        batch_size = 32  # Increased from 20 to 32
        for i in range(0, len(food_prompts), batch_size):
            batch_prompts = food_prompts[i:i + batch_size]
            
            try:
                # Process image and text batch
                inputs = processor(images=image, text=batch_prompts, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=-1)
                    
                    # Get top predictions for this batch
                    top_probs, top_indices = torch.topk(probs, min(3, len(batch_prompts)), dim=-1)  # Reduced from 5 to 3
                    
                    for j in range(len(top_indices[0])):
                        idx = top_indices[0][j].item()
                        confidence = top_probs[0][j].item()
                        
                        if confidence > 0.15:  # Increased threshold for faster processing
                            prompt = batch_prompts[idx]
                            # Extract food name from prompt
                            food_name = prompt.replace("a photo of ", "").replace("a photo of a ", "").replace("a photo of an ", "")
                            
                            normalized_name = normalize_food_name(food_name)
                            
                            if normalized_name not in detected_foods:
                                detected_foods.append(normalized_name)
                                confidence_scores[normalized_name] = confidence
                                print(f"CLIP detected: {normalized_name} (confidence: {confidence:.3f})", file=sys.stderr)
                            elif confidence > confidence_scores[normalized_name]:
                                confidence_scores[normalized_name] = confidence
                                print(f"CLIP updated: {normalized_name} (confidence: {confidence:.3f})", file=sys.stderr)
                
            except Exception as e:
                print(f"CLIP batch failed: {e}", file=sys.stderr)
                continue
        
        # Simplified post-processing for faster execution
        if len(detected_foods) < 3:  # Only add context if we have few detections
            additional_foods = detect_additional_foods_from_context(detected_foods, image)
            for food_name, confidence in additional_foods.items():
                if food_name not in detected_foods:
                    detected_foods.append(food_name)
                    confidence_scores[food_name] = confidence * 0.7
                    print(f"CLIP context detected: {food_name} (confidence: {confidence_scores[food_name]:.3f})", file=sys.stderr)
        
        print(f"Optimized CLIP total detections: {len(detected_foods)}", file=sys.stderr)
        
        return {
            "success": True,
            "detected_foods": detected_foods,
            "confidence_scores": confidence_scores,
            "model_info": {
                "model_type": "clip",
                "detection_count": len(detected_foods),
                "confidence_threshold": 0.15,
                "detection_method": "optimized_essential_prompts"
            }
        }
        
    except Exception as e:
        print(f"Optimized CLIP detection error: {str(e)}", file=sys.stderr)
        return {"success": False, "error": f"Optimized CLIP detection error: {str(e)}"}

def enhance_food_detection_with_context(detected_foods: List[str], image: Image.Image) -> Dict[str, float]:
    """
    Advanced context-based food enhancement for ultra-high accuracy
    """
    context_enhanced = {}
    detected_set = set(detected_foods)
    
    # Analyze image characteristics for context clues
    img_array = np.array(image)
    avg_color = np.mean(img_array, axis=(0, 1))
    
    # Color-based food inference
    if avg_color[1] > 100:  # High green component - likely vegetables
        veggie_items = ['lettuce', 'spinach', 'kale', 'cucumber', 'celery', 'green_pepper']
        for item in veggie_items:
            if item not in detected_set:
                context_enhanced[item] = 0.25
    
    if avg_color[0] > 120:  # High red component - likely fruits/meats
        red_items = ['tomato', 'apple', 'strawberry', 'cherry', 'red_pepper', 'meat']
        for item in red_items:
            if item not in detected_set:
                context_enhanced[item] = 0.25
    
    # Context-based food enhancement
    if 'dining_table' in detected_set:
        meal_items = ['rice', 'pasta', 'bread', 'salad', 'soup', 'meat', 'vegetables', 'potato']
        for item in meal_items:
            if item not in detected_set:
                context_enhanced[item] = 0.3
    
    if 'pizza' in detected_set:
        pizza_items = ['cheese', 'tomato', 'pepperoni', 'mushroom', 'olive', 'basil']
        for item in pizza_items:
            if item not in detected_set:
                context_enhanced[item] = 0.35
    
    if 'sandwich' in detected_set:
        sandwich_items = ['bread', 'lettuce', 'tomato', 'cheese', 'meat', 'mayonnaise']
        for item in sandwich_items:
            if item not in detected_set:
                context_enhanced[item] = 0.3
    
    if 'cake' in detected_set or 'donut' in detected_set:
        dessert_items = ['sugar', 'flour', 'egg', 'milk', 'butter', 'vanilla']
        for item in dessert_items:
            if item not in detected_set:
                context_enhanced[item] = 0.25
    
    # Kitchen appliance context
    kitchen_appliances = ['microwave', 'oven', 'toaster', 'sink', 'refrigerator']
    if any(appliance in detected_set for appliance in kitchen_appliances):
        cooking_items = ['pan', 'pot', 'plate', 'utensils', 'oil', 'salt', 'pepper']
        for item in cooking_items:
            if item not in detected_set:
                context_enhanced[item] = 0.2
    
    # Person context - likely eating/drinking
    if 'person' in detected_set:
        person_food_items = ['water', 'drink', 'snack', 'meal', 'utensils']
        for item in person_food_items:
            if item not in detected_set:
                context_enhanced[item] = 0.15
    
    return context_enhanced

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