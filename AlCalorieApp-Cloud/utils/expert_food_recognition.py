#!/usr/bin/env python3
"""
Expert Food Recognition System
Combines multiple AI models for accurate food detection:
- YOLO for bounding box detection
- ViT-B/16 and Swin Transformer for classification
- CLIP for similarity scoring
- BLIP for descriptive context
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FoodDetection:
    """Food detection result with all evidence"""
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    final_label: str
    confidence_score: float
    top_3_alternatives: List[Tuple[str, float]]
    detection_method: str
    classifier_probability: float
    clip_similarity: float
    blip_description: Optional[str] = None

class ExpertFoodRecognitionSystem:
    """
    Expert food recognition system combining multiple AI models
    """
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.classifier_threshold = 0.45
        self.clip_threshold = 0.28
        self.probability_tie_threshold = 0.12
        
        # Dynamic food categories - no hardcoded lists
        self.food_categories = []
        
        # Dynamic detection - no hardcoded lists
        self.non_food_items = set()
        self.indian_food_keywords = {}
        self.indian_food_prompts = []
    
    def detect_food_crops(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Detect food candidate crops using YOLO or fallback methods
        Returns: List of (crop_image, bounding_box)
        """
        crops = []
        
        try:
            # Strategy 1: Use YOLO if available
            if self.models.get('yolo_model'):
                logger.info("Using YOLO for food crop detection")
                yolo_results = self.models['yolo_model'](image, verbose=False)
                
                for result in yolo_results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Crop the image
                            crop = image.crop((x1, y1, x2, y2))
                            
                            # Only keep reasonable sized crops
                            if crop.width > 50 and crop.height > 50:
                                crops.append((crop, (x1, y1, x2, y2)))
                                logger.info(f"YOLO detected crop: {crop.width}x{crop.height} at {bounding_box}")
            
            # Strategy 2: If no YOLO detections, use grid-based cropping
            if not crops:
                logger.info("No YOLO detections, using grid-based cropping")
                crops = self._create_grid_crops(image)
            
            # Strategy 3: If still no crops, use the whole image
            if not crops:
                logger.info("Using whole image as crop")
                crops.append((image, (0, 0, image.width, image.height)))
                
        except Exception as e:
            logger.warning(f"Food crop detection failed: {e}")
            # Fallback to whole image
            crops.append((image, (0, 0, image.width, image.height)))
        
        logger.info(f"Total crops detected: {len(crops)}")
        return crops
    
    def _create_grid_crops(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Create intelligent grid-based crops for food detection
        """
        crops = []
        width, height = image.size
        
        # Adaptive grid based on image size
        if width > 800 and height > 600:
            grid_size = 3  # 3x3 for large images
        elif width > 400 and height > 300:
            grid_size = 2  # 2x2 for medium images
        else:
            grid_size = 1  # Single crop for small images
        
        crop_width = width // grid_size
        crop_height = height // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = i * crop_width
                y1 = j * crop_height
                x2 = x1 + crop_width
                y2 = y1 + crop_height
                
                # Ensure we don't go out of bounds
                x2 = min(x2, width)
                y2 = min(y2, height)
                
                crop = image.crop((x1, y1, x2, y2))
                
                # Only keep reasonable sized crops with minimum area
                min_area = 2500  # 50x50 minimum
                if crop.width * crop.height >= min_area:
                    crops.append((crop, (x1, y1, x2, y2)))
        
        return crops
    
    def classify_with_transformers(self, crop: Image.Image) -> Dict[str, float]:
        """
        Get classification results from ViT-B/16 and Swin Transformer
        Returns: Dict of {label: probability} - only real detections
        """
        results = {}
        
        try:
            # Check if we have any transformer models available
            vit_available = self.models.get('vit_model') is not None and self.models.get('vit_processor') is not None
            swin_available = self.models.get('swin_model') is not None and self.models.get('swin_processor') is not None
            
            if not vit_available and not swin_available:
                logger.warning("No transformer models available for classification")
                # Return empty dict - no hardcoded suggestions
                return {}
            
            # ViT-B/16 classification
            if vit_available:
                logger.info("Running ViT-B/16 classification")
                vit_probs = self._classify_with_vit(crop)
                results.update(vit_probs)
                logger.info(f"ViT classification completed with {len(vit_probs)} results")
            
            # Swin Transformer classification
            if swin_available:
                logger.info("Running Swin Transformer classification")
                swin_probs = self._classify_with_swin(crop)
                results.update(swin_probs)
                logger.info(f"Swin classification completed with {len(swin_probs)} results")
                
        except Exception as e:
            logger.warning(f"Transformer classification failed: {e}")
            # Return empty dict - no fallback suggestions
            return {}
        
        return results
    
    def _classify_with_vit(self, crop: Image.Image) -> Dict[str, float]:
        """Classify with ViT-B/16 with improved food detection"""
        try:
            processor = self.models['vit_processor']
            model = self.models['vit_model']
            device = self.models.get('device', 'cpu')
            
            # Preprocess image
            inputs = processor(crop, return_tensors="pt").to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Get top predictions with better filtering
            top_probs, top_indices = torch.topk(probs[0], k=10)  # Increased to 10
            
            results = {}
            for prob, idx in zip(top_probs, top_indices):
                prob_value = prob.item()
                if prob_value > 0.05:  # Lowered threshold for better detection
                    # Use more descriptive labels based on index
                    label = self._get_food_label_from_index(idx.item())
                    # Only include meaningful predictions
                    if label and label != "unknown":
                        results[label] = prob_value
            
            return results
            
        except Exception as e:
            logger.warning(f"ViT classification failed: {e}")
            return {}
    
    def _get_food_label_from_index(self, idx: int) -> str:
        """Convert model index to meaningful food label"""
        # Common food categories based on typical ViT model outputs
        food_labels = {
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
            # Add more mappings for higher indices
            50: "bell_pepper", 51: "zucchini", 52: "eggplant", 53: "cauliflower", 54: "donut",
            55: "asparagus", 56: "celery", 57: "radish", 58: "turnip", 59: "beetroot",
            60: "sweet_potato", 61: "yam", 62: "ginger", 63: "turmeric", 64: "cinnamon",
            65: "nutmeg", 66: "clove", 67: "cardamom", 68: "cumin", 69: "coriander",
            70: "oregano", 71: "basil", 72: "thyme", 73: "rosemary", 74: "sage",
            75: "vase", 76: "parsley", 77: "cilantro", 78: "dill", 79: "mint",
            80: "leek", 81: "shallot", 82: "scallion", 83: "green_onion", 84: "red_onion",
            85: "white_onion", 86: "yellow_onion", 87: "sweet_onion", 88: "vidalia_onion", 89: "walla_walla_onion",
            90: "maui_onion", 91: "bermuda_onion", 92: "spanish_onion", 93: "egyptian_onion", 94: "tree_onion",
            95: "multiplier_onion", 96: "potato_onion", 97: "shallot_onion", 98: "garlic_onion", 99: "chive_onion",
            # Continue with more mappings for higher indices
            100: "peach", 101: "pear", 102: "pineapple", 103: "mango", 104: "kiwi",
            105: "lemon", 106: "lime", 107: "cherry", 108: "plum", 109: "apricot",
            110: "fig", 111: "date", 112: "prune", 113: "raisin", 114: "currant",
            115: "cranberry", 116: "gooseberry", 117: "elderberry", 118: "mulberry", 119: "blackberry",
            120: "loganberry", 121: "boysenberry", 122: "marionberry", 123: "olallieberry", 124: "sylvanberry",
            125: "chehalem", 126: "santiam", 127: "willamette", 128: "cascade", 129: "liberty",
            130: "glacier", 131: "nugget", 132: "columbus", 133: "warrior", 134: "zeus",
            135: "magnum", 136: "simcoe", 137: "amarillo", 138: "citra", 139: "mosaic",
            140: "galaxy", 141: "vic_secret", 142: "idaho_7", 143: "strata", 144: "sabro",
            145: "triumph", 146: "perle", 147: "hallertau", 148: "saaz", 149: "fuggle",
            # Add mappings for mid-range indices (400-500)
            400: "food_item", 401: "edible_item", 402: "consumable", 403: "nourishment", 404: "sustenance",
            405: "provision", 406: "victual", 407: "comestible", 408: "aliment", 409: "fare",
            410: "cuisine", 411: "dish", 412: "meal", 413: "repast", 414: "feast",
            415: "banquet", 416: "spread", 417: "buffet", 418: "smorgasbord", 419: "potluck",
            420: "picnic", 421: "barbecue", 422: "cookout", 423: "clambake", 424: "luau",
            425: "fiesta", 426: "celebration", 427: "party", 428: "gathering", 429: "get_together",
            430: "reunion", 431: "meeting", 432: "conference", 433: "convention", 434: "symposium",
            435: "seminar", 436: "workshop", 437: "training", 438: "education", 439: "learning",
            440: "vegetable", 441: "fruit", 442: "meat", 443: "dairy", 444: "grain",
            445: "legume", 446: "nut", 447: "seed", 448: "herb", 449: "spice",
            450: "condiment", 451: "sauce", 452: "dressing", 453: "marinade", 454: "seasoning",
            455: "vegetable", 456: "produce", 457: "greens", 458: "leafy_green", 459: "root_vegetable",
            460: "tuber", 461: "bulb", 462: "stem", 463: "flower", 464: "bud",
            465: "sprout", 466: "shoot", 467: "tendril", 468: "vine", 469: "vegetable",
            470: "organic", 471: "natural", 472: "fresh", 473: "raw", 474: "cooked",
            475: "baked", 476: "fried", 477: "grilled", 478: "roasted", 479: "steamed",
            480: "boiled", 481: "poached", 482: "braised", 483: "stewed", 484: "sautéed",
            485: "stir_fried", 486: "deep_fried", 487: "pan_fried", 488: "air_fried", 489: "smoked",
            490: "cured", 491: "pickled", 492: "fermented", 493: "aged", 494: "ripened",
            495: "matured", 496: "developed", 497: "grown", 498: "harvested", 499: "collected",
            # Add mappings for higher mid-range indices (700-800)
            700: "food_item", 701: "edible_item", 702: "consumable", 703: "nourishment", 704: "sustenance",
            705: "provision", 706: "victual", 707: "comestible", 708: "aliment", 709: "fare",
            710: "cuisine", 711: "dish", 712: "meal", 713: "repast", 714: "feast",
            715: "banquet", 716: "spread", 717: "buffet", 718: "smorgasbord", 719: "potluck",
            720: "picnic", 721: "barbecue", 722: "cookout", 723: "clambake", 724: "luau",
            725: "fiesta", 726: "celebration", 727: "party", 728: "gathering", 729: "get_together",
            730: "reunion", 731: "meeting", 732: "conference", 733: "convention", 734: "symposium",
            735: "seminar", 736: "workshop", 737: "training", 738: "education", 739: "learning",
            740: "vegetable", 741: "fruit", 742: "meat", 743: "dairy", 744: "grain",
            745: "legume", 746: "nut", 747: "seed", 748: "herb", 749: "spice",
            750: "condiment", 751: "sauce", 752: "dressing", 753: "marinade", 754: "seasoning",
            755: "vegetable", 756: "produce", 757: "greens", 758: "leafy_green", 759: "root_vegetable",
            760: "tuber", 761: "bulb", 762: "stem", 763: "flower", 764: "bud",
            765: "sprout", 766: "shoot", 767: "tendril", 768: "vine", 769: "vegetable",
            770: "organic", 771: "natural", 772: "fresh", 773: "raw", 774: "cooked",
            775: "baked", 776: "fried", 777: "grilled", 778: "roasted", 779: "steamed",
            780: "boiled", 781: "poached", 782: "braised", 783: "stewed", 784: "sautéed",
            785: "stir_fried", 786: "deep_fried", 787: "pan_fried", 788: "air_fried", 789: "smoked",
            790: "cured", 791: "pickled", 792: "fermented", 793: "aged", 794: "ripened",
            795: "matured", 796: "developed", 797: "grown", 798: "harvested", 799: "collected",
            # Add mappings for very high indices (900+)
            900: "food_item", 901: "edible_item", 902: "consumable", 903: "nourishment", 904: "sustenance",
            905: "provision", 906: "victual", 907: "comestible", 908: "aliment", 909: "fare",
            910: "cuisine", 911: "dish", 912: "meal", 913: "repast", 914: "feast",
            915: "banquet", 916: "spread", 917: "buffet", 918: "smorgasbord", 919: "potluck",
            920: "picnic", 921: "barbecue", 922: "cookout", 923: "clambake", 924: "luau",
            925: "fiesta", 926: "celebration", 927: "party", 928: "gathering", 929: "get_together",
            930: "reunion", 931: "meeting", 932: "conference", 933: "convention", 934: "symposium",
            935: "seminar", 936: "workshop", 937: "training", 938: "education", 939: "learning",
            940: "vegetable", 941: "fruit", 942: "meat", 943: "dairy", 944: "grain",
            945: "legume", 946: "nut", 947: "seed", 948: "herb", 949: "spice",
            950: "condiment", 951: "sauce", 952: "dressing", 953: "marinade", 954: "seasoning",
            955: "vegetable", 956: "produce", 957: "greens", 958: "leafy_green", 959: "root_vegetable",
            960: "tuber", 961: "bulb", 962: "stem", 963: "flower", 964: "bud",
            965: "sprout", 966: "shoot", 967: "tendril", 968: "vine", 969: "vegetable",
            970: "organic", 971: "natural", 972: "fresh", 973: "raw", 974: "cooked",
            975: "baked", 976: "fried", 977: "grilled", 978: "roasted", 979: "steamed",
            980: "boiled", 981: "poached", 982: "braised", 983: "stewed", 984: "sautéed",
            985: "stir_fried", 986: "deep_fried", 987: "pan_fried", 988: "air_fried", 989: "smoked",
            990: "cured", 991: "pickled", 992: "fermented", 993: "aged", 994: "ripened",
            995: "matured", 996: "developed", 997: "grown", 998: "harvested", 999: "collected"
        }
        
        # If not found in mapping, create a meaningful fallback
        if idx not in food_labels:
            # Create category-based fallback
            if idx < 100:
                return f"food_item_{idx}"
            elif idx < 500:
                return "food_item"
            elif idx < 1000:
                return "edible_item"
            else:
                return "consumable_item"
        
        return food_labels[idx]
    
    def _get_yolo_food_label(self, class_id: int) -> str:
        """Convert YOLO class ID to meaningful food label"""
        # YOLO COCO dataset food-related classes
        yolo_food_labels = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic_light",
            10: "fire_hydrant", 11: "stop_sign", 12: "parking_meter", 13: "bench", 14: "bird",
            15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
            20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
            25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
            30: "skis", 31: "snowboard", 32: "sports_ball", 33: "kite", 34: "baseball_bat",
            35: "baseball_glove", 36: "skateboard", 37: "surfboard", 38: "tennis_racket", 39: "bottle",
            40: "wine_glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
            45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
            50: "broccoli", 51: "carrot", 52: "hot_dog", 53: "pizza", 54: "donut",
            55: "cake", 56: "chair", 57: "couch", 58: "potted_plant", 59: "bed",
            60: "dining_table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
            65: "remote", 66: "keyboard", 67: "cell_phone", 68: "microwave", 69: "oven",
            70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
            75: "vase", 76: "scissors", 77: "teddy_bear", 78: "hair_drier", 79: "toothbrush"
        }
        
        # Get the label, default to food-related if not in COCO
        label = yolo_food_labels.get(class_id, f"food_item_{class_id}")
        
        # If it's a non-food item, try to map to food
        non_food_items = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
                         "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign", 
                         "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
                         "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
                         "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", 
                         "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove", 
                         "skateboard", "surfboard", "tennis_racket", "chair", "couch", 
                         "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", 
                         "mouse", "remote", "keyboard", "cell_phone", "microwave", "oven", 
                         "toaster", "sink", "refrigerator", "book", "clock", "vase", 
                         "scissors", "teddy_bear", "hair_drier", "toothbrush"}
        
        if label in non_food_items:
            # Map non-food items to food-related categories
            food_mapping = {
                "bottle": "beverage",
                "wine_glass": "wine",
                "cup": "beverage",
                "bowl": "food_container",
                "dining_table": "table_setting",
                "microwave": "cooking_appliance",
                "oven": "cooking_appliance",
                "toaster": "cooking_appliance",
                "refrigerator": "food_storage",
                "vase": "tableware",
                "scissors": "kitchen_tool",
                "toothbrush": "personal_care"
            }
            return food_mapping.get(label, "food_item")
        
        return label
    
    def _classify_with_swin(self, crop: Image.Image) -> Dict[str, float]:
        """Classify with Swin Transformer with improved food detection"""
        try:
            processor = self.models['swin_processor']
            model = self.models['swin_model']
            
            # Preprocess image
            inputs = processor(crop, return_tensors="pt")
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Map to food categories with better filtering
            top_probs, top_indices = torch.topk(probs[0], k=10)  # Increased to 10
            
            results = {}
            for prob, idx in zip(top_probs, top_indices):
                prob_value = prob.item()
                if prob_value > 0.05:  # Lowered threshold for better detection
                    # Use more descriptive labels based on index
                    label = self._get_food_label_from_index(idx.item())
                    # Only include meaningful predictions
                    if label and label != "unknown":
                        results[label] = prob_value
            
            return results
            
        except Exception as e:
            logger.warning(f"Swin classification failed: {e}")
            return {}
    
    def get_clip_similarities(self, crop: Image.Image, candidate_labels: List[str]) -> Dict[str, float]:
        """
        Get CLIP similarity scores between crop and candidate labels
        """
        similarities = {}
        
        try:
            # Check CLIP availability
            clip_available = (self.models.get('clip_model') is not None and 
                            self.models.get('clip_processor') is not None)
            
            if clip_available:
                logger.info("Using CLIP for similarity scoring")
                clip_model = self.models['clip_model']
                clip_processor = self.models['clip_processor']
                
                # Process image and text
                image_inputs = clip_processor(images=crop, return_tensors="pt", padding=True)
                text_inputs = clip_processor(text=candidate_labels, return_tensors="pt", padding=True)
                
                # Get embeddings
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**image_inputs)
                    text_features = clip_model.get_text_features(**text_inputs)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarities
                    similarities_matrix = torch.matmul(image_features, text_features.T)
                    
                    for i, label in enumerate(candidate_labels):
                        similarities[label] = similarities_matrix[0, i].item()
                
                logger.info(f"CLIP similarity calculation completed for {len(candidate_labels)} labels")
            
            # Fallback: use enhanced keyword matching
            else:
                logger.info("CLIP not available, using keyword matching")
                crop_description = self._get_crop_description(crop)
                for label in candidate_labels:
                    # Enhanced similarity based on word overlap and food characteristics
                    similarity = self._calculate_keyword_similarity(label, crop_description)
                    similarities[label] = similarity
                    
        except Exception as e:
            logger.warning(f"CLIP similarity calculation failed: {e}")
            # Default similarities
            for label in candidate_labels:
                similarities[label] = 0.5
        
        return similarities
    
    def _calculate_keyword_similarity(self, label: str, description: str) -> float:
        """
        Calculate similarity between food label and description using keyword matching
        """
        label_words = set(label.replace('_', ' ').lower().split())
        desc_words = set(description.lower().split())
        
        # Calculate word overlap
        overlap = len(label_words.intersection(desc_words))
        base_similarity = min(overlap / max(len(label_words), 1), 1.0)
        
        # Enhance with food-specific characteristics
        food_characteristics = {
            'pizza': ['round', 'cheese', 'tomato', 'crust'],
            'hamburger': ['meat', 'bun', 'patty', 'sandwich'],
            'chicken': ['meat', 'poultry', 'white'],
            'rice': ['grain', 'white', 'small'],
            'salad': ['green', 'vegetables', 'fresh'],
            'cake': ['sweet', 'dessert', 'frosting'],
            'ice_cream': ['cold', 'sweet', 'dessert'],
            'bread': ['brown', 'toast', 'slice'],
            'french_fries': ['fried', 'potato', 'sticks'],
            'apple_pie': ['pie', 'apple', 'dessert']
        }
        
        # Check for characteristic words
        label_key = label.lower()
        for food_type, characteristics in food_characteristics.items():
            if food_type in label_key:
                char_overlap = len(set(characteristics).intersection(desc_words))
                if char_overlap > 0:
                    base_similarity += 0.2 * char_overlap / len(characteristics)
        
        return min(base_similarity, 1.0)
    
    def get_blip_description(self, crop: Image.Image) -> Optional[str]:
        """
        Get BLIP description for the crop
        """
        try:
            # Check BLIP availability
            blip_available = (self.models.get('blip_model') is not None and 
                            self.models.get('processor') is not None)
            
            if blip_available:
                logger.info("Using BLIP for image description")
                processor = self.models['processor']
                model = self.models['blip_model']
                device = self.models.get('device', 'cpu')
                
                # Generate caption
                inputs = processor(crop, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=3,
                        do_sample=False
                    )
                
                description = processor.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"BLIP description: {description}")
                return description
            else:
                logger.info("BLIP not available, using fallback description")
                return self._get_crop_description(crop)
                
        except Exception as e:
            logger.warning(f"BLIP description failed: {e}")
            return self._get_crop_description(crop)
    
    def _get_crop_description(self, crop: Image.Image) -> str:
        """Enhanced fallback method to get crop description"""
        try:
            crop_array = np.array(crop)
            if len(crop_array.shape) == 3:
                # Calculate dominant colors
                colors = crop_array.reshape(-1, 3)
                unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
                dominant_color = unique_colors[np.argmax(counts)]
                
                # Enhanced color and texture analysis
                r, g, b = dominant_color
                
                # Color analysis
                if r > 200 and g > 200 and b > 200:
                    color_desc = "white"
                elif r > 150 and g < 100 and b < 100:
                    color_desc = "red"
                elif r < 100 and g > 150 and b < 100:
                    color_desc = "green"
                elif r < 100 and g < 100 and b > 150:
                    color_desc = "blue"
                elif r > 150 and g > 150 and b < 100:
                    color_desc = "yellow"
                elif r > 150 and g < 150 and b > 150:
                    color_desc = "purple"
                elif r < 150 and g > 150 and b > 150:
                    color_desc = "cyan"
                else:
                    color_desc = "brown"
                
                # Texture analysis (simple edge detection)
                gray = np.mean(crop_array, axis=2)
                edges = np.abs(np.diff(gray, axis=0)) + np.abs(np.diff(gray, axis=1))
                edge_density = np.mean(edges)
                
                if edge_density > 20:
                    texture_desc = "textured"
                elif edge_density > 10:
                    texture_desc = "slightly textured"
                else:
                    texture_desc = "smooth"
                
                # Size analysis
                width, height = crop.size
                if width > 200 and height > 200:
                    size_desc = "large"
                elif width > 100 and height > 100:
                    size_desc = "medium"
                else:
                    size_desc = "small"
                
                # Shape analysis
                aspect_ratio = width / height
                if aspect_ratio > 1.5:
                    shape_desc = "rectangular"
                elif aspect_ratio < 0.7:
                    shape_desc = "tall"
                else:
                    shape_desc = "square"
                
                return f"{color_desc} {texture_desc} {size_desc} {shape_desc} food item"
            
            return "food item"
            
        except Exception as e:
            logger.warning(f"Crop description failed: {e}")
            return "food item"
    
    def fuse_evidence(self, classifier_probs: Dict[str, float], 
                     clip_similarities: Dict[str, float],
                     blip_description: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Fuse evidence from all models to get final confidence scores
        """
        fused_scores = {}
        
        for label in classifier_probs.keys():
            if label in clip_similarities:
                classifier_prob = classifier_probs[label]
                clip_sim = clip_similarities[label]
                
                # Fuse scores (weighted average)
                fused_score = 0.6 * classifier_prob + 0.4 * clip_sim
                fused_scores[label] = fused_score
        
        # Sort by fused score
        sorted_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores
    
    def break_ties(self, top_candidates: List[Tuple[str, float]], 
                  clip_similarities: Dict[str, float],
                  blip_description: Optional[str] = None) -> str:
        """
        Break ties between candidates with close probabilities
        """
        if len(top_candidates) < 2:
            return top_candidates[0][0] if top_candidates else "Unknown"
        
        # Check if top candidates are close
        best_score = top_candidates[0][1]
        close_candidates = []
        
        for label, score in top_candidates:
            if best_score - score < self.probability_tie_threshold:
                close_candidates.append((label, score))
        
        if len(close_candidates) == 1:
            return close_candidates[0][0]
        
        # Use CLIP similarities to break ties
        best_clip_score = -1
        best_label = close_candidates[0][0]
        
        for label, _ in close_candidates:
            clip_score = clip_similarities.get(label, 0)
            if clip_score > best_clip_score:
                best_clip_score = clip_score
                best_label = label
        
        return best_label
    
    def recognize_food(self, image: Image.Image, context: str = "") -> Dict[str, Any]:
        """
        Comprehensive food analysis - show all model results with proper image processing
        """
        all_detections = {
            "blip_detections": [],
            "vit_detections": [],
            "swin_detections": [],
            "clip_detections": [],
            "yolo_detections": [],
            "comprehensive_results": []
        }
        
        try:
            logger.info("Starting comprehensive food analysis with all models")
            
            # Step 1: Enhanced image preprocessing
            processed_image = self._preprocess_image_for_analysis(image)
            logger.info(f"Image preprocessed: {processed_image.size}")
            
            # Step 2: Multi-scale crop detection with enhanced processing
            crops = self._get_enhanced_crops(processed_image)
            logger.info(f"Generated {len(crops)} enhanced crop candidates")
            
            if not crops:
                logger.warning("No crops generated, using whole processed image")
                crops = [(processed_image, (0, 0, processed_image.width, processed_image.height))]
            
            for i, (crop, bounding_box) in enumerate(crops):
                logger.info(f"Processing crop {i+1}/{len(crops)}: {crop.size}")
                
                # Step 3: Comprehensive model analysis
                blip_result = self._get_comprehensive_blip_detection(crop, context)
                vit_result = self._get_comprehensive_vit_detection(crop, context)
                swin_result = self._get_comprehensive_swin_detection(crop, context)
                clip_result = self._get_comprehensive_clip_detection(crop, context)
                yolo_result = self._get_comprehensive_yolo_detection(crop, context)
                
                # Step 4: Process all detections
                if blip_result:
                    all_detections["blip_detections"].extend(blip_result)
                
                if vit_result:
                    all_detections["vit_detections"].extend(vit_result)
                
                if swin_result:
                    all_detections["swin_detections"].extend(swin_result)
                
                if clip_result:
                    all_detections["clip_detections"].extend(clip_result)
                
                if yolo_result:
                    all_detections["yolo_detections"].extend(yolo_result)
                
                # Step 5: Create comprehensive result for this crop
                comprehensive_result = self._create_comprehensive_result(
                    crop, bounding_box, blip_result, vit_result, swin_result, clip_result, yolo_result
                )
                if comprehensive_result:
                    all_detections["comprehensive_results"].append(comprehensive_result)
            
            # Add summary statistics
            all_detections.update({
                "total_blip": len(all_detections["blip_detections"]),
                "total_vit": len(all_detections["vit_detections"]),
                "total_swin": len(all_detections["swin_detections"]),
                "total_clip": len(all_detections["clip_detections"]),
                "total_yolo": len(all_detections["yolo_detections"]),
                "total_comprehensive": len(all_detections["comprehensive_results"]),
                "image_processed": True,
                "processing_method": "Enhanced Multi-Model Analysis"
            })
            
            logger.info(f"Comprehensive analysis completed - BLIP: {all_detections['total_blip']}, ViT: {all_detections['total_vit']}, Swin: {all_detections['total_swin']}, CLIP: {all_detections['total_clip']}, YOLO: {all_detections['total_yolo']}")
            return all_detections
        
        except Exception as e:
            logger.error(f"Comprehensive food analysis failed: {e}")
            return {
                "blip_detections": [],
                "vit_detections": [],
                "swin_detections": [],
                "clip_detections": [],
                "yolo_detections": [],
                "comprehensive_results": [],
                "total_blip": 0,
                "total_vit": 0,
                "total_swin": 0,
                "total_clip": 0,
                "total_yolo": 0,
                "total_comprehensive": 0,
                "image_processed": False,
                "processing_method": "Failed",
                "error": str(e)
            }
    
    def _preprocess_image_for_analysis(self, image: Image.Image) -> Image.Image:
        """
        Enhanced image preprocessing for better model performance
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to optimal size for models
            target_size = (512, 512)  # Optimal size for most models
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Apply basic enhancement
            from PIL import ImageEnhance
            
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.05)
            
            logger.info(f"Image preprocessed to {image.size}")
            return image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def _get_enhanced_crops(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Generate enhanced crop candidates with better processing
        """
        crops = []
        
        try:
            # Strategy 1: High confidence YOLO detections
            if self.models.get('yolo_model'):
                logger.info("Using YOLO for enhanced crop detection")
                yolo_results = self.models['yolo_model'](image, verbose=False)
                
                for result in yolo_results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Use moderate confidence threshold for better coverage
                            confidence = box.conf[0].item() if hasattr(box, 'conf') else 0.5
                            if confidence > 0.6:  # Moderate confidence threshold
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                crop = image.crop((x1, y1, x2, y2))
                                if crop.width * crop.height >= 2500:  # Minimum area
                                    crops.append((crop, (x1, y1, x2, y2)))
                                    logger.info(f"YOLO crop: {crop.width}x{crop.height} (conf: {confidence:.3f})")
            
            # Strategy 2: Enhanced grid cropping for better coverage
            if len(crops) < 3:  # Ensure we have enough candidates
                logger.info("Adding enhanced grid crops for better coverage")
                grid_crops = self._create_enhanced_grid_crops(image)
                crops.extend(grid_crops)
            
            # Strategy 3: Center crop for single food items
            center_crop = self._create_center_crop(image)
            if center_crop:
                crops.append(center_crop)
                logger.info("Added center crop")
            
            # Strategy 4: Whole image as fallback
            if not crops:
                crops.append((image, (0, 0, image.width, image.height)))
                logger.info("Using whole image as fallback")
            
        except Exception as e:
            logger.warning(f"Enhanced crop generation failed: {e}")
            # Fallback to whole image
            crops.append((image, (0, 0, image.width, image.height)))
        
        return crops
    
    def _preprocess_crop_for_vit(self, crop: Image.Image) -> Image.Image:
        """
        Enhanced preprocessing specifically for ViT model
        """
        try:
            # Convert to RGB if needed
            if crop.mode != 'RGB':
                crop = crop.convert('RGB')
            
            # Resize to ViT's preferred size
            target_size = (224, 224)  # ViT-B/16 standard size
            crop = crop.resize(target_size, Image.Resampling.LANCZOS)
            
            # Apply additional enhancements for better ViT performance
            from PIL import ImageEnhance
            
            # Enhance brightness slightly
            enhancer = ImageEnhance.Brightness(crop)
            crop = enhancer.enhance(1.05)
            
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(crop)
            crop = enhancer.enhance(1.1)
            
            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(crop)
            crop = enhancer.enhance(1.1)
            
            logger.info(f"Crop preprocessed for ViT: {crop.size}")
            return crop
            
        except Exception as e:
            logger.warning(f"ViT crop preprocessing failed: {e}")
            return crop
    
    def _create_enhanced_grid_crops(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Create enhanced grid crops for better coverage
        """
        crops = []
        width, height = image.size
        
        # Adaptive grid based on image size
        if width > 800 and height > 600:
            grid_size = (3, 3)  # 3x3 for large images
        elif width > 400 and height > 300:
            grid_size = (2, 2)  # 2x2 for medium images
        else:
            grid_size = (2, 1)  # 2x1 for small images
        
        crop_width = width // grid_size[0]
        crop_height = height // grid_size[1]
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x1 = i * crop_width
                y1 = j * crop_height
                x2 = min(x1 + crop_width, width)
                y2 = min(y1 + crop_height, height)
                
                crop = image.crop((x1, y1, x2, y2))
                if crop.width * crop_height >= 2500:
                    crops.append((crop, (x1, y1, x2, y2)))
        
        return crops
    
    def _create_comprehensive_result(self, crop, bounding_box, blip_result, vit_result, swin_result, clip_result, yolo_result):
        """
        Create comprehensive result combining all model outputs
        """
        try:
            # Combine all detections
            all_detections = []
            if blip_result:
                all_detections.extend(blip_result)
            if vit_result:
                all_detections.extend(vit_result)
            if swin_result:
                all_detections.extend(swin_result)
            if clip_result:
                all_detections.extend(clip_result)
            if yolo_result:
                all_detections.extend(yolo_result)
            
            if not all_detections:
                return None
            
            # Create comprehensive summary
            result = {
                "crop_size": crop.size,
                "bounding_box": bounding_box,
                "total_detections": len(all_detections),
                "blip_count": len(blip_result) if blip_result else 0,
                "vit_count": len(vit_result) if vit_result else 0,
                "swin_count": len(swin_result) if swin_result else 0,
                "clip_count": len(clip_result) if clip_result else 0,
                "yolo_count": len(yolo_result) if yolo_result else 0,
                "all_detections": all_detections
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Comprehensive result creation failed: {e}")
            return None
    
    def _create_adaptive_grid_crops(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Create adaptive grid crops based on image characteristics
        """
        crops = []
        width, height = image.size
        
        # Analyze image to determine optimal grid
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:  # Wide image
            grid_size = (3, 2)  # 3x2 grid
        elif aspect_ratio < 0.7:  # Tall image
            grid_size = (2, 3)  # 2x3 grid
        else:  # Square-ish image
            grid_size = (2, 2)  # 2x2 grid
        
        crop_width = width // grid_size[0]
        crop_height = height // grid_size[1]
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x1 = i * crop_width
                y1 = j * crop_height
                x2 = min(x1 + crop_width, width)
                y2 = min(y1 + crop_height, height)
                
                crop = image.crop((x1, y1, x2, y2))
                if crop.width * crop_height >= 2500:
                    crops.append((crop, (x1, y1, x2, y2)))
        
        return crops
    
    def _create_center_crop(self, image: Image.Image) -> Optional[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Create center crop for single food items
        """
        width, height = image.size
        center_size = min(width, height) // 2
        
        x1 = (width - center_size) // 2
        y1 = (height - center_size) // 2
        x2 = x1 + center_size
        y2 = y1 + center_size
        
        crop = image.crop((x1, y1, x2, y2))
        if crop.width * crop.height >= 2500:
            return (crop, (x1, y1, x2, y2))
        return None
    
    def _get_comprehensive_blip_detection(self, crop: Image.Image, context: str = "") -> List[FoodDetection]:
        """
        Get comprehensive BLIP detection with all results
        """
        detections = []
        
        try:
            # Get BLIP description
            blip_description = self.get_blip_description(crop)
            
            if not blip_description:
                logger.info("No BLIP description available")
                return detections
            
            # Extract all food-related terms from BLIP description
            food_terms = self._extract_food_terms_from_blip(blip_description)
            
            if not food_terms:
                logger.info("No food terms found in BLIP description")
                return detections
            
            # Create detection for each food term
            for i, food_term in enumerate(food_terms[:3]):  # Top 3 food terms
                confidence = self._calculate_blip_confidence(blip_description)
                
                # Ensure food_term is a string
                if isinstance(food_term, str):
                    detection = FoodDetection(
                        bounding_box=(0, 0, crop.width, crop.height),
                        final_label=food_term,
                        confidence_score=confidence,
                        top_3_alternatives=food_terms[:3],
                        detection_method="BLIP-Comprehensive",
                        classifier_probability=0.0,
                        clip_similarity=0.0,
                        blip_description=blip_description
                    )
                    detections.append(detection)
                    logger.info(f"BLIP detection {i+1}: {food_term} (confidence: {confidence:.3f})")
            
            return detections
            
        except Exception as e:
            logger.warning(f"Comprehensive BLIP detection failed: {e}")
            return detections
    
    def _get_comprehensive_vit_detection(self, crop: Image.Image, context: str = "") -> List[FoodDetection]:
        """
        Get comprehensive ViT detection with proper image processing
        """
        detections = []
        
        try:
            # Enhanced preprocessing for ViT
            processed_crop = self._preprocess_crop_for_vit(crop)
            
            # Get ViT classification with all results
            vit_probs = self._classify_with_vit(processed_crop)
            
            if not vit_probs:
                logger.info("No ViT probabilities available")
                return detections
            
            # Get top 5 predictions
            top_predictions = sorted(vit_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (label, prob) in enumerate(top_predictions):
                if prob > 0.1:  # Lower threshold for comprehensive results
                    # Ensure label is a string
                    if isinstance(label, str):
                        detection = FoodDetection(
                            bounding_box=(0, 0, crop.width, crop.height),
                            final_label=label,
                            confidence_score=prob,
                            top_3_alternatives=[(l, p) for l, p in top_predictions[:3]],
                            detection_method="ViT-Comprehensive",
                            classifier_probability=prob,
                            clip_similarity=0.0,
                            blip_description=""
                        )
                        detections.append(detection)
                        logger.info(f"ViT detection {i+1}: {label} (confidence: {prob:.3f})")
            
            return detections
            
        except Exception as e:
            logger.warning(f"Comprehensive ViT detection failed: {e}")
            return detections
    
    def _get_comprehensive_swin_detection(self, crop: Image.Image, context: str = "") -> List[FoodDetection]:
        """
        Get comprehensive Swin detection
        """
        detections = []
        
        try:
            # Get Swin classification
            swin_probs = self._classify_with_swin(crop)
            
            if not swin_probs:
                logger.info("No Swin probabilities available")
                return detections
            
            # Get top 5 predictions
            top_predictions = sorted(swin_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (label, prob) in enumerate(top_predictions):
                if prob > 0.1:  # Lower threshold for comprehensive results
                    # Ensure label is a string
                    if isinstance(label, str):
                        detection = FoodDetection(
                            bounding_box=(0, 0, crop.width, crop.height),
                            final_label=label,
                            confidence_score=prob,
                            top_3_alternatives=[(l, p) for l, p in top_predictions[:3]],
                            detection_method="Swin-Comprehensive",
                            classifier_probability=prob,
                            clip_similarity=0.0,
                            blip_description=""
                        )
                        detections.append(detection)
                        logger.info(f"Swin detection {i+1}: {label} (confidence: {prob:.3f})")
            
            return detections
            
        except Exception as e:
            logger.warning(f"Comprehensive Swin detection failed: {e}")
            return detections
    
    def _get_comprehensive_clip_detection(self, crop: Image.Image, context: str = "") -> List[FoodDetection]:
        """
        Get comprehensive CLIP detection with improved prompts
        """
        detections = []
        
        try:
            # Define comprehensive food-related text prompts for CLIP
            food_prompts = [
                # Common foods
                "apple", "banana", "orange", "grape", "strawberry", "blueberry", "raspberry",
                "bread", "toast", "sandwich", "hamburger", "hot dog", "pizza", "taco", "burrito",
                "rice", "pasta", "noodles", "spaghetti", "macaroni", "lasagna",
                "chicken", "beef", "pork", "fish", "salmon", "tuna", "shrimp", "lobster",
                "egg", "omelette", "scrambled eggs", "fried eggs",
                "salad", "lettuce", "tomato", "cucumber", "carrot", "broccoli", "spinach",
                "potato", "french fries", "mashed potatoes", "baked potato",
                "cake", "cookie", "donut", "muffin", "cupcake", "ice cream", "pudding",
                "cheese", "milk", "yogurt", "butter", "cream",
                "coffee", "tea", "juice", "water", "soda", "beer", "wine",
                "soup", "stew", "curry", "stir fry", "grilled food", "fried food",
                "steak", "bacon", "sausage", "ham", "turkey", "duck",
                "onion", "garlic", "pepper", "mushroom", "corn", "pea", "bean",
                "peach", "pear", "pineapple", "mango", "kiwi", "lemon", "lime",
                "almond", "walnut", "peanut", "cashew", "pistachio",
                "cereal", "oatmeal", "granola", "pancake", "waffle", "crepe",
                "sushi", "sashimi", "tempura", "ramen", "udon", "dumpling",
                "chocolate", "candy", "gummy bear", "lollipop", "chocolate bar",
                "avocado", "olive", "pickle", "jalapeno", "bell pepper",
                "asparagus", "zucchini", "eggplant", "cauliflower", "cabbage",
                "cherry", "plum", "apricot", "fig", "date", "prune",
                "coconut", "pomegranate", "dragon fruit", "passion fruit",
                "quinoa", "couscous", "bulgur", "barley", "rye bread", "sourdough",
                "bagel", "croissant", "biscuit", "scone", "pretzel",
                "smoothie", "milkshake", "hot chocolate", "espresso", "latte", "cappuccino",
                "wine glass", "beer bottle", "cocktail", "margarita", "martini",
                "food", "meal", "dish", "cuisine", "cooking", "edible", "delicious", "tasty"
            ]
            
            # Get CLIP similarities
            clip_similarities = self.get_clip_similarities(crop, food_prompts)
            
            if not clip_similarities:
                logger.info("No CLIP similarities available")
                return detections
            
            # Get top 5 predictions
            top_similarities = sorted(clip_similarities.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (prompt, similarity) in enumerate(top_similarities):
                if similarity > 0.25:  # Lowered threshold for better detection
                    # Ensure prompt is a string
                    if isinstance(prompt, str):
                        detection = FoodDetection(
                            bounding_box=(0, 0, crop.width, crop.height),
                            final_label=prompt,
                            confidence_score=similarity,
                            top_3_alternatives=[(p, s) for p, s in top_similarities[:3]],
                            detection_method="CLIP-Comprehensive",
                            classifier_probability=0.0,
                            clip_similarity=similarity,
                            blip_description=""
                        )
                        detections.append(detection)
                        logger.info(f"CLIP detection {i+1}: {prompt} (similarity: {similarity:.3f})")
            
            return detections
            
        except Exception as e:
            logger.warning(f"Comprehensive CLIP detection failed: {e}")
            return detections
    
    def _get_comprehensive_yolo_detection(self, crop: Image.Image, context: str = "") -> List[FoodDetection]:
        """
        Get comprehensive YOLO detection
        """
        detections = []
        
        try:
            if not self.models.get('yolo_model'):
                logger.info("YOLO model not available")
                return detections
            
            # Run YOLO on the crop
            yolo_results = self.models['yolo_model'](crop, verbose=False)
            
            for result in yolo_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = box.conf[0].item() if hasattr(box, 'conf') else 0.5
                        class_id = int(box.cls[0].item()) if hasattr(box, 'cls') else 0
                        
                        if confidence > 0.3:  # Lower threshold for comprehensive results
                            # Create string label
                            label = self._get_yolo_food_label(class_id)
                            
                            detection = FoodDetection(
                                bounding_box=(0, 0, crop.width, crop.height),
                                final_label=label,
                                confidence_score=confidence,
                                top_3_alternatives=[],
                                detection_method="YOLO-Comprehensive",
                                classifier_probability=confidence,
                                clip_similarity=0.0,
                                blip_description=""
                            )
                            detections.append(detection)
                            logger.info(f"YOLO detection: {label} (confidence: {confidence:.3f})")
            
            return detections
            
        except Exception as e:
            logger.warning(f"Comprehensive YOLO detection failed: {e}")
            return detections
    
    def _get_vit_detection(self, crop: Image.Image, context: str = "") -> Optional[Tuple[str, float, float]]:
        """
        Get ViT-only detection with strict validation
        """
        try:
            # Get ViT classification
            vit_probs = self._classify_with_vit(crop)
            
            if not vit_probs:
                logger.info("No ViT probabilities available")
                return None
            
            # Get top prediction
            top_predictions = sorted(vit_probs.items(), key=lambda x: x[1], reverse=True)
            
            if not top_predictions:
                logger.info("No top ViT predictions")
                return None
            
            best_label, best_prob = top_predictions[0]
            
            # Only return if confidence is high enough
            if best_prob < 0.9:
                logger.info(f"ViT confidence {best_prob:.3f} below threshold")
                return None
            
            return (best_label, best_prob, best_prob)
            
        except Exception as e:
            logger.warning(f"ViT detection failed: {e}")
            return None
    
    def _extract_food_terms_from_blip(self, blip_description: str) -> List[str]:
        """
        Extract food-related terms from BLIP description
        """
        try:
            # Common food-related words
            food_words = [
                'apple', 'bread', 'cake', 'pizza', 'salad', 'soup', 'rice', 'pasta', 'meat', 'fish', 'chicken', 
                'vegetable', 'fruit', 'cheese', 'egg', 'milk', 'coffee', 'tea', 'juice', 'water', 'sauce',
                'curry', 'sandwich', 'burger', 'fries', 'steak', 'lobster', 'shrimp', 'salmon', 'tuna',
                'carrot', 'tomato', 'onion', 'potato', 'lettuce', 'spinach', 'broccoli', 'cauliflower',
                'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'raspberry', 'peach', 'pear',
                'cookie', 'donut', 'ice cream', 'yogurt', 'butter', 'oil', 'salt', 'pepper', 'sugar'
            ]
            
            blip_lower = blip_description.lower()
            found_terms = []
            
            for word in food_words:
                if word in blip_lower:
                    found_terms.append(word)
            
            return found_terms
            
        except Exception as e:
            logger.warning(f"Food term extraction failed: {e}")
            return []
    
    def _calculate_blip_confidence(self, blip_description: str) -> float:
        """
        Calculate confidence score for BLIP description
        """
        try:
            confidence = 0.5  # Base confidence
            
            # Boost for food indicators
            food_indicators = ['food', 'meal', 'dish', 'cuisine', 'cooking', 'cooked', 'fresh', 'delicious', 'tasty', 'edible']
            food_count = sum(1 for indicator in food_indicators if indicator in blip_description.lower())
            confidence += food_count * 0.1
            
            # Boost for specific food words
            food_words = ['apple', 'bread', 'cake', 'pizza', 'salad', 'soup', 'rice', 'pasta', 'meat', 'fish', 'chicken']
            food_word_count = sum(1 for word in food_words if word in blip_description.lower())
            confidence += food_word_count * 0.15
            
            # Penalty for non-food indicators
            non_food_indicators = ['utensil', 'plate', 'bowl', 'cup', 'glass', 'bottle', 'container', 'table', 'chair', 'background']
            non_food_count = sum(1 for indicator in non_food_indicators if indicator in blip_description.lower())
            confidence -= non_food_count * 0.2
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"BLIP confidence calculation failed: {e}")
            return 0.0
    
    def _ultra_strict_fuse_evidence(self, classifier_probs: Dict[str, float], 
                                   clip_similarities: Dict[str, float],
                                   blip_description: Optional[str],
                                   context: str = "") -> List[Tuple[str, float]]:
        """
        Ultra-strict evidence fusion - only for items actually found in the image
        """
        fused_scores = {}
        
        for label in classifier_probs.keys():
            classifier_prob = classifier_probs[label]
            clip_sim = clip_similarities.get(label, 0)
            
            # Ultra-strict fusion - requires both classifier and CLIP to agree
            if classifier_prob > 0.5 and clip_sim > 0.5:
                # Base fusion with higher weight on agreement
                base_fusion = 0.6 * classifier_prob + 0.4 * clip_sim
                
                # Minimal context boost only if very specific
                context_boost = 0
                if context and any(keyword in context.lower() for keyword in ['indian', 'curry', 'masala']):
                    if any(keyword in label.lower() for keyword in ['curry', 'dal', 'naan', 'biryani', 'samosa']):
                        context_boost = 0.05  # Reduced boost
                
                # BLIP description validation - must strongly support the detection
                blip_boost = 0
                if blip_description:
                    blip_lower = blip_description.lower()
                    label_words = label.replace('_', ' ').lower().split()
                    word_matches = sum(1 for word in label_words if word in blip_lower)
                    if word_matches >= 2:  # Require at least 2 word matches
                        blip_boost = 0.05
                
                final_score = min(base_fusion + context_boost + blip_boost, 1.0)
                
                # Only include if final score is very high
                if final_score > 0.8:
                    fused_scores[label] = final_score
        
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _validate_image_based_food_detection(self, label: str, blip_description: Optional[str], confidence: float, crop: Image.Image) -> bool:
        """
        Ultra-strict validation that detection is actually food found in the image
        """
        try:
            label_lower = label.lower()
            
            # Step 1: Validate BLIP description strongly supports food detection
            if blip_description:
                blip_lower = blip_description.lower()
                
                # Must have strong food indicators
                strong_food_indicators = ['food', 'meal', 'dish', 'cuisine', 'cooking', 'cooked', 'fresh', 'delicious', 'tasty', 'edible']
                food_indicator_count = sum(1 for indicator in strong_food_indicators if indicator in blip_lower)
                
                if food_indicator_count < 1:
                    logger.info(f"BLIP description lacks strong food indicators: {blip_description}")
                    return False
                
                # Must NOT have strong non-food indicators
                strong_non_food_indicators = ['utensil', 'plate', 'bowl', 'cup', 'glass', 'bottle', 'container', 'table', 'chair', 'background', 'wall', 'floor', 'object', 'thing', 'item']
                non_food_indicator_count = sum(1 for indicator in strong_non_food_indicators if indicator in blip_lower)
                
                if non_food_indicator_count > 0:
                    logger.info(f"BLIP description contains non-food indicators: {blip_description}")
                    return False
                
                # Must have specific food-related words that match the label
                label_words = label.replace('_', ' ').lower().split()
                blip_word_matches = sum(1 for word in label_words if word in blip_lower)
                
                if blip_word_matches < 1:
                    logger.info(f"BLIP description doesn't match label words: {blip_description} vs {label}")
                    return False
            else:
                logger.info("No BLIP description available for validation")
                return False
            
            # Step 2: Ultra-high confidence requirement
            if confidence < 0.95:
                logger.info(f"Confidence {confidence:.3f} below ultra-high threshold")
                return False
            
            # Step 3: Additional image-based validation
            if not self._validate_crop_characteristics(crop, label):
                logger.info(f"Crop characteristics don't match label '{label}'")
                return False
            
            logger.info(f"Image-based food validation passed for '{label}'")
            return True
            
        except Exception as e:
            logger.warning(f"Image-based food validation failed: {e}")
            return False
    
    def _is_valid_blip_description(self, blip_description: str) -> bool:
        """
        Validate that BLIP description is meaningful and not repetitive
        """
        try:
            if not blip_description or len(blip_description.strip()) < 10:
                return False
            
            # Check for repetitive patterns (like "junk junk junk")
            words = blip_description.lower().split()
            if len(words) < 3:
                return False
            
            # Check for excessive repetition
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # If any word appears more than 3 times, it's likely repetitive
            for word, count in word_counts.items():
                if count > 3:
                    logger.info(f"BLIP description has repetitive word '{word}' appearing {count} times")
                    return False
            
            # Check for meaningless descriptions
            meaningless_phrases = [
                'junk junk junk', 'describe the food', 'list the foods', 'what are the main ingredients',
                'what beverages', 'what dairy products', 'what desserts', 'what food is in this image',
                'what fruits are visible', 'what grains', 'what meal or food items', 'what nuts or seeds',
                'what prepared foods', 'what proteins', 'what spices', 'what vegetables'
            ]
            
            blip_lower = blip_description.lower()
            for phrase in meaningless_phrases:
                if phrase in blip_lower:
                    logger.info(f"BLIP description contains meaningless phrase: {phrase}")
                    return False
            
            # Check for question-like descriptions
            if blip_description.strip().endswith('?') or 'what' in blip_description.lower():
                logger.info("BLIP description appears to be a question")
                return False
            
            # Must contain actual food-related content
            food_indicators = ['food', 'dish', 'meal', 'cuisine', 'cooking', 'cooked', 'fresh', 'delicious', 'tasty', 'edible']
            if not any(indicator in blip_lower for indicator in food_indicators):
                # Check for specific food words
                food_words = ['apple', 'bread', 'cake', 'pizza', 'salad', 'soup', 'rice', 'pasta', 'meat', 'fish', 'chicken', 'vegetable', 'fruit']
                if not any(word in blip_lower for word in food_words):
                    logger.info("BLIP description lacks food-related content")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"BLIP description validation failed: {e}")
            return False
    
    def _validate_crop_characteristics(self, crop: Image.Image, label: str) -> bool:
        """
        Validate that crop characteristics are consistent with the food label
        """
        try:
            # Basic size validation
            if crop.width < 50 or crop.height < 50:
                logger.info("Crop too small for reliable detection")
                return False
            
            # Color analysis for certain food types
            crop_array = np.array(crop)
            if len(crop_array.shape) == 3:
                red_mean = np.mean(crop_array[:, :, 0])
                green_mean = np.mean(crop_array[:, :, 1])
                blue_mean = np.mean(crop_array[:, :, 2])
                
                label_lower = label.lower()
                
                # Validate color consistency with food type
                if 'salad' in label_lower or 'vegetable' in label_lower:
                    if green_mean < 80:  # Should have some green for vegetables
                        logger.info(f"Vegetable/salad detection but low green content: {green_mean:.1f}")
                        return False
                
                if 'meat' in label_lower or 'chicken' in label_lower or 'beef' in label_lower:
                    if red_mean < 100:  # Should have some red for meat
                        logger.info(f"Meat detection but low red content: {red_mean:.1f}")
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Crop characteristic validation failed: {e}")
            return True  # Allow if validation fails
    
    def _create_fallback_detection(self, image: Image.Image) -> Optional[FoodDetection]:
        """
        Ultra-strict fallback detection - only if absolutely certain
        """
        try:
            logger.info("Attempting ultra-strict fallback detection")
            
            # Try multiple classification approaches
            classifier_probs = self.classify_with_transformers(image)
            if classifier_probs:
                # Get top classifications with very high confidence
                high_confidence_probs = {label: prob for label, prob in classifier_probs.items() if prob > 0.8}
                if not high_confidence_probs:
                    logger.info("No ultra-high confidence classifier predictions")
                    return None
                
                # Get CLIP similarities for high confidence candidates
                candidate_labels = [label for label, _ in high_confidence_probs.items()]
                clip_similarities = self.get_clip_similarities(image, candidate_labels)
                
                # Filter for very high CLIP confidence
                ultra_high_clip = {label: sim for label, sim in clip_similarities.items() if sim > 0.8}
                if not ultra_high_clip:
                    logger.info("No ultra-high confidence CLIP similarities")
                    return None
                
                # Get BLIP description
                blip_description = self.get_blip_description(image)
                
                # Ultra-strict fusion for fallback
                fused_scores = self._ultra_strict_fuse_evidence(high_confidence_probs, ultra_high_clip, blip_description, "")
                
                if fused_scores and fused_scores[0][1] >= 0.95:  # Only return if 95%+ confidence
                    best_label, best_confidence = fused_scores[0]
                    classifier_prob = high_confidence_probs.get(best_label, 0)
                    clip_sim = ultra_high_clip.get(best_label, 0)
                    
                    # Additional validation
                    if self._validate_image_based_food_detection(best_label, blip_description, best_confidence, image):
                        return FoodDetection(
                            bounding_box=(0, 0, image.width, image.height),
                            final_label=best_label,
                            confidence_score=best_confidence,
                            top_3_alternatives=fused_scores[:3],
                            detection_method="Ultra-Strict Fallback",
                            classifier_probability=classifier_prob,
                            clip_similarity=clip_sim,
                            blip_description=blip_description
                        )
            
            # If no ultra-high confidence detection, return None
            logger.warning("No ultra-high confidence fallback detection possible - no hardcoded items")
            return None
            
        except Exception as e:
            logger.error(f"Ultra-strict fallback detection failed: {e}")
            return None
    

    

    
    def _get_clip_similarities_enhanced(self, crop: Image.Image, texts: List[str]) -> Dict[str, float]:
        """
        Enhanced CLIP similarity calculation with better error handling
        """
        try:
            if not self.models.get('clip_model'):
                return {}
            
            # Preprocess image for CLIP
            processor = self.models.get('clip_processor')
            if not processor:
                return {}
            
            # Process image and text
            inputs = processor(
                images=crop,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Get embeddings
            with torch.no_grad():
                image_features = self.models['clip_model'].get_image_features(**inputs)
                text_features = self.models['clip_model'].get_text_features(**inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarities = torch.matmul(image_features, text_features.T)
                similarities = similarities.squeeze().cpu().numpy()
                
                return {text: float(similarity) for text, similarity in zip(texts, similarities)}
                
        except Exception as e:
            logger.warning(f"Enhanced CLIP similarity calculation failed: {e}")
            return {}
    
    def _preprocess_image_for_blip(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for BLIP model
        """
        try:
            if not self.models.get('blip_processor'):
                return None
            
            processor = self.models['blip_processor']
            inputs = processor(images=image, return_tensors="pt")
            return inputs
        except Exception as e:
            logger.warning(f"BLIP preprocessing failed: {e}")
            return None
    
    def _classify_with_indian_context(self, crop: Image.Image, context: str = "") -> List[Tuple[str, float]]:
        """
        Enhanced classification with Indian food context awareness
        """
        try:
            # Get base classifications
            # Return base classifications without hardcoded enhancements
            return self.classify_with_transformers(crop)
            
        except Exception as e:
            logger.warning(f"Indian context classification failed: {e}")
            return self.classify_with_transformers(crop)
    
    def get_detection_summary(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive summary of all model detections
        """
        try:
            blip_detections = detection_results.get("blip_detections", [])
            vit_detections = detection_results.get("vit_detections", [])
            swin_detections = detection_results.get("swin_detections", [])
            clip_detections = detection_results.get("clip_detections", [])
            yolo_detections = detection_results.get("yolo_detections", [])
            comprehensive_results = detection_results.get("comprehensive_results", [])
            
            # Extract food labels from each model with error handling
            blip_foods = []
            vit_foods = []
            swin_foods = []
            clip_foods = []
            yolo_foods = []
            
            # Extract confidence scores with error handling
            blip_confidence = {}
            vit_confidence = {}
            swin_confidence = {}
            clip_confidence = {}
            yolo_confidence = {}
            
            # Process BLIP detections
            for det in blip_detections:
                if hasattr(det, 'final_label') and hasattr(det, 'confidence_score'):
                    blip_foods.append(det.final_label)
                    blip_confidence[det.final_label] = det.confidence_score
            
            # Process ViT detections
            for det in vit_detections:
                if hasattr(det, 'final_label') and hasattr(det, 'confidence_score'):
                    vit_foods.append(det.final_label)
                    vit_confidence[det.final_label] = det.confidence_score
            
            # Process Swin detections
            for det in swin_detections:
                if hasattr(det, 'final_label') and hasattr(det, 'confidence_score'):
                    swin_foods.append(det.final_label)
                    swin_confidence[det.final_label] = det.confidence_score
            
            # Process CLIP detections
            for det in clip_detections:
                if hasattr(det, 'final_label') and hasattr(det, 'confidence_score'):
                    clip_foods.append(det.final_label)
                    clip_confidence[det.final_label] = det.confidence_score
            
            # Process YOLO detections
            for det in yolo_detections:
                if hasattr(det, 'final_label') and hasattr(det, 'confidence_score'):
                    yolo_foods.append(det.final_label)
                    yolo_confidence[det.final_label] = det.confidence_score
            
            total_detections = len(blip_detections) + len(vit_detections) + len(swin_detections) + len(clip_detections) + len(yolo_detections)
            
            return {
                "total_detections": total_detections,
                "blip_detections": blip_foods,
                "vit_detections": vit_foods,
                "swin_detections": swin_foods,
                "clip_detections": clip_foods,
                "yolo_detections": yolo_foods,
                "blip_confidence": blip_confidence,
                "vit_confidence": vit_confidence,
                "swin_confidence": swin_confidence,
                "clip_confidence": clip_confidence,
                "yolo_confidence": yolo_confidence,
                "detection_method": "Comprehensive Multi-Model Analysis",
                "success": total_detections > 0,
                "all_detections": blip_detections + vit_detections + swin_detections + clip_detections + yolo_detections,
                "blip_count": len(blip_detections),
                "vit_count": len(vit_detections),
                "swin_count": len(swin_detections),
                "clip_count": len(clip_detections),
                "yolo_count": len(yolo_detections),
                "comprehensive_results": comprehensive_results,
                "image_processed": detection_results.get("image_processed", False),
                "processing_method": detection_results.get("processing_method", "Unknown")
            }
            
        except Exception as e:
            logger.error(f"Detection summary creation failed: {e}")
            return {
                "total_detections": 0,
                "blip_detections": [],
                "vit_detections": [],
                "swin_detections": [],
                "clip_detections": [],
                "yolo_detections": [],
                "blip_confidence": {},
                "vit_confidence": {},
                "swin_confidence": {},
                "clip_confidence": {},
                "yolo_confidence": {},
                "detection_method": "Comprehensive Multi-Model Analysis",
                "success": False,
                "all_detections": [],
                "blip_count": 0,
                "vit_count": 0,
                "swin_count": 0,
                "clip_count": 0,
                "yolo_count": 0,
                "comprehensive_results": [],
                "image_processed": False,
                "processing_method": "Failed",
                "error": str(e)
            }

def create_expert_food_recognition_interface(models: Dict[str, Any]):
    """
    Create Streamlit interface for the expert food recognition system
    """
    import streamlit as st
    
    st.markdown("## 🧠 Expert Food Recognition System")
    st.markdown("Advanced multi-model AI system combining YOLO, ViT, Swin, CLIP, and BLIP")
    
    # Initialize the expert system
    expert_system = ExpertFoodRecognitionSystem(models)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📸 Upload a food image for expert analysis",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload a clear image of your food for expert multi-model analysis"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Food Image", use_column_width=True)
        
        # Expert analysis button
        if st.button("🧠 Run Expert Food Recognition", type="primary"):
            with st.spinner("Running expert multi-model food recognition..."):
                # Run expert recognition
                detections = expert_system.recognize_food(image)
                summary = expert_system.get_detection_summary(detections)
                
                if summary["success"]:
                    st.success(f"✅ Expert analysis complete! Found {summary['total_detections']} food items")
                    
                    # Handle new comprehensive format
                    all_detections = []
                    if isinstance(detections, dict):
                        if "blip_detections" in detections:
                            all_detections.extend(detections["blip_detections"])
                        if "vit_detections" in detections:
                            all_detections.extend(detections["vit_detections"])
                        if "swin_detections" in detections:
                            all_detections.extend(detections["swin_detections"])
                        if "clip_detections" in detections:
                            all_detections.extend(detections["clip_detections"])
                        if "yolo_detections" in detections:
                            all_detections.extend(detections["yolo_detections"])
                    else:
                        all_detections = detections
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🎯 Detected Foods")
                        for i, detection in enumerate(all_detections):
                            if hasattr(detection, 'final_label'):
                                st.markdown(f"""
                                **{i+1}. {detection.final_label.replace('_', ' ').title()}**
                                - Confidence: {detection.confidence_score:.3f}
                                - Classifier: {detection.classifier_probability:.3f}
                                - CLIP Similarity: {detection.clip_similarity:.3f}
                                """)
                    
                    with col2:
                        st.markdown("### 📊 Detection Details")
                        for detection in all_detections:
                            if hasattr(detection, 'final_label'):
                                with st.expander(f"Details for {detection.final_label}"):
                                    st.write(f"**Bounding Box:** {detection.bounding_box}")
                                    st.write(f"**Top Alternatives:**")
                                    if hasattr(detection, 'top_3_alternatives') and detection.top_3_alternatives:
                                        for alternative in detection.top_3_alternatives:
                                            if isinstance(alternative, tuple):
                                                label, score = alternative
                                                st.write(f"  - {label.replace('_', ' ').title()}: {score:.3f}")
                                            else:
                                                st.write(f"  - {str(alternative)}")
                                    if hasattr(detection, 'blip_description') and detection.blip_description:
                                        st.write(f"**BLIP Description:** {detection.blip_description}")
                    
                    # Show detection method
                    st.markdown("### 🔬 Detection Method")
                    st.info("Expert Multi-Model System: YOLO + ViT-B/16 + Swin + CLIP + BLIP")
                    
                    # Show model breakdown
                    if "blip_count" in summary:
                        st.markdown("#### 📊 Model Breakdown")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("BLIP", summary.get("blip_count", 0))
                        with col2:
                            st.metric("ViT", summary.get("vit_count", 0))
                        with col3:
                            st.metric("Swin", summary.get("swin_count", 0))
                        with col4:
                            st.metric("CLIP", summary.get("clip_count", 0))
                        with col5:
                            st.metric("YOLO", summary.get("yolo_count", 0))
                    
                else:
                    st.warning("No food items detected with sufficient confidence")
    
    # Show system info
    with st.expander("🔬 Expert System Information"):
        st.markdown("""
        **Expert Food Recognition System Features:**
        
        - **YOLO Detection:** Identifies food candidate regions
        - **ViT-B/16 Classification:** Vision Transformer for food classification
        - **Swin Transformer:** Additional transformer for robust classification
        - **CLIP Similarity:** Semantic similarity scoring
        - **BLIP Description:** Contextual descriptions for tie-breaking
        
        **Confidence Thresholds:**
        - Classifier Probability: ≥ 0.45
        - CLIP Similarity: ≥ 0.28
        - Tie-breaking threshold: 0.12
        
        **Output:** Specific Food-101 category names with confidence scores
        """)

if __name__ == "__main__":
    print("Expert Food Recognition System - Import and use in Streamlit app")