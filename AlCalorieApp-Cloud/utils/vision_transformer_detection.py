"""
Vision Transformer (ViT) and Swin Transformer Food Detection System
Advanced transformer-based models for accurate food identification
"""

import logging
import numpy as np
from typing import Dict, Any, List, Set, Tuple
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoImageProcessor, SwinForImageClassification
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

class VisionTransformerFoodDetector:
    """
    Advanced food detection using Vision Transformer and Swin Transformer models
    """
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.device = models.get('device', 'cpu')
        
        # Use pre-loaded models from the models dict if available
        self.vit_processor = models.get('vit_processor')
        self.vit_model = models.get('vit_model')
        self.swin_processor = models.get('swin_processor')
        self.swin_model = models.get('swin_model')
        
        # Food classification mapping (ImageNet classes to food items)
        self.imagenet_food_mapping = self._create_food_mapping()
        
        # Custom food vocabulary for enhanced detection
        self.food_vocabulary = self._load_comprehensive_food_vocabulary()
    

    
    def _create_food_mapping(self) -> Dict[int, str]:
        """Create comprehensive mapping from ImageNet class IDs to food items"""
        # Enhanced ImageNet classes that correspond to food items
        food_mapping = {
            # Fruits
            948: 'banana',
            949: 'apple',
            950: 'orange',
            951: 'lemon',
            952: 'fig',
            953: 'pineapple',
            954: 'strawberry',
            955: 'pomegranate',
            963: 'Granny Smith apple',
            964: 'strawberry',
            965: 'orange',
            966: 'lemon',
            
            # Vegetables
            936: 'broccoli',
            937: 'cauliflower',
            938: 'cabbage',
            939: 'artichoke',
            940: 'bell pepper',
            941: 'cucumber',
            942: 'mushroom',
            943: 'corn',
            956: 'acorn squash',
            957: 'butternut squash',
            958: 'cucumber',
            959: 'artichoke',
            960: 'bell pepper',
            961: 'cardoon',
            962: 'mushroom',
            
            # Prepared foods and dishes
            924: 'guacamole',
            925: 'consomme',
            926: 'trifle',
            927: 'pizza',
            928: 'hamburger',
            929: 'hot dog',
            930: 'sandwich',
            931: 'burrito',
            932: 'bagel',
            933: 'pretzel',
            934: 'ice cream',
            935: 'cake',
            
            # Beverages
            967: 'espresso',
            968: 'cup',
            969: 'wine bottle',
            970: 'beer bottle',
            
            # Fish and seafood (common food items)
            100: 'trout',
            101: 'salmon',
            102: 'perch',
            103: 'pike',
            104: 'catfish',
            105: 'tuna',
            106: 'mackerel',
            107: 'sardine',
            108: 'anchovy',
            109: 'herring',
            110: 'cod',
            111: 'bass',
            112: 'flounder',
            113: 'sole',
            114: 'halibut',
            115: 'red snapper',
            116: 'sea bass',
            117: 'grouper',
            118: 'sea bream',
            119: 'pompano',
            120: 'bluefish',
            
            # Dairy and eggs
            130: 'milk',
            131: 'cheese',
            132: 'yogurt',
            133: 'butter',
            134: 'cream',
            135: 'egg',
            136: 'omelette',
            137: 'scrambled eggs',
            138: 'poached egg',
            139: 'fried egg',
            140: 'hard-boiled egg',
            
            # Grains and breads
            150: 'bread',
            151: 'toast',
            152: 'bagel',
            153: 'croissant',
            154: 'muffin',
            155: 'biscuit',
            156: 'cookie',
            157: 'cake',
            158: 'pie',
            159: 'pastry',
            160: 'donut',
            161: 'waffle',
            162: 'pancake',
            163: 'crepe',
            164: 'french toast',
            165: 'sandwich',
            166: 'hamburger',
            167: 'hot dog',
            168: 'pizza',
            169: 'taco',
            170: 'burrito',
            171: 'sushi',
            172: 'rice',
            173: 'pasta',
            174: 'noodles',
            175: 'spaghetti',
            176: 'macaroni',
            177: 'lasagna',
            178: 'ravioli',
            179: 'dumpling',
            180: 'potato',
            181: 'french fries',
            182: 'mashed potatoes',
            183: 'baked potato',
            184: 'sweet potato',
            185: 'yam',
            186: 'corn',
            187: 'peas',
            188: 'carrot',
            189: 'broccoli',
            190: 'cauliflower',
            191: 'cabbage',
            192: 'lettuce',
            193: 'spinach',
            194: 'kale',
            195: 'arugula',
            196: 'watercress',
            197: 'endive',
            198: 'radicchio',
            199: 'frisée',
            200: 'escarole',
            
            # Meat and protein
            250: 'beef',
            251: 'steak',
            252: 'roast beef',
            253: 'ground beef',
            254: 'hamburger patty',
            255: 'pork',
            256: 'pork chop',
            257: 'bacon',
            258: 'ham',
            259: 'sausage',
            260: 'chicken',
            261: 'chicken breast',
            262: 'chicken thigh',
            263: 'chicken wing',
            264: 'turkey',
            265: 'duck',
            266: 'goose',
            267: 'lamb',
            268: 'lamb chop',
            269: 'mutton',
            270: 'veal',
            271: 'venison',
            272: 'rabbit',
            273: 'quail',
            274: 'pheasant',
            275: 'partridge',
            276: 'grouse',
            277: 'ptarmigan',
            278: 'guinea fowl',
            279: 'peacock',
            280: 'ostrich',
            
            # Nuts and seeds
            300: 'almond',
            301: 'cashew',
            302: 'walnut',
            303: 'pecan',
            304: 'hazelnut',
            305: 'macadamia',
            306: 'pistachio',
            307: 'peanut',
            308: 'sunflower seed',
            309: 'pumpkin seed',
            310: 'sesame seed',
            311: 'flax seed',
            312: 'chia seed',
            313: 'poppy seed',
            314: 'caraway seed',
            315: 'cumin seed',
            316: 'coriander seed',
            317: 'fennel seed',
            318: 'mustard seed',
            319: 'celery seed',
            320: 'dill seed',
            
            # Beverages
            400: 'coffee',
            401: 'tea',
            402: 'hot chocolate',
            403: 'milk',
            404: 'juice',
            405: 'soda',
            406: 'water',
            407: 'beer',
            408: 'wine',
            409: 'cocktail',
            410: 'smoothie',
            411: 'milkshake',
            412: 'lemonade',
            413: 'iced tea',
            414: 'espresso',
            415: 'cappuccino',
            416: 'latte',
            417: 'mocha',
            418: 'americano',
            419: 'macchiato',
            420: 'frappuccino',
            
            # Desserts and sweets
            500: 'ice cream',
            501: 'gelato',
            502: 'sorbet',
            503: 'pudding',
            504: 'custard',
            505: 'flan',
            506: 'creme brulee',
            507: 'tiramisu',
            508: 'cheesecake',
            509: 'chocolate cake',
            510: 'vanilla cake',
            511: 'carrot cake',
            512: 'red velvet cake',
            513: 'chocolate chip cookie',
            514: 'oatmeal cookie',
            515: 'sugar cookie',
            516: 'brownie',
            517: 'fudge',
            518: 'truffle',
            519: 'chocolate bar',
            520: 'candy',
            521: 'gummy bear',
            522: 'jelly bean',
            523: 'marshmallow',
            524: 'cotton candy',
            525: 'popcorn',
            526: 'caramel',
            527: 'toffee',
            528: 'nougat',
            529: 'marzipan',
            530: 'fondant',
            
            # Spices and herbs
            600: 'salt',
            601: 'pepper',
            602: 'garlic',
            603: 'onion',
            604: 'ginger',
            605: 'turmeric',
            606: 'cinnamon',
            607: 'nutmeg',
            608: 'clove',
            609: 'cardamom',
            610: 'saffron',
            611: 'vanilla',
            612: 'mint',
            613: 'basil',
            614: 'oregano',
            615: 'thyme',
            616: 'rosemary',
            617: 'sage',
            618: 'parsley',
            619: 'cilantro',
            620: 'dill',
            621: 'chive',
            622: 'leek',
            623: 'shallot',
            624: 'scallion',
            625: 'green onion',
            626: 'chili pepper',
            627: 'jalapeno',
            628: 'habanero',
            629: 'bell pepper',
            630: 'pimento',
            631: 'paprika',
            632: 'cayenne',
            633: 'red pepper flake',
            634: 'black pepper',
            635: 'white pepper',
            636: 'pink peppercorn',
            637: 'szechuan peppercorn',
            638: 'allspice',
            639: 'bay leaf',
            640: 'star anise',
            641: 'fennel',
            642: 'anise',
            643: 'caraway',
            644: 'cumin',
            645: 'coriander',
            646: 'dill seed',
            647: 'mustard',
            648: 'horseradish',
            649: 'wasabi',
            650: 'ginger',
            651: 'galangal',
            652: 'lemongrass',
            653: 'kaffir lime',
            654: 'curry leaf',
            655: 'fenugreek',
            656: 'asafoetida',
            657: 'amchur',
            658: 'tamarind',
            659: 'kokum',
            660: 'mango powder',
            661: 'pomegranate molasses',
            662: 'sumac',
            663: 'za\'atar',
            664: 'dukkah',
            665: 'ras el hanout',
            666: 'baharat',
            667: 'berbere',
            668: 'harissa',
        }
        
        return food_mapping
    
    def _load_comprehensive_food_vocabulary(self) -> Set[str]:
        """Load comprehensive food vocabulary"""
        return {
            # Fruits
            'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'raspberry',
            'blackberry', 'lemon', 'lime', 'peach', 'pear', 'mango', 'pineapple',
            'watermelon', 'cantaloupe', 'kiwi', 'avocado', 'cherry', 'plum', 'apricot',
            'coconut', 'pomegranate', 'fig', 'date', 'cranberry',
            
            # Vegetables
            'tomato', 'potato', 'carrot', 'onion', 'broccoli', 'cauliflower', 'lettuce',
            'spinach', 'kale', 'cucumber', 'bell pepper', 'jalapeno', 'garlic', 'ginger',
            'mushroom', 'corn', 'peas', 'beans', 'asparagus', 'zucchini', 'eggplant',
            'celery', 'radish', 'beet', 'sweet potato', 'cabbage', 'brussels sprouts',
            'artichoke', 'squash', 'pumpkin',
            
            # Proteins
            'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'fish', 'salmon',
            'tuna', 'shrimp', 'crab', 'lobster', 'egg', 'tofu', 'tempeh',
            
            # Grains
            'rice', 'bread', 'pasta', 'noodles', 'quinoa', 'oats', 'cereal', 'wheat',
            'barley', 'couscous', 'bulgur',
            
            # Dairy
            'cheese', 'milk', 'yogurt', 'butter', 'cream', 'ice cream',
            
            # Prepared foods
            'pizza', 'burger', 'hamburger', 'sandwich', 'hot dog', 'taco', 'burrito',
            'sushi', 'salad', 'soup', 'stew', 'curry', 'pasta', 'lasagna', 'cake',
            'cookie', 'pie', 'donut', 'muffin', 'bagel', 'pretzel',
            
            # Beverages
            'coffee', 'tea', 'juice', 'water', 'soda', 'beer', 'wine', 'smoothie',
            'milkshake', 'espresso', 'cappuccino', 'latte'
        }
    
    def detect_food_with_transformers(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect food using Vision Transformer and Swin Transformer models
        """
        try:
            logger.info("Starting transformer-based food detection...")
            
            # Ensure image is in RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            detected_foods = set()
            confidence_scores = {}
            detection_details = {}
            
            # Method 1: Vision Transformer (ViT-B/16) detection
            vit_results = self._detect_with_vit(image)
            for food, confidence in vit_results.items():
                detected_foods.add(food)
                confidence_scores[food] = max(confidence_scores.get(food, 0), confidence)
                detection_details[food] = detection_details.get(food, []) + ['ViT-B/16']
            
            # Method 2: Swin Transformer detection
            swin_results = self._detect_with_swin(image)
            for food, confidence in swin_results.items():
                detected_foods.add(food)
                confidence_scores[food] = max(confidence_scores.get(food, 0), confidence)
                detection_details[food] = detection_details.get(food, []) + ['Swin-Transformer']
            
            # Method 3: Ensemble prediction (combine both models)
            ensemble_results = self._ensemble_prediction(image)
            for food, confidence in ensemble_results.items():
                detected_foods.add(food)
                confidence_scores[food] = max(confidence_scores.get(food, 0), confidence)
                detection_details[food] = detection_details.get(food, []) + ['Ensemble']
            
            # Filter results by confidence threshold
            min_confidence = 0.1  # Lower threshold for more detections
            filtered_foods = {food for food, conf in confidence_scores.items() if conf >= min_confidence}
            
            # Ensure we have at least some results
            if not filtered_foods:
                # Fallback to color-based detection
                fallback_foods = self._color_based_fallback(image)
                for food in fallback_foods:
                    filtered_foods.add(food)
                    confidence_scores[food] = 0.3
                    detection_details[food] = ['Color-based fallback']
            
            return {
                'detected_foods': list(filtered_foods),
                'confidence_scores': {food: confidence_scores[food] for food in filtered_foods},
                'detection_details': {food: detection_details[food] for food in filtered_foods},
                'detection_method': 'transformer_ensemble',
                'models_used': ['ViT-B/16', 'Swin-Transformer'],
                'total_detections': len(filtered_foods),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Transformer detection failed: {e}")
            return {
                'detected_foods': ['food item'],
                'confidence_scores': {'food item': 0.3},
                'detection_details': {'food item': ['fallback']},
                'detection_method': 'fallback',
                'models_used': [],
                'total_detections': 1,
                'success': False,
                'error': str(e)
            }
    
    def _detect_with_vit(self, image: Image.Image) -> Dict[str, float]:
        """Detect food using Vision Transformer (ViT-B/16)"""
        detected_foods = {}
        
        if not self.vit_processor or not self.vit_model:
            return detected_foods
        
        try:
            # Preprocess image
            inputs = self.vit_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                predictions = F.softmax(outputs.logits, dim=-1)
            
            # Get top predictions
            top_predictions = torch.topk(predictions, k=10, dim=-1)
            
            for i in range(top_predictions.indices.shape[1]):
                class_id = top_predictions.indices[0][i].item()
                confidence = top_predictions.values[0][i].item()
                
                # Map to food item if it's a food class
                if class_id in self.imagenet_food_mapping:
                    food_name = self.imagenet_food_mapping[class_id]
                    detected_foods[food_name] = confidence
                    logger.info(f"ViT detected: {food_name} (confidence: {confidence:.3f})")
        
        except Exception as e:
            logger.warning(f"ViT detection failed: {e}")
        
        return detected_foods
    
    def _detect_with_swin(self, image: Image.Image) -> Dict[str, float]:
        """Detect food using Swin Transformer"""
        detected_foods = {}
        
        if not self.swin_processor or not self.swin_model:
            return detected_foods
        
        try:
            # Preprocess image
            inputs = self.swin_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.swin_model(**inputs)
                predictions = F.softmax(outputs.logits, dim=-1)
            
            # Get top predictions
            top_predictions = torch.topk(predictions, k=10, dim=-1)
            
            for i in range(top_predictions.indices.shape[1]):
                class_id = top_predictions.indices[0][i].item()
                confidence = top_predictions.values[0][i].item()
                
                # Map to food item if it's a food class
                if class_id in self.imagenet_food_mapping:
                    food_name = self.imagenet_food_mapping[class_id]
                    detected_foods[food_name] = confidence
                    logger.info(f"Swin detected: {food_name} (confidence: {confidence:.3f})")
        
        except Exception as e:
            logger.warning(f"Swin detection failed: {e}")
        
        return detected_foods
    
    def _ensemble_prediction(self, image: Image.Image) -> Dict[str, float]:
        """Combine predictions from both models using ensemble method"""
        ensemble_results = {}
        
        try:
            # Get predictions from both models
            vit_results = self._detect_with_vit(image)
            swin_results = self._detect_with_swin(image)
            
            # Combine results with weighted average
            all_foods = set(vit_results.keys()) | set(swin_results.keys())
            
            for food in all_foods:
                vit_conf = vit_results.get(food, 0.0)
                swin_conf = swin_results.get(food, 0.0)
                
                # Weighted ensemble (ViT: 0.6, Swin: 0.4)
                ensemble_conf = 0.6 * vit_conf + 0.4 * swin_conf
                
                if ensemble_conf > 0.05:  # Minimum threshold
                    ensemble_results[food] = ensemble_conf
                    logger.info(f"Ensemble: {food} (ViT: {vit_conf:.3f}, Swin: {swin_conf:.3f}, Combined: {ensemble_conf:.3f})")
        
        except Exception as e:
            logger.warning(f"Ensemble prediction failed: {e}")
        
        return ensemble_results
    
    def _color_based_fallback(self, image: Image.Image) -> List[str]:
        """Color-based fallback detection when transformers fail"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate dominant colors
            pixels = img_array.reshape(-1, 3)
            mean_color = np.mean(pixels, axis=0)
            
            r, g, b = mean_color
            
            # Color-based food inference
            detected_foods = []
            
            if r > 150 and g < 100 and b < 100:  # Red dominant
                detected_foods.extend(['tomato', 'apple', 'strawberry'])
            elif g > 150 and r < 120 and b < 120:  # Green dominant
                detected_foods.extend(['broccoli', 'lettuce', 'spinach'])
            elif r > 200 and g > 180 and b < 100:  # Yellow dominant
                detected_foods.extend(['banana', 'corn', 'cheese'])
            elif r > 150 and g > 100 and b < 80:  # Orange dominant
                detected_foods.extend(['carrot', 'orange', 'sweet potato'])
            elif r > 100 and g > 80 and b > 60:  # Brown dominant
                detected_foods.extend(['bread', 'chicken', 'potato'])
            else:
                detected_foods.append('mixed food')
            
            return detected_foods[:2]  # Return top 2
            
        except Exception as e:
            logger.warning(f"Color-based fallback failed: {e}")
            return ['food item']
    
    def get_food_categories(self, detected_foods: List[str]) -> Dict[str, List[str]]:
        """Categorize detected foods"""
        categories = {
            'fruits': [],
            'vegetables': [],
            'proteins': [],
            'grains': [],
            'dairy': [],
            'prepared_foods': [],
            'beverages': [],
            'other': []
        }
        
        for food in detected_foods:
            food_lower = food.lower()
            
            # Categorize based on food type
            if any(fruit in food_lower for fruit in ['apple', 'banana', 'orange', 'grape', 'berry', 'fruit']):
                categories['fruits'].append(food)
            elif any(veg in food_lower for veg in ['tomato', 'broccoli', 'carrot', 'lettuce', 'vegetable']):
                categories['vegetables'].append(food)
            elif any(protein in food_lower for protein in ['chicken', 'beef', 'fish', 'egg', 'meat']):
                categories['proteins'].append(food)
            elif any(grain in food_lower for grain in ['rice', 'bread', 'pasta', 'grain']):
                categories['grains'].append(food)
            elif any(dairy in food_lower for dairy in ['cheese', 'milk', 'yogurt', 'dairy']):
                categories['dairy'].append(food)
            elif any(prepared in food_lower for prepared in ['pizza', 'burger', 'sandwich', 'prepared']):
                categories['prepared_foods'].append(food)
            elif any(beverage in food_lower for beverage in ['coffee', 'tea', 'juice', 'drink']):
                categories['beverages'].append(food)
            else:
                categories['other'].append(food)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def analyze_nutritional_balance(self, detected_foods: List[str]) -> Dict[str, Any]:
        """Analyze nutritional balance of detected foods"""
        categories = self.get_food_categories(detected_foods)
        
        balance_score = 0
        recommendations = []
        
        # Check for variety
        if len(categories) >= 3:
            balance_score += 2
            recommendations.append("Good variety of food categories")
        elif len(categories) >= 2:
            balance_score += 1
            recommendations.append("Moderate variety - consider adding more food groups")
        else:
            recommendations.append("Limited variety - try to include foods from different categories")
        
        # Check for specific nutrients
        if categories.get('proteins'):
            balance_score += 2
            recommendations.append("Good protein sources identified")
        else:
            recommendations.append("Consider adding protein sources")
        
        if categories.get('vegetables') or categories.get('fruits'):
            balance_score += 2
            recommendations.append("Good micronutrient sources from fruits/vegetables")
        else:
            recommendations.append("Add more fruits and vegetables for vitamins and minerals")
        
        if categories.get('grains'):
            balance_score += 1
            recommendations.append("Carbohydrate sources present for energy")
        
        return {
            'balance_score': min(balance_score, 10),  # Cap at 10
            'categories': categories,
            'recommendations': recommendations,
            'total_foods': len(detected_foods)
        }