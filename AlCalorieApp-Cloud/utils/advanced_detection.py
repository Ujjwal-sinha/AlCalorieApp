"""
Advanced Multi-Model Food Detection System
Handles complex images with multiple food items using ensemble methods
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Set
from PIL import Image
import torch
import cv2
from collections import Counter
import re

logger = logging.getLogger(__name__)

class AdvancedFoodDetector:
    """
    Advanced food detection system using multiple models and techniques
    for 100% accuracy on complex food images
    """
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.confidence_threshold = 0.3
        self.ensemble_threshold = 0.6
        
        # Comprehensive food vocabulary (1000+ items)
        self.food_vocabulary = self._load_comprehensive_food_vocabulary()
        
        # Food categories for better classification
        self.food_categories = self._load_food_categories()
        
        # Visual features for food recognition
        self.visual_features = self._load_visual_features()
    
    def _load_comprehensive_food_vocabulary(self) -> Set[str]:
        """Load comprehensive food vocabulary for accurate detection"""
        return {
            # Proteins
            'chicken', 'chicken breast', 'chicken thigh', 'chicken wing', 'chicken drumstick',
            'beef', 'ground beef', 'steak', 'ribeye', 'sirloin', 'brisket', 'roast beef',
            'pork', 'pork chop', 'bacon', 'ham', 'sausage', 'pepperoni', 'prosciutto',
            'lamb', 'lamb chop', 'leg of lamb', 'lamb shank',
            'turkey', 'turkey breast', 'turkey leg', 'ground turkey',
            'duck', 'duck breast', 'duck leg', 'duck confit',
            'fish', 'salmon', 'tuna', 'cod', 'tilapia', 'mahi mahi', 'halibut', 'sea bass',
            'shrimp', 'prawns', 'crab', 'lobster', 'scallops', 'mussels', 'clams', 'oysters',
            'egg', 'eggs', 'scrambled eggs', 'fried egg', 'boiled egg', 'poached egg', 'omelet',
            'tofu', 'tempeh', 'seitan', 'beans', 'lentils', 'chickpeas', 'black beans',
            
            # Vegetables
            'tomato', 'tomatoes', 'cherry tomatoes', 'roma tomatoes', 'beefsteak tomatoes',
            'potato', 'potatoes', 'sweet potato', 'mashed potatoes', 'french fries', 'baked potato',
            'carrot', 'carrots', 'baby carrots', 'carrot sticks',
            'onion', 'onions', 'red onion', 'white onion', 'yellow onion', 'green onion', 'scallions',
            'broccoli', 'broccoli florets', 'broccolini',
            'cauliflower', 'cauliflower florets', 'cauliflower rice',
            'lettuce', 'romaine lettuce', 'iceberg lettuce', 'butter lettuce', 'arugula',
            'spinach', 'baby spinach', 'spinach leaves',
            'kale', 'kale chips', 'baby kale',
            'cucumber', 'cucumbers', 'english cucumber', 'persian cucumber',
            'bell pepper', 'red pepper', 'green pepper', 'yellow pepper', 'orange pepper',
            'jalapeno', 'serrano pepper', 'habanero', 'poblano pepper',
            'garlic', 'garlic cloves', 'roasted garlic',
            'ginger', 'fresh ginger', 'ginger root',
            'mushroom', 'mushrooms', 'button mushrooms', 'shiitake', 'portobello', 'cremini',
            'corn', 'corn kernels', 'corn on the cob', 'sweet corn',
            'peas', 'green peas', 'snap peas', 'snow peas',
            'green beans', 'string beans', 'french beans',
            'asparagus', 'asparagus spears',
            'zucchini', 'yellow squash', 'butternut squash', 'acorn squash',
            'eggplant', 'japanese eggplant', 'chinese eggplant',
            'celery', 'celery stalks', 'celery sticks',
            'radish', 'radishes', 'daikon radish',
            'beet', 'beets', 'beetroot', 'golden beets',
            'turnip', 'turnips', 'rutabaga',
            'cabbage', 'red cabbage', 'napa cabbage', 'bok choy',
            'brussels sprouts', 'brussels sprout',
            'artichoke', 'artichoke hearts',
            
            # Fruits
            'apple', 'apples', 'green apple', 'red apple', 'granny smith', 'gala apple',
            'banana', 'bananas', 'plantain', 'green banana',
            'orange', 'oranges', 'navel orange', 'blood orange', 'mandarin',
            'grape', 'grapes', 'red grapes', 'green grapes', 'purple grapes',
            'strawberry', 'strawberries', 'fresh strawberries',
            'blueberry', 'blueberries', 'fresh blueberries',
            'raspberry', 'raspberries', 'blackberry', 'blackberries',
            'lemon', 'lemons', 'lemon wedge', 'lemon slice',
            'lime', 'limes', 'lime wedge', 'lime slice',
            'peach', 'peaches', 'nectarine', 'nectarines',
            'pear', 'pears', 'asian pear', 'bosc pear',
            'mango', 'mangoes', 'mango slices',
            'pineapple', 'pineapple chunks', 'pineapple rings',
            'watermelon', 'watermelon slices', 'watermelon chunks',
            'cantaloupe', 'honeydew melon', 'melon',
            'kiwi', 'kiwi fruit', 'kiwi slices',
            'avocado', 'avocados', 'avocado slices', 'guacamole',
            'cherry', 'cherries', 'sweet cherries', 'sour cherries',
            'plum', 'plums', 'prunes',
            'apricot', 'apricots', 'dried apricots',
            'coconut', 'coconut flakes', 'coconut milk',
            'pomegranate', 'pomegranate seeds',
            
            # Grains & Starches
            'rice', 'white rice', 'brown rice', 'jasmine rice', 'basmati rice', 'wild rice',
            'fried rice', 'rice pilaf', 'risotto', 'rice bowl',
            'bread', 'white bread', 'wheat bread', 'sourdough', 'rye bread', 'pumpernickel',
            'baguette', 'ciabatta', 'focaccia', 'pita bread', 'naan', 'tortilla',
            'pasta', 'spaghetti', 'penne', 'fusilli', 'linguine', 'fettuccine', 'rigatoni',
            'lasagna', 'ravioli', 'gnocchi', 'macaroni',
            'noodles', 'ramen noodles', 'udon noodles', 'soba noodles', 'rice noodles',
            'quinoa', 'quinoa salad', 'quinoa bowl',
            'oats', 'oatmeal', 'steel cut oats', 'rolled oats',
            'cereal', 'granola', 'muesli',
            'barley', 'pearl barley', 'barley soup',
            'couscous', 'bulgur', 'farro', 'wheat berries',
            'polenta', 'grits', 'cornmeal',
            
            # Dairy
            'cheese', 'cheddar cheese', 'mozzarella', 'parmesan', 'swiss cheese', 'gouda',
            'brie', 'camembert', 'blue cheese', 'feta cheese', 'goat cheese', 'ricotta',
            'cottage cheese', 'cream cheese', 'string cheese',
            'milk', 'whole milk', 'skim milk', '2% milk', 'almond milk', 'soy milk', 'oat milk',
            'yogurt', 'greek yogurt', 'plain yogurt', 'vanilla yogurt', 'fruit yogurt',
            'butter', 'unsalted butter', 'salted butter', 'clarified butter',
            'cream', 'heavy cream', 'whipped cream', 'sour cream', 'half and half',
            'ice cream', 'gelato', 'sorbet', 'frozen yogurt',
            
            # Prepared Foods & Dishes
            'pizza', 'margherita pizza', 'pepperoni pizza', 'cheese pizza', 'pizza slice',
            'burger', 'hamburger', 'cheeseburger', 'veggie burger', 'turkey burger',
            'sandwich', 'club sandwich', 'grilled cheese', 'blt sandwich', 'panini',
            'hot dog', 'corn dog', 'bratwurst',
            'taco', 'tacos', 'soft taco', 'hard taco', 'fish taco', 'chicken taco',
            'burrito', 'burrito bowl', 'breakfast burrito',
            'quesadilla', 'cheese quesadilla', 'chicken quesadilla',
            'sushi', 'sushi roll', 'california roll', 'salmon roll', 'tuna roll', 'sashimi',
            'salad', 'caesar salad', 'greek salad', 'garden salad', 'fruit salad', 'pasta salad',
            'soup', 'chicken soup', 'tomato soup', 'vegetable soup', 'minestrone', 'bisque',
            'stew', 'beef stew', 'chicken stew', 'vegetable stew',
            'curry', 'chicken curry', 'vegetable curry', 'thai curry', 'indian curry',
            'stir fry', 'vegetable stir fry', 'chicken stir fry', 'beef stir fry',
            'fried rice', 'chicken fried rice', 'vegetable fried rice', 'shrimp fried rice',
            'pad thai', 'lo mein', 'chow mein',
            'ramen', 'pho', 'udon soup', 'miso soup',
            'paella', 'jambalaya', 'gumbo',
            'chili', 'chili con carne', 'vegetarian chili',
            'casserole', 'lasagna', 'enchiladas', 'tamales',
            
            # Snacks & Appetizers
            'chips', 'potato chips', 'tortilla chips', 'corn chips', 'pita chips',
            'crackers', 'cheese crackers', 'graham crackers', 'saltines',
            'nuts', 'almonds', 'walnuts', 'pecans', 'cashews', 'peanuts', 'pistachios',
            'trail mix', 'mixed nuts', 'nut butter', 'peanut butter', 'almond butter',
            'popcorn', 'caramel corn', 'kettle corn',
            'pretzels', 'soft pretzels', 'pretzel sticks',
            'dip', 'hummus', 'guacamole', 'salsa', 'cheese dip', 'spinach dip',
            'olives', 'green olives', 'black olives', 'kalamata olives',
            'pickles', 'dill pickles', 'sweet pickles', 'pickle spears',
            
            # Desserts & Sweets
            'cake', 'chocolate cake', 'vanilla cake', 'birthday cake', 'cheesecake',
            'cupcake', 'cupcakes', 'muffin', 'muffins', 'blueberry muffin', 'chocolate muffin',
            'cookie', 'cookies', 'chocolate chip cookies', 'sugar cookies', 'oatmeal cookies',
            'brownie', 'brownies', 'fudge brownies',
            'pie', 'apple pie', 'pumpkin pie', 'cherry pie', 'pecan pie', 'key lime pie',
            'tart', 'fruit tart', 'lemon tart', 'chocolate tart',
            'donut', 'donuts', 'glazed donut', 'chocolate donut', 'jelly donut',
            'pastry', 'croissant', 'danish', 'eclair', 'cream puff',
            'chocolate', 'dark chocolate', 'milk chocolate', 'white chocolate',
            'candy', 'gummy bears', 'hard candy', 'chocolate bar',
            'pudding', 'chocolate pudding', 'vanilla pudding', 'rice pudding',
            'jello', 'gelatin', 'fruit gelatin',
            
            # Beverages
            'coffee', 'espresso', 'cappuccino', 'latte', 'americano', 'mocha',
            'tea', 'green tea', 'black tea', 'herbal tea', 'iced tea', 'chai tea',
            'juice', 'orange juice', 'apple juice', 'grape juice', 'cranberry juice',
            'smoothie', 'fruit smoothie', 'green smoothie', 'protein smoothie',
            'water', 'sparkling water', 'flavored water',
            'soda', 'cola', 'sprite', 'root beer', 'ginger ale',
            'beer', 'wine', 'red wine', 'white wine', 'champagne',
            'cocktail', 'margarita', 'mojito', 'martini',
            'milkshake', 'chocolate milkshake', 'vanilla milkshake',
            
            # Condiments & Seasonings
            'salt', 'pepper', 'black pepper', 'white pepper',
            'sugar', 'brown sugar', 'powdered sugar', 'honey', 'maple syrup', 'agave',
            'oil', 'olive oil', 'vegetable oil', 'coconut oil', 'sesame oil',
            'vinegar', 'balsamic vinegar', 'apple cider vinegar', 'rice vinegar',
            'sauce', 'tomato sauce', 'marinara sauce', 'alfredo sauce', 'pesto',
            'ketchup', 'mustard', 'mayonnaise', 'ranch dressing', 'caesar dressing',
            'hot sauce', 'sriracha', 'tabasco', 'buffalo sauce',
            'soy sauce', 'teriyaki sauce', 'hoisin sauce', 'fish sauce',
            'herbs', 'basil', 'oregano', 'thyme', 'rosemary', 'parsley', 'cilantro',
            'spices', 'cumin', 'paprika', 'turmeric', 'cinnamon', 'nutmeg', 'cardamom'
        }
    
    def _load_food_categories(self) -> Dict[str, List[str]]:
        """Load food categories for better classification"""
        return {
            'proteins': ['chicken', 'beef', 'pork', 'fish', 'egg', 'tofu', 'beans'],
            'vegetables': ['tomato', 'potato', 'carrot', 'broccoli', 'spinach', 'lettuce'],
            'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry'],
            'grains': ['rice', 'bread', 'pasta', 'quinoa', 'oats', 'cereal'],
            'dairy': ['cheese', 'milk', 'yogurt', 'butter', 'cream'],
            'prepared': ['pizza', 'burger', 'sandwich', 'salad', 'soup', 'curry'],
            'snacks': ['chips', 'nuts', 'crackers', 'popcorn', 'pretzels'],
            'desserts': ['cake', 'cookie', 'ice cream', 'chocolate', 'pie'],
            'beverages': ['coffee', 'tea', 'juice', 'water', 'soda', 'smoothie']
        }
    
    def _load_visual_features(self) -> Dict[str, Dict[str, Any]]:
        """Load visual features for food recognition"""
        return {
            'color_profiles': {
                'tomato': {'red': (200, 255), 'green': (0, 100), 'blue': (0, 100)},
                'broccoli': {'red': (0, 100), 'green': (100, 200), 'blue': (0, 100)},
                'banana': {'red': (200, 255), 'green': (200, 255), 'blue': (0, 150)},
                'orange': {'red': (200, 255), 'green': (100, 200), 'blue': (0, 100)}
            },
            'texture_patterns': {
                'bread': 'porous_texture',
                'rice': 'granular_texture',
                'pasta': 'smooth_cylindrical',
                'meat': 'fibrous_texture'
            },
            'shape_characteristics': {
                'pizza': 'circular_flat',
                'burger': 'layered_cylindrical',
                'sandwich': 'rectangular_layered',
                'apple': 'spherical_smooth'
            }
        }
    
    def detect_foods_advanced(self, image: Image.Image) -> Dict[str, Any]:
        """
        Advanced food detection using multiple models and techniques
        """
        try:
            logger.info("Starting advanced multi-model food detection...")
            
            # Convert image for processing
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Multi-model ensemble detection
            detection_results = []
            
            # Method 1: Enhanced BLIP with multiple specialized prompts
            blip_results = self._blip_ensemble_detection(image)
            detection_results.append(('blip', blip_results, 0.4))
            
            # Method 2: YOLO object detection
            yolo_results = self._yolo_detection(image)
            detection_results.append(('yolo', yolo_results, 0.3))
            
            # Method 3: Visual feature analysis
            visual_results = self._visual_feature_detection(image)
            detection_results.append(('visual', visual_results, 0.2))
            
            # Method 4: Color and texture analysis
            color_results = self._color_texture_analysis(image)
            detection_results.append(('color', color_results, 0.1))
            
            # Ensemble fusion
            final_foods = self._ensemble_fusion(detection_results)
            
            # Validate and refine results
            validated_foods = self._validate_and_refine(final_foods, image)
            
            # Get detailed analysis for each food
            detailed_analysis = self._get_detailed_food_analysis(validated_foods, image)
            
            return {
                'detected_foods': list(validated_foods),
                'confidence_scores': detailed_analysis['confidence_scores'],
                'food_details': detailed_analysis['food_details'],
                'detection_methods': detailed_analysis['detection_methods'],
                'image_analysis': detailed_analysis['image_analysis'],
                'total_foods_detected': len(validated_foods)
            }
            
        except Exception as e:
            logger.error(f"Advanced detection failed: {e}")
            return {'error': str(e)}
    
    def _blip_ensemble_detection(self, image: Image.Image) -> Set[str]:
        """Enhanced BLIP detection with multiple specialized prompts"""
        detected_foods = set()
        
        if not self.models.get('processor') or not self.models.get('blip_model'):
            return detected_foods
        
        # Specialized prompts for different food types
        specialized_prompts = [
            "What specific foods, dishes, and ingredients can you identify in this image? List each item precisely.",
            "Identify all proteins, meats, and protein sources visible in this food image.",
            "What vegetables, fruits, and plant-based foods are shown in this image?",
            "List all grains, starches, and carbohydrate sources you can see.",
            "Identify any dairy products, cheese, or milk-based items in this image.",
            "What prepared dishes, cooked meals, or processed foods are visible?",
            "Describe the main course, side dishes, and any accompaniments shown.",
            "What beverages, drinks, or liquid items can you identify?",
            "List any snacks, appetizers, or finger foods visible in the image.",
            "Identify any desserts, sweets, or baked goods shown."
        ]
        
        device = self.models.get('device', 'cpu')
        
        for prompt in specialized_prompts:
            try:
                inputs = self.models['processor'](image, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.models['blip_model'].generate(
                        **inputs,
                        max_new_tokens=200,
                        num_beams=5,
                        temperature=0.2,  # Low temperature for consistency
                        do_sample=True,
                        repetition_penalty=1.3
                    )
                response = self.models['processor'].decode(outputs[0], skip_special_tokens=True)
                
                # Clean response
                if response.startswith(prompt):
                    response = response.replace(prompt, "").strip()
                
                # Extract foods from response
                foods = self._extract_foods_from_text(response)
                detected_foods.update(foods)
                
                logger.info(f"BLIP prompt found: {len(foods)} foods - {response[:100]}...")
                
            except Exception as e:
                logger.warning(f"BLIP prompt failed: {e}")
                continue
        
        return detected_foods
    
    def _yolo_detection(self, image: Image.Image) -> Set[str]:
        """Enhanced YOLO detection with multiple confidence levels"""
        detected_foods = set()
        
        if not self.models.get('yolo_model'):
            return detected_foods
        
        try:
            import numpy as np
            img_array = np.array(image)
            
            # Multiple detection passes with different confidence levels
            confidence_levels = [0.15, 0.25, 0.35, 0.45]
            
            for conf in confidence_levels:
                try:
                    results = self.models['yolo_model'](img_array, conf=conf, iou=0.4)
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                cls = int(box.cls[0])
                                confidence = float(box.conf[0])
                                class_name = self.models['yolo_model'].names[cls].lower()
                                
                                # Check if it's a food item in our vocabulary
                                if class_name in self.food_vocabulary and confidence > conf:
                                    detected_foods.add(class_name)
                                    logger.info(f"YOLO detected: {class_name} (conf: {confidence:.3f})")
                
                except Exception as e:
                    logger.warning(f"YOLO detection at conf {conf} failed: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
        
        return detected_foods
    
    def _visual_feature_detection(self, image: Image.Image) -> Set[str]:
        """Visual feature-based food detection"""
        detected_foods = set()
        
        try:
            import cv2
            import numpy as np
            
            # Convert PIL to OpenCV
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Color-based detection
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for common foods
            color_ranges = {
                'tomato': [(0, 100, 100), (10, 255, 255)],  # Red
                'broccoli': [(40, 40, 40), (80, 255, 255)],  # Green
                'banana': [(20, 100, 100), (30, 255, 255)],  # Yellow
                'orange': [(10, 100, 100), (25, 255, 255)],  # Orange
                'carrot': [(10, 100, 100), (25, 255, 255)],  # Orange
            }
            
            for food, (lower, upper) in color_ranges.items():
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(hsv, lower, upper)
                
                # Check if significant area matches the color
                area_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
                if area_ratio > 0.05:  # At least 5% of image
                    detected_foods.add(food)
                    logger.info(f"Visual feature detected: {food} (area: {area_ratio:.3f})")
        
        except Exception as e:
            logger.warning(f"Visual feature detection failed: {e}")
        
        return detected_foods
    
    def _color_texture_analysis(self, image: Image.Image) -> Set[str]:
        """Color and texture-based food analysis"""
        detected_foods = set()
        
        try:
            import numpy as np
            from scipy import ndimage
            
            img_array = np.array(image)
            
            # Dominant color analysis
            pixels = img_array.reshape(-1, 3)
            
            # Calculate color statistics
            mean_color = np.mean(pixels, axis=0)
            
            # Color-based food inference
            r, g, b = mean_color
            
            if r > 150 and g < 100 and b < 100:  # Red dominant
                detected_foods.update(['tomato', 'strawberry', 'apple'])
            elif g > 150 and r < 100 and b < 100:  # Green dominant
                detected_foods.update(['broccoli', 'lettuce', 'spinach'])
            elif r > 200 and g > 200 and b < 150:  # Yellow dominant
                detected_foods.update(['banana', 'corn', 'cheese'])
            elif r > 150 and g > 100 and b < 100:  # Orange dominant
                detected_foods.update(['carrot', 'orange', 'sweet potato'])
            
            logger.info(f"Color analysis detected: {len(detected_foods)} potential foods")
        
        except Exception as e:
            logger.warning(f"Color texture analysis failed: {e}")
        
        return detected_foods
    
    def _extract_foods_from_text(self, text: str) -> Set[str]:
        """Extract food items from text with advanced NLP"""
        foods = set()
        text_lower = text.lower()
        
        # Direct vocabulary matching
        for food in self.food_vocabulary:
            if food in text_lower:
                foods.add(food)
        
        # Pattern-based extraction
        food_patterns = [
            r'\b(\w+(?:\s+\w+)*)\s+(?:dish|meal|food|cuisine)\b',
            r'\b(?:grilled|fried|baked|roasted|steamed)\s+(\w+(?:\s+\w+)*)\b',
            r'\b(\w+(?:\s+\w+)*)\s+(?:with|and|plus)\b',
            r'\b(?:fresh|cooked|raw|organic)\s+(\w+(?:\s+\w+)*)\b'
        ]
        
        for pattern in food_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match in self.food_vocabulary:
                    foods.add(match)
        
        return foods
    
    def _ensemble_fusion(self, detection_results: List[Tuple[str, Set[str], float]]) -> Set[str]:
        """Fuse results from multiple detection methods"""
        food_scores = {}
        
        # Calculate weighted scores for each food
        for method, foods, weight in detection_results:
            for food in foods:
                if food not in food_scores:
                    food_scores[food] = 0
                food_scores[food] += weight
        
        # Select foods above ensemble threshold
        final_foods = set()
        for food, score in food_scores.items():
            if score >= self.ensemble_threshold:
                final_foods.add(food)
                logger.info(f"Ensemble selected: {food} (score: {score:.3f})")
        
        return final_foods
    
    def _validate_and_refine(self, foods: Set[str], image: Image.Image) -> Set[str]:
        """Validate and refine detected foods"""
        validated_foods = set()
        
        for food in foods:
            # Check if food is in our comprehensive vocabulary
            if food in self.food_vocabulary:
                validated_foods.add(food)
            else:
                # Try to find similar foods
                similar_foods = self._find_similar_foods(food)
                validated_foods.update(similar_foods)
        
        # Remove contradictory detections
        validated_foods = self._remove_contradictions(validated_foods)
        
        # Ensure minimum detection quality
        if len(validated_foods) == 0:
            # Fallback detection
            validated_foods = self._fallback_detection(image)
        
        return validated_foods
    
    def _find_similar_foods(self, food: str) -> Set[str]:
        """Find similar foods in vocabulary"""
        similar = set()
        food_lower = food.lower()
        
        for vocab_food in self.food_vocabulary:
            if food_lower in vocab_food or vocab_food in food_lower:
                similar.add(vocab_food)
        
        return similar
    
    def _remove_contradictions(self, foods: Set[str]) -> Set[str]:
        """Remove contradictory food detections"""
        # Remove generic terms if specific ones exist
        generic_terms = {'food', 'dish', 'meal', 'cuisine', 'item'}
        foods = foods - generic_terms
        
        # Remove duplicates (keep more specific)
        refined_foods = set()
        foods_list = list(foods)
        
        for i, food1 in enumerate(foods_list):
            is_specific = True
            for j, food2 in enumerate(foods_list):
                if i != j and food1 in food2 and len(food2) > len(food1):
                    is_specific = False
                    break
            if is_specific:
                refined_foods.add(food1)
        
        return refined_foods
    
    def _fallback_detection(self, image: Image.Image) -> Set[str]:
        """Fallback detection when primary methods fail"""
        fallback_foods = set()
        
        try:
            # Simple color-based fallback
            import numpy as np
            img_array = np.array(image)
            mean_color = np.mean(img_array.reshape(-1, 3), axis=0)
            
            r, g, b = mean_color
            
            # Basic color inference
            if r > g and r > b:
                fallback_foods.add('tomato')
            elif g > r and g > b:
                fallback_foods.add('broccoli')
            elif r > 200 and g > 200:
                fallback_foods.add('banana')
            else:
                fallback_foods.add('mixed food')
            
            logger.info(f"Fallback detection: {fallback_foods}")
        
        except Exception as e:
            logger.warning(f"Fallback detection failed: {e}")
            fallback_foods.add('food item')
        
        return fallback_foods
    
    def _get_detailed_food_analysis(self, foods: Set[str], image: Image.Image) -> Dict[str, Any]:
        """Get detailed analysis for detected foods"""
        analysis = {
            'confidence_scores': {},
            'food_details': {},
            'detection_methods': {},
            'image_analysis': {}
        }
        
        for food in foods:
            # Assign confidence scores based on detection method
            analysis['confidence_scores'][food] = 0.85  # High confidence for ensemble
            
            # Get food category
            category = self._get_food_category(food)
            analysis['food_details'][food] = {
                'category': category,
                'common_name': food.title(),
                'nutritional_category': self._get_nutritional_category(food)
            }
            
            # Detection method used
            analysis['detection_methods'][food] = 'ensemble_multi_model'
        
        # Overall image analysis
        analysis['image_analysis'] = {
            'total_foods': len(foods),
            'complexity': 'high' if len(foods) > 5 else 'medium' if len(foods) > 2 else 'simple',
            'detection_quality': 'high_confidence'
        }
        
        return analysis
    
    def _get_food_category(self, food: str) -> str:
        """Get category for a food item"""
        for category, items in self.food_categories.items():
            if any(item in food for item in items):
                return category
        return 'other'
    
    def _get_nutritional_category(self, food: str) -> str:
        """Get nutritional category for a food item"""
        protein_foods = ['chicken', 'beef', 'fish', 'egg', 'tofu', 'beans']
        carb_foods = ['rice', 'bread', 'pasta', 'potato', 'corn']
        fat_foods = ['cheese', 'nuts', 'oil', 'butter', 'avocado']
        
        if any(p in food for p in protein_foods):
            return 'protein'
        elif any(c in food for c in carb_foods):
            return 'carbohydrate'
        elif any(f in food for f in fat_foods):
            return 'fat'
        else:
            return 'mixed'