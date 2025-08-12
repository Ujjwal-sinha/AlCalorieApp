import logging
import re
from typing import Dict, Any, List, Tuple
from PIL import Image
import uuid

# Try to import torch for model operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

def validate_food_items(items: set, context: str) -> set:
    """Validate detected food items to ensure accuracy and remove false positives."""
    validated_items = set()
    
    # Comprehensive list of actual food items
    valid_food_items = {
        # Proteins
        'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'fish', 'salmon', 'tuna', 'cod',
        'shrimp', 'crab', 'lobster', 'egg', 'eggs', 'tofu', 'tempeh', 'bacon', 'sausage', 'ham',
        'steak', 'ground beef', 'chicken breast', 'chicken thigh', 'pork chop', 'lamb chop',
        
        # Vegetables
        'tomato', 'tomatoes', 'potato', 'potatoes', 'carrot', 'carrots', 'onion', 'onions',
        'broccoli', 'cauliflower', 'lettuce', 'spinach', 'kale', 'cucumber', 'bell pepper',
        'peppers', 'garlic', 'ginger', 'mushroom', 'mushrooms', 'corn', 'peas', 'beans',
        'green beans', 'asparagus', 'zucchini', 'eggplant', 'celery', 'radish', 'beet',
        'sweet potato', 'cabbage', 'brussels sprouts', 'artichoke',
        
        # Fruits
        'apple', 'apples', 'banana', 'bananas', 'orange', 'oranges', 'grape', 'grapes',
        'strawberry', 'strawberries', 'blueberry', 'blueberries', 'lemon', 'lime', 'peach',
        'pear', 'mango', 'pineapple', 'watermelon', 'cantaloupe', 'kiwi', 'avocado',
        'cherry', 'cherries', 'plum', 'apricot', 'coconut',
        
        # Grains & Starches
        'rice', 'bread', 'pasta', 'noodles', 'quinoa', 'oats', 'oatmeal', 'cereal',
        'wheat', 'barley', 'couscous', 'bulgur', 'tortilla', 'bagel', 'croissant',
        
        # Dairy
        'cheese', 'milk', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese',
        'mozzarella', 'cheddar', 'parmesan', 'feta', 'ricotta',
        
        # Prepared Foods
        'pizza', 'burger', 'sandwich', 'salad', 'soup', 'stew', 'curry', 'pasta',
        'spaghetti', 'lasagna', 'tacos', 'burrito', 'sushi', 'ramen', 'stir fry',
        
        # Beverages
        'water', 'juice', 'coffee', 'tea', 'milk', 'soda', 'beer', 'wine', 'smoothie',
        
        # Condiments & Seasonings
        'salt', 'pepper', 'oil', 'olive oil', 'butter', 'sauce', 'ketchup', 'mustard',
        'mayonnaise', 'vinegar', 'soy sauce', 'hot sauce', 'herbs', 'spices',
        
        # Desserts
        'cake', 'cookie', 'cookies', 'ice cream', 'chocolate', 'candy', 'pie', 'pastry'
    }
    
    # Items that are definitely NOT food
    non_food_items = {
        'plate', 'bowl', 'cup', 'glass', 'fork', 'knife', 'spoon', 'napkin', 'table',
        'chair', 'wall', 'background', 'surface', 'container', 'dish', 'utensil',
        'cutlery', 'placemat', 'tablecloth', 'decoration', 'garnish', 'presentation',
        'lighting', 'shadow', 'reflection', 'texture', 'color', 'pattern', 'style',
        'arrangement', 'display', 'setting', 'environment', 'scene', 'photo', 'image'
    }
    
    for item in items:
        item_clean = item.strip().lower()
        
        # Skip if too short or empty
        if len(item_clean) < 3:
            continue
            
        # Skip if it's a non-food item
        if any(non_food in item_clean for non_food in non_food_items):
            continue
            
        # Include if it's a known food item
        if any(food in item_clean for food in valid_food_items):
            validated_items.add(item_clean)
            continue
            
        # Include if it contains food-related keywords and context supports it
        food_keywords = ['meat', 'vegetable', 'fruit', 'grain', 'dairy', 'protein', 'sauce', 'seasoning']
        if any(keyword in item_clean for keyword in food_keywords):
            validated_items.add(item_clean)
            continue
            
        # Include if the context strongly suggests it's food
        if any(food_context in context.lower() for food_context in ['cooked', 'fried', 'grilled', 'baked', 'fresh', 'seasoned']):
            if len(item_clean) > 3 and not any(non_food in item_clean for non_food in non_food_items):
                validated_items.add(item_clean)
    
    return validated_items

def extract_food_items_from_text(text: str) -> set:
    """Extract individual food items from descriptive text with comprehensive parsing."""
    items = set()
    text = text.lower().strip()
    
    # Remove common prefixes and phrases
    prefixes_to_remove = [
        "a photo of", "an image of", "this image shows", "i can see", "there is", "there are",
        "the image contains", "visible in the image", "in this image", "this appears to be",
        "looking at this", "from what i can see", "it looks like", "this seems to be",
        "i can see", "there appears to be", "the image shows", "this picture shows",
        "visible are", "present are", "shown are", "included are", "featured are"
    ]
    
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text.replace(prefix, "").strip()
    
    # Enhanced separators for better splitting
    separators = [
        ',', ';', ' and ', ' with ', ' including ', ' plus ', ' also ', ' as well as ',
        ' along with ', ' together with ', ' accompanied by ', ' served with ', ' topped with ',
        ' garnished with ', ' mixed with ', ' combined with ', ' containing ', ' featuring ',
        ' such as ', ' like ', ' for example ', ' e.g. ', ' furthermore ', ' additionally ',
        ' moreover ', ' besides ', ' in addition ', ' what\'s more ', ' on top of that '
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
        'probably', 'possibly', 'maybe', 'perhaps', 'likely', 'unlikely', 'just',
        'only', 'even', 'still', 'already', 'yet', 'again', 'ever', 'never', 'always',
        'usually', 'often', 'sometimes', 'rarely', 'seldom', 'hardly', 'barely',
        'almost', 'nearly', 'about', 'around', 'approximately', 'roughly', 'exactly',
        'precisely', 'definitely', 'certainly', 'surely', 'obviously', 'clearly',
        'apparently', 'evidently', 'seemingly', 'supposedly', 'allegedly', 'reportedly'
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
            r'^(a|an|some|many|few|several|various|different|fresh|cooked|raw|fried|grilled|baked|roasted|steamed|boiled|sauteed|stir-fried|deep-fried|pan-fried|air-fried)\s+',
            r'^(small|medium|large|big|huge|tiny|little|sliced|diced|chopped|minced|whole|half|quarter|crushed|grated|shredded|julienned|cubed|wedged|spiralized)\s+',
            r'^(hot|cold|warm|cool|spicy|mild|sweet|sour|salty|bitter|savory|delicious|tasty|flavorful|aromatic|fresh|ripe|unripe|overripe|firm|soft|crunchy|crispy|tender|juicy|dry|moist|creamy|smooth|chunky|thick|thin)\s+',
            r'^\d+\s*(pieces?|slices?|cups?|tablespoons?|teaspoons?|ounces?|grams?|pounds?|lbs?|oz|g|kg|ml|l|tbsp|tsp|pinch|dash|handful|bunch|clove|head|stalk|sprig)\s+(of\s+)?',
            r'^(organic|natural|artificial|synthetic|homemade|store-bought|pre-packaged|frozen|canned|dried|dehydrated|fermented|pickled|smoked|cured|aged|fresh|ripe|green|yellow|red|orange|purple|brown|white|black)\s+'
        ]
        
        for pattern in quantity_patterns:
            part = re.sub(pattern, '', part, flags=re.IGNORECASE).strip()
        
        # Skip if became too short after cleaning
        if len(part) <= 2:
            continue
        
        # Additional cleaning for common food-related phrases
        part = re.sub(r'\s+', ' ', part)  # Normalize whitespace
        part = part.strip()
        
        # Skip if empty after cleaning
        if not part:
            continue
        
        # Add the cleaned food item
        items.add(part)
    
    return items

def describe_image_enhanced(image: Image.Image, models: Dict[str, Any]) -> str:
    """Enhanced food detection with multiple strategies for better accuracy."""
    if not models.get('processor') or not models.get('blip_model'):
        return "Image analysis unavailable. Please check model loading."
    
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        device = models.get('device')
        if not device:
            return "Device not available for image processing."
        
        logger.info("Starting enhanced food detection...")
        
        # Strategy 1: Direct food identification with comprehensive prompts
        food_prompts = [
            "What food is in this image?",
            "Describe the food items you can see.",
            "What are the main ingredients or dishes shown?",
            "List the foods visible in this picture.",
            "What meal or food items are displayed?",
            "What vegetables can you see?",
            "What fruits are visible?",
            "What proteins or meat items are shown?",
            "What grains or carbohydrates are present?",
            "What dairy products can you identify?",
            "What beverages or drinks are visible?",
            "What spices or seasonings can you see?",
            "What nuts or seeds are present?",
            "What prepared foods or dishes are shown?",
            "What desserts or sweets are visible?"
        ]
        
        all_detected_foods = set()
        
        for prompt in food_prompts:
            try:
                inputs = models['processor'](image, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = models['blip_model'].generate(
                        **inputs,
                        max_new_tokens=100,
                        num_beams=3,
                        do_sample=False  # Disable sampling for more deterministic results
                    )
                response = models['processor'].decode(outputs[0], skip_special_tokens=True)
                
                # Clean the response
                if response.startswith(prompt):
                    response = response.replace(prompt, "").strip()
                
                logger.info(f"Prompt '{prompt}' response: {response}")
                
                # Extract foods from response
                foods = extract_food_items_from_text(response)
                all_detected_foods.update(foods)
                
            except Exception as e:
                logger.warning(f"Prompt '{prompt}' failed: {e}")
                continue
        
        # Strategy 2: Vision Transformer detection (ViT-B/16 and Swin Transformer)
        vit_foods = set()
        if models.get('vit_model') or models.get('swin_model'):
            try:
                from utils.vision_transformer_detection import VisionTransformerFoodDetector
                
                vit_detector = VisionTransformerFoodDetector(models)
                vit_results = vit_detector.detect_food_with_transformers(image)
                
                if vit_results.get('success', False):
                    vit_foods = set(vit_results.get('detected_foods', []))
                    logger.info(f"Vision Transformer detected: {vit_foods}")
                    logger.info(f"Models used: {vit_results.get('models_used', [])}")
                    logger.info(f"Detection method: {vit_results.get('detection_method', 'unknown')}")
                else:
                    logger.warning(f"Vision Transformer detection failed: {vit_results.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.warning(f"Vision Transformer detection failed: {e}")
        
        # Strategy 3: YOLO detection as fallback
        yolo_foods = set()
        if models.get('yolo_model'):
            try:
                import numpy as np
                img_array = np.array(image)
                
                # Try multiple confidence levels
                for conf_level in [0.1, 0.2, 0.3]:
                    try:
                        results = models['yolo_model'](img_array, conf=conf_level)
                        
                        # YOLO COCO classes that are food items (expanded)
                        food_classes = {
                            'apple', 'banana', 'sandwich', 'orange', 'broccoli', 'carrot', 
                            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'dining table',
                            'cup', 'fork', 'knife', 'spoon', 'bowl', 'wine glass', 'bottle',
                            'cell phone', 'laptop', 'mouse', 'remote', 'keyboard', 'book',
                            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                        }
                        
                        # Expanded actual food items (including utensils and containers that might contain food)
                        actual_food_classes = {
                            'apple', 'banana', 'sandwich', 'orange', 'broccoli', 'carrot', 
                            'hot dog', 'pizza', 'donut', 'cake', 'cup', 'bowl', 'wine glass', 'bottle',
                            'fork', 'knife', 'spoon'  # Utensils indicate food presence
                        }
                        
                        for result in results:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    cls = int(box.cls[0])
                                    confidence = float(box.conf[0])
                                    class_name = models['yolo_model'].names[cls].lower()
                                    
                                    if class_name in actual_food_classes and confidence > conf_level:
                                        yolo_foods.add(class_name)
                                        logger.info(f"YOLO detected: {class_name} (confidence: {confidence:.3f})")
                        
                        if yolo_foods:  # If we found foods, use this confidence level
                            break
                            
                    except Exception as e:
                        logger.warning(f"YOLO at confidence {conf_level} failed: {e}")
                        continue
                
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
        
        # Strategy 3: Image captioning without specific prompts
        try:
            inputs = models['processor'](image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = models['blip_model'].generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=3,
                    do_sample=False
                )
            caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Image caption: {caption}")
            
            # Extract foods from caption
            caption_foods = extract_food_items_from_text(caption)
            all_detected_foods.update(caption_foods)
            
        except Exception as e:
            logger.warning(f"Image captioning failed: {e}")
        
        # Combine all detection methods with comprehensive strategy
        combined_foods = set()
        detection_sources = []
        food_sources = {}  # Track which model detected which food
        
        # Priority 1: Vision Transformer results (highest confidence)
        if vit_foods:
            combined_foods.update(vit_foods)
            detection_sources.append("Vision Transformer")
            for food in vit_foods:
                food_sources[food] = food_sources.get(food, []) + ["ViT"]
            logger.info(f"Added ViT foods: {vit_foods}")
        
        # Priority 2: BLIP prompt results (add all, don't filter)
        if all_detected_foods:
            combined_foods.update(all_detected_foods)
            detection_sources.append("BLIP Prompts")
            for food in all_detected_foods:
                food_sources[food] = food_sources.get(food, []) + ["BLIP"]
            logger.info(f"Added BLIP foods: {all_detected_foods}")
        
        # Priority 3: YOLO results (add all, don't filter)
        if yolo_foods:
            combined_foods.update(yolo_foods)
            detection_sources.append("YOLO")
            for food in yolo_foods:
                food_sources[food] = food_sources.get(food, []) + ["YOLO"]
            logger.info(f"Added YOLO foods: {yolo_foods}")
        
        # NO HARDCODED VALUES: Only use real model predictions
        logger.info("Using only real model predictions - no color-based fallback")
        if len(combined_foods) == 0:
            logger.warning("No real detections found - this indicates a model issue")
        
        # Validate detected foods but be more permissive
        validated_foods = validate_food_items(combined_foods, "food image")
        
        # If validation is too strict, include more foods
        if len(validated_foods) < len(combined_foods) * 0.7:  # If we lost more than 30%
            logger.info("Validation too strict, including more foods")
            validated_foods = combined_foods  # Include all detected foods
        
        if validated_foods:
            # Don't limit the number of foods - include all detected
            food_list = sorted(list(validated_foods))
            detection_method = " + ".join(detection_sources) if detection_sources else "multiple models"
            
            # Create detailed result with source information
            detailed_foods = []
            for food in food_list:
                sources = food_sources.get(food, ["Unknown"])
                detailed_foods.append(f"{food} ({', '.join(sources)})")
            
            result = f"Detected foods: {', '.join(detailed_foods)} (via {detection_method})"
            logger.info(f"Final detection result: {result}")
            logger.info(f"Detection sources: {detection_sources}")
            logger.info(f"Total foods detected: {len(food_list)}")
            return result
        
        # NO HARDCODED VALUES: Only return real detections
        if len(combined_foods) == 0:
            return "No food items detected. Please try a clearer image with better lighting."
        else:
            return "Food detection completed with available models."
        
    except Exception as e:
        logger.error(f"Enhanced food detection error: {e}")
        return "Food detection failed. Please try a different image."

def _detect_common_foods_by_color(image: Image.Image) -> List[str]:
    """Detect common foods based on dominant colors as fallback"""
    try:
        import numpy as np
        from collections import Counter
        
        # Convert to numpy array
        img_array = np.array(image.resize((100, 100)))  # Resize for faster processing
        
        # Get dominant colors
        pixels = img_array.reshape(-1, 3)
        
        # Calculate average color
        avg_color = np.mean(pixels, axis=0)
        r, g, b = avg_color
        
        detected_foods = []
        
        # Comprehensive color-based food detection
        if r > 150 and g < 100 and b < 100:  # Red dominant
            detected_foods.extend(['tomato', 'apple', 'strawberry', 'red pepper', 'beet', 'cherry', 'raspberry', 'red onion', 'red cabbage'])
        elif g > 150 and r < 120 and b < 120:  # Green dominant
            detected_foods.extend(['broccoli', 'lettuce', 'spinach', 'green beans', 'cucumber', 'zucchini', 'kale', 'green pepper', 'peas', 'asparagus'])
        elif r > 200 and g > 180 and b < 100:  # Yellow dominant
            detected_foods.extend(['banana', 'corn', 'cheese', 'lemon', 'pineapple', 'mango', 'yellow pepper', 'squash', 'yellow onion'])
        elif r > 150 and g > 100 and b < 80:  # Orange dominant
            detected_foods.extend(['carrot', 'orange', 'sweet potato', 'pumpkin', 'apricot', 'peach', 'butternut squash', 'orange pepper'])
        elif r > 100 and g > 80 and b > 60:  # Brown dominant
            detected_foods.extend(['bread', 'chicken', 'potato', 'coffee', 'chocolate', 'beef', 'mushroom', 'nuts', 'grains', 'rice'])
        elif r > 200 and g > 200 and b > 200:  # White dominant
            detected_foods.extend(['rice', 'milk', 'yogurt', 'cauliflower', 'pasta', 'fish', 'egg', 'onion', 'garlic', 'mushroom'])
        elif r > 100 and g > 100 and b > 150:  # Blue/Purple dominant
            detected_foods.extend(['blueberry', 'eggplant', 'grape', 'plum', 'purple cabbage', 'purple onion', 'purple potato'])
        elif r > 180 and g > 150 and b > 100:  # Pink/Peach dominant
            detected_foods.extend(['salmon', 'shrimp', 'pink grapefruit', 'watermelon', 'pink pepper', 'pink onion'])
        else:
            # Mixed colors - suggest comprehensive meal components
            detected_foods.extend(['mixed food', 'meal', 'dish', 'salad', 'soup', 'stew', 'curry', 'stir fry', 'casserole'])
        
        # Add generic food categories for comprehensive coverage
        detected_foods.extend(['vegetables', 'protein', 'grains', 'fruits', 'dairy', 'beverages', 'spices', 'herbs'])
        
        return detected_foods[:10]  # Return top 10 for comprehensive coverage
        
    except Exception as e:
        logger.warning(f"Color-based detection failed: {e}")
        return ['food item']
        
        # Strategy 2: Ultra-Enhanced YOLO Detection with Multiple Passes
        yolo_detected_items = set()
        if models.get('yolo_model') and models.get('NUMPY_AVAILABLE'):
            try:
                import numpy as np
                img_np = np.array(image)
                
                # Balanced detection passes for accuracy and coverage
                detection_configs = [
                    {'conf': 0.25, 'iou': 0.45},  # High confidence for accuracy
                    {'conf': 0.20, 'iou': 0.40},  # Medium confidence
                    {'conf': 0.15, 'iou': 0.35},  # Lower confidence for coverage
                ]
                
                # Comprehensive food database for maximum detection
                comprehensive_food_items = {
                    # Fruits
                    'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'lemon', 'lime',
                    'peach', 'pear', 'mango', 'pineapple', 'watermelon', 'cantaloupe', 'kiwi', 'avocado',
                    'cherry', 'plum', 'apricot', 'nectarine', 'fig', 'date', 'raisin', 'cranberry',
                    'raspberry', 'blackberry', 'gooseberry', 'currant', 'pomegranate', 'guava', 'papaya',
                    'dragon fruit', 'star fruit', 'lychee', 'longan', 'rambutan', 'durian', 'jackfruit',
                    'breadfruit', 'plantain', 'persimmon', 'quince', 'medlar', 'loquat', 'mulberry',
                    # Vegetables
                    'tomato', 'potato', 'carrot', 'onion', 'broccoli', 'cauliflower', 'lettuce', 'spinach',
                    'cucumber', 'bell pepper', 'jalapeno', 'garlic', 'ginger', 'mushroom', 'corn', 'peas',
                    'beans', 'asparagus', 'zucchini', 'eggplant', 'celery', 'radish', 'beet', 'turnip',
                    'sweet potato', 'yam', 'taro', 'cassava', 'parsnip', 'rutabaga', 'kohlrabi', 'fennel',
                    'artichoke', 'brussels sprout', 'kale', 'collard greens', 'swiss chard', 'arugula',
                    'watercress', 'endive', 'escarole', 'bok choy', 'napa cabbage', 'savoy cabbage',
                    'red cabbage', 'green cabbage', 'white cabbage', 'chinese cabbage', 'pak choi',
                    'mustard greens', 'turnip greens', 'beet greens', 'dandelion greens', 'purslane',
                    # Proteins
                    'chicken', 'fish', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'egg', 'tofu', 'tempeh',
                    'shrimp', 'crab', 'lobster', 'salmon', 'tuna', 'cod', 'tilapia', 'bacon', 'sausage',
                    'ham', 'prosciutto', 'salami', 'pepperoni', 'chorizo', 'pastrami', 'corned beef',
                    'roast beef', 'steak', 'ground beef', 'mince', 'liver', 'kidney', 'heart', 'tongue',
                    'tripe', 'oxtail', 'short ribs', 'brisket', 'flank steak', 'skirt steak', 'hanger steak',
                    'flat iron steak', 'ribeye', 'strip steak', 'tenderloin', 'filet mignon', 'porterhouse',
                    't-bone', 'sirloin', 'round steak', 'chuck roast', 'shoulder roast', 'leg of lamb',
                    'lamb chops', 'lamb shank', 'goat', 'venison', 'bison', 'elk', 'rabbit', 'quail',
                    'pheasant', 'partridge', 'guinea fowl', 'geese', 'pigeon', 'squab', 'ostrich',
                    'emu', 'kangaroo', 'crocodile', 'alligator', 'frog legs', 'snail', 'escargot',
                    # Dairy
                    'cheese', 'milk', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese',
                    'ricotta', 'mozzarella', 'cheddar', 'parmesan', 'gouda', 'brie', 'camembert',
                    'blue cheese', 'feta', 'halloumi', 'paneer', 'queso fresco', 'manchego', 'pecorino',
                    'asiago', 'provolone', 'swiss cheese', 'havarti', 'muenster', 'colby', 'monterey jack',
                    'pepper jack', 'string cheese', 'cream cheese', 'mascarpone', 'quark', 'kefir',
                    'buttermilk', 'evaporated milk', 'condensed milk', 'powdered milk', 'almond milk',
                    'soy milk', 'oat milk', 'coconut milk', 'rice milk', 'hemp milk', 'cashew milk',
                    # Grains
                    'bread', 'rice', 'pasta', 'noodles', 'quinoa', 'oatmeal', 'cereal', 'flour', 'wheat',
                    'barley', 'rye', 'oats', 'corn', 'millet', 'sorghum', 'teff', 'amaranth', 'buckwheat',
                    'farro', 'spelt', 'kamut', 'freekeh', 'bulgur', 'couscous', 'polenta', 'grits',
                    'cornmeal', 'semolina', 'durum wheat', 'whole wheat', 'white flour', 'bread flour',
                    'cake flour', 'pastry flour', 'all-purpose flour', 'self-rising flour', 'rye flour',
                    'buckwheat flour', 'almond flour', 'coconut flour', 'chickpea flour', 'rice flour',
                    'tapioca flour', 'potato flour', 'arrowroot flour', 'cassava flour', 'tigernut flour',
                    # Processed Foods
                    'pizza', 'burger', 'sandwich', 'hot dog', 'taco', 'burrito', 'sushi', 'salad', 'soup',
                    'stew', 'curry', 'stir fry', 'lasagna', 'spaghetti', 'macaroni', 'cake', 'cookie',
                    'brownie', 'muffin', 'donut', 'croissant', 'bagel', 'toast', 'pancake', 'waffle',
                    'crepe', 'french toast', 'eggs benedict', 'omelette', 'frittata', 'quiche', 'pie',
                    'tart', 'pastry', 'danish', 'strudel', 'baklava', 'cannoli', 'tiramisu', 'cheesecake',
                    'ice cream', 'gelato', 'sorbet', 'pudding', 'custard', 'flan', 'creme brulee',
                    'chocolate', 'candy', 'chocolate bar', 'truffle', 'praline', 'fudge', 'caramel',
                    'toffee', 'nougat', 'marshmallow', 'gummy bear', 'jelly bean', 'licorice', 'lollipop',
                    # Beverages
                    'coffee', 'tea', 'juice', 'water', 'soda', 'beer', 'wine', 'milk', 'smoothie',
                    'milkshake', 'hot chocolate', 'cocoa', 'espresso', 'cappuccino', 'latte', 'americano',
                    'mocha', 'macchiato', 'flat white', 'cortado', 'piccolo', 'ristretto', 'lungo',
                    'black tea', 'green tea', 'oolong tea', 'white tea', 'herbal tea', 'chai', 'mate',
                    'rooibos', 'hibiscus', 'chamomile', 'peppermint', 'ginger tea', 'lemon tea',
                    'orange juice', 'apple juice', 'grape juice', 'cranberry juice', 'tomato juice',
                    'carrot juice', 'beet juice', 'celery juice', 'wheatgrass juice', 'aloe vera juice',
                    # Condiments & Seasonings
                    'sauce', 'ketchup', 'mustard', 'mayonnaise', 'hot sauce', 'soy sauce', 'vinegar',
                    'oil', 'olive oil', 'butter', 'salt', 'pepper', 'sugar', 'honey', 'syrup',
                    'maple syrup', 'agave nectar', 'stevia', 'splenda', 'aspartame', 'saccharin',
                    'balsamic vinegar', 'apple cider vinegar', 'red wine vinegar', 'white wine vinegar',
                    'rice vinegar', 'malt vinegar', 'distilled vinegar', 'coconut oil', 'avocado oil',
                    'sesame oil', 'peanut oil', 'canola oil', 'sunflower oil', 'safflower oil',
                    'grapeseed oil', 'walnut oil', 'almond oil', 'hazelnut oil', 'pistachio oil',
                    'black pepper', 'white pepper', 'pink pepper', 'green pepper', 'szechuan pepper',
                    'cayenne pepper', 'paprika', 'smoked paprika', 'chili powder', 'cumin', 'coriander',
                    'turmeric', 'ginger', 'garlic', 'onion powder', 'oregano', 'basil', 'thyme',
                    'rosemary', 'sage', 'marjoram', 'tarragon', 'dill', 'parsley', 'cilantro',
                    'mint', 'chives', 'bay leaf', 'cardamom', 'cinnamon', 'nutmeg', 'cloves',
                    'allspice', 'star anise', 'fennel seed', 'caraway seed', 'celery seed',
                    'poppy seed', 'sesame seed', 'sunflower seed', 'pumpkin seed', 'chia seed',
                    'flax seed', 'hemp seed', 'quinoa seed', 'amaranth seed', 'buckwheat seed'
                }
                
                # Multiple detection passes with different parameters for maximum coverage
                for config in detection_configs:
                    try:
                        results = models['yolo_model'](img_np, conf=config['conf'], iou=config['iou'])
                        
                        for result in results:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    class_name = models['yolo_model'].names[cls].lower()
                                    
                                    # Check if it's a comprehensive food item
                                    if class_name in comprehensive_food_items:
                                        all_food_items.add(class_name)
                                        logger.info(f"YOLO detected: {class_name} (confidence: {conf:.2f}, config: {config})")
                    except Exception as e:
                        logger.warning(f"YOLO detection pass failed with config {config}: {e}")
                        continue
                                
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
        else:
            logger.info("YOLO model not available - using BLIP-only detection")
        
        # Strategy 3: Ultra-Detailed Category-Specific Prompts for Maximum Coverage
        ultra_category_prompts = [
            "What vegetables are in this image? Include fresh vegetables, cooked vegetables, pickled vegetables, fermented vegetables, and any vegetable-based ingredients:",
            "What fruits are visible? Include fresh fruits, dried fruits, cooked fruits, fruit juices, fruit sauces, and any fruit-based components:",
            "What proteins and meats are present? Include all types of meat, poultry, fish, seafood, eggs, legumes, nuts, seeds, and plant-based proteins:",
            "What grains and carbohydrates can you see? Include bread, rice, pasta, noodles, cereals, flour-based items, and any grain-based ingredients:",
            "What dairy products are visible? Include milk, cheese, yogurt, butter, cream, and any dairy-based ingredients or preparations:",
            "What sauces, condiments, and seasonings are present? Include all liquid seasonings, dry spices, herbs, oils, vinegars, and flavor enhancers:",
            "What beverages and drinks are in this image? Include hot drinks, cold drinks, alcoholic beverages, juices, and any liquid consumables:",
            "What desserts, sweets, and baked goods are visible? Include cakes, cookies, pastries, candies, chocolates, and any sweet preparations:",
            "What cooking oils, fats, and lipids can you identify? Include butter, olive oil, vegetable oils, animal fats, and any fat-based ingredients:",
            "What fermented foods are present? Include pickles, sauerkraut, kimchi, fermented sauces, aged cheeses, and any fermented products:",
            "What cooking methods and preparation techniques are visible? Identify grilled, fried, baked, steamed, boiled, raw, and other preparation styles:",
            "What garnishes, toppings, and decorative elements are present? Include herbs, seeds, nuts, sauces, and any finishing touches:"
        ]
        
        # Use ultra-detailed category prompts for additional detection
        for i, prompt in enumerate(ultra_category_prompts):
            try:
                inputs = models['processor'](image, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = models['blip_model'].generate(
                        **inputs, 
                        max_new_tokens=250,  # Increased for detailed responses
                        num_beams=6,         # Higher for better quality
                        do_sample=True,
                        temperature=0.4,     # Lower for more focused detection
                        top_p=0.95,
                        repetition_penalty=1.1
                    )
                caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                
                if caption.startswith(prompt):
                    caption = caption.replace(prompt, "").strip()
                
                items = extract_food_items_from_text(caption)
                all_food_items.update(items)
                logger.info(f"Category prompt {i+1} found: {len(items)} items - {caption[:50]}...")
                
            except Exception as e:
                logger.warning(f"Category prompt {i+1} failed: {e}")
                continue
        
        # Enhanced filtering with broader food keywords
        if all_food_items:
            # Much broader essential food keywords
            essential_food_keywords = {
                # Fruits
                'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'lemon', 'lime',
                'peach', 'pear', 'mango', 'pineapple', 'watermelon', 'cantaloupe', 'kiwi', 'fruit',
                # Vegetables
                'tomato', 'potato', 'carrot', 'onion', 'broccoli', 'cauliflower', 'lettuce', 'spinach',
                'cucumber', 'bell pepper', 'jalapeno', 'garlic', 'ginger', 'mushroom', 'corn', 'peas',
                'beans', 'asparagus', 'zucchini', 'eggplant', 'celery', 'radish', 'beet', 'turnip', 'vegetable',
                # Proteins
                'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'egg', 'tofu', 'tempeh',
                'shrimp', 'crab', 'lobster', 'salmon', 'tuna', 'cod', 'tilapia', 'bacon', 'sausage', 'meat', 'fish',
                # Dairy
                'cheese', 'milk', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese', 'dairy',
                # Grains
                'bread', 'rice', 'pasta', 'noodles', 'quinoa', 'oatmeal', 'cereal', 'flour', 'wheat', 'grain',
                # Processed foods
                'pizza', 'burger', 'sandwich', 'hot dog', 'taco', 'burrito', 'sushi', 'salad', 'soup',
                'stew', 'curry', 'stir fry', 'lasagna', 'spaghetti', 'macaroni', 'cake', 'cookie',
                'brownie', 'muffin', 'donut', 'croissant', 'bagel', 'toast', 'pancake', 'waffle',
                # Beverages
                'coffee', 'tea', 'juice', 'water', 'soda', 'beer', 'wine', 'milk', 'smoothie', 'drink',
                # Condiments
                'sauce', 'ketchup', 'mustard', 'mayonnaise', 'hot sauce', 'soy sauce', 'vinegar',
                'oil', 'olive oil', 'butter', 'salt', 'pepper', 'sugar', 'honey', 'syrup', 'seasoning',
                # General food terms
                'food', 'meal', 'dish', 'ingredient', 'spice', 'herb', 'garnish', 'topping', 'filling'
            }
            
            final_items = []
            for item in all_food_items:
                item_clean = item.strip().lower()
                # More permissive filtering - include items that are longer than 2 chars and not obvious non-food
                if (len(item_clean) > 2 and 
                    (any(keyword in item_clean for keyword in essential_food_keywords) or 
                     not any(non_food in item_clean for non_food in ['plate', 'bowl', 'cup', 'glass', 'table', 'chair', 'wall', 'floor', 'ceiling', 'window', 'door', 'light', 'shadow']))):
                    final_items.append(item_clean)
            
            if final_items:
                unique_items = sorted(set(final_items))
                result = ', '.join(unique_items)
                logger.info(f"Enhanced detection found {len(unique_items)} items: {result}")
                return result
        
        # Enhanced fallback with multiple strategies
        fallback_prompts = [
            "What food is in this image?",
            "Describe the food items you can see:",
            "What are the main food components?",
            "List the visible food items:"
        ]
        
        for prompt in fallback_prompts:
            try:
                inputs = models['processor'](image, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = models['blip_model'].generate(
                        **inputs, 
                        max_new_tokens=200,
                        num_beams=6,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9
                    )
                caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                
                if caption.startswith(prompt):
                    caption = caption.replace(prompt, "").strip()
                
                items = extract_food_items_from_text(caption)
                all_food_items.update(items)
                logger.info(f"Fallback prompt found: {len(items)} items - {caption[:50]}...")
                
            except Exception as e:
                logger.warning(f"Fallback prompt failed: {e}")
                continue
        
        # Apply validation to all detected items
        if all_food_items:
            # Validate all detected items for accuracy
            validated_items = validate_food_items(all_food_items, "")
            
            if validated_items:
                # Sort and limit to most relevant items
                unique_items = sorted(list(validated_items))
                
                # Limit to reasonable number for accuracy
                if len(unique_items) > 10:
                    unique_items = unique_items[:10]
                
                # Format as "Main Food Items Identified: item1, item2, item3..."
                result_description = "Main Food Items Identified: " + ", ".join(unique_items)
                
                # Add detection method info
                detection_methods = []
                if models.get('yolo_model'):
                    detection_methods.append("YOLO")
                detection_methods.append("BLIP")
                
                result_description += f" (Detected using: {', '.join(detection_methods)})"
                
                logger.info(f"Validated food items ({len(unique_items)}): {result_description}")
                return result_description
        
        return "Food items detected. Add context for better identification."
            
    except Exception as e:
        logger.error(f"Food detection error: {e}")
        return "Detection failed. Please try again."

def query_langchain(prompt: str, models: Dict[str, Any]) -> str:
    """Query LLM for nutrition analysis"""
    if not models.get('llm'):
        return "LLM service unavailable. Please check your API key."
    try:
        from langchain.schema import HumanMessage
        response = models['llm']([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return f"Error querying LLM: {str(e)}"

def extract_items_and_nutrients(text: str) -> Tuple[List[Dict], Dict]:
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

def analyze_food_with_enhanced_prompt(image_description: str, context: str, models: Dict[str, Any]) -> Dict[str, Any]:
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
        response = query_langchain(prompt, models)
        
        # Extract structured data with enhanced parsing
        items, totals = extract_items_and_nutrients(response)
        
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
            
            response = fallback_analysis
        
        # Ensure minimum detail level
        if len(response) < 500:  # If analysis is too brief, enhance it
            enhancement_prompt = f"""The previous analysis was too brief. Please provide a more detailed analysis of: {image_description}

Include:
- Detailed breakdown of each food component
- Comprehensive nutritional information
- Health assessment and recommendations
- Meal composition analysis

Make it thorough and informative."""
            
            try:
                enhanced_response = query_langchain(enhancement_prompt, models)
                response = enhanced_response
            except:
                pass  # Keep original if enhancement fails
        
        return {
            "success": True,
            "analysis": response,
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

def analyze_food_image(image: Image.Image, context: str, models: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to analyze food image with all features"""
    try:
        # Get food description
        food_description = describe_image_enhanced(image, models)
        
        # Analyze the food
        analysis_result = analyze_food_with_enhanced_prompt(food_description, context, models)
        
        # Add description to result
        analysis_result["description"] = food_description
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Food image analysis error: {e}")
        return {
            "success": False,
            "error": str(e),
            "description": "Analysis failed",
            "analysis": "Unable to analyze image. Please try again.",
            "nutritional_data": {
                "total_calories": 0,
                "total_protein": 0,
                "total_carbs": 0,
                "total_fats": 0,
                "items": []
            }
        }

def analyze_manual_food_items(food_items: List[Dict], models: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze manually selected food items for nutrition data"""
    try:
        # Create a description from manual selections with location/bbox info
        items_description = []
        for item in food_items:
            name = item['name']
            quantity = item.get('quantity', '1 serving')
            bbox = item.get('bbox', None)
            location = item.get('location', '')
            
            if bbox:
                # Use bounding box coordinates
                left, top, width, height = bbox
                items_description.append(f"{name} ({quantity}) at coordinates ({left}, {top}, {width}, {height})")
            elif location:
                # Use location description
                items_description.append(f"{name} ({quantity}) at {location}")
            else:
                items_description.append(f"{name} ({quantity})")
        
        description = f"Main Food Items Identified: {', '.join(items_description)}"
        
        # Query LLM for nutrition analysis
        if models.get('llm'):
            # Create detailed food items list with location/bbox
            food_items_detail = []
            for item in food_items:
                name = item['name']
                quantity = item.get('quantity', '1 serving')
                bbox = item.get('bbox', None)
                location = item.get('location', '')
                
                if bbox:
                    # Use bounding box coordinates
                    left, top, width, height = bbox
                    food_items_detail.append(f"{name} ({quantity}) at coordinates ({left}, {top}, {width}, {height})")
                elif location:
                    # Use location description
                    food_items_detail.append(f"{name} ({quantity}) at {location}")
                else:
                    food_items_detail.append(f"{name} ({quantity})")
            
            prompt = f"""
            Analyze the following food items and provide detailed nutritional information:
            
            Food Items: {', '.join(food_items_detail)}
            
            For each food item, provide:
            1. Estimated calories
            2. Protein content (grams)
            3. Carbohydrate content (grams)
            4. Fat content (grams)
            5. Additional nutritional notes
            
            Consider the quantities specified for each item.
            Format the response as a structured analysis with total nutritional values.
            """
            
            try:
                response = query_langchain(prompt, models)
                analysis_text = response
            except Exception as e:
                analysis_text = f"Analysis based on manual selection: {', '.join([item['name'] for item in food_items])}"
        else:
            analysis_text = f"Analysis based on manual selection: {', '.join([item['name'] for item in food_items])}"
        
        # Extract nutrition data from the analysis
        items, nutrition = extract_nutrition_data(analysis_text)
        
        # If no nutrition data extracted, provide estimates
        if nutrition['total_calories'] == 0:
            # Provide basic estimates based on food names
            total_cals = 0
            total_protein = 0
            total_carbs = 0
            total_fats = 0
            
            for item in food_items:
                name = item['name'].lower()
                quantity = item.get('quantity', '1 serving')
                
                # Basic nutrition estimates (simplified)
                if any(word in name for word in ['apple', 'banana', 'orange', 'fruit']):
                    total_cals += 80
                    total_carbs += 20
                elif any(word in name for word in ['chicken', 'beef', 'fish', 'meat']):
                    total_cals += 200
                    total_protein += 25
                    total_fats += 10
                elif any(word in name for word in ['rice', 'pasta', 'bread', 'potato']):
                    total_cals += 150
                    total_carbs += 30
                elif any(word in name for word in ['salad', 'vegetable', 'lettuce']):
                    total_cals += 50
                    total_carbs += 10
                else:
                    total_cals += 100
                    total_carbs += 15
                    total_protein += 5
                    total_fats += 5
            
            nutrition = {
                'total_calories': total_cals,
                'total_protein': total_protein,
                'total_carbs': total_carbs,
                'total_fats': total_fats
            }
        
        return {
            "success": True,
            "description": description,
            "analysis": analysis_text,
            "nutritional_data": nutrition,
            "items": items,
            "manual_selection": True
        }
        
    except Exception as e:
        logger.error(f"Error in manual food analysis: {e}")
        return {
            "success": False,
            "error": f"Manual analysis failed: {str(e)}",
            "description": f"Main Food Items Identified: {', '.join([item['name'] for item in food_items])}",
            "nutritional_data": {
                "total_calories": 0,
                "total_protein": 0,
                "total_carbs": 0,
                "total_fats": 0
            }
        }

def extract_nutrition_data(text: str) -> Tuple[List[Dict], Dict]:
    """Extract nutritional data from text (alias for compatibility)"""
    return extract_items_and_nutrients(text)
