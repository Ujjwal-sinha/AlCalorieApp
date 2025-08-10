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
    """Enhanced food detection with global search, advanced parsing, and comprehensive model integration."""
    if not models.get('processor') or not models.get('blip_model'):
        return "Image analysis unavailable. Please check model loading."
    
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        device = models.get('device')
        if not device:
            return "Device not available for image processing."
        
        all_food_items = set()
        
        # Strategy 1: Global Search with Multi-Scale Analysis
        global_search_prompts = [
            "Perform a global search and identify EVERY food item, ingredient, dish, sauce, beverage, condiment, garnish, seasoning, and edible component visible in this image. Be exhaustive and thorough:",
            "Conduct a comprehensive scan of this image and list ALL food-related items including main dishes, sides, appetizers, desserts, drinks, spices, herbs, oils, sauces, toppings, and any edible elements:",
            "Search this entire image systematically and identify each food component, ingredient, preparation method, cooking style, and nutritional element present:",
            "Analyze this image globally and extract every food item, cooking ingredient, dietary component, meal element, and consumable item visible:",
            "Perform a thorough global examination and catalog all food items, beverages, condiments, seasonings, garnishes, and edible components in this image:",
            "Conduct an exhaustive search of this image and identify every food-related element including dishes, ingredients, preparations, and nutritional components:",
            "Search the entire image comprehensively and list all food items, drinks, sauces, spices, herbs, and edible elements with their preparation methods:",
            "Analyze this image with global perspective and identify every food component, ingredient, cooking method, and dietary element present:",
            "Perform a complete global scan and extract all food items, beverages, condiments, seasonings, and edible components from this image:",
            "Conduct a thorough global analysis and identify every food-related item, preparation method, cooking style, and nutritional component visible:"
        ]
        
        for i, prompt in enumerate(global_search_prompts):
            try:
                inputs = models['processor'](image, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = models['blip_model'].generate(
                        **inputs, 
                        max_new_tokens=500,  # Increased for global search
                        num_beams=10,        # Increased for better quality
                        do_sample=True,
                        temperature=0.3,     # Lower for more focused detection
                        top_p=0.98,         # Higher for comprehensive results
                        repetition_penalty=1.1,
                        length_penalty=1.2   # Encourage longer, more detailed responses
                    )
                caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                
                if caption.startswith(prompt):
                    caption = caption.replace(prompt, "").strip()
                
                items = extract_food_items_from_text(caption)
                all_food_items.update(items)
                logger.info(f"Global search prompt {i+1} found: {len(items)} items")
                
            except Exception as e:
                logger.warning(f"Global search prompt {i+1} failed: {e}")
        
        # Strategy 2: Advanced YOLO Detection with Comprehensive Food Database
        if models.get('yolo_model') and models.get('NUMPY_AVAILABLE'):
            try:
                import numpy as np
                img_np = np.array(image)
                # Ultra-low confidence threshold for maximum detection
                results = models['yolo_model'](img_np, conf=0.01, iou=0.2)
                
                # Comprehensive global food database
                global_food_database = {
                    # Fruits (Global)
                    'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'lemon', 'lime',
                    'peach', 'pear', 'mango', 'pineapple', 'watermelon', 'cantaloupe', 'kiwi', 'avocado',
                    'cherry', 'plum', 'apricot', 'nectarine', 'fig', 'date', 'raisin', 'cranberry',
                    'raspberry', 'blackberry', 'gooseberry', 'currant', 'pomegranate', 'guava', 'papaya',
                    'dragon fruit', 'star fruit', 'lychee', 'longan', 'rambutan', 'durian', 'jackfruit',
                    'breadfruit', 'plantain', 'persimmon', 'quince', 'medlar', 'loquat', 'mulberry',
                    # Vegetables (Global)
                    'tomato', 'potato', 'carrot', 'onion', 'broccoli', 'cauliflower', 'lettuce', 'spinach',
                    'cucumber', 'bell pepper', 'jalapeno', 'garlic', 'ginger', 'mushroom', 'corn', 'peas',
                    'beans', 'asparagus', 'zucchini', 'eggplant', 'celery', 'radish', 'beet', 'turnip',
                    'sweet potato', 'yam', 'taro', 'cassava', 'parsnip', 'rutabaga', 'kohlrabi', 'fennel',
                    'artichoke', 'brussels sprout', 'kale', 'collard greens', 'swiss chard', 'arugula',
                    'watercress', 'endive', 'escarole', 'bok choy', 'napa cabbage', 'savoy cabbage',
                    'red cabbage', 'green cabbage', 'white cabbage', 'chinese cabbage', 'pak choi',
                    'mustard greens', 'turnip greens', 'beet greens', 'dandelion greens', 'purslane',
                    'amaranth', 'malabar spinach', 'new zealand spinach', 'orach', 'good king henry',
                    # Proteins (Global)
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
                    # Dairy (Global)
                    'cheese', 'milk', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese',
                    'ricotta', 'mozzarella', 'cheddar', 'parmesan', 'gouda', 'brie', 'camembert',
                    'blue cheese', 'feta', 'halloumi', 'paneer', 'queso fresco', 'manchego', 'pecorino',
                    'asiago', 'provolone', 'swiss cheese', 'havarti', 'muenster', 'colby', 'monterey jack',
                    'pepper jack', 'string cheese', 'cream cheese', 'mascarpone', 'quark', 'kefir',
                    'buttermilk', 'evaporated milk', 'condensed milk', 'powdered milk', 'almond milk',
                    'soy milk', 'oat milk', 'coconut milk', 'rice milk', 'hemp milk', 'cashew milk',
                    # Grains (Global)
                    'bread', 'rice', 'pasta', 'noodles', 'quinoa', 'oatmeal', 'cereal', 'flour', 'wheat',
                    'barley', 'rye', 'oats', 'corn', 'millet', 'sorghum', 'teff', 'amaranth', 'buckwheat',
                    'farro', 'spelt', 'kamut', 'freekeh', 'bulgur', 'couscous', 'polenta', 'grits',
                    'cornmeal', 'semolina', 'durum wheat', 'whole wheat', 'white flour', 'bread flour',
                    'cake flour', 'pastry flour', 'all-purpose flour', 'self-rising flour', 'rye flour',
                    'buckwheat flour', 'almond flour', 'coconut flour', 'chickpea flour', 'rice flour',
                    'tapioca flour', 'potato flour', 'arrowroot flour', 'cassava flour', 'tigernut flour',
                    # Processed Foods (Global)
                    'pizza', 'burger', 'sandwich', 'hot dog', 'taco', 'burrito', 'sushi', 'salad', 'soup',
                    'stew', 'curry', 'stir fry', 'lasagna', 'spaghetti', 'macaroni', 'cake', 'cookie',
                    'brownie', 'muffin', 'donut', 'croissant', 'bagel', 'toast', 'pancake', 'waffle',
                    'crepe', 'french toast', 'eggs benedict', 'omelette', 'frittata', 'quiche', 'pie',
                    'tart', 'pastry', 'danish', 'strudel', 'baklava', 'cannoli', 'tiramisu', 'cheesecake',
                    'ice cream', 'gelato', 'sorbet', 'pudding', 'custard', 'flan', 'creme brulee',
                    'chocolate', 'candy', 'chocolate bar', 'truffle', 'praline', 'fudge', 'caramel',
                    'toffee', 'nougat', 'marshmallow', 'gummy bear', 'jelly bean', 'licorice', 'lollipop',
                    # Beverages (Global)
                    'coffee', 'tea', 'juice', 'water', 'soda', 'beer', 'wine', 'milk', 'smoothie',
                    'milkshake', 'hot chocolate', 'cocoa', 'espresso', 'cappuccino', 'latte', 'americano',
                    'mocha', 'macchiato', 'flat white', 'cortado', 'piccolo', 'ristretto', 'lungo',
                    'black tea', 'green tea', 'oolong tea', 'white tea', 'herbal tea', 'chai', 'mate',
                    'rooibos', 'hibiscus', 'chamomile', 'peppermint', 'ginger tea', 'lemon tea',
                    'orange juice', 'apple juice', 'grape juice', 'cranberry juice', 'tomato juice',
                    'carrot juice', 'beet juice', 'celery juice', 'wheatgrass juice', 'aloe vera juice',
                    # Condiments & Seasonings (Global)
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
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = models['yolo_model'].names[cls].lower()
                            
                            # More permissive detection
                            if conf > 0.05 and any(term in class_name for term in food_terms):
                                all_food_items.add(class_name)
                                logger.info(f"YOLO detected: {class_name} (confidence: {conf:.2f})")
                                
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
        
        # Strategy 3: Additional BLIP prompts for specific food categories
        category_prompts = [
            "What vegetables and fruits are in this image?",
            "What proteins and meats are visible?",
            "What grains and carbohydrates can you see?",
            "What sauces, condiments, and seasonings are present?",
            "What beverages and drinks are in this image?",
            "What desserts and sweets are visible?",
            "What herbs and spices can you identify?"
        ]
        
        for prompt in category_prompts:
            try:
                inputs = models['processor'](image, text=prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = models['blip_model'].generate(
                        **inputs, 
                        max_new_tokens=200,
                        num_beams=6,
                        do_sample=True,
                        temperature=0.5,
                        top_p=0.9
                    )
                caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                
                if caption.startswith(prompt):
                    caption = caption.replace(prompt, "").strip()
                
                items = extract_food_items_from_text(caption)
                all_food_items.update(items)
                logger.info(f"Category prompt found: {len(items)} items")
                
            except Exception as e:
                logger.warning(f"Category prompt failed: {e}")
        
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
                    outputs = models['blip_model'].generate(**inputs, max_new_tokens=150, num_beams=5)
                caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
                
                if prompt.lower() in caption.lower():
                    caption = caption.lower().replace(prompt.lower(), "").strip()
                
                if len(caption.split()) >= 3:  # More permissive
                    return caption
                    
            except Exception as e:
                logger.warning(f"Fallback prompt failed: {e}")
        
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

def extract_nutrition_data(text: str) -> Tuple[List[Dict], Dict]:
    """Extract nutritional data from text (alias for compatibility)"""
    return extract_items_and_nutrients(text)
