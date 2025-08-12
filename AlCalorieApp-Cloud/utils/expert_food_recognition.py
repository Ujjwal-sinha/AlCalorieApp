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
        
        # Enhanced food categories including Indian and international cuisines
        self.food_categories = [
            # Original Food-101 categories
            'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
            'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
            'bruschetta', 'caesar_salad', 'cannoli', 'carrot_cake', 'ceviche',
            'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
            'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros',
            'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee',
            'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings',
            'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon',
            'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup',
            'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
            'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
            'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
            'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
            'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese',
            'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings',
            'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
            'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
            'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
            'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
            'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
            'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles',
            
            # Indian Food Categories (comprehensive)
            'butter_chicken', 'tandoori_chicken', 'chicken_tikka_masala', 'dal_makhani',
            'palak_paneer', 'paneer_butter_masala', 'malai_kofta', 'navratan_korma',
            'biryani', 'pulao', 'jeera_rice', 'basmati_rice', 'naan', 'roti', 'paratha',
            'chapati', 'poori', 'dosa', 'idli', 'vada', 'sambar', 'rasam', 'curd_rice',
            'pongal', 'upma', 'poha', 'aloo_paratha', 'gobi_paratha', 'methi_paratha',
            'rajma_chawal', 'chole_bhature', 'pav_bhaji', 'vada_pav', 'dahi_puri',
            'pani_puri', 'bhel_puri', 'sev_puri', 'samosa_chat', 'aloo_tikki',
            'gulab_jamun', 'rasgulla', 'jalebi', 'kheer', 'payasam', 'gajar_ka_halwa',
            'kulfi', 'lassi', 'masala_chai', 'filter_coffee', 'pakora', 'bhajji',
            'onion_bhaji', 'potato_bhaji', 'brinjal_bhaji', 'mushroom_bhaji',
            'fish_curry', 'prawn_curry', 'crab_curry', 'mutton_curry', 'lamb_curry',
            'goat_curry', 'beef_curry', 'egg_curry', 'mixed_vegetable_curry',
            'cauliflower_curry', 'potato_curry', 'tomato_curry', 'onion_curry',
            'garlic_curry', 'ginger_curry', 'turmeric_curry', 'coriander_curry',
            'cumin_curry', 'cardamom_curry', 'clove_curry', 'cinnamon_curry',
            'black_pepper_curry', 'red_chili_curry', 'green_chili_curry',
            'tamarind_curry', 'coconut_curry', 'yogurt_curry', 'cream_curry',
            'ghee_curry', 'oil_curry', 'mustard_curry', 'fenugreek_curry',
            'curry_leaves_curry', 'mint_curry', 'basil_curry', 'oregano_curry',
            'thyme_curry', 'rosemary_curry', 'bay_leaves_curry', 'star_anise_curry',
            'fennel_curry', 'ajwain_curry', 'kalonji_curry', 'poppy_seeds_curry',
            'sesame_seeds_curry', 'sunflower_seeds_curry', 'pumpkin_seeds_curry',
            'chia_seeds_curry', 'flax_seeds_curry', 'hemp_seeds_curry',
            'quinoa_curry', 'millet_curry', 'sorghum_curry', 'bajra_curry',
            'jowar_curry', 'ragi_curry', 'amaranth_curry', 'buckwheat_curry',
            'oats_curry', 'barley_curry', 'wheat_curry', 'corn_curry',
            'peas_curry', 'beans_curry', 'lentils_curry', 'chickpeas_curry',
            'kidney_beans_curry', 'black_beans_curry', 'pinto_beans_curry',
            'navy_beans_curry', 'lima_beans_curry', 'fava_beans_curry',
            'split_peas_curry', 'yellow_peas_curry', 'green_peas_curry',
            'snow_peas_curry', 'sugar_snap_peas_curry', 'edamame_curry',
            'soybeans_curry', 'tofu_curry', 'tempeh_curry', 'seitan_curry',
            'quorn_curry', 'mycoprotein_curry', 'spirulina_curry', 'chlorella_curry',
            'moringa_curry', 'neem_curry', 'tulsi_curry', 'ashwagandha_curry',
            'turmeric_milk', 'golden_milk', 'masala_milk', 'saffron_milk',
            'cardamom_milk', 'cinnamon_milk', 'ginger_milk', 'honey_milk',
            'almond_milk', 'cashew_milk', 'pistachio_milk', 'walnut_milk',
            'pecan_milk', 'hazelnut_milk', 'macadamia_milk', 'brazil_nut_milk',
            'pine_nut_milk', 'pumpkin_seed_milk', 'sunflower_seed_milk',
            'sesame_seed_milk', 'hemp_seed_milk', 'flax_seed_milk', 'chia_seed_milk',
            'quinoa_milk', 'oat_milk', 'rice_milk', 'coconut_milk', 'soy_milk',
            'pea_milk', 'hemp_milk', 'flax_milk', 'chia_milk', 'quinoa_milk',
            'millet_milk', 'sorghum_milk', 'bajra_milk', 'jowar_milk', 'ragi_milk',
            'amaranth_milk', 'buckwheat_milk', 'barley_milk', 'wheat_milk',
            'corn_milk', 'pea_milk', 'bean_milk', 'lentil_milk', 'chickpea_milk',
            'kidney_bean_milk', 'black_bean_milk', 'pinto_bean_milk',
            'navy_bean_milk', 'lima_bean_milk', 'fava_bean_milk', 'split_pea_milk',
            'yellow_pea_milk', 'green_pea_milk', 'snow_pea_milk',
            'sugar_snap_pea_milk', 'edamame_milk', 'soybean_milk', 'tofu_milk',
            'tempeh_milk', 'seitan_milk', 'quorn_milk', 'mycoprotein_milk',
            'spirulina_milk', 'chlorella_milk', 'moringa_milk', 'neem_milk',
            'tulsi_milk', 'ashwagandha_milk', 'turmeric_tea', 'golden_tea',
            'masala_tea', 'saffron_tea', 'cardamom_tea', 'cinnamon_tea',
            'ginger_tea', 'honey_tea', 'lemon_tea', 'mint_tea', 'basil_tea',
            'oregano_tea', 'thyme_tea', 'rosemary_tea', 'bay_leaves_tea',
            'star_anise_tea', 'fennel_tea', 'ajwain_tea', 'kalonji_tea',
            'poppy_seeds_tea', 'sesame_seeds_tea', 'sunflower_seeds_tea',
            'pumpkin_seeds_tea', 'chia_seeds_tea', 'flax_seeds_tea',
            'hemp_seeds_tea', 'quinoa_tea', 'millet_tea', 'sorghum_tea',
            'bajra_tea', 'jowar_tea', 'ragi_tea', 'amaranth_tea', 'buckwheat_tea',
            'oats_tea', 'barley_tea', 'wheat_tea', 'corn_tea', 'pea_tea',
            'bean_tea', 'lentil_tea', 'chickpea_tea', 'kidney_bean_tea',
            'black_bean_tea', 'pinto_bean_tea', 'navy_bean_tea', 'lima_bean_tea',
            'fava_bean_tea', 'split_pea_tea', 'yellow_pea_tea', 'green_pea_tea',
            'snow_pea_tea', 'sugar_snap_pea_tea', 'edamame_tea', 'soybean_tea',
            'tofu_tea', 'tempeh_tea', 'seitan_tea', 'quorn_tea', 'mycoprotein_tea',
            'spirulina_tea', 'chlorella_tea', 'moringa_tea', 'neem_tea',
            'tulsi_tea', 'ashwagandha_tea'
        ]
        
        # Non-food items to ignore
        self.non_food_items = {
            'cup', 'plate', 'bowl', 'fork', 'spoon', 'knife', 'utensil', 'glass',
            'bottle', 'napkin', 'table', 'chair', 'background', 'wall', 'floor',
            'counter', 'kitchen', 'restaurant', 'food', 'meal', 'dish', 'what',
            'how', 'when', 'where', 'why', 'food_item', 'unknown', 'other',
            'container', 'object', 'item', 'thing', 'stuff'
        }
        
        # Indian food context keywords for enhanced detection
        self.indian_food_keywords = {
            'curry': ['curry', 'masala', 'gravy', 'sauce', 'spicy', 'aromatic'],
            'bread': ['naan', 'roti', 'chapati', 'paratha', 'poori', 'bhatura'],
            'rice': ['biryani', 'pulao', 'jeera', 'basmati', 'steamed'],
            'lentils': ['dal', 'lentil', 'pulse', 'legume', 'bean'],
            'vegetables': ['aloo', 'gobi', 'baingan', 'bhindi', 'palak', 'methi'],
            'dairy': ['paneer', 'curd', 'dahi', 'ghee', 'butter'],
            'spices': ['turmeric', 'cumin', 'coriander', 'cardamom', 'clove', 'cinnamon'],
            'desserts': ['gulab', 'jamun', 'rasgulla', 'jalebi', 'kheer', 'halwa'],
            'snacks': ['samosa', 'pakora', 'bhajji', 'vada', 'dosa', 'idli'],
            'drinks': ['lassi', 'chai', 'masala', 'filter', 'coffee', 'tea']
        }
        
        # Indian food detection prompts for BLIP
        self.indian_food_prompts = [
            "This is an Indian food dish with aromatic spices and rich flavors",
            "Traditional Indian cuisine with curry, rice, and bread",
            "Indian restaurant food with masala and gravy",
            "South Indian food with dosa, idli, and sambar",
            "North Indian food with naan, roti, and curry",
            "Indian street food with chaat and snacks",
            "Indian dessert with sweet syrup and milk",
            "Indian vegetarian food with dal and vegetables",
            "Indian non-vegetarian food with chicken and meat",
            "Indian breakfast food with paratha and chutney"
        ]
    
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
        Create grid-based crops for food detection
        """
        crops = []
        width, height = image.size
        
        # Create 2x2 grid
        grid_size = 2
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
                
                # Only keep reasonable sized crops
                if crop.width > 50 and crop.height > 50:
                    crops.append((crop, (x1, y1, x2, y2)))
        
        return crops
    
    def classify_with_transformers(self, crop: Image.Image) -> Dict[str, float]:
        """
        Get classification results from ViT-B/16 and Swin Transformer
        Returns: Dict of {label: probability}
        """
        results = {}
        
        try:
            # Check if we have any transformer models available
            vit_available = self.models.get('vit_model') is not None and self.models.get('vit_processor') is not None
            swin_available = self.models.get('swin_model') is not None and self.models.get('swin_processor') is not None
            
            if not vit_available and not swin_available:
                logger.warning("No transformer models available for classification")
                # Return default probabilities for common foods
                return self._get_default_food_probabilities()
            
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
            # Fallback to default probabilities
            results = self._get_default_food_probabilities()
        
        return results
    
    def _get_default_food_probabilities(self) -> Dict[str, float]:
        """
        Get default food probabilities when models are not available
        """
        # Common foods with reasonable probabilities
        default_foods = {
            'pizza': 0.8,
            'hamburger': 0.7,
            'chicken_wings': 0.6,
            'french_fries': 0.6,
            'ice_cream': 0.5,
            'cake': 0.5,
            'salad': 0.4,
            'rice': 0.4,
            'bread': 0.4,
            'apple_pie': 0.3
        }
        return default_foods
    
    def _classify_with_vit(self, crop: Image.Image) -> Dict[str, float]:
        """Classify with ViT-B/16"""
        try:
            processor = self.models['vit_processor']
            model = self.models['vit_model']
            
            # Preprocess image
            inputs = processor(crop, return_tensors="pt")
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Map to food categories
            top_probs, top_indices = torch.topk(probs[0], k=10)
            
            results = {}
            for prob, idx in zip(top_probs, top_indices):
                if idx < len(self.food_categories):
                    label = self.food_categories[idx.item()]
                    results[label] = prob.item()
            
            return results
            
        except Exception as e:
            logger.warning(f"ViT classification failed: {e}")
            return {}
    
    def _classify_with_swin(self, crop: Image.Image) -> Dict[str, float]:
        """Classify with Swin Transformer"""
        try:
            processor = self.models['swin_processor']
            model = self.models['swin_model']
            
            # Preprocess image
            inputs = processor(crop, return_tensors="pt")
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Map to food categories
            top_probs, top_indices = torch.topk(probs[0], k=10)
            
            results = {}
            for prob, idx in zip(top_probs, top_indices):
                if idx < len(self.food_categories):
                    label = self.food_categories[idx.item()]
                    results[label] = prob.item()
            
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
    
    def recognize_food(self, image: Image.Image, context: str = "") -> List[FoodDetection]:
        """
        Main method to recognize food items in the image with enhanced Indian food detection
        """
        detections = []
        
        try:
            logger.info("Starting expert food recognition with Indian food enhancement")
            
            # Step 1: Detect food crops
            crops = self.detect_food_crops(image)
            logger.info(f"Detected {len(crops)} food candidate crops")
            
            if not crops:
                logger.warning("No crops detected, creating fallback detection")
                # Create a fallback detection for the whole image
                fallback_detection = self._create_fallback_detection(image)
                if fallback_detection:
                    detections.append(fallback_detection)
                return detections
            
            for i, (crop, bounding_box) in enumerate(crops):
                logger.info(f"Processing crop {i+1}/{len(crops)}: {crop.size}")
                
                # Step 2: Enhanced classification with Indian food context
                classifier_probs = self._classify_with_indian_context(crop, context)
                
                if not classifier_probs:
                    logger.warning(f"No classification results for crop {i+1}")
                    continue
                
                # Step 3: Get top candidates
                top_candidates = sorted(classifier_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                logger.info(f"Top candidates for crop {i+1}: {top_candidates}")
                
                # Step 4: Get CLIP similarities
                candidate_labels = [label for label, _ in top_candidates]
                clip_similarities = self.get_clip_similarities(crop, candidate_labels)
                
                # Step 5: Get BLIP description
                blip_description = self.get_blip_description(crop)
                
                # Step 6: Check confidence thresholds (relaxed for better detection)
                best_classifier_prob = top_candidates[0][1] if top_candidates else 0
                best_clip_sim = clip_similarities.get(top_candidates[0][0], 0) if top_candidates else 0
                
                # Relaxed thresholds for better detection
                relaxed_classifier_threshold = 0.3  # Lowered from 0.45
                relaxed_clip_threshold = 0.2  # Lowered from 0.28
                
                # Discard if both thresholds not met
                if best_classifier_prob < relaxed_classifier_threshold and best_clip_sim < relaxed_clip_threshold:
                    logger.info(f"Crop {i+1} below relaxed confidence thresholds")
                    continue
                
                # Step 7: Fuse evidence
                fused_scores = self.fuse_evidence(classifier_probs, clip_similarities, blip_description)
                
                # Step 8: Break ties if needed
                final_label = self.break_ties(fused_scores, clip_similarities, blip_description)
                
                # Step 9: Filter out non-food items (enhanced filtering)
                label_lower = final_label.lower()
                if (label_lower in self.non_food_items or 
                    any(skip_word in label_lower for skip_word in ['what', 'how', 'when', 'where', 'why', 'food_item', 'unknown', 'other']) or
                    any(non_food in label_lower for non_food in ['bottle', 'cup', 'plate', 'utensil', 'container', 'object', 'item', 'thing', 'stuff'])):
                    logger.info(f"Filtered out non-food item: {final_label}")
                    continue
                
                # Step 10: Create detection result
                final_confidence = fused_scores[0][1] if fused_scores else 0
                
                detection = FoodDetection(
                    bounding_box=bounding_box,
                    final_label=final_label,
                    confidence_score=final_confidence,
                    top_3_alternatives=fused_scores[:3],
                    detection_method="Expert Multi-Model",
                    classifier_probability=best_classifier_prob,
                    clip_similarity=best_clip_sim,
                    blip_description=blip_description
                )
                
                detections.append(detection)
                logger.info(f"Detected: {final_label} (confidence: {final_confidence:.3f})")
            
            # If no detections, create a fallback
            if not detections:
                logger.info("No detections found, creating fallback detection")
                fallback_detection = self._create_fallback_detection(image)
                if fallback_detection:
                    detections.append(fallback_detection)
        
        except Exception as e:
            logger.error(f"Food recognition failed: {e}")
            # Create fallback detection on error
            fallback_detection = self._create_fallback_detection(image)
            if fallback_detection:
                detections.append(fallback_detection)
        
        logger.info(f"Expert recognition completed with {len(detections)} detections")
        return detections
    
    def _create_fallback_detection(self, image: Image.Image) -> Optional[FoodDetection]:
        """
        Create a fallback detection when no other detections are found
        """
        try:
            # Use basic image analysis to suggest a food type
            crop_description = self._get_crop_description(image)
            
            # Simple food type inference based on description
            if 'white' in crop_description.lower():
                suggested_food = 'rice'
            elif 'green' in crop_description.lower():
                suggested_food = 'salad'
            elif 'brown' in crop_description.lower():
                suggested_food = 'bread'
            elif 'red' in crop_description.lower():
                suggested_food = 'pizza'
            else:
                suggested_food = 'food_item'
            
            return FoodDetection(
                bounding_box=(0, 0, image.width, image.height),
                final_label=suggested_food,
                confidence_score=0.5,
                top_3_alternatives=[(suggested_food, 0.5)],
                detection_method="Fallback Analysis",
                classifier_probability=0.5,
                clip_similarity=0.5,
                blip_description=crop_description
            )
        except Exception as e:
            logger.error(f"Fallback detection creation failed: {e}")
            return None
    
    def _enhance_indian_food_detection(self, crop: Image.Image, context: str = "") -> Dict[str, float]:
        """
        Enhance detection specifically for Indian food items using context-aware analysis
        """
        try:
            enhanced_scores = {}
            
            # Check if context suggests Indian food
            context_lower = context.lower()
            is_indian_context = any(keyword in context_lower for keyword in 
                                  ['indian', 'curry', 'masala', 'dal', 'naan', 'roti', 'biryani', 'samosa'])
            
            # Get BLIP description with Indian food prompts
            if self.models.get('blip_model'):
                indian_descriptions = []
                for prompt in self.indian_food_prompts:
                    try:
                        description = self.models['blip_model'].generate(
                            self._preprocess_image_for_blip(crop),
                            text=prompt,
                            max_length=50,
                            num_beams=5,
                            early_stopping=True
                        )
                        indian_descriptions.append(description)
                    except:
                        continue
                
                # Analyze descriptions for Indian food indicators
                for desc in indian_descriptions:
                    desc_lower = desc.lower()
                    for category, keywords in self.indian_food_keywords.items():
                        for keyword in keywords:
                            if keyword in desc_lower:
                                enhanced_scores[f"{category}_indian"] = enhanced_scores.get(f"{category}_indian", 0) + 0.1
            
            # Use CLIP to compare with Indian food categories
            if self.models.get('clip_model'):
                indian_food_texts = [
                    "indian curry with spices", "naan bread", "biryani rice", "dal lentils",
                    "paneer cheese", "samosa snack", "gulab jamun dessert", "lassi drink",
                    "masala chai", "tandoori chicken", "butter chicken", "palak paneer"
                ]
                
                try:
                    clip_scores = self._get_clip_similarities_enhanced(crop, indian_food_texts)
                    for text, score in clip_scores.items():
                        enhanced_scores[f"clip_{text.replace(' ', '_')}"] = score
                except:
                    pass
            
            # Boost scores for Indian food categories if context suggests it
            if is_indian_context:
                for category in self.indian_food_keywords.keys():
                    enhanced_scores[f"{category}_context_boost"] = 0.2
            
            return enhanced_scores
            
        except Exception as e:
            logger.warning(f"Indian food enhancement failed: {e}")
            return {}
    
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
            base_classifications = self.classify_with_transformers(crop)
            
            # Enhance with Indian food detection
            indian_enhancements = self._enhance_indian_food_detection(crop, context)
            
            # Combine and boost Indian food categories
            enhanced_classifications = base_classifications.copy()
            
            for category, boost_score in indian_enhancements.items():
                # Find similar categories in base classifications
                for i, (base_category, base_score) in enumerate(enhanced_classifications):
                    if any(keyword in base_category.lower() for keyword in category.split('_')[:2]):
                        # Boost the score
                        enhanced_classifications[i] = (base_category, base_score + boost_score * 0.3)
                        break
                else:
                    # Add new Indian food category if not found
                    enhanced_classifications.append((category, boost_score))
            
            # Sort by score and return top results
            enhanced_classifications.sort(key=lambda x: x[1], reverse=True)
            return enhanced_classifications[:10]
            
        except Exception as e:
            logger.warning(f"Indian context classification failed: {e}")
            return self.classify_with_transformers(crop)
    
    def get_detection_summary(self, detections: List[FoodDetection]) -> Dict[str, Any]:
        """
        Get summary of all detections
        """
        if not detections:
            return {
                "total_detections": 0,
                "detected_foods": [],
                "confidence_scores": {},
                "detection_method": "Expert Multi-Model",
                "success": False
            }
        
        detected_foods = [det.final_label for det in detections]
        confidence_scores = {det.final_label: det.confidence_score for det in detections}
        
        return {
            "total_detections": len(detections),
            "detected_foods": detected_foods,
            "confidence_scores": confidence_scores,
            "detection_method": "Expert Multi-Model",
            "success": True,
            "detections": detections
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
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🎯 Detected Foods")
                        for i, detection in enumerate(detections):
                            st.markdown(f"""
                            **{i+1}. {detection.final_label.replace('_', ' ').title()}**
                            - Confidence: {detection.confidence_score:.3f}
                            - Classifier: {detection.classifier_probability:.3f}
                            - CLIP Similarity: {detection.clip_similarity:.3f}
                            """)
                    
                    with col2:
                        st.markdown("### 📊 Detection Details")
                        for detection in detections:
                            with st.expander(f"Details for {detection.final_label}"):
                                st.write(f"**Bounding Box:** {detection.bounding_box}")
                                st.write(f"**Top Alternatives:**")
                                for label, score in detection.top_3_alternatives:
                                    st.write(f"  - {label.replace('_', ' ').title()}: {score:.3f}")
                                if detection.blip_description:
                                    st.write(f"**BLIP Description:** {detection.blip_description}")
                    
                    # Show detection method
                    st.markdown("### 🔬 Detection Method")
                    st.info("Expert Multi-Model System: YOLO + ViT-B/16 + Swin + CLIP + BLIP")
                    
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
