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
    
    def _get_default_food_probabilities(self, crop: Image.Image = None) -> Dict[str, float]:
        """
        Return empty dict - no hardcoded food suggestions
        """
        # Return empty dict to ensure no hardcoded food items are suggested
        return {}
    
    def _classify_with_vit(self, crop: Image.Image) -> Dict[str, float]:
        """Classify with ViT-B/16 with enhanced accuracy"""
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
            
            # Get top predictions with higher confidence
            top_probs, top_indices = torch.topk(probs[0], k=15)
            
            results = {}
            for prob, idx in zip(top_probs, top_indices):
                prob_value = prob.item()
                if prob_value > 0.1:  # Only include predictions with >10% confidence
                    if idx < len(self.food_categories):
                        label = self.food_categories[idx.item()]
                        results[label] = prob_value
            
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
        Comprehensive food analysis - only returns items actually found in the image
        """
        detections = []
        
        try:
            logger.info("Starting comprehensive food analysis - image-only detection")
            
            # Step 1: Multi-scale crop detection for comprehensive coverage
            crops = self._get_comprehensive_crops(image)
            logger.info(f"Generated {len(crops)} comprehensive crop candidates")
            
            if not crops:
                logger.warning("No crops generated, using whole image")
                crops = [(image, (0, 0, image.width, image.height))]
            
            for i, (crop, bounding_box) in enumerate(crops):
                logger.info(f"Processing crop {i+1}/{len(crops)}: {crop.size}")
                
                # Step 2: Multi-model classification with strict validation
                classification_result = self._get_strict_image_based_classification(crop, context)
                
                if not classification_result:
                    logger.info(f"No valid image-based classification for crop {i+1}")
                    continue
                
                final_label, final_confidence, classifier_prob, clip_sim, blip_desc, alternatives = classification_result
                
                # Step 3: Ultra-strict confidence validation (95%+ requirement)
                if final_confidence < 0.95:
                    logger.info(f"Crop {i+1} confidence {final_confidence:.3f} below 95% threshold")
                    continue
                
                # Step 4: Validate that this is actually food found in the image
                if not self._validate_image_based_food_detection(final_label, blip_desc, final_confidence, crop):
                    logger.info(f"Crop {i+1} failed image-based food validation")
                    continue
                
                # Step 5: Create image-based detection
                detection = FoodDetection(
                    bounding_box=bounding_box,
                    final_label=final_label,
                    confidence_score=final_confidence,
                    top_3_alternatives=alternatives[:3],
                    detection_method="Image-Based Multi-Model",
                    classifier_probability=classifier_prob,
                    clip_similarity=clip_sim,
                    blip_description=blip_desc
                )
                
                detections.append(detection)
                logger.info(f"Image-based detection: {final_label} (confidence: {final_confidence:.3f})")
            
            # Only return detections with 95%+ confidence
            ultra_high_confidence_detections = [d for d in detections if d.confidence_score >= 0.95]
            
            if not ultra_high_confidence_detections:
                logger.warning("No detections met 95% confidence threshold - no hardcoded items returned")
                return []
            
            logger.info(f"Comprehensive food analysis completed with {len(ultra_high_confidence_detections)} image-based detections")
            return ultra_high_confidence_detections
        
        except Exception as e:
            logger.error(f"Comprehensive food analysis failed: {e}")
            return []
    
    def _get_comprehensive_crops(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Generate comprehensive crop candidates using multiple strategies
        """
        crops = []
        
        try:
            # Strategy 1: YOLO detection with high confidence
            if self.models.get('yolo_model'):
                logger.info("Using YOLO for high-confidence crop detection")
                yolo_results = self.models['yolo_model'](image, verbose=False)
                
                for result in yolo_results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Only use high-confidence YOLO detections
                            confidence = box.conf[0].item() if hasattr(box, 'conf') else 0.5
                            if confidence > 0.7:  # High confidence threshold
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                crop = image.crop((x1, y1, x2, y2))
                                if crop.width * crop.height >= 2500:  # Minimum area
                                    crops.append((crop, (x1, y1, x2, y2)))
                                    logger.info(f"YOLO crop: {crop.width}x{crop.height} (conf: {confidence:.3f})")
            
            # Strategy 2: Intelligent grid cropping
            if len(crops) < 3:  # Ensure we have enough candidates
                grid_crops = self._create_adaptive_grid_crops(image)
                crops.extend(grid_crops)
            
            # Strategy 3: Center crop for single food items
            center_crop = self._create_center_crop(image)
            if center_crop:
                crops.append(center_crop)
            
            # Strategy 4: Whole image as fallback
            if not crops:
                crops.append((image, (0, 0, image.width, image.height)))
            
        except Exception as e:
            logger.warning(f"Comprehensive crop generation failed: {e}")
            crops.append((image, (0, 0, image.width, image.height)))
        
        return crops
    
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
    
    def _get_strict_image_based_classification(self, crop: Image.Image, context: str = "") -> Optional[Tuple]:
        """
        Get strict image-based classification - only real detections from the image
        """
        try:
            # Get all available classifications
            classifier_probs = self.classify_with_transformers(crop)
            if not classifier_probs:
                logger.info("No classifier probabilities available")
                return None
            
            # Filter out low-confidence predictions
            high_confidence_probs = {label: prob for label, prob in classifier_probs.items() if prob > 0.3}
            if not high_confidence_probs:
                logger.info("No high-confidence classifier predictions")
                return None
            
            # Get top candidates
            top_candidates = sorted(high_confidence_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Get CLIP similarities for top candidates
            candidate_labels = [label for label, _ in top_candidates]
            clip_similarities = self.get_clip_similarities(crop, candidate_labels)
            
            # Filter CLIP similarities for high confidence
            high_confidence_clip = {label: sim for label, sim in clip_similarities.items() if sim > 0.4}
            if not high_confidence_clip:
                logger.info("No high-confidence CLIP similarities")
                return None
            
            # Get BLIP description
            blip_description = self.get_blip_description(crop)
            
            # Fuse evidence with ultra-strict weighting
            fused_scores = self._ultra_strict_fuse_evidence(high_confidence_probs, high_confidence_clip, blip_description, context)
            
            if not fused_scores:
                return None
            
            # Get best result
            best_label, best_confidence = fused_scores[0]
            classifier_prob = high_confidence_probs.get(best_label, 0)
            clip_sim = high_confidence_clip.get(best_label, 0)
            
            return (best_label, best_confidence, classifier_prob, clip_sim, blip_description, fused_scores)
            
        except Exception as e:
            logger.warning(f"Strict image-based classification failed: {e}")
            return None
    
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
            
            # Step 1: Check against non-food items
            if label_lower in self.non_food_items:
                logger.info(f"Label '{label}' is in non-food items list")
                return False
            
            # Step 2: Validate BLIP description strongly supports food detection
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
            
            # Step 3: Validate label contains food-related terms
            food_terms = ['food', 'meal', 'dish', 'cuisine', 'cooking', 'cooked', 'fresh', 'delicious', 'tasty', 'edible']
            if not any(term in label_lower for term in food_terms):
                # Check if it's a specific food name from our categories
                if label not in self.food_categories:
                    logger.info(f"Label '{label}' is not a recognized food category")
                    return False
            
            # Step 4: Ultra-high confidence requirement
            if confidence < 0.95:
                logger.info(f"Confidence {confidence:.3f} below ultra-high threshold")
                return False
            
            # Step 5: Additional image-based validation
            if not self._validate_crop_characteristics(crop, label):
                logger.info(f"Crop characteristics don't match label '{label}'")
                return False
            
            logger.info(f"Image-based food validation passed for '{label}'")
            return True
            
        except Exception as e:
            logger.warning(f"Image-based food validation failed: {e}")
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
