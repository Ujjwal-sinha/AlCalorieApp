"""
Lightweight Vision Transformer Food Detection
Memory-efficient implementation using only ViT-B/16
"""

import logging
import numpy as np
from typing import Dict, Any, List, Set
from PIL import Image
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class LightweightViTFoodDetector:
    """
    Lightweight food detection using Vision Transformer (ViT-B/16) only
    """
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.device = models.get('device', 'cpu')
        
        # Use pre-loaded ViT model
        self.vit_processor = models.get('vit_processor')
        self.vit_model = models.get('vit_model')
        
        # Food classification mapping (ImageNet classes to food items)
        self.imagenet_food_mapping = self._create_food_mapping()
        
        # Food vocabulary
        self.food_vocabulary = self._load_food_vocabulary()
    
    def _create_food_mapping(self) -> Dict[int, str]:
        """Create mapping from ImageNet class IDs to food items"""
        return {
            # Fruits
            948: 'banana',
            949: 'apple',
            950: 'orange',
            951: 'lemon',
            952: 'fig',
            953: 'pineapple',
            954: 'strawberry',
            955: 'pomegranate',
            
            # Vegetables
            936: 'broccoli',
            937: 'cauliflower',
            938: 'cabbage',
            939: 'artichoke',
            940: 'bell pepper',
            941: 'cucumber',
            942: 'mushroom',
            943: 'corn',
            
            # Prepared foods
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
            
            # Other foods
            924: 'guacamole',
            925: 'consomme',
            926: 'trifle',
            963: 'apple',
            964: 'strawberry',
            965: 'orange',
            966: 'lemon'
        }
    
    def _load_food_vocabulary(self) -> Set[str]:
        """Load basic food vocabulary"""
        return {
            'apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry',
            'tomato', 'potato', 'carrot', 'onion', 'broccoli', 'lettuce',
            'chicken', 'beef', 'fish', 'egg', 'cheese', 'milk',
            'rice', 'bread', 'pasta', 'pizza', 'burger', 'sandwich',
            'cake', 'cookie', 'ice cream', 'coffee', 'tea'
        }
    
    def detect_food_lightweight(self, image: Image.Image) -> Dict[str, Any]:
        """
        Lightweight food detection using ViT-B/16
        """
        try:
            logger.info("Starting lightweight ViT food detection...")
            
            # Ensure image is in RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            detected_foods = set()
            confidence_scores = {}
            
            # Method 1: ViT detection
            if self.vit_processor and self.vit_model:
                vit_results = self._detect_with_vit(image)
                for food, confidence in vit_results.items():
                    detected_foods.add(food)
                    confidence_scores[food] = confidence
            
            # Method 2: Color-based fallback
            if not detected_foods:
                fallback_foods = self._color_based_detection(image)
                for food in fallback_foods:
                    detected_foods.add(food)
                    confidence_scores[food] = 0.4
            
            # Ensure we have at least one result
            if not detected_foods:
                detected_foods.add('food item')
                confidence_scores['food item'] = 0.3
            
            return {
                'detected_foods': list(detected_foods),
                'confidence_scores': confidence_scores,
                'detection_method': 'lightweight_vit',
                'models_used': ['ViT-B/16'] if self.vit_model else ['Color-based'],
                'total_detections': len(detected_foods)
            }
            
        except Exception as e:
            logger.error(f"Lightweight ViT detection failed: {e}")
            return {
                'detected_foods': ['food item'],
                'confidence_scores': {'food item': 0.3},
                'detection_method': 'fallback',
                'models_used': ['Fallback'],
                'total_detections': 1
            }
    
    def _detect_with_vit(self, image: Image.Image) -> Dict[str, float]:
        """Detect food using Vision Transformer (ViT-B/16)"""
        detected_foods = {}
        
        try:
            # Preprocess image
            inputs = self.vit_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                predictions = F.softmax(outputs.logits, dim=-1)
            
            # Get top predictions
            top_predictions = torch.topk(predictions, k=5, dim=-1)
            
            for i in range(top_predictions.indices.shape[1]):
                class_id = top_predictions.indices[0][i].item()
                confidence = top_predictions.values[0][i].item()
                
                # Map to food item if it's a food class
                if class_id in self.imagenet_food_mapping and confidence > 0.01:
                    food_name = self.imagenet_food_mapping[class_id]
                    detected_foods[food_name] = confidence
                    logger.info(f"ViT detected: {food_name} (confidence: {confidence:.3f})")
        
        except Exception as e:
            logger.warning(f"ViT detection failed: {e}")
        
        return detected_foods
    
    def _color_based_detection(self, image: Image.Image) -> List[str]:
        """Simple color-based food detection"""
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
                detected_foods.extend(['broccoli', 'lettuce'])
            elif r > 200 and g > 180 and b < 100:  # Yellow dominant
                detected_foods.extend(['banana', 'corn'])
            elif r > 150 and g > 100 and b < 80:  # Orange dominant
                detected_foods.extend(['carrot', 'orange'])
            elif r > 100 and g > 80 and b > 60:  # Brown dominant
                detected_foods.extend(['bread', 'chicken'])
            else:
                detected_foods.append('mixed food')
            
            return detected_foods[:2]  # Return top 2
            
        except Exception as e:
            logger.warning(f"Color-based detection failed: {e}")
            return ['food item']