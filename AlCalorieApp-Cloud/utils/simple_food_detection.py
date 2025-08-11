"""
Simple Food Detection System
Lightweight fallback when transformer models are not available
"""

import logging
import numpy as np
from typing import Dict, Any, List, Set
from PIL import Image
import cv2

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimpleFoodDetector:
    """
    Simple food detection system using basic computer vision techniques
    """
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        
        # Basic food database with visual characteristics
        self.food_database = {
            'pizza': {'colors': [(200, 150, 100), (255, 200, 150)], 'keywords': ['round', 'flat', 'cheese']},
            'burger': {'colors': [(150, 100, 50), (200, 150, 100)], 'keywords': ['layered', 'round', 'meat']},
            'sandwich': {'colors': [(200, 180, 120), (150, 120, 80)], 'keywords': ['rectangular', 'bread']},
            'chicken': {'colors': [(200, 180, 140), (180, 160, 120)], 'keywords': ['meat', 'protein']},
            'rice': {'colors': [(240, 240, 230), (255, 255, 245)], 'keywords': ['white', 'grain']},
            'bread': {'colors': [(200, 160, 100), (220, 180, 120)], 'keywords': ['brown', 'loaf']},
            'tomato': {'colors': [(200, 50, 50), (255, 100, 100)], 'keywords': ['red', 'round']},
            'apple': {'colors': [(200, 50, 50), (100, 150, 50)], 'keywords': ['red', 'green', 'round']},
            'banana': {'colors': [(255, 220, 50), (255, 240, 100)], 'keywords': ['yellow', 'curved']},
            'orange': {'colors': [(255, 150, 50), (255, 180, 80)], 'keywords': ['orange', 'round']},
            'broccoli': {'colors': [(50, 150, 50), (80, 180, 80)], 'keywords': ['green', 'tree']},
            'carrot': {'colors': [(255, 140, 50), (255, 160, 80)], 'keywords': ['orange', 'long']},
            'potato': {'colors': [(200, 180, 140), (180, 160, 120)], 'keywords': ['brown', 'oval']},
            'cheese': {'colors': [(255, 220, 100), (255, 240, 150)], 'keywords': ['yellow', 'block']},
            'salad': {'colors': [(100, 200, 100), (150, 220, 150)], 'keywords': ['green', 'mixed']},
            'pasta': {'colors': [(255, 240, 200), (240, 220, 180)], 'keywords': ['white', 'noodles']},
            'egg': {'colors': [(255, 255, 240), (255, 220, 100)], 'keywords': ['white', 'yellow', 'oval']},
            'fish': {'colors': [(200, 180, 160), (180, 160, 140)], 'keywords': ['silver', 'protein']},
            'soup': {'colors': [(200, 150, 100), (180, 130, 80)], 'keywords': ['liquid', 'bowl']},
            'cake': {'colors': [(255, 240, 220), (200, 150, 100)], 'keywords': ['sweet', 'layered']}
        }
    
    def detect_food_simple(self, image: Image.Image) -> Dict[str, Any]:
        """
        Simple food detection using multiple basic methods
        """
        try:
            logger.info("Starting simple food detection...")
            
            # Convert image
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            detected_foods = set()
            confidence_scores = {}
            
            # Method 1: YOLO detection (if available)
            yolo_foods = self._yolo_detection(image)
            for food in yolo_foods:
                detected_foods.add(food)
                confidence_scores[food] = 0.8
            
            # Method 2: Color-based detection
            color_foods = self._color_based_detection(image)
            for food in color_foods:
                detected_foods.add(food)
                if food not in confidence_scores:
                    confidence_scores[food] = 0.6
            
            # Method 3: BLIP detection (if available)
            blip_foods = self._blip_detection(image)
            for food in blip_foods:
                detected_foods.add(food)
                if food not in confidence_scores:
                    confidence_scores[food] = 0.5
            
            # Method 4: Shape-based detection
            shape_foods = self._shape_based_detection(image)
            for food in shape_foods:
                detected_foods.add(food)
                if food not in confidence_scores:
                    confidence_scores[food] = 0.4
            
            # Ensure we always have some result
            if not detected_foods:
                detected_foods = {'food item'}
                confidence_scores['food item'] = 0.3
            
            return {
                'detected_foods': list(detected_foods),
                'confidence_scores': confidence_scores,
                'detection_method': 'simple_multi_method',
                'models_used': self._get_available_methods(),
                'total_foods': len(detected_foods)
            }
            
        except Exception as e:
            logger.error(f"Simple detection failed: {e}")
            return {
                'detected_foods': ['food item'],
                'confidence_scores': {'food item': 0.3},
                'detection_method': 'fallback',
                'models_used': ['Basic Detection'],
                'total_foods': 1
            }
    
    def _yolo_detection(self, image: Image.Image) -> Set[str]:
        """YOLO detection if available"""
        detected_foods = set()
        
        if not self.models.get('yolo_model'):
            return detected_foods
        
        try:
            img_array = np.array(image)
            results = self.models['yolo_model'](img_array, conf=0.2)
            
            # COCO food classes
            food_classes = {
                'apple', 'banana', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake'
            }
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.models['yolo_model'].names[cls].lower()
                        
                        if class_name in food_classes and confidence > 0.2:
                            detected_foods.add(class_name)
                            logger.info(f"YOLO detected: {class_name} (conf: {confidence:.3f})")
        
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
        
        return detected_foods
    
    def _color_based_detection(self, image: Image.Image) -> Set[str]:
        """Enhanced color-based detection"""
        detected_foods = set()
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate dominant colors
            pixels = img_array.reshape(-1, 3)
            mean_color = np.mean(pixels, axis=0)
            
            # Check against food database
            for food, characteristics in self.food_database.items():
                colors = characteristics.get('colors', [])
                
                for color in colors:
                    # Calculate color distance
                    distance = np.linalg.norm(mean_color - np.array(color))
                    
                    if distance < 60:  # Threshold for color similarity
                        detected_foods.add(food)
                        logger.info(f"Color detection: {food} (distance: {distance:.1f})")
                        break
        
        except Exception as e:
            logger.warning(f"Color detection failed: {e}")
        
        return detected_foods
    
    def _blip_detection(self, image: Image.Image) -> Set[str]:
        """BLIP detection if available"""
        detected_foods = set()
        
        if not self.models.get('processor') or not self.models.get('blip_model'):
            return detected_foods
        
        try:
            device = self.models.get('device', 'cpu')
            
            # Simple prompts
            prompts = ["What food is this?", "Describe this image."]
            
            for prompt in prompts:
                try:
                    inputs = self.models['processor'](image, text=prompt, return_tensors="pt").to(device)
                    if TORCH_AVAILABLE:
                        with torch.no_grad():
                            outputs = self.models['blip_model'].generate(**inputs, max_new_tokens=20)
                    else:
                        outputs = self.models['blip_model'].generate(**inputs, max_new_tokens=20)
                    
                    response = self.models['processor'].decode(outputs[0], skip_special_tokens=True)
                    
                    # Clean response
                    if response.startswith(prompt):
                        response = response.replace(prompt, "").strip()
                    
                    # Check if response contains food words
                    response_lower = response.lower()
                    for food in self.food_database.keys():
                        if food in response_lower:
                            detected_foods.add(food)
                            logger.info(f"BLIP detected: {food}")
                
                except Exception as e:
                    logger.warning(f"BLIP prompt '{prompt}' failed: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"BLIP detection failed: {e}")
        
        return detected_foods
    
    def _shape_based_detection(self, image: Image.Image) -> Set[str]:
        """Basic shape-based detection"""
        detected_foods = set()
        
        try:
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Detect circles (pizza, apple, orange)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, 
                                     param1=50, param2=30, minRadius=10, maxRadius=100)
            
            if circles is not None:
                detected_foods.add('pizza')  # Assume circular food is pizza
                logger.info("Shape detection: circular shape detected (pizza)")
            
            # Detect rectangles (sandwich, bread)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Rectangular shape
                    detected_foods.add('sandwich')
                    logger.info("Shape detection: rectangular shape detected (sandwich)")
                    break
        
        except Exception as e:
            logger.warning(f"Shape detection failed: {e}")
        
        return detected_foods
    
    def _get_available_methods(self) -> List[str]:
        """Get list of available detection methods"""
        methods = ['Color Analysis', 'Shape Detection']
        
        if self.models.get('yolo_model'):
            methods.append('YOLO')
        
        if self.models.get('blip_model'):
            methods.append('BLIP')
        
        return methods