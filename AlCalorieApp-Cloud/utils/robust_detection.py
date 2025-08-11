"""
Robust Food Detection System
Uses multiple detection methods with strong fallbacks
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

class RobustFoodDetector:
    """
    Robust food detection system with multiple fallback methods
    """
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        
        # Comprehensive food database
        self.food_database = {
            # Common foods with visual characteristics
            'pizza': {'colors': [(200, 150, 100), (255, 200, 150)], 'shapes': 'circular'},
            'burger': {'colors': [(150, 100, 50), (200, 150, 100)], 'shapes': 'layered'},
            'sandwich': {'colors': [(200, 180, 120), (150, 120, 80)], 'shapes': 'rectangular'},
            'chicken': {'colors': [(200, 180, 140), (180, 160, 120)], 'shapes': 'irregular'},
            'rice': {'colors': [(240, 240, 230), (255, 255, 245)], 'shapes': 'granular'},
            'bread': {'colors': [(200, 160, 100), (220, 180, 120)], 'shapes': 'loaf'},
            'tomato': {'colors': [(200, 50, 50), (255, 100, 100)], 'shapes': 'round'},
            'apple': {'colors': [(200, 50, 50), (100, 150, 50)], 'shapes': 'round'},
            'banana': {'colors': [(255, 220, 50), (255, 240, 100)], 'shapes': 'curved'},
            'orange': {'colors': [(255, 150, 50), (255, 180, 80)], 'shapes': 'round'},
            'broccoli': {'colors': [(50, 150, 50), (80, 180, 80)], 'shapes': 'tree-like'},
            'carrot': {'colors': [(255, 140, 50), (255, 160, 80)], 'shapes': 'elongated'},
            'potato': {'colors': [(200, 180, 140), (180, 160, 120)], 'shapes': 'oval'},
            'cheese': {'colors': [(255, 220, 100), (255, 240, 150)], 'shapes': 'block'},
            'salad': {'colors': [(100, 200, 100), (150, 220, 150)], 'shapes': 'mixed'},
            'pasta': {'colors': [(255, 240, 200), (240, 220, 180)], 'shapes': 'stringy'},
            'egg': {'colors': [(255, 255, 240), (255, 220, 100)], 'shapes': 'oval'},
            'fish': {'colors': [(200, 180, 160), (180, 160, 140)], 'shapes': 'fish-like'},
            'soup': {'colors': [(200, 150, 100), (180, 130, 80)], 'shapes': 'liquid'},
            'cake': {'colors': [(255, 240, 220), (200, 150, 100)], 'shapes': 'layered'}
        }
    
    def detect_food_robust(self, image: Image.Image) -> Dict[str, Any]:
        """
        Robust food detection using multiple methods
        """
        try:
            logger.info("Starting robust food detection...")
            
            # Convert image
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            detected_foods = set()
            confidence_scores = {}
            
            # Method 1: YOLO detection (most reliable)
            yolo_foods = self._yolo_detection_robust(image)
            for food in yolo_foods:
                detected_foods.add(food)
                confidence_scores[food] = 0.9
            
            # Method 2: Color-based detection
            color_foods = self._color_based_detection(image)
            for food in color_foods:
                detected_foods.add(food)
                if food not in confidence_scores:
                    confidence_scores[food] = 0.7
            
            # Method 3: Simple BLIP detection (if available)
            blip_foods = self._simple_blip_detection(image)
            for food in blip_foods:
                detected_foods.add(food)
                if food not in confidence_scores:
                    confidence_scores[food] = 0.6
            
            # Method 4: Pattern-based detection
            pattern_foods = self._pattern_based_detection(image)
            for food in pattern_foods:
                detected_foods.add(food)
                if food not in confidence_scores:
                    confidence_scores[food] = 0.5
            
            # Ensure we always have some result
            if not detected_foods:
                detected_foods = {'food item'}
                confidence_scores['food item'] = 0.3
            
            return {
                'detected_foods': list(detected_foods),
                'confidence_scores': confidence_scores,
                'detection_method': 'robust_multi_method',
                'total_foods': len(detected_foods)
            }
            
        except Exception as e:
            logger.error(f"Robust detection failed: {e}")
            return {
                'detected_foods': ['food item'],
                'confidence_scores': {'food item': 0.3},
                'detection_method': 'fallback',
                'total_foods': 1
            }
    
    def _yolo_detection_robust(self, image: Image.Image) -> Set[str]:
        """Robust YOLO detection"""
        detected_foods = set()
        
        if not self.models.get('yolo_model'):
            return detected_foods
        
        try:
            img_array = np.array(image)
            
            # Try multiple confidence levels
            for conf in [0.1, 0.2, 0.3, 0.4]:
                try:
                    results = self.models['yolo_model'](img_array, conf=conf)
                    
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
                                
                                if class_name in food_classes and confidence > conf:
                                    detected_foods.add(class_name)
                                    logger.info(f"YOLO detected: {class_name} (conf: {confidence:.3f})")
                    
                    if detected_foods:  # If we found something, stop
                        break
                        
                except Exception as e:
                    logger.warning(f"YOLO at conf {conf} failed: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
        
        return detected_foods
    
    def _color_based_detection(self, image: Image.Image) -> Set[str]:
        """Color-based food detection"""
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
                    
                    if distance < 80:  # Threshold for color similarity
                        detected_foods.add(food)
                        logger.info(f"Color detection: {food} (distance: {distance:.1f})")
                        break
        
        except Exception as e:
            logger.warning(f"Color detection failed: {e}")
        
        return detected_foods
    
    def _simple_blip_detection(self, image: Image.Image) -> Set[str]:
        """Simple BLIP detection with basic prompts"""
        detected_foods = set()
        
        if not self.models.get('processor') or not self.models.get('blip_model'):
            return detected_foods
        
        try:
            device = self.models.get('device', 'cpu')
            
            # Simple prompts that work
            prompts = ["What is this?", "Describe this image."]
            
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
    
    def _pattern_based_detection(self, image: Image.Image) -> Set[str]:
        """Pattern-based detection using image analysis"""
        detected_foods = set()
        
        try:
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Detect circles (could be pizza, apple, orange)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
            
            if circles is not None:
                # Analyze color of circular regions
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Extract color from circular region
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, -1)
                    mean_color = cv2.mean(img_cv, mask=mask)[:3]
                    
                    # Check against circular foods
                    circular_foods = ['pizza', 'apple', 'orange', 'tomato']
                    for food in circular_foods:
                        food_colors = self.food_database.get(food, {}).get('colors', [])
                        for color in food_colors:
                            distance = np.linalg.norm(np.array(mean_color) - np.array(color))
                            if distance < 60:
                                detected_foods.add(food)
                                logger.info(f"Pattern detection: {food} (circular)")
                                break
            
            # Detect rectangular shapes (could be sandwich, bread)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If it's roughly rectangular
                if len(approx) == 4:
                    rect_foods = ['sandwich', 'bread', 'pizza']
                    detected_foods.update(rect_foods[:1])  # Add one rectangular food
                    logger.info("Pattern detection: rectangular shape detected")
                    break
        
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
        
        return detected_foods