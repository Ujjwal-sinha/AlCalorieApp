#!/usr/bin/env python3
"""
Simplified Expert Food Recognition System
Uses only YOLO11m for accurate food detection and classification
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
    """Food detection result with YOLO evidence"""
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    final_label: str
    confidence_score: float
    detection_method: str = "YOLO11m"
    classifier_probability: float = 0.0
    clip_similarity: float = 0.0
    top_3_alternatives: List[Tuple[str, float]] = None
    blip_description: str = None
    
    def __post_init__(self):
        if self.top_3_alternatives is None:
            self.top_3_alternatives = []

class YOLO11mFoodRecognitionSystem:
    """
    Simplified food recognition system using only YOLO11m
    """
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.confidence_threshold = 0.15  # Lower threshold to detect more items
        self.min_crop_size = 30  # Smaller minimum crop size
        
        # Food-related COCO classes (YOLO11m is trained on COCO dataset)
        self.food_classes = {
            # Fruits
            'apple', 'orange', 'banana', 'strawberry', 'grape', 'pineapple', 'mango', 'peach', 'pear', 'kiwi',
            # Vegetables
            'carrot', 'broccoli', 'tomato', 'potato', 'onion', 'garlic', 'pepper', 'cucumber', 'celery', 'mushroom',
            'corn', 'pea', 'bean', 'cabbage', 'cauliflower', 'asparagus', 'zucchini', 'eggplant', 'lettuce', 'spinach',
            # Prepared Foods
            'pizza', 'hot dog', 'sandwich', 'cake', 'donut', 'cookie', 'bread', 'pasta', 'rice', 'noodle',
            'burger', 'hamburger', 'cheeseburger', 'taco', 'burrito', 'sushi', 'salad', 'soup', 'stew',
            # Meat & Protein
            'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'egg', 'meat', 'steak', 'bacon', 'ham', 'turkey',
            'duck', 'lamb', 'veal', 'sausage', 'meatball', 'shrimp', 'lobster', 'crab',
            # Dairy
            'milk', 'cheese', 'yogurt', 'butter', 'cream', 'ice cream',
            # Beverages
            'coffee', 'tea', 'juice', 'soda', 'water', 'wine', 'beer',
            # Utensils & Containers (food-related)
            'cup', 'bowl', 'spoon', 'fork', 'knife', 'wine glass', 'bottle', 'plate', 'dish', 'mug',
            # Appliances (kitchen-related)
            'microwave', 'oven', 'toaster', 'refrigerator', 'sink', 'stove', 'blender', 'mixer',
            # Furniture (dining-related)
            'dining table', 'chair', 'table'
        }
        
        # Non-food items to exclude
        self.non_food_items = {
            'cat', 'dog', 'bird', 'horse', 'cow', 'sheep', 'pig', 'elephant', 'giraffe', 'zebra', 'lion', 'tiger',
            'bear', 'wolf', 'fox', 'deer', 'rabbit', 'squirrel', 'mouse', 'rat', 'hamster', 'guinea pig',
            'book', 'cell phone', 'remote', 'keyboard', 'mouse', 'laptop', 'tv', 'computer', 'monitor',
            'chair', 'couch', 'bed', 'sofa', 'toilet', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
            'skateboard', 'surfboard', 'tennis racket', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush', 'potted plant', 'tree', 'flower', 'grass', 'sky', 'cloud', 'sun',
            'moon', 'star', 'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'airplane', 'train', 'boat',
            'traffic light', 'stop sign', 'fire hydrant', 'bench', 'person', 'child', 'man', 'woman'
        }
        
        # Enhanced food keywords for better classification
        self.food_keywords = {
            'apple': ['apple', 'red apple', 'green apple', 'fruit'],
            'orange': ['orange', 'citrus', 'fruit'],
            'banana': ['banana', 'yellow fruit'],
            'carrot': ['carrot', 'vegetable', 'orange vegetable'],
            'broccoli': ['broccoli', 'green vegetable', 'vegetable'],
            'pizza': ['pizza', 'cheese pizza', 'pepperoni pizza'],
            'hot dog': ['hot dog', 'sausage', 'frankfurter'],
            'sandwich': ['sandwich', 'burger', 'hamburger', 'cheeseburger'],
            'cake': ['cake', 'birthday cake', 'dessert'],
            'donut': ['donut', 'doughnut', 'pastry'],
            'cookie': ['cookie', 'biscuit', 'dessert'],
            'cup': ['cup', 'mug', 'drink'],
            'bowl': ['bowl', 'dish', 'plate'],
            'spoon': ['spoon', 'utensil'],
            'fork': ['fork', 'utensil'],
            'knife': ['knife', 'utensil'],
            'wine glass': ['wine glass', 'glass', 'drink'],
            'bottle': ['bottle', 'water bottle', 'drink']
        }
    
    def detect_food_crops(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Detect food candidate crops using YOLO11m with enhanced strategy
        Returns: List of (crop_image, bounding_box)
        """
        crops = []
        
        try:
            if not self.models.get('yolo_model'):
                logger.warning("YOLO11m model not available")
                # Fallback to whole image
                crops.append((image, (0, 0, image.width, image.height)))
                return crops
            
            logger.info("Using YOLO11m for enhanced food crop detection")
            
            # Strategy 1: Run YOLO11m on the full image
            yolo_results = self.models['yolo_model'](image, verbose=False)
            
            for result in yolo_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.models['yolo_model'].names[class_id]
                        
                        # Process if confidence is above threshold and it's a food item
                        if confidence >= self.confidence_threshold and self._is_food_item(class_name):
                            # Crop the image
                            crop = image.crop((x1, y1, x2, y2))
                            
                            # Keep reasonable sized crops
                            if crop.width > self.min_crop_size and crop.height > self.min_crop_size:
                                crops.append((crop, (x1, y1, x2, y2)))
                                logger.info(f"YOLO11m detected food: {class_name} (conf: {confidence:.2f}) at {crop.width}x{crop.height}")
            
            # Strategy 2: If we have crops, run YOLO11m on each crop for more detailed detection
            additional_crops = []
            for crop, bbox in crops[:3]:  # Limit to first 3 crops to avoid too many detections
                try:
                    crop_results = self.models['yolo_model'](crop, verbose=False)
                    for result in crop_results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                confidence = float(box.conf[0].cpu().numpy())
                                class_id = int(box.cls[0].cpu().numpy())
                                class_name = self.models['yolo_model'].names[class_id]
                                
                                if confidence >= self.confidence_threshold and self._is_food_item(class_name):
                                    # Adjust coordinates relative to original image
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    
                                    # Convert to original image coordinates
                                    orig_x1 = bbox[0] + x1
                                    orig_y1 = bbox[1] + y1
                                    orig_x2 = bbox[0] + x2
                                    orig_y2 = bbox[1] + y2
                                    
                                    # Create new crop
                                    new_crop = image.crop((orig_x1, orig_y1, orig_x2, orig_y2))
                                    if new_crop.width > self.min_crop_size and new_crop.height > self.min_crop_size:
                                        additional_crops.append((new_crop, (orig_x1, orig_y1, orig_x2, orig_y2)))
                except Exception as e:
                    logger.warning(f"Error processing crop: {e}")
            
            # Add additional crops
            crops.extend(additional_crops)
            
            # Strategy 3: If still no crops, use grid-based cropping
            if not crops:
                logger.info("No YOLO11m detections, using grid-based cropping")
                crops = self._create_grid_crops(image)
            
            # Strategy 4: If still no crops, use the whole image
            if not crops:
                logger.info("No crops detected, using whole image")
                crops.append((image, (0, 0, image.width, image.height)))
                
        except Exception as e:
            logger.warning(f"Food crop detection failed: {e}")
            # Fallback to whole image
            crops.append((image, (0, 0, image.width, image.height)))
        
        logger.info(f"Total crops detected: {len(crops)}")
        return crops
    
    def recognize_food(self, image: Image.Image) -> Dict[str, Any]:
        """
        Main food recognition function using only YOLO11m
        """
        try:
            logger.info("Starting YOLO11m food recognition")
            
            # Get crops from the image
            crops = self.detect_food_crops(image)
            
            # Process each crop with YOLO11m
            all_detections = []
            
            for crop, bounding_box in crops:
                crop_detections = self._get_yolo11m_detection(crop, bounding_box)
                all_detections.extend(crop_detections)
            
            # Remove duplicates and low confidence detections
            filtered_detections = self._filter_detections(all_detections)
            
            logger.info(f"YOLO11m recognition complete: {len(filtered_detections)} detections")
            
            return {
                "success": True,
                "detections": filtered_detections,
                "total_detections": len(filtered_detections),
                "method": "YOLO11m"
            }
            
        except Exception as e:
            logger.error(f"Food recognition failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "detections": [],
                "total_detections": 0,
                "method": "YOLO11m"
            }
    
    def _get_yolo11m_detection(self, crop: Image.Image, bounding_box: Tuple[int, int, int, int]) -> List[FoodDetection]:
        """
        Get YOLO11m detections for a specific crop with multiple confidence thresholds
        """
        detections = []
        
        try:
            if not self.models.get('yolo_model'):
                return detections
            
            # Run YOLO11m on the crop with different confidence thresholds
            confidence_thresholds = [0.15, 0.25, 0.35]  # Multiple thresholds to catch more items
            
            for threshold in confidence_thresholds:
                results = self.models['yolo_model'](crop, verbose=False, conf=threshold)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.models['yolo_model'].names[class_id]
                            
                            # Only include food items above our base threshold
                            if confidence >= self.confidence_threshold and self._is_food_item(class_name):
                                # Improve the label
                                improved_label = self._improve_food_label(class_name)
                                
                                detection = FoodDetection(
                                    bounding_box=bounding_box,
                                    final_label=improved_label,
                                    confidence_score=confidence,
                                    detection_method="YOLO11m"
                                )
                                detections.append(detection)
                                
                                logger.info(f"Food detection: {improved_label} (conf: {confidence:.2f})")
        
        except Exception as e:
            logger.warning(f"YOLO11m detection failed for crop: {e}")
        
        return detections
    
    def _is_food_item(self, class_name: str) -> bool:
        """
        Check if a detected item is food-related
        """
        class_lower = class_name.lower()
        
        # Check if it's in our food classes
        if class_lower in self.food_classes:
            return True
        
        # Check if it's in our non-food items (exclude these)
        if class_lower in self.non_food_items:
            return False
        
        # Additional food-related keywords
        food_keywords = [
            'food', 'meal', 'dish', 'cuisine', 'ingredient', 'spice', 'herb', 'seasoning',
            'sauce', 'dressing', 'condiment', 'spread', 'dip', 'snack', 'dessert', 'sweet',
            'drink', 'beverage', 'liquid', 'soup', 'stew', 'curry', 'stir fry', 'grill',
            'bake', 'roast', 'fry', 'steam', 'boil', 'cook', 'prepared', 'fresh', 'organic'
        ]
        
        return any(keyword in class_lower for keyword in food_keywords)
    
    def _create_grid_crops(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Create grid-based crops for food detection
        """
        crops = []
        width, height = image.size
        
        # Create a 3x3 grid
        grid_size = 3
        crop_width = width // grid_size
        crop_height = height // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = i * crop_width
                y1 = j * crop_height
                x2 = x1 + crop_width
                y2 = y1 + crop_height
                
                # Ensure we don't go beyond image boundaries
                x2 = min(x2, width)
                y2 = min(y2, height)
                
                crop = image.crop((x1, y1, x2, y2))
                if crop.width > self.min_crop_size and crop.height > self.min_crop_size:
                    crops.append((crop, (x1, y1, x2, y2)))
        
        return crops
    
    def _improve_food_label(self, class_name: str) -> str:
        """
        Improve the food label for better readability
        """
        # Convert to title case and replace underscores
        improved = class_name.replace('_', ' ').title()
        
        # Handle specific cases
        if improved.lower() in ['hot dog']:
            return 'Hot Dog'
        elif improved.lower() in ['wine glass']:
            return 'Wine Glass'
        elif improved.lower() in ['dining table']:
            return 'Dining Table'
        elif improved.lower() in ['potted plant']:
            return 'Potted Plant'
        elif improved.lower() in ['hair drier']:
            return 'Hair Dryer'
        elif improved.lower() in ['cell phone']:
            return 'Cell Phone'
        elif improved.lower() in ['baseball bat']:
            return 'Baseball Bat'
        elif improved.lower() in ['baseball glove']:
            return 'Baseball Glove'
        elif improved.lower() in ['tennis racket']:
            return 'Tennis Racket'
        elif improved.lower() in ['sports ball']:
            return 'Sports Ball'
        elif improved.lower() in ['water bottle']:
            return 'Water Bottle'
        
        return improved
    
    def _filter_detections(self, detections: List[FoodDetection]) -> List[FoodDetection]:
        """
        Filter and deduplicate detections with enhanced logic
        """
        if not detections:
            return []
        
        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x.confidence_score, reverse=True)
        
        # Enhanced deduplication - allow similar items if confidence is high enough
        filtered = []
        seen_labels = {}
        
        for detection in sorted_detections:
            label_lower = detection.final_label.lower()
            
            # If we haven't seen this label, add it
            if label_lower not in seen_labels:
                seen_labels[label_lower] = detection
                filtered.append(detection)
            else:
                # If we have seen this label, only replace if new detection has higher confidence
                existing_confidence = seen_labels[label_lower].confidence_score
                if detection.confidence_score > existing_confidence + 0.1:  # 10% improvement threshold
                    # Replace the existing detection
                    filtered.remove(seen_labels[label_lower])
                    seen_labels[label_lower] = detection
                    filtered.append(detection)
        
        # Sort again by confidence
        filtered = sorted(filtered, key=lambda x: x.confidence_score, reverse=True)
        
        return filtered
    
    def get_detection_summary(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the detection results
        """
        if not detection_results.get("success", False):
            return {
                "success": False,
                "error": detection_results.get("error", "Unknown error"),
                "total_detections": 0,
                "detected_items": [],
                "method": "YOLO11m"
            }
        
        detections = detection_results.get("detections", [])
        
        # Extract detected items
        detected_items = [detection.final_label for detection in detections]
        
        # Calculate average confidence
        avg_confidence = 0
        if detections:
            avg_confidence = sum(detection.confidence_score for detection in detections) / len(detections)
        
        return {
            "success": True,
            "total_detections": len(detections),
            "detected_items": detected_items,
            "average_confidence": avg_confidence,
            "method": "YOLO11m",
            "detection_details": [
                {
                    "label": detection.final_label,
                    "confidence": detection.confidence_score,
                    "method": detection.detection_method
                }
                for detection in detections
            ]
        }

# Alias for backward compatibility
ExpertFoodRecognitionSystem = YOLO11mFoodRecognitionSystem

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
                        st.markdown("#### �� Model Breakdown")
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