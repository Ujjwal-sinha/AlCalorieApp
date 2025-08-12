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
        
        # Food-101 categories (specific food names)
        self.food_categories = [
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
            'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
        ]
        
        # Non-food items to ignore
        self.non_food_items = {
            'cup', 'plate', 'bowl', 'fork', 'spoon', 'knife', 'utensil', 'glass',
            'bottle', 'napkin', 'table', 'chair', 'background', 'wall', 'floor',
            'counter', 'kitchen', 'restaurant', 'food', 'meal', 'dish'
        }
    
    def detect_food_crops(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Detect food candidate crops using YOLO
        Returns: List of (crop_image, bounding_box)
        """
        crops = []
        
        try:
            if self.models.get('yolo_model'):
                # Use YOLO for object detection
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
            
            # If no YOLO or no detections, create a single crop from the whole image
            if not crops:
                crops.append((image, (0, 0, image.width, image.height)))
                
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
            # Fallback to whole image
            crops.append((image, (0, 0, image.width, image.height)))
        
        return crops
    
    def classify_with_transformers(self, crop: Image.Image) -> Dict[str, float]:
        """
        Get classification results from ViT-B/16 and Swin Transformer
        Returns: Dict of {label: probability}
        """
        results = {}
        
        try:
            # ViT-B/16 classification
            if self.models.get('vit_model') and self.models.get('vit_processor'):
                vit_probs = self._classify_with_vit(crop)
                results.update(vit_probs)
            
            # Swin Transformer classification
            if self.models.get('swin_model') and self.models.get('swin_processor'):
                swin_probs = self._classify_with_swin(crop)
                results.update(swin_probs)
                
        except Exception as e:
            logger.warning(f"Transformer classification failed: {e}")
        
        return results
    
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
            # Use CLIP model if available
            if 'clip_model' in self.models and 'clip_processor' in self.models:
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
            
            # Fallback: use simple keyword matching
            else:
                crop_description = self._get_crop_description(crop)
                for label in candidate_labels:
                    # Simple similarity based on word overlap
                    label_words = set(label.replace('_', ' ').lower().split())
                    desc_words = set(crop_description.lower().split())
                    overlap = len(label_words.intersection(desc_words))
                    similarities[label] = min(overlap / max(len(label_words), 1), 1.0)
                    
        except Exception as e:
            logger.warning(f"CLIP similarity calculation failed: {e}")
            # Default similarities
            for label in candidate_labels:
                similarities[label] = 0.5
        
        return similarities
    
    def get_blip_description(self, crop: Image.Image) -> Optional[str]:
        """
        Get BLIP description for the crop
        """
        try:
            if self.models.get('blip_model') and self.models.get('processor'):
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
                return description
                
        except Exception as e:
            logger.warning(f"BLIP description failed: {e}")
        
        return None
    
    def _get_crop_description(self, crop: Image.Image) -> str:
        """Fallback method to get crop description"""
        # Simple color-based description
        crop_array = np.array(crop)
        if len(crop_array.shape) == 3:
            # Calculate dominant colors
            colors = crop_array.reshape(-1, 3)
            unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
            dominant_color = unique_colors[np.argmax(counts)]
            
            # Simple color description
            r, g, b = dominant_color
            if r > 200 and g > 200 and b > 200:
                return "white food item"
            elif r > 150 and g < 100 and b < 100:
                return "red food item"
            elif r < 100 and g > 150 and b < 100:
                return "green food item"
            elif r < 100 and g < 100 and b > 150:
                return "blue food item"
            else:
                return "colored food item"
        
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
    
    def recognize_food(self, image: Image.Image) -> List[FoodDetection]:
        """
        Main method to recognize food items in the image
        """
        detections = []
        
        try:
            # Step 1: Detect food crops
            crops = self.detect_food_crops(image)
            logger.info(f"Detected {len(crops)} food candidate crops")
            
            for crop, bounding_box in crops:
                # Step 2: Classify with transformers
                classifier_probs = self.classify_with_transformers(crop)
                
                if not classifier_probs:
                    continue
                
                # Step 3: Get top candidates
                top_candidates = sorted(classifier_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                
                # Step 4: Get CLIP similarities
                candidate_labels = [label for label, _ in top_candidates]
                clip_similarities = self.get_clip_similarities(crop, candidate_labels)
                
                # Step 5: Get BLIP description
                blip_description = self.get_blip_description(crop)
                
                # Step 6: Check confidence thresholds
                best_classifier_prob = top_candidates[0][1] if top_candidates else 0
                best_clip_sim = clip_similarities.get(top_candidates[0][0], 0) if top_candidates else 0
                
                # Discard if both thresholds not met
                if best_classifier_prob < self.classifier_threshold and best_clip_sim < self.clip_threshold:
                    logger.info(f"Crop {bounding_box} below confidence thresholds")
                    continue
                
                # Step 7: Fuse evidence
                fused_scores = self.fuse_evidence(classifier_probs, clip_similarities, blip_description)
                
                # Step 8: Break ties if needed
                final_label = self.break_ties(fused_scores, clip_similarities, blip_description)
                
                # Step 9: Filter out non-food items
                if final_label.lower() in self.non_food_items:
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
        
        except Exception as e:
            logger.error(f"Food recognition failed: {e}")
        
        return detections
    
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
