#!/usr/bin/env python3
"""
Test script for Expert Food Recognition System
"""

import logging
from PIL import Image
import numpy as np
from utils.expert_food_recognition import ExpertFoodRecognitionSystem
from utils.models import load_models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image():
    """Create a simple test image"""
    # Create a simple colored image that looks like food
    img_array = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add some food-like colors
    img_array[50:250, 100:300] = [255, 200, 100]  # Orange/brown (like pizza)
    img_array[100:200, 150:250] = [255, 255, 255]  # White (like cheese)
    
    return Image.fromarray(img_array)

def test_expert_system():
    """Test the expert food recognition system"""
    print("🧪 Testing Expert Food Recognition System")
    
    # Load models
    print("📦 Loading models...")
    models = load_models()
    
    # Check model availability
    available_models = {k: v is not None for k, v in models.items() 
                       if 'model' in k or 'processor' in k}
    print(f"✅ Available models: {available_models}")
    
    # Initialize expert system
    print("🔧 Initializing expert system...")
    expert_system = ExpertFoodRecognitionSystem(models)
    
    # Create test image
    print("🖼️ Creating test image...")
    test_image = create_test_image()
    print(f"   Image size: {test_image.size}")
    
    # Test food recognition
    print("🔍 Running expert food recognition...")
    detections = expert_system.recognize_food(test_image)
    
    # Display results
    print(f"\n📊 Recognition Results:")
    print(f"   Total detections: {len(detections)}")
    
    for i, detection in enumerate(detections):
        print(f"\n   Detection {i+1}:")
        print(f"     Label: {detection.final_label}")
        print(f"     Confidence: {detection.confidence_score:.3f}")
        print(f"     Method: {detection.detection_method}")
        print(f"     Bounding Box: {detection.bounding_box}")
        print(f"     Classifier Prob: {detection.classifier_probability:.3f}")
        print(f"     CLIP Similarity: {detection.clip_similarity:.3f}")
        if detection.blip_description:
            print(f"     BLIP Description: {detection.blip_description}")
        print(f"     Top Alternatives:")
        for label, score in detection.top_3_alternatives:
            print(f"       - {label}: {score:.3f}")
    
    # Test summary
    print("\n📋 Testing summary generation...")
    summary = expert_system.get_detection_summary(detections)
    print(f"   Summary: {summary}")
    
    print("\n✅ Expert system test completed!")

if __name__ == "__main__":
    test_expert_system()
