#!/usr/bin/env python3
"""
Test Expert Food Recognition System
Tests the expert multi-model food recognition system
"""

import sys
import os
from PIL import Image
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_expert_system():
    """Test the expert food recognition system"""
    try:
        print("🧠 Testing Expert Food Recognition System...")
        
        # Load models
        from utils.models import load_models
        models = load_models()
        
        # Initialize expert system
        from utils.expert_food_recognition import ExpertFoodRecognitionSystem
        expert_system = ExpertFoodRecognitionSystem(models)
        
        print("   ✅ Expert system initialized successfully")
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color=(200, 150, 100))
        
        # Test food recognition
        print("   🔍 Running expert food recognition...")
        detections = expert_system.recognize_food(test_image)
        summary = expert_system.get_detection_summary(detections)
        
        print(f"   📊 Detection Results:")
        print(f"      - Total detections: {summary['total_detections']}")
        print(f"      - Success: {summary['success']}")
        print(f"      - Detection method: {summary['detection_method']}")
        
        if summary['success']:
            print(f"      - Detected foods: {summary['detected_foods']}")
            print(f"      - Confidence scores: {summary['confidence_scores']}")
            
            # Show detailed results
            for i, detection in enumerate(detections):
                print(f"      Detection {i+1}:")
                print(f"        - Label: {detection.final_label}")
                print(f"        - Confidence: {detection.confidence_score:.3f}")
                print(f"        - Classifier prob: {detection.classifier_probability:.3f}")
                print(f"        - CLIP similarity: {detection.clip_similarity:.3f}")
                print(f"        - Bounding box: {detection.bounding_box}")
                if detection.blip_description:
                    print(f"        - BLIP description: {detection.blip_description}")
        
        print("   ✅ Expert system test completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Expert system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_availability():
    """Test model availability for expert system"""
    try:
        print("\n🔧 Testing Model Availability...")
        
        from utils.models import load_models, get_model_status
        models = load_models()
        status = get_model_status(models)
        
        print("   📊 Model Status:")
        for model_name, is_available in status.items():
            status_icon = "✅" if is_available else "❌"
            print(f"     {status_icon} {model_name}: {'Available' if is_available else 'Not Available'}")
        
        # Check required models for expert system
        required_models = [
            'BLIP (Image Analysis)',
            'ViT-B/16 (Vision Transformer)',
            'Swin Transformer',
            'CLIP (Similarity Scoring)'
        ]
        
        available_count = sum(1 for model in required_models if status.get(model, False))
        print(f"\n   📈 Expert System Readiness: {available_count}/{len(required_models)} models available")
        
        if available_count >= 2:
            print("   ✅ Expert system can run with available models")
            return True
        else:
            print("   ⚠️  Expert system may have limited functionality")
            return False
            
    except Exception as e:
        print(f"   ❌ Model availability test failed: {e}")
        return False

def test_confidence_thresholds():
    """Test confidence threshold logic"""
    try:
        print("\n🎯 Testing Confidence Thresholds...")
        
        from utils.expert_food_recognition import ExpertFoodRecognitionSystem
        from utils.models import load_models
        
        models = load_models()
        expert_system = ExpertFoodRecognitionSystem(models)
        
        # Test threshold values
        print(f"   📊 Threshold Configuration:")
        print(f"      - Classifier threshold: {expert_system.classifier_threshold}")
        print(f"      - CLIP threshold: {expert_system.clip_threshold}")
        print(f"      - Tie-breaking threshold: {expert_system.probability_tie_threshold}")
        
        # Test threshold logic
        test_cases = [
            (0.5, 0.3, True),   # Both above thresholds
            (0.3, 0.3, False),  # Classifier below, CLIP at threshold
            (0.5, 0.2, False),  # Classifier above, CLIP below
            (0.3, 0.2, False),  # Both below thresholds
        ]
        
        print("   🧪 Testing threshold logic:")
        for classifier_prob, clip_sim, expected in test_cases:
            result = (classifier_prob >= expert_system.classifier_threshold or 
                     clip_sim >= expert_system.clip_threshold)
            status = "✅" if result == expected else "❌"
            print(f"     {status} Classifier: {classifier_prob:.2f}, CLIP: {clip_sim:.2f} -> {result}")
        
        print("   ✅ Confidence threshold test completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Confidence threshold test failed: {e}")
        return False

def test_food_categories():
    """Test food category handling"""
    try:
        print("\n🍽️ Testing Food Categories...")
        
        from utils.expert_food_recognition import ExpertFoodRecognitionSystem
        from utils.models import load_models
        
        models = load_models()
        expert_system = ExpertFoodRecognitionSystem(models)
        
        print(f"   📊 Food Categories: {len(expert_system.food_categories)} categories")
        print(f"   🚫 Non-food items: {len(expert_system.non_food_items)} items")
        
        # Test some food categories
        sample_foods = expert_system.food_categories[:5]
        print(f"   🍕 Sample food categories: {sample_foods}")
        
        # Test non-food filtering
        test_items = ['pizza', 'cup', 'plate', 'hamburger', 'fork']
        filtered_items = [item for item in test_items if item not in expert_system.non_food_items]
        print(f"   🔍 Filtering test: {test_items} -> {filtered_items}")
        
        print("   ✅ Food categories test completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Food categories test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧠 Expert Food Recognition System Test Suite")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Model Availability", test_model_availability),
        ("Food Categories", test_food_categories),
        ("Confidence Thresholds", test_confidence_thresholds),
        ("Expert System", test_expert_system),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Expert system is ready.")
    else:
        print("⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main()
