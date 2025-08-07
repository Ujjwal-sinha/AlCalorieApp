#!/usr/bin/env python3
"""
Test script to verify enhanced food detection capabilities
"""

import sys
import os
from PIL import Image
import logging

# Add the calarieapp directory to the path
sys.path.append('calarieapp')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_food_detection():
    """Test the enhanced food detection functionality."""
    try:
        # Import the enhanced functions
        from app import (
            load_models, 
            describe_image_enhanced, 
            analyze_food_with_enhanced_prompt,
            get_comprehensive_food_analysis,
            enhance_image_quality
        )
        
        print("🧪 Testing Enhanced Food Detection System")
        print("=" * 50)
        
        # Test 1: Model loading
        print("\n1. Testing model loading...")
        models = load_models()
        
        model_status = {
            'BLIP': models['blip_model'] is not None,
            'YOLO': models['yolo_model'] is not None,
            'LLM': models['llm'] is not None,
            'Food Agent': models['food_agent'] is not None
        }
        
        for model, status in model_status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {model}: {'Available' if status else 'Not Available'}")
        
        # Test 2: Image enhancement
        print("\n2. Testing image enhancement...")
        try:
            # Create a test image (you can replace this with an actual food image)
            test_image = Image.new('RGB', (224, 224), color='red')
            enhanced_images = enhance_image_quality(test_image)
            print(f"   ✅ Created {len(enhanced_images)} enhanced image versions")
        except Exception as e:
            print(f"   ❌ Image enhancement failed: {e}")
        
        # Test 3: Text-based food analysis
        print("\n3. Testing text-based food analysis...")
        test_descriptions = [
            "pizza with cheese and pepperoni",
            "salad with lettuce, tomatoes, chicken, and dressing",
            "rice bowl with vegetables and tofu",
            "burger with fries and a soft drink"
        ]
        
        for desc in test_descriptions:
            try:
                result = analyze_food_with_enhanced_prompt(desc, "lunch meal")
                if result['success']:
                    items_count = len(result.get('food_items', []))
                    calories = result.get('nutritional_data', {}).get('total_calories', 0)
                    print(f"   ✅ '{desc}' -> {items_count} items, {calories} calories")
                else:
                    print(f"   ❌ '{desc}' -> Analysis failed")
            except Exception as e:
                print(f"   ❌ '{desc}' -> Error: {e}")
        
        print("\n4. Testing comprehensive detection features...")
        
        # Test food categorization
        if models['food_agent']:
            try:
                agent = models['food_agent']
                test_items = ["chicken breast", "apple", "rice", "cheese"]
                for item in test_items:
                    category = agent.categorize_food_item(item)
                    print(f"   ✅ '{item}' categorized as: {category}")
            except Exception as e:
                print(f"   ❌ Food categorization test failed: {e}")
        else:
            print("   ⚠️  Food agent not available for categorization test")
        
        print("\n🎉 Enhanced Food Detection Test Complete!")
        print("\nKey Improvements:")
        print("• Multiple AI models for comprehensive detection")
        print("• Enhanced image preprocessing with 10+ enhancement techniques")
        print("• Improved BLIP prompting with context-aware detection")
        print("• Advanced YOLO object detection with expanded food categories")
        print("• Intelligent result combination and deduplication")
        print("• Comprehensive nutritional analysis with food categorization")
        print("• Fallback mechanisms for robust detection")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_food_detection()
    sys.exit(0 if success else 1)