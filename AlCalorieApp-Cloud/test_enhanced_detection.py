#!/usr/bin/env python3
"""
Test Enhanced Detection - Verify All Food Items Are Detected
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

def test_enhanced_detection():
    """Test that enhanced detection captures all food items"""
    try:
        print("🧪 Testing Enhanced Food Detection...")
        print("=" * 60)
        
        # Load models
        print("🔄 Loading models...")
        from utils.models import load_models
        models = load_models()
        
        if "error" in models:
            print("❌ Models failed to load:", models["error"])
            return False
        
        print("✅ Models loaded successfully")
        
        # Test Vision Transformer detection
        print("\n🔍 Testing Vision Transformer Detection...")
        from utils.vision_transformer_detection import VisionTransformerFoodDetector
        vit_detector = VisionTransformerFoodDetector(models)
        
        # Test with different food colors
        test_images = [
            ('Red Food (Apple/Tomato)', Image.new('RGB', (224, 224), color=(200, 50, 50))),
            ('Green Food (Broccoli/Lettuce)', Image.new('RGB', (224, 224), color=(50, 150, 50))),
            ('Yellow Food (Banana/Corn)', Image.new('RGB', (224, 224), color=(255, 220, 50))),
            ('Orange Food (Carrot/Orange)', Image.new('RGB', (224, 224), color=(255, 165, 0))),
            ('Brown Food (Bread/Chocolate)', Image.new('RGB', (224, 224), color=(139, 69, 19))),
            ('White Food (Rice/Milk)', Image.new('RGB', (224, 224), color=(240, 240, 240))),
            ('Purple Food (Eggplant/Grape)', Image.new('RGB', (224, 224), color=(128, 0, 128))),
            ('Pink Food (Salmon/Shrimp)', Image.new('RGB', (224, 224), color=(255, 192, 203))),
            ('Mixed Food (Complex)', Image.new('RGB', (224, 224), color=(150, 100, 75)))
        ]
        
        total_detections = 0
        for name, test_image in test_images:
            print(f"\n📸 Testing {name}...")
            result = vit_detector.detect_food_with_transformers(test_image)
            
            detected_foods = result.get('detected_foods', [])
            confidence_scores = result.get('confidence_scores', {})
            detection_method = result.get('detection_method', 'unknown')
            models_used = result.get('models_used', [])
            success = result.get('success', False)
            
            print(f"   📊 Results:")
            print(f"      • Success: {'✅' if success else '❌'}")
            print(f"      • Detected foods: {detected_foods}")
            print(f"      • Detection method: {detection_method}")
            print(f"      • Models used: {models_used}")
            print(f"      • Total detections: {len(detected_foods)}")
            print(f"      • Confidence scores: {confidence_scores}")
            
            total_detections += len(detected_foods)
            
            if not success:
                error = result.get('error', 'Unknown error')
                print(f"      • Error: {error}")
        
        print(f"\n📈 Vision Transformer Summary:")
        print(f"   • Total foods detected: {total_detections}")
        print(f"   • Average per image: {total_detections / len(test_images):.1f}")
        
        # Test Enhanced Analysis
        print(f"\n🔗 Testing Enhanced Analysis Integration...")
        from utils.analysis import describe_image_enhanced
        
        # Test with a complex image
        mixed_image = Image.new('RGB', (224, 224), color=(150, 100, 75))
        description = describe_image_enhanced(mixed_image, models)
        
        print(f"   📝 Analysis Description:")
        print(f"      • {description}")
        
        # Test Food Agent
        print(f"\n🤖 Testing Food Agent Integration...")
        from utils.food_agent import FoodAgent
        
        agent = FoodAgent(models)
        
        # Test with different food types
        agent_test_images = [
            ('Protein-rich (Red/Brown)', Image.new('RGB', (224, 224), color=(180, 100, 80))),
            ('Vegetable-rich (Green)', Image.new('RGB', (224, 224), color=(60, 180, 60))),
            ('Grain-rich (Yellow/White)', Image.new('RGB', (224, 224), color=(220, 200, 150)))
        ]
        
        for name, test_image in agent_test_images:
            print(f"\n   📸 Testing {name} with Food Agent...")
            result = agent.get_comprehensive_analysis(test_image)
            
            if "error" in result:
                print(f"      ⚠️ Agent had issues: {result['error']}")
            else:
                detected_foods = result.get('detected_foods', [])
                vit_detected = result.get('vit_detected_foods', [])
                description_foods = result.get('description_foods', [])
                detection_method = result.get('detection_method', 'unknown')
                models_used = result.get('models_used', [])
                
                print(f"      📊 Agent Results:")
                print(f"         • All detected foods: {detected_foods}")
                print(f"         • ViT detected foods: {vit_detected}")
                print(f"         • Description foods: {description_foods}")
                print(f"         • Detection method: {detection_method}")
                print(f"         • Models used: {models_used}")
                print(f"         • Total foods: {len(detected_foods)}")
        
        # Test comprehensive food vocabulary
        print(f"\n🍽️ Testing Comprehensive Food Vocabulary...")
        comprehensive_foods = [
            # Fruits
            'apple', 'banana', 'orange', 'strawberry', 'pineapple', 'mango', 'grape', 'peach',
            # Vegetables
            'broccoli', 'carrot', 'spinach', 'tomato', 'cucumber', 'bell pepper', 'onion', 'garlic',
            # Proteins
            'chicken', 'beef', 'salmon', 'egg', 'tofu', 'lentil', 'chickpea', 'quinoa',
            # Grains
            'rice', 'pasta', 'bread', 'oatmeal', 'corn', 'wheat', 'barley', 'millet',
            # Dairy
            'milk', 'cheese', 'yogurt', 'butter', 'cream', 'ice cream', 'cottage cheese',
            # Nuts & Seeds
            'almond', 'walnut', 'peanut', 'sunflower seed', 'chia seed', 'flax seed',
            # Beverages
            'coffee', 'tea', 'juice', 'water', 'smoothie', 'milkshake',
            # Prepared Foods
            'pizza', 'hamburger', 'sandwich', 'soup', 'salad', 'stew', 'curry', 'sushi',
            # Generic Categories
            'vegetables', 'protein', 'grains', 'fruits', 'dairy', 'beverages', 'spices', 'herbs'
        ]
        
        print(f"   📋 Food vocabulary coverage...")
        print(f"      • Total foods in vocabulary: {len(comprehensive_foods)}")
        
        # Test how many foods can be detected
        detected_count = 0
        for food in comprehensive_foods:
            if food in vit_detector.food_vocabulary:
                detected_count += 1
        
        print(f"      • Foods in detection vocabulary: {detected_count}/{len(comprehensive_foods)}")
        print(f"      • Coverage percentage: {(detected_count/len(comprehensive_foods))*100:.1f}%")
        
        # Test minimum detection requirements
        print(f"\n✅ Enhanced Detection Test Complete!")
        
        # Check if we meet minimum requirements
        min_detections_per_image = 5
        min_total_detections = len(test_images) * min_detections_per_image
        
        if total_detections >= min_total_detections:
            print(f"✅ SUCCESS: Detected {total_detections} foods (minimum: {min_total_detections})")
            return True
        else:
            print(f"❌ FAILED: Only detected {total_detections} foods (minimum: {min_total_detections})")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_detection()
    if success:
        print("\n🎉 Enhanced detection is working! All food items are being detected.")
    else:
        print("\n💥 Enhanced detection needs improvement. Check the logs above.")
        sys.exit(1)
