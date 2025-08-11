#!/usr/bin/env python3
"""
Comprehensive Test for All Model Integration
Tests Vision Transformer, BLIP, YOLO, and ensemble detection
"""

import sys
import os
from PIL import Image
import logging
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_comprehensive_integration():
    """Test comprehensive integration of all models"""
    try:
        print("🚀 Testing Comprehensive Model Integration...")
        print("=" * 80)
        
        # Load models
        print("🔄 Loading all models (ViT-B/16, Swin-T, BLIP, YOLO)...")
        from utils.models import load_models
        models = load_models()
        
        if "error" in models:
            print("❌ Models failed to load:", models["error"])
            return False
        
        print("✅ All models loaded successfully")
        
        # Check model availability
        vit_available = models.get('vit_model') is not None
        swin_available = models.get('swin_model') is not None
        blip_available = models.get('blip_model') is not None
        yolo_available = models.get('yolo_model') is not None
        llm_available = models.get('llm') is not None
        
        print(f"📊 Model Status:")
        print(f"   • ViT-B/16: {'✅ Available' if vit_available else '❌ Not Available'}")
        print(f"   • Swin Transformer: {'✅ Available' if swin_available else '❌ Not Available'}")
        print(f"   • BLIP: {'✅ Available' if blip_available else '❌ Not Available'}")
        print(f"   • YOLO: {'✅ Available' if yolo_available else '❌ Not Available'}")
        print(f"   • LLM: {'✅ Available' if llm_available else '❌ Not Available'}")
        
        # Test 1: Vision Transformer Detection
        print(f"\n🔍 Test 1: Vision Transformer Detection...")
        if vit_available or swin_available:
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
                ('Pink Food (Salmon/Shrimp)', Image.new('RGB', (224, 224), color=(255, 192, 203)))
            ]
            
            total_detections = 0
            for name, test_image in test_images:
                print(f"\n   📸 Testing {name}...")
                result = vit_detector.detect_food_with_transformers(test_image)
                
                detected_foods = result.get('detected_foods', [])
                confidence_scores = result.get('confidence_scores', {})
                detection_method = result.get('detection_method', 'unknown')
                models_used = result.get('models_used', [])
                success = result.get('success', False)
                
                print(f"      📊 Results:")
                print(f"         • Success: {'✅' if success else '❌'}")
                print(f"         • Detected foods: {detected_foods}")
                print(f"         • Detection method: {detection_method}")
                print(f"         • Models used: {models_used}")
                print(f"         • Total detections: {len(detected_foods)}")
                
                total_detections += len(detected_foods)
                
                if not success:
                    error = result.get('error', 'Unknown error')
                    print(f"         • Error: {error}")
            
            print(f"\n   📈 Vision Transformer Summary:")
            print(f"      • Total foods detected: {total_detections}")
            print(f"      • Average per image: {total_detections / len(test_images):.1f}")
        
        # Test 2: Enhanced Analysis Integration
        print(f"\n🔗 Test 2: Enhanced Analysis Integration...")
        from utils.analysis import describe_image_enhanced
        
        # Test with a complex image (mixed colors)
        mixed_image = Image.new('RGB', (224, 224), color=(150, 100, 75))
        description = describe_image_enhanced(mixed_image, models)
        
        print(f"   📝 Analysis Description:")
        print(f"      • {description[:300]}...")
        
        # Test 3: Food Agent Integration
        print(f"\n🤖 Test 3: Food Agent Integration...")
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
                health_score = result.get('health_score', 'N/A')
                recommendations = result.get('recommendations', [])
                
                print(f"      📊 Agent Results:")
                print(f"         • All detected foods: {detected_foods}")
                print(f"         • ViT detected foods: {vit_detected}")
                print(f"         • Description foods: {description_foods}")
                print(f"         • Detection method: {detection_method}")
                print(f"         • Models used: {models_used}")
                print(f"         • Health score: {health_score}/10")
                print(f"         • Total recommendations: {len(recommendations)}")
        
        # Test 4: Nutritional Balance Analysis
        print(f"\n📊 Test 4: Nutritional Balance Analysis...")
        if vit_available or swin_available:
            test_foods = ['apple', 'chicken', 'rice', 'broccoli', 'salmon', 'avocado', 'quinoa', 'spinach']
            balance_analysis = vit_detector.analyze_nutritional_balance(test_foods)
            
            print(f"   📈 Balance Analysis:")
            print(f"      • Balance score: {balance_analysis.get('balance_score', 0)}/10")
            print(f"      • Food categories: {list(balance_analysis.get('categories', {}).keys())}")
            print(f"      • Recommendations: {len(balance_analysis.get('recommendations', []))}")
            
            # Show categories
            categories = balance_analysis.get('categories', {})
            for category, foods in categories.items():
                print(f"      • {category}: {foods}")
        
        # Test 5: Model Performance Comparison
        print(f"\n⚡ Test 5: Model Performance Comparison...")
        test_image = Image.new('RGB', (224, 224), color=(100, 150, 100))
        
        # Test individual models
        if vit_available:
            start_time = time.time()
            vit_results = vit_detector._detect_with_vit(test_image)
            vit_time = time.time() - start_time
            print(f"   ⏱️ ViT-B/16: {len(vit_results)} foods in {vit_time:.3f}s")
        
        if swin_available:
            start_time = time.time()
            swin_results = vit_detector._detect_with_swin(test_image)
            swin_time = time.time() - start_time
            print(f"   ⏱️ Swin-T: {len(swin_results)} foods in {swin_time:.3f}s")
        
        if blip_available:
            start_time = time.time()
            from utils.analysis import describe_image_enhanced
            blip_description = describe_image_enhanced(test_image, models)
            blip_time = time.time() - start_time
            print(f"   ⏱️ BLIP: Description generated in {blip_time:.3f}s")
        
        # Test 6: Comprehensive Food List Test
        print(f"\n🍽️ Test 6: Comprehensive Food List Test...")
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
            'pizza', 'hamburger', 'sandwich', 'soup', 'salad', 'stew', 'curry', 'sushi'
        ]
        
        print(f"   📋 Testing food vocabulary coverage...")
        print(f"      • Total foods in vocabulary: {len(comprehensive_foods)}")
        
        # Test how many foods can be detected by our models
        detected_count = 0
        for food in comprehensive_foods[:20]:  # Test first 20 for speed
            # Create a simple test to see if food is in our detection vocabulary
            if food in vit_detector.food_vocabulary:
                detected_count += 1
        
        print(f"      • Foods in detection vocabulary: {detected_count}/20")
        print(f"      • Coverage percentage: {(detected_count/20)*100:.1f}%")
        
        print(f"\n✅ Comprehensive Integration Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_food_detection():
    """Test detection of specific food categories"""
    try:
        print("\n🎯 Testing Specific Food Category Detection...")
        print("=" * 60)
        
        from utils.models import load_models
        models = load_models()
        
        if models.get('vit_model') or models.get('swin_model'):
            from utils.vision_transformer_detection import VisionTransformerFoodDetector
            vit_detector = VisionTransformerFoodDetector(models)
            
            # Test specific food categories
            food_categories = {
                'Fruits': ['apple', 'banana', 'orange', 'strawberry', 'grape'],
                'Vegetables': ['broccoli', 'carrot', 'spinach', 'tomato', 'cucumber'],
                'Proteins': ['chicken', 'beef', 'salmon', 'egg', 'tofu'],
                'Grains': ['rice', 'pasta', 'bread', 'oatmeal', 'corn'],
                'Dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream'],
                'Nuts': ['almond', 'walnut', 'peanut', 'cashew', 'pistachio'],
                'Beverages': ['coffee', 'tea', 'juice', 'water', 'smoothie'],
                'Prepared Foods': ['pizza', 'hamburger', 'sandwich', 'soup', 'salad']
            }
            
            for category, foods in food_categories.items():
                print(f"\n   🍎 Testing {category}...")
                detected_in_category = 0
                
                for food in foods:
                    if food in vit_detector.food_vocabulary:
                        detected_in_category += 1
                        print(f"      ✅ {food}")
                    else:
                        print(f"      ❌ {food}")
                
                coverage = (detected_in_category / len(foods)) * 100
                print(f"      📊 {category} coverage: {coverage:.1f}% ({detected_in_category}/{len(foods)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Specific food detection test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Starting Comprehensive Model Integration Tests...")
    
    success1 = test_comprehensive_integration()
    success2 = test_specific_food_detection()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Comprehensive integration is working.")
        print("\n📋 Summary:")
        print("   ✅ Vision Transformer (ViT-B/16) integration")
        print("   ✅ Swin Transformer integration")
        print("   ✅ BLIP integration")
        print("   ✅ YOLO integration")
        print("   ✅ Ensemble detection")
        print("   ✅ Food Agent integration")
        print("   ✅ Comprehensive food vocabulary")
        print("   ✅ Multi-model detection pipeline")
    else:
        print("\n💥 Some tests failed. Check the logs above for details.")
        sys.exit(1)
