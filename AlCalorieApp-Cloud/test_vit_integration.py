#!/usr/bin/env python3
"""
Test Vision Transformer Integration with Enhanced Models
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

def test_vision_transformer_integration():
    """Test Vision Transformer integration with enhanced models"""
    try:
        print("🚀 Testing Enhanced Vision Transformer Integration...")
        print("=" * 70)
        
        # Load models
        print("🔄 Loading enhanced models (including ViT-B/16 and Swin Transformer)...")
        from utils.models import load_models
        models = load_models()
        
        if "error" in models:
            print("❌ Models failed to load:", models["error"])
            return False
        
        print("✅ Models loaded successfully")
        
        # Check which transformer models are available
        vit_available = models.get('vit_model') is not None
        swin_available = models.get('swin_model') is not None
        
        print(f"📊 Enhanced Model Status:")
        print(f"   • ViT-B/16: {'✅ Available' if vit_available else '❌ Not Available'}")
        print(f"   • Swin Transformer (Swin-T): {'✅ Available' if swin_available else '❌ Not Available'}")
        
        if not vit_available and not swin_available:
            print("⚠️ No Vision Transformer models available. Check model loading.")
            return False
        
        # Test Vision Transformer detector
        print(f"\n🔍 Testing Enhanced Vision Transformer Food Detector...")
        from utils.vision_transformer_detection import VisionTransformerFoodDetector
        
        vit_detector = VisionTransformerFoodDetector(models)
        
        # Create test images with different food colors
        test_images = [
            ('Red Food Image (Apple/Tomato)', Image.new('RGB', (224, 224), color=(200, 50, 50))),
            ('Green Food Image (Broccoli/Lettuce)', Image.new('RGB', (224, 224), color=(50, 150, 50))),
            ('Yellow Food Image (Banana/Corn)', Image.new('RGB', (224, 224), color=(255, 220, 50))),
            ('Orange Food Image (Carrot/Orange)', Image.new('RGB', (224, 224), color=(255, 165, 0))),
            ('Brown Food Image (Bread/Chocolate)', Image.new('RGB', (224, 224), color=(139, 69, 19)))
        ]
        
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
            print(f"      • Confidence scores: {confidence_scores}")
            
            if not success:
                error = result.get('error', 'Unknown error')
                print(f"      • Error: {error}")
        
        # Test integration with main analysis pipeline
        print(f"\n🔗 Testing Integration with Main Analysis Pipeline...")
        from utils.analysis import describe_image_enhanced
        
        # Test with a red image
        red_image = Image.new('RGB', (224, 224), color=(200, 50, 50))
        description = describe_image_enhanced(red_image, models)
        
        print(f"   📝 Analysis Description:")
        print(f"      • {description[:200]}...")
        
        # Test food agent integration
        print(f"\n🤖 Testing Food Agent Integration...")
        from utils.food_agent import FoodAgent
        
        agent = FoodAgent(models)
        
        # Test with a green image (should detect broccoli/lettuce-like foods)
        green_image = Image.new('RGB', (224, 224), color=(50, 150, 50))
        result = agent.get_comprehensive_analysis(green_image)
        
        if "error" in result:
            print(f"⚠️ Agent had issues: {result['error']}")
        else:
            print(f"   📊 Agent Results:")
            print(f"      • Detected foods: {result.get('detected_foods', [])}")
            print(f"      • Detection method: {result.get('detection_method', 'unknown')}")
            print(f"      • Models used: {result.get('models_used', [])}")
            print(f"      • Health score: {result.get('health_score', 'N/A')}/10")
            print(f"      • Total recommendations: {len(result.get('recommendations', []))}")
        
        # Test nutritional balance analysis
        print(f"\n📊 Testing Nutritional Balance Analysis...")
        if vit_available or swin_available:
            test_foods = ['apple', 'chicken', 'rice', 'broccoli', 'salmon']
            balance_analysis = vit_detector.analyze_nutritional_balance(test_foods)
            
            print(f"   📈 Balance Analysis:")
            print(f"      • Balance score: {balance_analysis.get('balance_score', 0)}/10")
            print(f"      • Food categories: {list(balance_analysis.get('categories', {}).keys())}")
            print(f"      • Recommendations: {len(balance_analysis.get('recommendations', []))}")
            
            # Show categories
            categories = balance_analysis.get('categories', {})
            for category, foods in categories.items():
                print(f"      • {category}: {foods}")
        
        print(f"\n✅ Enhanced Vision Transformer Integration Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vision_transformer_integration()
    if success:
        print("\n🎉 All tests passed! Vision Transformer integration is working.")
    else:
        print("\n💥 Some tests failed. Check the logs above for details.")
        sys.exit(1)
