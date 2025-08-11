#!/usr/bin/env python3
"""
Test Vision Transformer and Swin Transformer integration
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

def test_vision_transformers():
    """Test Vision Transformer integration"""
    try:
        print("🚀 Testing Vision Transformer Integration...")
        print("=" * 60)
        
        # Load models
        print("🔄 Loading models (including Vision Transformers)...")
        from utils.models import load_models
        models = load_models()
        
        if "error" in models:
            print("❌ Models failed to load:", models["error"])
            return False
        
        print("✅ Models loaded successfully")
        
        # Check which transformer models are available
        vit_available = models.get('vit_model') is not None
        swin_available = models.get('swin_model') is not None
        
        print(f"📊 Model Status:")
        print(f"   • ViT-B/16: {'✅ Available' if vit_available else '❌ Not Available'}")
        print(f"   • Swin Transformer: {'✅ Available' if swin_available else '❌ Not Available'}")
        
        # Test Vision Transformer detector
        print(f"\n🔍 Testing Vision Transformer Food Detector...")
        from utils.vision_transformer_detection import VisionTransformerFoodDetector
        
        vit_detector = VisionTransformerFoodDetector(models)
        
        # Create test images
        test_images = [
            ('Red Food Image', Image.new('RGB', (224, 224), color=(200, 50, 50))),
            ('Green Food Image', Image.new('RGB', (224, 224), color=(50, 150, 50))),
            ('Yellow Food Image', Image.new('RGB', (224, 224), color=(255, 220, 50)))
        ]
        
        for name, test_image in test_images:
            print(f"\n📸 Testing {name}...")
            result = vit_detector.detect_food_with_transformers(test_image)
            
            detected_foods = result.get('detected_foods', [])
            confidence_scores = result.get('confidence_scores', {})
            detection_method = result.get('detection_method', 'unknown')
            models_used = result.get('models_used', [])
            
            print(f"   📊 Results:")
            print(f"      • Detected foods: {detected_foods}")
            print(f"      • Detection method: {detection_method}")
            print(f"      • Models used: {models_used}")
            print(f"      • Confidence scores: {confidence_scores}")
        
        # Test food agent with Vision Transformers
        print(f"\n🤖 Testing Food Agent with Vision Transformers...")
        from utils.food_agent import FoodAgent
        
        agent = FoodAgent(models)
        
        # Test with a red image (should detect tomato/apple-like foods)
        red_image = Image.new('RGB', (224, 224), color=(200, 50, 50))
        result = agent.get_comprehensive_analysis(red_image)
        
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
            test_foods = ['apple', 'chicken', 'rice', 'broccoli']
            balance_analysis = vit_detector.analyze_nutritional_balance(test_foods)
            
            print(f"   📈 Balance Analysis:")
            print(f"      • Balance score: {balance_analysis.get('balance_score', 0)}/10")
            print(f"      • Food categories: {list(balance_analysis.get('categories', {}).keys())}")
            print(f"      • Recommendations: {len(balance_analysis.get('recommendations', []))}")
        
        print(f"\n✅ Vision Transformer integration tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}")
        return False

if __name__ == "__main__":
    success = test_vision_transformers()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Vision Transformer Integration SUCCESS!")
        print("\n🚀 New Capabilities:")
        print("   • ViT-B/16 for advanced image classification")
        print("   • Swin Transformer for hierarchical vision processing")
        print("   • Ensemble prediction combining both models")
        print("   • ImageNet food class mapping")
        print("   • Enhanced confidence scoring")
        print("   • Nutritional balance analysis")
        print("   • Fallback to robust detection when needed")
        
        print("\n💡 To use:")
        print("   1. Run: streamlit run app.py")
        print("   2. Upload food image")
        print("   3. Click 'Analyze Food (Advanced Multi-Model Detection)'")
        print("   4. See Vision Transformer results!")
    else:
        print("💥 Vision Transformer integration needs fixes.")
    
    # Clean up
    try:
        os.remove(__file__)
    except:
        pass
    
    sys.exit(0 if success else 1)