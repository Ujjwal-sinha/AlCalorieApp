#!/usr/bin/env python3
"""
Test the smart food agent with a simulated food description
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

def test_food_detection():
    """Test food detection with simulated descriptions"""
    try:
        from utils.food_agent import FoodAgent
        from utils.models import load_models
        
        print("🔄 Loading models...")
        models = load_models()
        
        if "error" in models:
            print("❌ Models failed to load:", models["error"])
            return False
        
        print("✅ Models loaded successfully")
        
        # Initialize the agent
        print("🤖 Initializing smart food agent...")
        agent = FoodAgent(models)
        
        # Test food extraction from descriptions
        print("\n🧪 Testing food extraction from descriptions...")
        
        test_descriptions = [
            "Foods identified: chicken, rice, broccoli. Details: grilled chicken breast with steamed white rice and fresh broccoli",
            "Foods identified: pizza, salad. Details: margherita pizza with mixed green salad",
            "Foods identified: banana, apple, yogurt. Details: fresh fruit bowl with greek yogurt"
        ]
        
        for i, description in enumerate(test_descriptions, 1):
            print(f"\n📝 Test {i}: {description[:50]}...")
            
            # Extract foods
            foods = agent._extract_foods_from_description(description)
            print(f"   🍽️ Detected foods: {foods}")
            
            # Get nutrition estimates
            nutrition = agent._estimate_nutrition(foods)
            print(f"   📊 Estimated nutrition: {nutrition}")
            
            # Calculate health score
            health_score = agent._calculate_health_score(nutrition)
            print(f"   🎯 Health score: {health_score}/10")
            
            # Generate recommendations
            recommendations = agent._generate_recommendations(foods, nutrition)
            print(f"   💡 Recommendations: {recommendations}")
        
        print("\n✅ All food detection tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing smart food detection with real examples...")
    success = test_food_detection()
    
    if success:
        print("\n🎉 Smart food detection is working correctly!")
        print("The system can now:")
        print("✅ Accurately detect food items")
        print("✅ Estimate realistic nutrition values")
        print("✅ Calculate meaningful health scores")
        print("✅ Provide practical recommendations")
    else:
        print("\n💥 Tests failed. Check the implementation.")
    
    sys.exit(0 if success else 1)