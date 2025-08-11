#!/usr/bin/env python3
"""
Test the new smart food agent implementation
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

def test_new_agent():
    """Test the new smart food agent"""
    try:
        # Import the new agent
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
        
        # Test with a simple image (create a dummy image for testing)
        print("🖼️ Creating test image...")
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Test the comprehensive analysis
        print("🔍 Running comprehensive analysis...")
        result = agent.get_comprehensive_analysis(test_image)
        
        if "error" in result:
            print("❌ Analysis failed:", result["error"])
            return False
        
        print("✅ Analysis completed successfully!")
        print(f"📊 Detected foods: {result.get('detected_foods', [])}")
        print(f"🍽️ Description: {result.get('food_description', 'N/A')}")
        print(f"📈 Nutrition: {result.get('nutrition_data', {})}")
        print(f"🎯 Health score: {result.get('health_score', 'N/A')}")
        print(f"💡 Recommendations: {result.get('recommendations', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing new smart food agent...")
    success = test_new_agent()
    
    if success:
        print("🎉 All tests passed! The new agent is working correctly.")
    else:
        print("💥 Tests failed. Check the implementation.")
    
    sys.exit(0 if success else 1)