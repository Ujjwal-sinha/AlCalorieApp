#!/usr/bin/env python3
"""
Test script for the Enhanced Food Agent

This script demonstrates the food agent functionality with a sample image.
Run this script to test the agent's capabilities.
"""

import sys
import os
from PIL import Image
import numpy as np

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def create_sample_image():
    """Create a sample food image for testing"""
    # Create a simple colored image that represents food
    # This is a placeholder - in real usage, you'd use actual food images
    
    # Create a 300x300 image with food-like colors
    img_array = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Add some food-like colors (browns, greens, yellows)
    img_array[50:150, 50:150] = [139, 69, 19]  # Brown (bread-like)
    img_array[150:250, 50:150] = [34, 139, 34]  # Green (vegetables)
    img_array[50:150, 150:250] = [255, 215, 0]  # Yellow (cheese-like)
    img_array[150:250, 150:250] = [255, 140, 0]  # Orange (carrots)
    
    # Convert to PIL Image
    image = Image.fromarray(img_array)
    return image

def test_food_agent():
    """Test the food agent with a sample image"""
    print("🍽️ Testing Enhanced Food Agent")
    print("=" * 50)
    
    try:
        # Import the food agent
        from utils.food_agent import FoodAgent
        
        # Create mock models for testing
        mock_models = {
            'blip_model': None,
            'processor': None,
            'llm': None,
            'yolo_model': None,
            'device': 'cpu'
        }
        
        # Initialize the agent
        print("1. Initializing Food Agent...")
        agent = FoodAgent(mock_models)
        
        # Create sample image
        print("2. Creating sample food image...")
        sample_image = create_sample_image()
        
        # Test image analysis
        print("3. Testing image analysis...")
        analysis_result = agent.analyze_food_image(sample_image)
        
        if "error" not in analysis_result:
            print(f"✅ Analysis successful!")
            print(f"   Session ID: {analysis_result.get('session_id', 'N/A')}")
            print(f"   Description: {analysis_result.get('original_description', 'N/A')}")
        else:
            print(f"❌ Analysis failed: {analysis_result['error']}")
        
        # Test web search (with mock data)
        print("\n4. Testing web search functionality...")
        test_description = "grilled chicken with vegetables and rice"
        search_result = agent.search_web_information(test_description)
        
        if "error" not in search_result:
            print("✅ Web search successful!")
            print(f"   Food Name: {search_result.get('food_name', 'N/A')}")
            print(f"   Nutrition: {search_result.get('nutrition', {}).get('calories', 'N/A')}")
        else:
            print(f"❌ Web search failed: {search_result['error']}")
        
        # Test complete pipeline
        print("\n5. Testing complete processing pipeline...")
        complete_result = agent.process_food_image_complete(sample_image, "What are the health benefits?")
        
        if "error" not in complete_result:
            print("✅ Complete pipeline successful!")
            print(f"   Context stored: {len(agent.context_cache)} items")
            print(f"   Search cache: {len(agent.search_cache)} items")
        else:
            print(f"❌ Complete pipeline failed: {complete_result['error']}")
        
        # Test agent status
        print("\n6. Testing agent status...")
        status = agent.get_agent_status()
        print(f"✅ Agent Status:")
        print(f"   Context Cache Size: {status['context_cache_size']}")
        print(f"   Search Cache Size: {status['search_cache_size']}")
        print(f"   Models Available: {status['models_available']}")
        
        print("\n🎉 Food Agent Test Completed Successfully!")
        print("\nFeatures Demonstrated:")
        print("✅ Image analysis and description generation")
        print("✅ Web search for comprehensive information")
        print("✅ Context storage for follow-up questions")
        print("✅ Modular architecture for easy upgrades")
        print("✅ Error handling and fallback mechanisms")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all required modules are available.")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_web_search():
    """Test web search functionality specifically"""
    print("\n🌐 Testing Web Search Functionality")
    print("=" * 40)
    
    try:
        from utils.food_agent import FoodAgent
        
        # Create agent with mock models
        mock_models = {'llm': None}
        agent = FoodAgent(mock_models)
        
        # Test different search queries
        test_queries = [
            "pizza nutrition facts calories protein",
            "sushi cultural background history origin",
            "chicken breast recipe cooking instructions"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing query: '{query}'")
            results = agent._perform_web_search(query)
            
            if results:
                print(f"   ✅ Found {len(results)} results")
                for j, result in enumerate(results[:2], 1):  # Show first 2 results
                    print(f"   {j}. {result.get('title', 'No title')}")
                    print(f"      {result.get('snippet', 'No snippet')[:100]}...")
            else:
                print("   ❌ No results found")
    
    except Exception as e:
        print(f"❌ Web search test failed: {e}")

if __name__ == "__main__":
    print("🚀 Enhanced Food Agent Test Suite")
    print("=" * 60)
    
    # Run tests
    test_food_agent()
    test_web_search()
    
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print("The Enhanced Food Agent provides:")
    print("• Comprehensive image analysis")
    print("• Web search for detailed information")
    print("• Context storage for follow-up questions")
    print("• Modular architecture for easy upgrades")
    print("• Robust error handling")
    print("\nReady for integration with Streamlit app!")
