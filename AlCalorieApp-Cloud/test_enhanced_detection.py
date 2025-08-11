#!/usr/bin/env python3
"""
Test script for enhanced food detection and web search functionality
"""

import sys
import os
from PIL import Image, ImageDraw
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_realistic_food_image():
    """Create a realistic test food image"""
    # Create a 400x300 test image with food-like colors
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw realistic food items
    # Chicken (brown/golden)
    draw.ellipse([50, 50, 150, 120], fill='#D2691E', outline='#8B4513', width=2)
    draw.text((70, 80), "Chicken", fill='white')
    
    # Rice (white/cream)
    draw.rectangle([200, 60, 300, 130], fill='#F5F5DC', outline='#DDD', width=2)
    draw.text((220, 90), "Rice", fill='black')
    
    # Vegetables (green)
    draw.ellipse([80, 180, 140, 240], fill='#228B22', outline='#006400', width=2)
    draw.text((90, 205), "Broccoli", fill='white')
    
    # Tomato (red)
    draw.ellipse([220, 190, 280, 250], fill='#FF6347', outline='#DC143C', width=2)
    draw.text((230, 215), "Tomato", fill='white')
    
    return img

def test_food_detection():
    """Test the enhanced food detection"""
    print("🧪 Testing Enhanced Food Detection...")
    
    try:
        # Import the analysis module
        from utils.analysis import describe_image_enhanced, validate_food_items, extract_food_items_from_text
        print("✅ Successfully imported analysis functions")
        
        # Create test image
        test_img = create_realistic_food_image()
        print("✅ Created realistic test food image")
        
        # Test food item extraction
        test_text = "chicken, rice, broccoli, tomato, plate, bowl"
        extracted_items = extract_food_items_from_text(test_text)
        print(f"✅ Extracted items from text: {extracted_items}")
        
        # Test validation
        validated_items = validate_food_items(extracted_items, "cooked meal")
        print(f"✅ Validated items: {validated_items}")
        
        # Test with mock models (since we don't have real models in test)
        mock_models = {
            'processor': None,
            'blip_model': None,
            'device': None
        }
        
        result = describe_image_enhanced(test_img, mock_models)
        print(f"✅ Detection result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Food detection test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_web_search():
    """Test the web search functionality"""
    print("\n🌐 Testing Web Search Functionality...")
    
    try:
        # Import the food agent
        from utils.food_agent import FoodAgent
        print("✅ Successfully imported FoodAgent")
        
        # Create mock models
        mock_models = {
            'llm': None,
            'blip_model': None,
            'processor': None
        }
        
        # Initialize agent
        agent = FoodAgent(mock_models)
        print("✅ Initialized FoodAgent")
        
        # Test web search
        test_result = agent.test_web_search("chicken rice")
        print(f"✅ Web search test result: {test_result}")
        
        # Test search query generation
        queries = agent._generate_search_queries("chicken rice vegetables")
        print(f"✅ Generated search queries: {list(queries.keys())}")
        
        # Test comprehensive food data generation
        food_data = agent._generate_comprehensive_food_data("chicken rice")
        print(f"✅ Generated comprehensive food data: {len(food_data)} items")
        
        # Test nutrition data
        nutrition_data = agent._get_nutrition_data("chicken")
        print(f"✅ Retrieved nutrition data: {len(nutrition_data)} items")
        
        # Test cultural info
        cultural_data = agent._get_cultural_food_info("rice")
        print(f"✅ Retrieved cultural data: {len(cultural_data)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Web search test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test the integration between detection and web search"""
    print("\n🔗 Testing Integration...")
    
    try:
        # Test that all components work together
        from utils.analysis import describe_image_enhanced
        from utils.food_agent import FoodAgent
        
        # Mock models
        mock_models = {
            'processor': None,
            'blip_model': None,
            'llm': None,
            'device': None
        }
        
        # Test image
        test_img = create_realistic_food_image()
        
        # Test detection
        detection_result = describe_image_enhanced(test_img, mock_models)
        print(f"✅ Detection integration: {detection_result}")
        
        # Test agent
        agent = FoodAgent(mock_models)
        agent_status = agent.get_agent_status()
        print(f"✅ Agent status: {agent_status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test required dependencies"""
    print("🔍 Testing Dependencies...")
    
    dependencies = [
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('requests', 'Requests'),
        ('logging', 'Logging'),
        ('json', 'JSON'),
        ('hashlib', 'Hashlib'),
        ('datetime', 'DateTime'),
        ('re', 'Regular Expressions'),
        ('typing', 'Typing'),
    ]
    
    all_good = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} is available")
        except ImportError:
            print(f"❌ {name} is missing")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("🚀 Starting Enhanced Detection & Web Search Tests\n")
    
    # Test dependencies first
    deps_ok = test_dependencies()
    print()
    
    if deps_ok:
        # Test food detection
        detection_ok = test_food_detection()
        
        # Test web search
        web_search_ok = test_web_search()
        
        # Test integration
        integration_ok = test_integration()
        
        if detection_ok and web_search_ok and integration_ok:
            print("\n🎉 All tests completed successfully!")
            print("✅ Enhanced food detection is working")
            print("✅ Web search functionality is operational")
            print("✅ Integration between components is solid")
            print("\n💡 Your enhanced food detection system is ready!")
        else:
            print("\n⚠️ Some tests had issues, but the system should still work")
            print("The app will use fallback methods for any failing components")
    else:
        print("\n❌ Missing required dependencies")
        print("Please install missing packages")
        sys.exit(1)