#!/usr/bin/env python3
"""
Test GROQ Integration for Diet Report Generation
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_groq_integration():
    """Test GROQ service integration"""
    print("🧪 Testing GROQ Integration...")
    print("=" * 50)
    
    # Check if GROQ API key is available
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in environment variables")
        print("💡 Set GROQ_API_KEY to test the integration")
        return False
    
    print("✅ GROQ_API_KEY found")
    
    try:
        # Import GROQ service
        from utils.groq_service import groq_service
        
        # Test service availability
        if groq_service.is_available():
            print("✅ GROQ service is available")
        else:
            print("❌ GROQ service is not available")
            return False
        
        # Test with sample data
        sample_foods = ["apple", "chicken", "rice", "broccoli"]
        sample_nutrition = {
            "total_calories": 450,
            "total_protein": 25.5,
            "total_carbs": 45.2,
            "total_fats": 12.8
        }
        
        print("🧪 Testing diet report generation...")
        print(f"Sample foods: {sample_foods}")
        print(f"Sample nutrition: {sample_nutrition}")
        
        # Generate test report
        report = groq_service.generate_diet_report(
            detected_foods=sample_foods,
            nutritional_data=sample_nutrition,
            context="Test meal for lunch",
            meal_time="lunch"
        )
        
        if report["success"]:
            print("✅ Diet report generated successfully!")
            print(f"📝 Report length: {len(report['report'])} characters")
            print(f"🤖 Model used: {report['model_used']}")
            print(f"⏰ Generated at: {report['generated_at']}")
            
            # Show first 200 characters of report
            print("\n📋 Report Preview:")
            print("-" * 30)
            print(report['report'][:200] + "...")
            print("-" * 30)
            
            return True
        else:
            print(f"❌ Report generation failed: {report.get('error', 'Unknown error')}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_quick_insights():
    """Test quick insights generation"""
    print("\n🧪 Testing Quick Insights...")
    print("=" * 30)
    
    try:
        from utils.groq_service import groq_service
        
        sample_foods = ["pizza", "salad"]
        sample_nutrition = {
            "total_calories": 600,
            "total_protein": 20,
            "total_carbs": 65,
            "total_fats": 25
        }
        
        insights = groq_service.generate_quick_insights(sample_foods, sample_nutrition)
        
        if insights and "GROQ analysis not available" not in insights:
            print("✅ Quick insights generated!")
            print("💡 Insights:")
            print(insights)
            return True
        else:
            print("❌ Quick insights generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error generating insights: {e}")
        return False

if __name__ == "__main__":
    print("🚀 GROQ Integration Test Suite")
    print("=" * 50)
    
    # Test main functionality
    main_test = test_groq_integration()
    
    # Test quick insights
    insights_test = test_quick_insights()
    
    print("\n" + "=" * 50)
    if main_test and insights_test:
        print("🎉 All tests passed! GROQ integration is working correctly.")
        print("\n📋 Next steps:")
        print("1. Deploy to Streamlit Cloud")
        print("2. Set GROQ_API_KEY environment variable")
        print("3. Test the diet report generation in the app")
    else:
        print("❌ Some tests failed. Please check the configuration.")
        print("\n💡 Troubleshooting:")
        print("1. Ensure GROQ_API_KEY is set")
        print("2. Check internet connection")
        print("3. Verify GROQ API access")
    
    sys.exit(0 if (main_test and insights_test) else 1)
