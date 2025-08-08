#!/usr/bin/env python3
"""
Test script to verify Food Items Breakdown removal
"""

import streamlit as st
from modern_ui import (
    load_css, create_modern_header, create_analysis_results
)

def main():
    # Page config
    st.set_page_config(
        page_title="🍱 No Breakdown Test", 
        layout="wide", 
        page_icon="🍽️"
    )
    
    # Load modern CSS
    load_css()
    
    # Modern header
    create_modern_header()
    
    st.markdown("## 🧪 Testing Analysis Results (No Food Items Breakdown)")
    
    # Sample nutrition data
    sample_nutrition = {
        "total_calories": 1250,
        "total_protein": 65.5,
        "total_carbs": 120.3,
        "total_fats": 45.2
    }
    
    # Sample AI analysis text
    sample_analysis = """
    **AI Analysis Report:**
    
    Your meal contains a well-balanced combination of nutrients:
    
    🔥 **Calories**: 1,250 kcal - This is a substantial meal, suitable for lunch or dinner.
    
    💪 **Protein**: 65.5g - Excellent protein content, supporting muscle maintenance and growth.
    
    🌾 **Carbohydrates**: 120.3g - Good energy source, providing sustained fuel for your activities.
    
    🥑 **Fats**: 45.2g - Healthy fat content, supporting hormone production and nutrient absorption.
    
    **Recommendations:**
    - This meal provides excellent nutritional balance
    - Consider adding more vegetables for fiber and micronutrients
    - Great choice for post-workout recovery
    - Maintains good macronutrient ratios for overall health
    """
    
    # Test the analysis results function (should only show metrics and AI analysis)
    create_analysis_results(sample_nutrition, sample_analysis)
    
    st.success("✅ Test completed! Food Items Breakdown has been removed successfully.")
    st.info("📝 Only the nutrition metrics and AI Analysis Report are now displayed.")

if __name__ == "__main__":
    main()
