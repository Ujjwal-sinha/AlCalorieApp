#!/usr/bin/env python3
"""
Test script for modern UI components
"""

import streamlit as st
from modern_ui import (
    load_css, create_modern_header, create_metric_card, create_feature_card,
    create_ai_analysis_box, create_timeline_item, create_modern_footer, 
    create_modern_sidebar, create_upload_section, create_analysis_results, 
    create_modern_chart_container, create_loading_animation, create_empty_state
)

def main():
    # Page config
    st.set_page_config(
        page_title="🍱 Modern UI Test", 
        layout="wide", 
        page_icon="🍽️",
        initial_sidebar_state="expanded"
    )
    
    # Load modern CSS
    load_css()
    
    # Modern header
    create_modern_header()
    
    # Test modern sidebar
    selected_tab = create_modern_sidebar()
    
    st.write(f"Selected tab: {selected_tab}")
    
    # Test metric cards
    st.markdown("## 📊 Test Metric Cards")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card("850", "Calories", "🔥")
    with col2:
        create_metric_card("45.2g", "Protein", "💪")
    with col3:
        create_metric_card("78.5g", "Carbs", "🌾")
    with col4:
        create_metric_card("32.1g", "Fats", "🥑")
    
    # Test feature cards
    st.markdown("## 🎨 Test Feature Cards")
    col1, col2 = st.columns(2)
    with col1:
        create_feature_card("🤖", "AI Analysis", "Advanced AI-powered food recognition")
    with col2:
        create_feature_card("📊", "Visualizations", "Beautiful charts and graphs")
    
    # Test analysis results (without food items breakdown)
    st.markdown("## 📊 Test Analysis Results")
    sample_nutrition = {
        "total_calories": 850,
        "total_protein": 45.2,
        "total_carbs": 78.5,
        "total_fats": 32.1
    }
    create_analysis_results(sample_nutrition, "This is a sample AI analysis showing detailed nutritional insights and recommendations for your meal.")
    
    # Test timeline item
    st.markdown("## 📅 Test Timeline Item")
    create_timeline_item("2024-01-15 12:30", "Lunch", "Grilled chicken with rice - 450 calories")
    
    # Test AI analysis box
    st.markdown("## 🤖 Test AI Analysis Box")
    create_ai_analysis_box("Sample Analysis", "This is a sample AI analysis of your meal. It shows detailed nutritional information and recommendations.")
    
    # Test empty state
    st.markdown("## 📭 Test Empty State")
    create_empty_state("📊", "No Data Yet", "Start by analyzing your first meal!", "Go to Food Analysis tab")
    
    # Test modern footer
    st.markdown("## 🦶 Test Footer")
    create_modern_footer()

if __name__ == "__main__":
    main()
