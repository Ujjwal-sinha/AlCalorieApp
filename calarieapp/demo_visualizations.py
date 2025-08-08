#!/usr/bin/env python3
"""
Demo script to test visualization functions
Run this to see all the charts in action
"""

import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from visualizations import NutritionVisualizer, display_visualization_dashboard

# Sample data for testing
sample_nutrition_data = {
    "total_calories": 850,
    "total_protein": 45.2,
    "total_carbs": 78.5,
    "total_fats": 32.1,
    "total_fiber": 12.3
}

sample_food_items = [
    {"item": "Grilled Chicken Breast", "calories": 250, "protein": 35.0, "carbs": 0.0, "fats": 12.0, "fiber": 0.0},
    {"item": "Brown Rice", "calories": 200, "protein": 4.5, "carbs": 42.0, "fats": 1.5, "fiber": 3.5},
    {"item": "Steamed Broccoli", "calories": 55, "protein": 3.8, "carbs": 11.0, "fats": 0.6, "fiber": 5.2},
    {"item": "Olive Oil", "calories": 120, "protein": 0.0, "carbs": 0.0, "fats": 14.0, "fiber": 0.0},
    {"item": "Mixed Salad", "calories": 125, "protein": 1.9, "carbs": 25.5, "fats": 4.0, "fiber": 3.6}
]

# Generate sample daily calories for the past week
sample_daily_calories = {}
for i in range(7):
    day = date.today() - timedelta(days=i)
    sample_daily_calories[day.isoformat()] = 1800 + (i * 50)  # Varying calories

# Generate sample history data
sample_history = []
for i in range(5):
    entry = {
        "timestamp": datetime.now() - timedelta(hours=i*4),
        "type": "image",
        "description": f"Sample meal {i+1}",
        "nutritional_data": {
            "total_calories": 800 + (i * 100),
            "total_protein": 40 + (i * 5),
            "total_carbs": 70 + (i * 10),
            "total_fats": 30 + (i * 2),
            "total_fiber": 10 + i
        },
        "context": f"Sample context {i+1}"
    }
    sample_history.append(entry)

def main():
    st.title("🍱 AI Calorie Tracker - Visualization Demo")
    st.write("This demo shows all the visualization features with sample data.")
    
    # Test individual charts
    visualizer = NutritionVisualizer()
    
    st.subheader("📊 Individual Chart Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Pie Chart Test**")
        pie_fig = visualizer.create_nutrition_pie_chart(sample_nutrition_data)
        if pie_fig:
            st.pyplot(pie_fig)
            plt.close()
        
        st.write("**Bar Chart Test**")
        bar_fig = visualizer.create_nutrition_bar_chart(sample_nutrition_data)
        if bar_fig:
            st.pyplot(bar_fig)
            plt.close()
    
    with col2:
        st.write("**Radar Chart Test**")
        radar_fig = visualizer.create_nutrition_radar_chart(sample_nutrition_data)
        if radar_fig:
            st.pyplot(radar_fig)
            plt.close()
        
        st.write("**Meal Analysis Chart Test**")
        meal_fig = visualizer.create_meal_analysis_chart(sample_food_items)
        if meal_fig:
            st.pyplot(meal_fig)
            plt.close()
    
    # Test trend charts
    st.subheader("📈 Trend Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Daily Trend Test**")
        trend_fig = visualizer.create_daily_calorie_trend(sample_daily_calories)
        if trend_fig:
            st.pyplot(trend_fig)
            plt.close()
    
    with col2:
        st.write("**Gauge Chart Test**")
        gauge_fig = visualizer.create_calorie_gauge_chart(1850)
        if gauge_fig:
            st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Test weekly summary
    st.subheader("📊 Weekly Summary")
    weekly_fig = visualizer.create_weekly_summary_chart(sample_daily_calories)
    if weekly_fig:
        st.pyplot(weekly_fig)
        plt.close()
    
    # Test comparison chart
    st.subheader("🍽️ Meal Comparison")
    comparison_fig = visualizer.create_nutrition_comparison_chart(sample_history)
    if comparison_fig:
        st.pyplot(comparison_fig)
        plt.close()
    
    # Test interactive charts
    st.subheader("🎯 Interactive Charts")
    fig_pie, fig_bar, fig_trend = visualizer.create_plotly_interactive_charts(
        sample_nutrition_data, sample_daily_calories, sample_food_items
    )
    
    if fig_pie:
        st.plotly_chart(fig_pie, use_container_width=True)
    
    if fig_bar:
        st.plotly_chart(fig_bar, use_container_width=True)
    
    if fig_trend:
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Test full dashboard
    st.subheader("📊 Full Dashboard Demo")
    display_visualization_dashboard(
        nutrition_data=sample_nutrition_data,
        daily_calories=sample_daily_calories,
        food_items=sample_food_items,
        history_data=sample_history
    )

if __name__ == "__main__":
    main()
