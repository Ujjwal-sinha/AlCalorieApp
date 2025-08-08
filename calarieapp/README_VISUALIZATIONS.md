# 🍱 AI Calorie Tracker - Visualization Features

This document explains the comprehensive visualization features added to the AI Calorie Tracker app.

## 📊 Overview

The app now includes advanced analytics and visualization capabilities that provide detailed insights into your nutrition data through various charts and graphs.

## 🎯 Features Added

### 1. **Current Meal Analysis**
- **Pie Chart**: Shows macronutrient distribution (Protein, Carbs, Fats)
- **Bar Chart**: Displays nutritional values breakdown
- **Radar Chart**: Comprehensive nutrition profile visualization
- **Food Items Chart**: Individual food items and their calorie contributions

### 2. **Daily Trends**
- **Line Chart**: Daily calorie intake trends over time
- **Gauge Chart**: Interactive progress indicator for daily calorie goals
- **Area Charts**: Visual representation of calorie targets vs actual intake

### 3. **Historical Data Analysis**
- **Weekly Summary**: 4-panel comprehensive weekly analysis
- **Meal Comparison**: Side-by-side comparison of multiple meals
- **Distribution Charts**: Statistical analysis of calorie patterns

### 4. **Interactive Charts**
- **Plotly Visualizations**: Hover effects, zoom, and interactive features
- **Dynamic Updates**: Real-time chart updates as data changes
- **Customizable Views**: Multiple chart types for the same data

## 📁 File Structure

```
calarieapp/
├── app.py                    # Main application with visualization integration
├── visualizations.py         # Core visualization module
├── demo_visualizations.py    # Demo script for testing charts
├── requirements.txt          # Updated dependencies
└── README_VISUALIZATIONS.md  # This documentation
```

## 🚀 How to Use

### Running the Main App
```bash
cd calarieapp
streamlit run app.py
```

### Testing Visualizations (Demo)
```bash
cd calarieapp
streamlit run demo_visualizations.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## 📈 Chart Types Available

### 1. **NutritionVisualizer Class**
The main class that creates all visualization charts:

```python
from visualizations import NutritionVisualizer

visualizer = NutritionVisualizer()

# Create individual charts
pie_chart = visualizer.create_nutrition_pie_chart(nutrition_data)
bar_chart = visualizer.create_nutrition_bar_chart(nutrition_data)
radar_chart = visualizer.create_nutrition_radar_chart(nutrition_data)
trend_chart = visualizer.create_daily_calorie_trend(daily_calories)
gauge_chart = visualizer.create_calorie_gauge_chart(current_calories)
```

### 2. **Dashboard Functions**
Complete dashboard displays:

```python
from visualizations import display_visualization_dashboard

# Display full analytics dashboard
display_visualization_dashboard(
    nutrition_data=nutrition_data,
    daily_calories=daily_calories,
    food_items=food_items,
    history_data=history_data
)
```

## 🎨 Chart Features

### **Pie Chart**
- Shows macronutrient distribution
- Exploded segments for better visibility
- Percentage labels
- Color-coded segments

### **Bar Chart**
- Nutritional values breakdown
- Value labels on bars
- Color-coded categories
- Grid lines for readability

### **Radar Chart**
- 5-point nutrition profile
- Normalized values (0-100 scale)
- Filled area for visual impact
- Grid lines for reference

### **Line Chart (Trends)**
- Daily calorie trends
- Target line overlay
- Color-coded areas (above/below target)
- Interactive markers

### **Gauge Chart**
- Progress indicator
- Color zones (green/yellow/red)
- Target threshold line
- Percentage display

### **Weekly Summary**
- 4-panel comprehensive view:
  1. Daily calorie intake
  2. Weekly average vs target
  3. Calorie distribution histogram
  4. Progress pie chart

### **Interactive Charts (Plotly)**
- Hover tooltips
- Zoom and pan capabilities
- Click interactions
- Responsive design

## 🔧 Customization

### Color Schemes
The visualizer uses consistent color schemes:

```python
colors = {
    'primary': '#4CAF50',    # Green
    'secondary': '#2196F3',  # Blue
    'accent': '#FF9800',     # Orange
    'danger': '#F44336',     # Red
    'success': '#4CAF50',    # Green
    'warning': '#FFC107',    # Yellow
    'info': '#00BCD4',       # Cyan
}

nutrition_colors = {
    'calories': '#FF6B6B',   # Red
    'protein': '#4ECDC4',    # Teal
    'carbs': '#45B7D1',      # Blue
    'fats': '#96CEB4',       # Green
    'fiber': '#FFEAA7'       # Yellow
}
```

### Chart Sizes
All charts are optimized for Streamlit display:
- Standard charts: 8x6 inches
- Wide charts: 12x6 inches
- Dashboard charts: 15x10 inches

## 📊 Data Requirements

### Nutrition Data Format
```python
nutrition_data = {
    "total_calories": 850,
    "total_protein": 45.2,
    "total_carbs": 78.5,
    "total_fats": 32.1,
    "total_fiber": 12.3
}
```

### Food Items Format
```python
food_items = [
    {
        "item": "Grilled Chicken Breast",
        "calories": 250,
        "protein": 35.0,
        "carbs": 0.0,
        "fats": 12.0,
        "fiber": 0.0
    }
]
```

### Daily Calories Format
```python
daily_calories = {
    "2024-01-15": 1850,
    "2024-01-16": 1920,
    "2024-01-17": 1780
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Charts not displaying**
   - Check if matplotlib is installed: `pip install matplotlib`
   - Ensure plotly is installed: `pip install plotly`
   - Verify seaborn installation: `pip install seaborn`

2. **Memory issues with large datasets**
   - Charts automatically limit to last 7 days for trends
   - Use `plt.close()` after displaying charts
   - Consider reducing chart resolution for large datasets

3. **Interactive charts not working**
   - Ensure plotly is properly installed
   - Check browser compatibility
   - Verify data format is correct

### Performance Tips

1. **Optimize chart generation**
   - Use `plt.close()` after each chart
   - Limit data points for trend charts
   - Use appropriate chart sizes

2. **Memory management**
   - Clear old charts when updating
   - Use efficient data structures
   - Monitor memory usage with large datasets

## 🔮 Future Enhancements

### Planned Features
- [ ] Export charts as images/PDF
- [ ] Custom chart themes
- [ ] Advanced statistical analysis
- [ ] Machine learning insights
- [ ] Mobile-optimized charts
- [ ] Real-time data streaming
- [ ] Custom chart builder

### Contributing
To add new chart types or improve existing ones:

1. Add new methods to `NutritionVisualizer` class
2. Follow the existing naming conventions
3. Include proper error handling
4. Add documentation and examples
5. Test with sample data

## 📞 Support

For issues or questions about the visualization features:
- Check the demo script for examples
- Review the error messages in the console
- Ensure all dependencies are installed
- Verify data format matches requirements

---

**Developed by Ujjwal Sinha** 🚀
*Enhanced with Advanced AI Visualizations*
