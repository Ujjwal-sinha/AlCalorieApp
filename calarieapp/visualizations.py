import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    # Set style for matplotlib
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    SEABORN_AVAILABLE = False
    plt.style.use('default')

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class NutritionVisualizer:
    """Comprehensive visualization class for nutrition data"""
    
    def __init__(self):
        self.colors = {
            'primary': '#4CAF50',
            'secondary': '#2196F3', 
            'accent': '#FF9800',
            'danger': '#F44336',
            'success': '#4CAF50',
            'warning': '#FFC107',
            'info': '#00BCD4',
            'light': '#F5F5F5',
            'dark': '#212121'
        }
        
        # Nutrition color scheme
        self.nutrition_colors = {
            'calories': '#FF6B6B',
            'protein': '#4ECDC4', 
            'carbs': '#45B7D1',
            'fats': '#96CEB4',
            'fiber': '#FFEAA7'
        }
    
    def create_nutrition_pie_chart(self, nutrition_data):
        """Create a pie chart showing macronutrient distribution"""
        try:
            protein = nutrition_data.get('total_protein', 0) * 4  # 4 cal/g
            carbs = nutrition_data.get('total_carbs', 0) * 4     # 4 cal/g
            fats = nutrition_data.get('total_fats', 0) * 9       # 9 cal/g
            
            # Calculate percentages
            total_calories = protein + carbs + fats
            if total_calories == 0:
                return None
                
            labels = ['Protein', 'Carbs', 'Fats']
            sizes = [protein, carbs, fats]
            colors = [self.nutrition_colors['protein'], self.nutrition_colors['carbs'], self.nutrition_colors['fats']]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                             startangle=90, explode=(0.05, 0.05, 0.05))
            
            # Enhance text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Macronutrient Distribution', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating pie chart: {e}")
            return None
    
    def create_nutrition_bar_chart(self, nutrition_data):
        """Create a bar chart showing nutritional values"""
        try:
            categories = ['Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)']
            values = [
                nutrition_data.get('total_calories', 0),
                nutrition_data.get('total_protein', 0),
                nutrition_data.get('total_carbs', 0),
                nutrition_data.get('total_fats', 0)
            ]
            colors = [self.nutrition_colors['calories'], self.nutrition_colors['protein'], 
                     self.nutrition_colors['carbs'], self.nutrition_colors['fats']]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Nutritional Breakdown', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Amount', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating bar chart: {e}")
            return None
    
    def create_daily_calorie_trend(self, daily_calories, target=2000):
        """Create a line chart showing daily calorie trends"""
        try:
            if not daily_calories:
                return None
                
            # Get last 7 days
            dates = sorted(daily_calories.keys())[-7:]
            calories = [daily_calories.get(d, 0) for d in dates]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot calorie line
            ax.plot(dates, calories, marker='o', linewidth=3, markersize=8, 
                   color=self.colors['primary'], label='Actual Calories')
            
            # Plot target line
            target_line = [target] * len(dates)
            ax.plot(dates, target_line, '--', color=self.colors['danger'], 
                   linewidth=2, label=f'Target ({target} cal)')
            
            # Fill area between actual and target
            ax.fill_between(dates, calories, target_line, 
                          where=[c <= target for c in calories], 
                          alpha=0.3, color=self.colors['success'])
            ax.fill_between(dates, calories, target_line, 
                          where=[c > target for c in calories], 
                          alpha=0.3, color=self.colors['danger'])
            
            ax.set_title('Daily Calorie Intake Trend', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Calories', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating trend chart: {e}")
            return None
    
    def create_weekly_summary_chart(self, daily_calories, target=2000):
        """Create a comprehensive weekly summary chart"""
        try:
            if not daily_calories:
                return None
                
            # Get last 7 days
            dates = sorted(daily_calories.keys())[-7:]
            calories = [daily_calories.get(d, 0) for d in dates]
            
            # Calculate statistics
            avg_calories = np.mean(calories)
            max_calories = max(calories)
            min_calories = min(calories)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Daily calories bar chart
            bars = ax1.bar(dates, calories, color=self.colors['primary'], alpha=0.8)
            ax1.axhline(y=target, color=self.colors['danger'], linestyle='--', label=f'Target ({target})')
            ax1.set_title('Daily Calorie Intake', fontweight='bold')
            ax1.set_ylabel('Calories')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Weekly average vs target
            categories = ['Weekly Avg', 'Target']
            values = [avg_calories, target]
            colors = [self.colors['primary'], self.colors['danger']]
            ax2.bar(categories, values, color=colors, alpha=0.8)
            ax2.set_title('Weekly Average vs Target', fontweight='bold')
            ax2.set_ylabel('Calories')
            
            # 3. Calorie distribution
            ax3.hist(calories, bins=5, color=self.colors['secondary'], alpha=0.7, edgecolor='black')
            ax3.axvline(avg_calories, color=self.colors['danger'], linestyle='--', label=f'Avg: {avg_calories:.0f}')
            ax3.set_title('Calorie Distribution', fontweight='bold')
            ax3.set_xlabel('Calories')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            
            # 4. Progress summary
            progress = (sum(calories) / (target * 7)) * 100
            ax4.pie([progress, 100-progress], labels=['Achieved', 'Remaining'], 
                   colors=[self.colors['success'], self.colors['light']], autopct='%1.1f%%')
            ax4.set_title('Weekly Progress', fontweight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error creating weekly summary: {e}")
            return None
    
    def create_meal_analysis_chart(self, food_items):
        """Create a chart showing individual food items and their calories"""
        try:
            if not food_items:
                return None
                
            items = [item.get('item', 'Unknown')[:20] for item in food_items]  # Truncate long names
            calories = [item.get('calories', 0) for item in food_items]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(items, calories, color=self.colors['primary'], alpha=0.8)
            
            # Add value labels
            for i, (bar, cal) in enumerate(zip(bars, calories)):
                width = bar.get_width()
                ax.text(width + max(calories)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{cal} cal', ha='left', va='center', fontweight='bold')
            
            ax.set_title('Individual Food Items - Calorie Breakdown', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Calories', fontsize=12)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating meal analysis chart: {e}")
            return None
    
    def create_nutrition_radar_chart(self, nutrition_data):
        """Create a radar chart for comprehensive nutrition view"""
        try:
            # Normalize values for radar chart (0-100 scale)
            max_values = {
                'calories': 2000,
                'protein': 100,
                'carbs': 300,
                'fats': 100,
                'fiber': 30
            }
            
            categories = ['Calories', 'Protein', 'Carbs', 'Fats', 'Fiber']
            values = [
                min(nutrition_data.get('total_calories', 0) / max_values['calories'] * 100, 100),
                min(nutrition_data.get('total_protein', 0) / max_values['protein'] * 100, 100),
                min(nutrition_data.get('total_carbs', 0) / max_values['carbs'] * 100, 100),
                min(nutrition_data.get('total_fats', 0) / max_values['fats'] * 100, 100),
                min(nutrition_data.get('total_fiber', 0) / max_values['fiber'] * 100, 100)
            ]
            
            # Number of variables
            N = len(categories)
            
            # Create angles for each category
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Add the first value to the end to complete the polygon
            values += values[:1]
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # Plot the data
            ax.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
            ax.fill(angles, values, alpha=0.25, color=self.colors['primary'])
            
            # Set the labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)
            
            # Add grid
            ax.grid(True)
            
            ax.set_title('Nutrition Profile Radar Chart', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating radar chart: {e}")
            return None
    
    def create_plotly_interactive_charts(self, nutrition_data, daily_calories, food_items):
        """Create interactive Plotly charts"""
        if not PLOTLY_AVAILABLE:
            st.warning("Plotly not available. Interactive charts disabled.")
            return None, None, None
            
        try:
            # 1. Interactive pie chart
            protein = nutrition_data.get('total_protein', 0) * 4
            carbs = nutrition_data.get('total_carbs', 0) * 4
            fats = nutrition_data.get('total_fats', 0) * 9
            
            fig_pie = px.pie(
                values=[protein, carbs, fats],
                names=['Protein', 'Carbs', 'Fats'],
                title='Macronutrient Distribution (Interactive)',
                color_discrete_sequence=[self.nutrition_colors['protein'], 
                                       self.nutrition_colors['carbs'], 
                                       self.nutrition_colors['fats']]
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            
            # 2. Interactive bar chart for food items
            if food_items:
                items_df = pd.DataFrame(food_items)
                fig_bar = px.bar(
                    items_df, 
                    x='item', 
                    y='calories',
                    title='Food Items Calorie Breakdown (Interactive)',
                    color='calories',
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
            else:
                fig_bar = None
            
            # 3. Interactive daily trend
            if daily_calories:
                dates = sorted(daily_calories.keys())[-7:]
                calories = [daily_calories.get(d, 0) for d in dates]
                
                fig_trend = px.line(
                    x=dates, 
                    y=calories,
                    title='Daily Calorie Trend (Interactive)',
                    markers=True
                )
                fig_trend.add_hline(y=2000, line_dash="dash", line_color="red", 
                                  annotation_text="Target (2000 cal)")
            else:
                fig_trend = None
            
            return fig_pie, fig_bar, fig_trend
            
        except Exception as e:
            st.error(f"Error creating interactive charts: {e}")
            return None, None, None
    
    def create_calorie_gauge_chart(self, current_calories, target_calories=2000):
        """Create a gauge chart showing calorie progress"""
        if not PLOTLY_AVAILABLE:
            st.warning("Plotly not available. Gauge chart disabled.")
            return None
            
        try:
            percentage = min((current_calories / target_calories) * 100, 100)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = current_calories,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Daily Calorie Progress"},
                delta = {'reference': target_calories},
                gauge = {
                    'axis': {'range': [None, target_calories * 1.2]},
                    'bar': {'color': self.colors['primary']},
                    'steps': [
                        {'range': [0, target_calories * 0.6], 'color': self.colors['success']},
                        {'range': [target_calories * 0.6, target_calories], 'color': self.colors['warning']},
                        {'range': [target_calories, target_calories * 1.2], 'color': self.colors['danger']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': target_calories
                    }
                }
            ))
            
            fig.update_layout(height=400)
            return fig
            
        except Exception as e:
            st.error(f"Error creating gauge chart: {e}")
            return None
    
    def create_nutrition_comparison_chart(self, history_data):
        """Create a comparison chart of multiple meals"""
        try:
            if not history_data or len(history_data) < 2:
                return None
            
            # Get last 5 meals
            recent_meals = history_data[-5:]
            
            meal_names = []
            calories = []
            proteins = []
            carbs = []
            fats = []
            
            for meal in recent_meals:
                nutrition = meal.get('nutritional_data', {})
                meal_names.append(meal.get('description', 'Meal')[:15])
                calories.append(nutrition.get('total_calories', 0))
                proteins.append(nutrition.get('total_protein', 0))
                carbs.append(nutrition.get('total_carbs', 0))
                fats.append(nutrition.get('total_fats', 0))
            
            # Create grouped bar chart
            x = np.arange(len(meal_names))
            width = 0.2
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            ax.bar(x - width*1.5, calories, width, label='Calories', color=self.nutrition_colors['calories'])
            ax.bar(x - width*0.5, proteins, width, label='Protein (g)', color=self.nutrition_colors['protein'])
            ax.bar(x + width*0.5, carbs, width, label='Carbs (g)', color=self.nutrition_colors['carbs'])
            ax.bar(x + width*1.5, fats, width, label='Fats (g)', color=self.nutrition_colors['fats'])
            
            ax.set_xlabel('Meals')
            ax.set_ylabel('Amount')
            ax.set_title('Nutrition Comparison Across Meals', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(meal_names, rotation=45)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error creating comparison chart: {e}")
            return None

def display_visualization_dashboard(nutrition_data, daily_calories, food_items, history_data):
    """Main function to display all visualizations"""
    visualizer = NutritionVisualizer()
    
    st.subheader("📊 Advanced Analytics Dashboard")
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "🍽️ Current Meal Analysis",
        "📈 Daily Trends", 
        "📊 Historical Data",
        "🎯 Interactive Charts"
    ])
    
    with tab1:
        st.write("**Current Meal Nutritional Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            pie_fig = visualizer.create_nutrition_pie_chart(nutrition_data)
            if pie_fig:
                st.pyplot(pie_fig)
                plt.close()
            
            # Radar chart
            radar_fig = visualizer.create_nutrition_radar_chart(nutrition_data)
            if radar_fig:
                st.pyplot(radar_fig)
                plt.close()
        
        with col2:
            # Bar chart
            bar_fig = visualizer.create_nutrition_bar_chart(nutrition_data)
            if bar_fig:
                st.pyplot(bar_fig)
                plt.close()
            
            # Food items breakdown
            meal_fig = visualizer.create_meal_analysis_chart(food_items)
            if meal_fig:
                st.pyplot(meal_fig)
                plt.close()
    
    with tab2:
        st.write("**Daily Calorie Trends and Progress**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily trend
            trend_fig = visualizer.create_daily_calorie_trend(daily_calories)
            if trend_fig:
                st.pyplot(trend_fig)
                plt.close()
        
        with col2:
            # Gauge chart
            current_cals = daily_calories.get(date.today().isoformat(), 0)
            gauge_fig = visualizer.create_calorie_gauge_chart(current_cals)
            if gauge_fig:
                st.plotly_chart(gauge_fig, use_container_width=True)
            elif not PLOTLY_AVAILABLE:
                st.info("📊 **Daily Progress**: {current_cals} calories")
                st.progress(min(current_cals / 2000, 1.0))
                st.caption(f"Progress: {(current_cals / 2000) * 100:.1f}% of daily target")
    
    with tab3:
        st.write("**Historical Data Analysis**")
        
        # Weekly summary
        weekly_fig = visualizer.create_weekly_summary_chart(daily_calories)
        if weekly_fig:
            st.pyplot(weekly_fig)
            plt.close()
        
        # Meal comparison
        comparison_fig = visualizer.create_nutrition_comparison_chart(history_data)
        if comparison_fig:
            st.pyplot(comparison_fig)
            plt.close()
    
    with tab4:
        st.write("**Interactive Visualizations**")
        
        if not PLOTLY_AVAILABLE:
            st.warning("⚠️ **Plotly not available** - Interactive charts are disabled.")
            st.info("To enable interactive charts, install plotly: `pip install plotly`")
            st.write("**Alternative**: Use the static charts in other tabs for visualization.")
        else:
            # Interactive charts
            fig_pie, fig_bar, fig_trend = visualizer.create_plotly_interactive_charts(
                nutrition_data, daily_calories, food_items
            )
            
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
            
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
            
            if fig_trend:
                st.plotly_chart(fig_trend, use_container_width=True)
            
            if not any([fig_pie, fig_bar, fig_trend]):
                st.info("No interactive charts available. Add more data to see interactive visualizations.")

def create_quick_summary_charts(nutrition_data, daily_calories):
    """Create quick summary charts for sidebar or compact display"""
    visualizer = NutritionVisualizer()
    
    # Quick pie chart
    pie_fig = visualizer.create_nutrition_pie_chart(nutrition_data)
    if pie_fig:
        st.pyplot(pie_fig)
        plt.close()
    
    # Quick trend
    trend_fig = visualizer.create_daily_calorie_trend(daily_calories)
    if trend_fig:
        st.pyplot(trend_fig)
        plt.close()

# Export functions for easy import
__all__ = [
    'NutritionVisualizer',
    'display_visualization_dashboard', 
    'create_quick_summary_charts'
]
