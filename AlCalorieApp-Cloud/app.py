import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os
from datetime import datetime, date
import logging
import re
from typing import Dict, Any, List
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="🍱 AI Calorie Tracker", 
    layout="wide", 
    page_icon="🍽️",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Import utilities with error handling
try:
    from utils.models import load_models, get_model_status
    from utils.analysis import analyze_food_image, extract_nutrition_data
    from utils.ui import create_header, create_sidebar, create_upload_section
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    st.error("Utility modules not found. Please check the project structure.")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "daily_calories" not in st.session_state:
    st.session_state.daily_calories = {}
if "calorie_target" not in st.session_state:
    st.session_state.calorie_target = 2000

# Load models with caching
@st.cache_resource
def initialize_models():
    """Initialize AI models with proper error handling"""
    if UTILS_AVAILABLE:
        return load_models()
    else:
        return {"error": "Utils not available"}

# Load models
models = initialize_models()

# Custom CSS for better appearance
st.markdown("""
<style>
    .main { 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px; 
        border-radius: 10px; 
    }
    .stButton>button { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; 
        border-radius: 8px; 
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .header-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .analysis-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    .status-success { color: #4CAF50; font-weight: bold; }
    .status-error { color: #f44336; font-weight: bold; }
    .status-warning { color: #ff9800; font-weight: bold; }
    
    /* Enhanced Footer */
    .modern-footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem;
        margin: 3rem -1rem -1rem -1rem;
        border-radius: 30px 30px 0 0;
        text-align: center;
        box-shadow: 0 -8px 32px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .modern-footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    }
    
    .modern-footer h3 {
        margin-bottom: 1rem;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .modern-footer p {
        margin: 0.5rem 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .modern-footer a {
        transition: all 0.3s ease;
    }
    
    .modern-footer a:hover {
        opacity: 0.8;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

def create_complex_nutrition_charts(nutrition_data):
    """Create beautiful and complex nutrition charts with advanced visualizations"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Circle, Wedge, Rectangle, FancyBboxPatch
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patches as mpatches
        
        # Set style for beautiful charts
        plt.style.use('default')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        
        charts = {}
        
        # 1. Complex 3D-Style Bar Chart with Gradients
        fig1 = plt.figure(figsize=(14, 8))
        ax1 = fig1.add_subplot(111)
        
        categories = ['Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)', 'Fiber (g)']
        values = [
            nutrition_data.get('total_calories', 0),
            nutrition_data.get('total_protein', 0),
            nutrition_data.get('total_carbs', 0),
            nutrition_data.get('total_fats', 0),
            nutrition_data.get('total_fiber', 0) if 'total_fiber' in nutrition_data else 0
        ]
        
        # Create gradient colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D', '#9B59B6']
        gradient_colors = []
        
        for i, color in enumerate(colors):
            # Create gradient effect
            cmap = LinearSegmentedColormap.from_list('custom', [color, 'white'])
            gradient_colors.append(cmap)
        
        # Create 3D-style bars with gradients
        x_pos = np.arange(len(categories))
        bars = []
        
        for i, (value, color, cmap) in enumerate(zip(values, colors, gradient_colors)):
            # Main bar
            bar = ax1.bar(x_pos[i], value, color=color, alpha=0.8, edgecolor='white', linewidth=3, width=0.6)
            
            # Add gradient overlay
            gradient = np.linspace(0, 1, 100)
            for j, alpha in enumerate(gradient):
                height = (value * alpha)
                ax1.bar(x_pos[i], height, color=color, alpha=0.1, width=0.6)
            
            # Add 3D shadow effect
            shadow = ax1.bar(x_pos[i] + 0.02, value, color='black', alpha=0.2, width=0.6)
            
            bars.extend(bar)
        
        # Add value labels with beautiful styling
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.03,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=colors[i]))
        
        # Enhanced styling
        ax1.set_title('Advanced Nutritional Analysis', fontsize=20, fontweight='bold', pad=30, 
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        ax1.set_ylabel('Amount', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Nutritional Components', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.2, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add background gradient
        ax1.set_facecolor('#f8f9fa')
        fig1.patch.set_facecolor('white')
        
        plt.tight_layout()
        charts['complex_bar_chart'] = fig1
        
        # 2. Complex Enhanced Donut Chart with Multiple Rings
        total_cals = nutrition_data.get('total_calories', 0)
        if total_cals > 0:
            protein_cals = nutrition_data.get('total_protein', 0) * 4
            carbs_cals = nutrition_data.get('total_carbs', 0) * 4
            fats_cals = nutrition_data.get('total_fats', 0) * 9
            
            # Calculate percentages
            protein_pct = (protein_cals / total_cals) * 100
            carbs_pct = (carbs_cals / total_cals) * 100
            fats_pct = (fats_cals / total_cals) * 100
            
            fig2 = plt.figure(figsize=(12, 10))
            ax2 = fig2.add_subplot(111)
            
            # Create multiple rings for enhanced effect
            sizes = [protein_pct, carbs_pct, fats_pct]
            labels = [f'Protein\n{protein_pct:.1f}%', f'Carbs\n{carbs_pct:.1f}%', f'Fats\n{fats_pct:.1f}%']
            colors_pie = ['#4ECDC4', '#45B7D1', '#FFD93D']
            
            # Outer ring (main data)
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                              startangle=90, explode=(0.1, 0.1, 0.1), radius=1.0)
            
            # Middle ring (shadow effect)
            ax2.pie(sizes, colors=['black']*3, radius=0.8, startangle=90)
            
            # Inner ring (highlight effect)
            ax2.pie(sizes, colors=colors_pie, radius=0.6, startangle=90)
            
            # Create donut effect with multiple circles
            centre_circle = plt.Circle((0,0), 0.4, fc='white', edgecolor='gray', linewidth=2)
            ax2.add_artist(centre_circle)
            
            # Add center information
            ax2.text(0, 0.1, f'{total_cals:.0f}', ha='center', va='center', fontsize=24, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
            ax2.text(0, -0.1, 'Total kcal', ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Enhance text with beautiful styling
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(14)
                autotext.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            for text in texts:
                text.set_fontsize(12)
                text.set_fontweight('bold')
                text.set_color('white')
                text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            ax2.set_title('Enhanced Macronutrient Distribution', fontsize=18, fontweight='bold', pad=30,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.3))
            
            # Enhanced styling
            ax2.set_facecolor('#f8f9fa')
            fig2.patch.set_facecolor('white')
            
            plt.tight_layout()
            charts['enhanced_pie_chart'] = fig2
        
        # 3. Complex Horizontal Bar Chart with Patterns and Effects
        fig3 = plt.figure(figsize=(14, 6))
        ax3 = fig3.add_subplot(111)
        
        sources = ['Protein', 'Carbohydrates', 'Fats']
        calories = [protein_cals, carbs_cals, fats_cals]
        colors_hbar = ['#4ECDC4', '#45B7D1', '#FFD93D']
        
        # Create gradient bars with patterns
        y_pos = np.arange(len(sources))
        bars = []
        
        for i, (source, cal, color) in enumerate(zip(sources, calories, colors_hbar)):
            # Main bar with gradient
            bar = ax3.barh(y_pos[i], cal, color=color, alpha=0.8, edgecolor='white', linewidth=3, height=0.6)
            
            # Add pattern overlay
            pattern = np.linspace(0, cal, 50)
            for j, val in enumerate(pattern):
                ax3.barh(y_pos[i], val, color=color, alpha=0.1, height=0.6)
            
            # Add 3D shadow effect
            shadow = ax3.barh(y_pos[i] - 0.02, cal, color='black', alpha=0.3, height=0.6)
            
            bars.extend(bar)
        
        # Add value labels with beautiful styling
        for i, (bar, value) in enumerate(zip(bars, calories)):
            width = bar.get_width()
            ax3.text(width + max(calories)*0.03, bar.get_y() + bar.get_height()/2,
                   f'{value:.0f} kcal', ha='left', va='center', fontweight='bold', fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=colors_hbar[i]))
        
        # Enhanced styling
        ax3.set_title('Advanced Calorie Sources Analysis', fontsize=18, fontweight='bold', pad=25,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.3))
        ax3.set_xlabel('Calories (kcal)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Macronutrients', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.2, linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Add background gradient
        ax3.set_facecolor('#f8f9fa')
        fig3.patch.set_facecolor('white')
        
        plt.tight_layout()
        charts['complex_calorie_sources'] = fig3
        
        # 4. Complex Multi-Ring Donut Chart with Progress Indicators
        daily_target = st.session_state.get('calorie_target', 2000)
        current_cals = nutrition_data.get('total_calories', 0)
        remaining = max(0, daily_target - current_cals)
        
        fig4 = plt.figure(figsize=(12, 10))
        ax4 = fig4.add_subplot(111)
        
        # Create multiple rings for complex effect
        sizes = [current_cals, remaining]
        labels = [f'Consumed\n{current_cals} kcal', f'Remaining\n{remaining} kcal']
        colors_donut = ['#FF6B6B', '#E0E0E0']
        
        # Outer ring (main progress)
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_donut, autopct='%1.1f%%',
                                          startangle=90, pctdistance=0.85, radius=1.0)
        
        # Middle ring (shadow effect)
        ax4.pie(sizes, colors=['black']*2, radius=0.8, startangle=90)
        
        # Inner ring (highlight)
        ax4.pie(sizes, colors=colors_donut, radius=0.6, startangle=90)
        
        # Create donut effect with multiple circles
        centre_circle = plt.Circle((0,0), 0.5, fc='white', edgecolor='gray', linewidth=2)
        ax4.add_artist(centre_circle)
        
        # Add progress indicator in center
        progress_pct = (current_cals / daily_target * 100) if daily_target > 0 else 0
        ax4.text(0, 0.1, f'{progress_pct:.1f}%', ha='center', va='center', fontsize=24, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        ax4.text(0, -0.1, 'Complete', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Enhance text with beautiful styling
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)
            autotext.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
            text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax4.set_title(f'Advanced Daily Progress Analysis\n({daily_target} kcal target)', 
                     fontsize=18, fontweight='bold', pad=30,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.3))
        plt.tight_layout()
        charts['complex_daily_progress'] = fig4
        
        # 5. Complex Radar Chart for Nutritional Balance
        fig5 = plt.figure(figsize=(10, 10))
        ax5 = fig5.add_subplot(111, projection='polar')
        
        # Calculate nutritional scores (0-100 scale)
        protein_score = min(100, (nutrition_data.get('total_protein', 0) / 50) * 100)  # 50g = 100%
        carbs_score = min(100, (nutrition_data.get('total_carbs', 0) / 100) * 100)     # 100g = 100%
        fats_score = min(100, (nutrition_data.get('total_fats', 0) / 50) * 100)        # 50g = 100%
        fiber_score = min(100, (nutrition_data.get('total_fiber', 0) / 25) * 100) if 'total_fiber' in nutrition_data else 0
        calorie_score = min(100, (nutrition_data.get('total_calories', 0) / 800) * 100) # 800kcal = 100%
        
        categories = ['Protein', 'Carbs', 'Fats', 'Fiber', 'Calories']
        values = [protein_score, carbs_score, fats_score, fiber_score, calorie_score]
        
        # Close the plot by appending first value
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        # Plot radar chart
        ax5.plot(angles, values, 'o-', linewidth=3, color='#FF6B6B', label='Current Meal')
        ax5.fill(angles, values, alpha=0.25, color='#FF6B6B')
        
        # Add target line
        target_values = [80, 80, 80, 80, 80] + [80]
        ax5.plot(angles, target_values, 'o--', linewidth=2, color='#4ECDC4', label='Target (80%)')
        
        # Set labels
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories, fontsize=12, fontweight='bold')
        ax5.set_ylim(0, 100)
        ax5.set_yticks([20, 40, 60, 80, 100])
        ax5.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        
        # Add grid
        ax5.grid(True, alpha=0.3)
        
        # Add legend
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        ax5.set_title('Nutritional Balance Radar Analysis', fontsize=16, fontweight='bold', pad=30,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.3))
        plt.tight_layout()
        charts['radar_chart'] = fig5
        
        return charts
        
    except Exception as e:
        st.error(f"Error creating charts: {e}")
        return None

def create_simple_chart(nutrition_data):
    """Create a simple nutrition chart (for backward compatibility)"""
    charts = create_nutrition_charts(nutrition_data)
    return charts.get('bar_chart') if charts else None

def create_complex_history_trends(history_data):
    """Create beautiful and complex trend charts for history data"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime, timedelta
        from matplotlib.patches import Rectangle, FancyBboxPatch
        from matplotlib.colors import LinearSegmentedColormap
        
        if not history_data:
            return None
        
        # Set style for beautiful charts
        plt.style.use('default')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        
        charts = {}
        
        # Prepare data
        dates = []
        calories = []
        proteins = []
        carbs = []
        fats = []
        
        for entry in history_data[-10:]:  # Last 10 entries
            if isinstance(entry.get('timestamp'), datetime):
                dates.append(entry['timestamp'].strftime('%m/%d'))
            else:
                dates.append(f"Entry {len(dates)+1}")
            
            nutrition = entry.get('nutritional_data', {})
            calories.append(nutrition.get('total_calories', 0))
            proteins.append(nutrition.get('total_protein', 0))
            carbs.append(nutrition.get('total_carbs', 0))
            fats.append(nutrition.get('total_fats', 0))
        
        # 1. Complex 3D-Style Calorie Trend with Area Fill
        fig1 = plt.figure(figsize=(14, 8))
        ax1 = fig1.add_subplot(111)
        
        # Create gradient for area fill
        gradient = np.linspace(0, 1, len(dates))
        colors = plt.cm.Reds(gradient)
        
        # Plot with enhanced styling
        line = ax1.plot(dates, calories, marker='o', linewidth=4, markersize=12, 
                       color='#FF6B6B', label='Calories', zorder=3)
        
        # Add gradient area fill
        ax1.fill_between(dates, calories, alpha=0.6, color='#FF6B6B', zorder=1)
        
        # Add shadow effect
        ax1.plot(dates, calories, color='black', alpha=0.3, linewidth=4, zorder=2)
        
        # Add value labels
        for i, (date, cal) in enumerate(zip(dates, calories)):
            ax1.annotate(f'{cal:.0f}', (date, cal), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax1.set_title('Advanced Calorie Intake Trend Analysis', fontsize=18, fontweight='bold', pad=30,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.3))
        ax1.set_ylabel('Calories', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.2, linestyle='--', zorder=0)
        ax1.legend(fontsize=12)
        plt.xticks(rotation=45)
        
        # Enhanced styling
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_facecolor('#f8f9fa')
        fig1.patch.set_facecolor('white')
        
        plt.tight_layout()
        charts['complex_calorie_trend'] = fig1
        
        # 2. Complex 3D-Style Macronutrient Trend
        fig2 = plt.figure(figsize=(14, 8))
        ax2 = fig2.add_subplot(111)
        
        x = np.arange(len(dates))
        width = 0.25
        
        # Create 3D-style bars with gradients
        bars1 = ax2.bar(x - width, proteins, width, label='Protein', color='#4ECDC4', 
                       alpha=0.8, edgecolor='white', linewidth=2)
        bars2 = ax2.bar(x, carbs, width, label='Carbs', color='#45B7D1', 
                       alpha=0.8, edgecolor='white', linewidth=2)
        bars3 = ax2.bar(x + width, fats, width, label='Fats', color='#FFD93D', 
                       alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add shadow effects
        ax2.bar(x - width + 0.02, proteins, width, color='black', alpha=0.3)
        ax2.bar(x + 0.02, carbs, width, color='black', alpha=0.3)
        ax2.bar(x + width + 0.02, fats, width, color='black', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(max(proteins), max(carbs), max(fats))*0.02,
                       f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        ax2.set_title('Advanced Macronutrient Intake Trend Analysis', fontsize=18, fontweight='bold', pad=30,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        ax2.set_ylabel('Grams', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(dates, rotation=45)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.2, linestyle='--')
        
        # Enhanced styling
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_facecolor('#f8f9fa')
        fig2.patch.set_facecolor('white')
        
        plt.tight_layout()
        charts['complex_macro_trend'] = fig2
        
        # 3. Complex Progress Chart with Gradient Bars
        daily_target = st.session_state.get('calorie_target', 2000)
        daily_progress = [(cal / daily_target * 100) if daily_target > 0 else 0 for cal in calories]
        
        fig3 = plt.figure(figsize=(14, 8))
        ax3 = fig3.add_subplot(111)
        
        # Create gradient colors based on progress
        colors = ['#4ECDC4' if p <= 100 else '#FF6B6B' for p in daily_progress]
        
        bars = ax3.bar(dates, daily_progress, color=colors, alpha=0.8, edgecolor='white', linewidth=3)
        
        # Add shadow effects
        ax3.bar(dates, daily_progress, color='black', alpha=0.3, edgecolor='white', linewidth=3)
        
        # Add target line with enhanced styling
        ax3.axhline(y=100, color='#FF6B6B', linestyle='--', linewidth=3, label='Daily Target (100%)', alpha=0.8)
        
        # Add value labels
        for bar, progress in zip(bars, daily_progress):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(daily_progress)*0.02,
                   f'{progress:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax3.set_title('Advanced Daily Calorie Target Progress Analysis', fontsize=18, fontweight='bold', pad=30,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.3))
        ax3.set_ylabel('Progress (%)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, max(daily_progress) * 1.1)
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.2, linestyle='--')
        plt.xticks(rotation=45)
        
        # Enhanced styling
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.set_facecolor('#f8f9fa')
        fig3.patch.set_facecolor('white')
        
        plt.tight_layout()
        charts['complex_progress_trend'] = fig3
        
        return charts
        
    except Exception as e:
        st.error(f"Error creating complex history trends: {e}")
        return None

def create_history_trends(history_data):
    """Create trend charts for history data (backward compatibility)"""
    return create_complex_history_trends(history_data)

def main():
    # Header
    st.markdown("""
    <div class="header-card">
        <h1>🍱 AI Calorie Tracker</h1>
        <p>Track your nutrition with AI-powered food analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model status in sidebar
    with st.sidebar:
        st.markdown("### 🤖 AI Models Status")
        
        if UTILS_AVAILABLE and "error" not in models:
            model_status = get_model_status(models)
            for model, status in model_status.items():
                status_class = "status-success" if status else "status-error"
                status_icon = "✅" if status else "❌"
                status_text = "Available" if status else "Not Available"
                st.markdown(f'<span class="{status_class}">{status_icon} **{model}**: {status_text}</span>', unsafe_allow_html=True)
        else:
            st.error("Models not available. Check deployment configuration.")
        
        st.markdown("---")
        
        # User settings
        st.markdown("### 👤 Settings")
        st.number_input("Daily Calorie Target (kcal)", min_value=1000, max_value=5000, 
                       value=st.session_state.calorie_target, step=100, key="calorie_target")
        
        # Today's progress
        st.markdown("### 📊 Today's Progress")
        today = date.today().isoformat()
        today_cals = st.session_state.daily_calories.get(today, 0)
        progress = min(today_cals / st.session_state.calorie_target, 1.0) if st.session_state.calorie_target > 0 else 0
        
        st.metric("Calories", f"{today_cals} / {st.session_state.calorie_target}")
        st.progress(progress)
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.history.clear()
            st.session_state.daily_calories.clear()
            st.success("History cleared!")
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🍽️ Food Analysis & Enhanced Agent", "📊 History", "📈 Analytics"])
    
    with tab1:
        st.markdown("""
        <div class="metric-card">
            <h3>🍽️ Food Analysis & Enhanced Agent</h3>
            <p>Upload a food image for comprehensive analysis with AI-powered insights and web-sourced information.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tips
        with st.expander("💡 Tips for Better Results"):
            st.markdown("""
            - 📸 Take clear photos in good lighting
            - 🍽️ Include all food items in the frame
            - 📝 Add context description if needed
            - 🔄 Try different angles if detection is incomplete
            - 🌐 Use Enhanced Agent for web-sourced information
            """)
        
        # Single file upload for both analysis types
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of your food for analysis"
        )
        
        # Single context input for both analysis types
        context = st.text_area(
            "Additional Context (Optional)", 
            placeholder="Describe the meal if needed (e.g., 'chicken curry with rice')", 
            height=80
        )
        
        # Display uploaded image if available
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Food Image", use_column_width=True)
        
        # Single comprehensive analysis button
        st.markdown("### 🚀 Comprehensive Analysis")
        st.markdown("Get both standard nutritional analysis and enhanced web-sourced insights in one click!")
        
        # Single analysis button that does both
        if st.button("🔍 Analyze Food (Standard + Enhanced)", disabled=not uploaded_file, type="primary"):
                if uploaded_file and UTILS_AVAILABLE and "error" not in models:
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("📷 Loading image...")
                        progress_bar.progress(10)
                        
                        image = Image.open(uploaded_file)
                        
                        status_text.text("🔍 Running standard analysis...")
                        progress_bar.progress(30)
                        
                        # Standard analysis
                        analysis_result = analyze_food_image(image, context, models)
                        
                        status_text.text("🤖 Running enhanced agent...")
                        progress_bar.progress(60)
                        
                        # Enhanced agent analysis
                        enhanced_result = None
                        try:
                            from utils.food_agent import FoodAgent
                            agent = FoodAgent(models)
                            enhanced_result = agent.process_food_image_complete(image)
                        except Exception as e:
                            st.warning(f"Enhanced agent not available: {str(e)}")
                        
                        status_text.text("📊 Processing combined results...")
                        progress_bar.progress(90)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Comprehensive analysis complete!")
                        
                        # Clear progress
                        progress_bar.empty()
                        status_text.empty()
                        
                        if analysis_result["success"]:
                            st.success("✅ Comprehensive analysis completed!")
                            
                            # Create a beautiful results container with enhanced food items display
                            description = analysis_result.get('description', 'Food items detected')
                            
                            # Format the description for better display
                            if description.startswith("Main Food Items Identified:"):
                                # Extract just the food items for cleaner display
                                food_items = description.replace("Main Food Items Identified:", "").strip()
                                food_items_list = food_items.split(", ")
                                display_title = "🍽️ Main Food Items Identified"
                                display_content = food_items
                            else:
                                display_title = "🍽️ Comprehensive Analysis Results"
                                display_content = description
                                food_items_list = [description]
                            
                            # Enhanced main results container
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 25px; border-radius: 20px; color: white; margin: 20px 0; 
                                        box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.2);">
                                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                                    <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 50%; margin-right: 15px;">
                                        <span style="font-size: 24px;">🍽️</span>
                                    </div>
                                    <div>
                                        <h3 style="color: white; margin: 0; font-size: 24px; font-weight: 600;">{display_title}</h3>
                                        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 14px;">AI-Powered Food Detection</p>
                                    </div>
                                </div>
                                <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; margin-top: 15px;">
                                    <p style="font-size: 16px; margin: 0; line-height: 1.6; color: #333; font-weight: 500;">
                                        <strong style="color: #667eea;">Detected Items:</strong> {display_content}
                                    </p>
                                </div>
                                <div style="margin-top: 15px; display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-size: 12px; opacity: 0.8;">Standard Analysis + Enhanced Agent</span>
                                    <span style="background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 12px;">AI Verified</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Enhanced item count summary with better styling
                            if description.startswith("Main Food Items Identified:"):
                                item_count = len(food_items_list)
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                                            padding: 20px; border-radius: 15px; color: white; margin: 15px 0; 
                                            box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                                    <div style="display: flex; align-items: center; justify-content: space-between;">
                                        <div style="display: flex; align-items: center;">
                                            <div style="background: rgba(255,255,255,0.2); padding: 8px; border-radius: 50%; margin-right: 12px;">
                                                <span style="font-size: 18px;">📊</span>
                                            </div>
                                            <div>
                                                <h4 style="color: white; margin: 0; font-size: 18px; font-weight: 600;">Detection Summary</h4>
                                                <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 14px;">AI Analysis Results</p>
                                            </div>
                                        </div>
                                        <div style="text-align: center; background: rgba(255,255,255,0.2); padding: 15px; border-radius: 12px; min-width: 80px;">
                                            <div style="font-size: 24px; font-weight: bold; color: white;">{item_count}</div>
                                            <div style="font-size: 12px; color: rgba(255,255,255,0.8);">Items Found</div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Enhanced nutrition summary with better styling
                            nutrition = analysis_result["nutritional_data"]
                            
                            st.markdown("### 📈 Nutrition Analysis")
                            
                            # Create nutrition cards with enhanced styling
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%); 
                                            padding: 25px; border-radius: 18px; color: white; text-align: center; 
                                            box-shadow: 0 8px 25px rgba(255,107,107,0.3); border: 1px solid rgba(255,255,255,0.2);">
                                    <div style="font-size: 28px; margin-bottom: 12px;">🔥</div>
                                    <div style="font-size: 24px; font-weight: bold; margin-bottom: 8px;">{nutrition['total_calories']}</div>
                                    <div style="font-size: 14px; opacity: 0.9; font-weight: 500;">Calories</div>
                                    <div style="font-size: 12px; opacity: 0.7; margin-top: 5px;">kcal</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                                            padding: 25px; border-radius: 18px; color: white; text-align: center; 
                                            box-shadow: 0 8px 25px rgba(78,205,196,0.3); border: 1px solid rgba(255,255,255,0.2);">
                                    <div style="font-size: 28px; margin-bottom: 12px;">💪</div>
                                    <div style="font-size: 24px; font-weight: bold; margin-bottom: 8px;">{nutrition['total_protein']:.1f}g</div>
                                    <div style="font-size: 14px; opacity: 0.9; font-weight: 500;">Protein</div>
                                    <div style="font-size: 12px; opacity: 0.7; margin-top: 5px;">Building Blocks</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col3:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #45B7D1 0%, #96CEB4 100%); 
                                            padding: 25px; border-radius: 18px; color: white; text-align: center; 
                                            box-shadow: 0 8px 25px rgba(69,183,209,0.3); border: 1px solid rgba(255,255,255,0.2);">
                                    <div style="font-size: 28px; margin-bottom: 12px;">🌾</div>
                                    <div style="font-size: 24px; font-weight: bold; margin-bottom: 8px;">{nutrition['total_carbs']:.1f}g</div>
                                    <div style="font-size: 14px; opacity: 0.9; font-weight: 500;">Carbs</div>
                                    <div style="font-size: 12px; opacity: 0.7; margin-top: 5px;">Energy Source</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col4:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #FFD93D 0%, #FFB347 100%); 
                                            padding: 25px; border-radius: 18px; color: white; text-align: center; 
                                            box-shadow: 0 8px 25px rgba(255,217,61,0.3); border: 1px solid rgba(255,255,255,0.2);">
                                    <div style="font-size: 28px; margin-bottom: 12px;">🥑</div>
                                    <div style="font-size: 24px; font-weight: bold; margin-bottom: 8px;">{nutrition['total_fats']:.1f}g</div>
                                    <div style="font-size: 14px; opacity: 0.9; font-weight: 500;">Fats</div>
                                    <div style="font-size: 12px; opacity: 0.7; margin-top: 5px;">Essential Fats</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show beautiful and complex charts
                            st.markdown("### 🎨 Advanced Nutrition Visualizations")
                            
                            # Create all complex charts
                            charts = create_complex_nutrition_charts(nutrition)
                            
                            if charts:
                                # Create tabs for different chart types
                                chart_tab1, chart_tab2, chart_tab3, chart_tab4, chart_tab5 = st.tabs([
                                    "📊 3D Breakdown", "🥧 Enhanced Distribution", "🔥 Advanced Sources", "🎯 Multi-Ring Progress", "🎯 Radar Analysis"
                                ])
                                
                                with chart_tab1:
                                    st.markdown("#### 📊 Advanced 3D Nutritional Breakdown")
                                    st.pyplot(charts['complex_bar_chart'])
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                padding: 20px; border-radius: 15px; color: white; margin: 15px 0;">
                                        <h5 style="color: white; margin-bottom: 10px;">🎨 Advanced Features:</h5>
                                        <ul style="color: rgba(255,255,255,0.9); margin: 0; padding-left: 20px;">
                                            <li>3D-style bars with gradient effects</li>
                                            <li>Shadow effects for depth</li>
                                            <li>Enhanced value labels with rounded boxes</li>
                                            <li>Professional styling with background gradients</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with chart_tab2:
                                    st.markdown("#### 🥧 Enhanced Macronutrient Distribution")
                                    st.pyplot(charts['enhanced_pie_chart'])
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                                                padding: 20px; border-radius: 15px; color: white; margin: 15px 0;">
                                        <h5 style="color: white; margin-bottom: 10px;">🎨 Advanced Features:</h5>
                                        <ul style="color: rgba(255,255,255,0.9); margin: 0; padding-left: 20px;">
                                            <li>Multiple rings for depth effect</li>
                                            <li>Shadow and highlight effects</li>
                                            <li>Center total calorie display</li>
                                            <li>Enhanced text with background boxes</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with chart_tab3:
                                    st.markdown("#### 🔥 Advanced Calorie Sources Analysis")
                                    st.pyplot(charts['complex_calorie_sources'])
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%); 
                                                padding: 20px; border-radius: 15px; color: white; margin: 15px 0;">
                                        <h5 style="color: white; margin-bottom: 10px;">🎨 Advanced Features:</h5>
                                        <ul style="color: rgba(255,255,255,0.9); margin: 0; padding-left: 20px;">
                                            <li>Gradient bars with pattern overlays</li>
                                            <li>3D shadow effects</li>
                                            <li>Enhanced value labels with styling</li>
                                            <li>Professional grid and background</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with chart_tab4:
                                    st.markdown("#### 🎯 Multi-Ring Daily Progress Analysis")
                                    st.pyplot(charts['complex_daily_progress'])
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #FFD93D 0%, #FFB347 100%); 
                                                padding: 20px; border-radius: 15px; color: white; margin: 15px 0;">
                                        <h5 style="color: white; margin-bottom: 10px;">🎨 Advanced Features:</h5>
                                        <ul style="color: rgba(255,255,255,0.9); margin: 0; padding-left: 20px;">
                                            <li>Multiple rings for depth effect</li>
                                            <li>Center progress indicator</li>
                                            <li>Enhanced text with background boxes</li>
                                            <li>Professional styling with gradients</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with chart_tab5:
                                    st.markdown("#### 🎯 Nutritional Balance Radar Analysis")
                                    st.pyplot(charts['radar_chart'])
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #9B59B6 0%, #8E44AD 100%); 
                                                padding: 20px; border-radius: 15px; color: white; margin: 15px 0;">
                                        <h5 style="color: white; margin-bottom: 10px;">🎨 Advanced Features:</h5>
                                        <ul style="color: rgba(255,255,255,0.9); margin: 0; padding-left: 20px;">
                                            <li>Polar radar chart for balance analysis</li>
                                            <li>Target line comparison</li>
                                            <li>Filled area for visual impact</li>
                                            <li>Professional grid and legend</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Add a summary section
                                st.markdown("### 📈 Chart Summary")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%); 
                                                padding: 20px; border-radius: 15px; color: white; text-align: center; 
                                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                                        <div style="font-size: 24px; margin-bottom: 8px;">📊</div>
                                        <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">4 Charts</div>
                                        <div style="font-size: 14px; opacity: 0.9;">Created</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    total_cals = nutrition['total_calories']
                                    daily_target = st.session_state.get('calorie_target', 2000)
                                    progress_pct = (total_cals / daily_target * 100) if daily_target > 0 else 0
                                    
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                                                padding: 20px; border-radius: 15px; color: white; text-align: center; 
                                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                                        <div style="font-size: 24px; margin-bottom: 8px;">🎯</div>
                                        <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">{progress_pct:.1f}%</div>
                                        <div style="font-size: 14px; opacity: 0.9;">Daily Target</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    # Determine chart recommendation
                                    if nutrition['total_protein'] > 30:
                                        recommendation = "High Protein"
                                        icon = "💪"
                                        color = "#4ECDC4"
                                    elif nutrition['total_carbs'] > 50:
                                        recommendation = "High Carb"
                                        icon = "🌾"
                                        color = "#45B7D1"
                                    else:
                                        recommendation = "Balanced"
                                        icon = "⚖️"
                                        color = "#FFD93D"
                                    
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, {color} 0%, #FFB347 100%); 
                                                padding: 20px; border-radius: 15px; color: white; text-align: center; 
                                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                                        <div style="font-size: 24px; margin-bottom: 8px;">{icon}</div>
                                        <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">{recommendation}</div>
                                        <div style="font-size: 14px; opacity: 0.9;">Meal Type</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Enhanced detailed analysis display
                            st.markdown("### 📝 Detailed Analysis")
                            
                            # Parse and display analysis in a more structured way
                            analysis_text = analysis_result["analysis"]
                            
                            # Create expandable sections for better organization
                            with st.expander("🔍 Complete Analysis Report", expanded=True):
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                            padding: 25px; border-radius: 18px; color: white; margin: 15px 0; 
                                            box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
                                    <div style="display: flex; align-items: center; margin-bottom: 20px;">
                                        <div style="background: rgba(255,255,255,0.2); padding: 8px; border-radius: 50%; margin-right: 12px;">
                                            <span style="font-size: 18px;">📋</span>
                                        </div>
                                        <h4 style="color: white; margin: 0; font-size: 18px; font-weight: 600;">AI Analysis Report</h4>
                                    </div>
                                    <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 12px; color: #333;">
                                        <div style="line-height: 1.8; font-size: 15px;">
                                            {analysis_text.replace(chr(10), '<br>')}
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Add quick insights section
                            st.markdown("### 💡 Quick Insights")
                            
                            # Calculate some quick insights
                            total_cals = nutrition['total_calories']
                            protein_pct = (nutrition['total_protein'] * 4 / total_cals * 100) if total_cals > 0 else 0
                            carbs_pct = (nutrition['total_carbs'] * 4 / total_cals * 100) if total_cals > 0 else 0
                            fats_pct = (nutrition['total_fats'] * 9 / total_cals * 100) if total_cals > 0 else 0
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                                            padding: 25px; border-radius: 18px; color: white; 
                                            box-shadow: 0 8px 25px rgba(78,205,196,0.3);">
                                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                                        <div style="background: rgba(255,255,255,0.2); padding: 8px; border-radius: 50%; margin-right: 12px;">
                                            <span style="font-size: 18px;">📊</span>
                                        </div>
                                        <h5 style="color: white; margin: 0; font-size: 18px; font-weight: 600;">Macronutrient Balance</h5>
                                    </div>
                                    <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 12px; color: #333;">
                                        <p style="margin: 8px 0; font-size: 15px;"><strong>Protein:</strong> {protein_pct:.1f}%</p>
                                        <p style="margin: 8px 0; font-size: 15px;"><strong>Carbs:</strong> {carbs_pct:.1f}%</p>
                                        <p style="margin: 8px 0; font-size: 15px;"><strong>Fats:</strong> {fats_pct:.1f}%</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                # Determine meal type based on calories
                                if total_cals < 300:
                                    meal_type = "Light Snack"
                                    meal_icon = "🍎"
                                    meal_color = "#FF6B6B"
                                elif total_cals < 600:
                                    meal_type = "Regular Meal"
                                    meal_icon = "🍽️"
                                    meal_color = "#4ECDC4"
                                else:
                                    meal_type = "Hearty Meal"
                                    meal_icon = "🍖"
                                    meal_color = "#FFD93D"
                                
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, {meal_color} 0%, #FFB347 100%); 
                                            padding: 25px; border-radius: 18px; color: white; 
                                            box-shadow: 0 8px 25px rgba(255,107,107,0.3);">
                                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                                        <div style="background: rgba(255,255,255,0.2); padding: 8px; border-radius: 50%; margin-right: 12px;">
                                            <span style="font-size: 18px;">🍽️</span>
                                        </div>
                                        <h5 style="color: white; margin: 0; font-size: 18px; font-weight: 600;">Meal Classification</h5>
                                    </div>
                                    <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 12px; color: #333;">
                                        <p style="margin: 8px 0; font-size: 15px;"><strong>Type:</strong> {meal_icon} {meal_type}</p>
                                        <p style="margin: 8px 0; font-size: 15px;"><strong>Calorie Level:</strong> {total_cals} kcal</p>
                                        <p style="margin: 8px 0; font-size: 15px;"><strong>Analysis Quality:</strong> ✅ High</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Enhanced Agent Results (if available)
                            if enhanced_result and "error" not in enhanced_result:
                                st.markdown("---")
                                st.markdown("### 🤖 Enhanced Agent Insights")
                                
                                # Create beautiful enhanced results container
                                st.markdown("""
                                <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                                            padding: 20px; border-radius: 15px; color: white; margin: 20px 0;">
                                    <h3 style="color: white; margin-bottom: 15px;">🌐 Web-Enhanced Insights</h3>
                                    <p style="font-size: 16px; margin-bottom: 0;"><strong>Session ID:</strong> {}</p>
                                </div>
                                """.format(enhanced_result['image_analysis']['session_id']), unsafe_allow_html=True)
                                
                                # Enhanced results in organized sections
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("### 📸 Enhanced Analysis")
                                    st.markdown(f"""
                                    <div style="background: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 15px; 
                                                box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 5px solid #4ECDC4;">
                                        <h5 style="color: #4ECDC4; margin-bottom: 15px;">🔍 AI + Web Detection</h5>
                                        <p style="line-height: 1.6; color: #333; margin-bottom: 10px;">
                                            <strong>Enhanced Description:</strong><br>
                                            {enhanced_result['image_analysis']['enhanced_description']}
                                        </p>
                                        <p style="color: #666; font-size: 14px; margin: 0;">
                                            <strong>Analysis Quality:</strong> ✅ Enhanced with Web Data
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("### 🌐 Web Information")
                                    web_info = enhanced_result['web_information']
                                    st.markdown(f"""
                                    <div style="background: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 15px; 
                                                box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 5px solid #FFD93D;">
                                        <h5 style="color: #FFD93D; margin-bottom: 15px;">🌍 Web-Sourced Data</h5>
                                        <p style="margin: 5px 0;"><strong>Food Name:</strong> {web_info.get('food_name', 'Unknown')}</p>
                                        <p style="margin: 5px 0;"><strong>Nutrition:</strong> {web_info.get('nutrition', {}).get('calories', 'Variable')}</p>
                                        <p style="margin: 5px 0;"><strong>Origin:</strong> {web_info.get('cultural', {}).get('origin', 'Various regions')}</p>
                                        <p style="color: #666; font-size: 14px; margin: 10px 0 0 0;">
                                            <strong>Data Source:</strong> 🌐 Web Search + AI Analysis
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Quick web insights with charts
                                st.markdown("### 💡 Web Insights & Analytics")
                                web_info = enhanced_result['web_information']
                                
                                # Create web insights charts
                                try:
                                    import matplotlib.pyplot as plt
                                    
                                    # Nutrition comparison chart
                                    if 'nutrition' in web_info and web_info['nutrition']:
                                        fig_web, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                        
                                        # Chart 1: Web vs AI Nutrition
                                        nutrition_sources = ['AI Analysis', 'Web Data']
                                        calories_comparison = [
                                            nutrition['total_calories'],
                                            web_info['nutrition'].get('calories', nutrition['total_calories'])
                                        ]
                                        
                                        bars1 = ax1.bar(nutrition_sources, calories_comparison, 
                                                       color=['#667eea', '#4ECDC4'], alpha=0.8)
                                        ax1.set_title('Calories: AI vs Web Data', fontweight='bold')
                                        ax1.set_ylabel('Calories')
                                        
                                        # Add value labels
                                        for bar, value in zip(bars1, calories_comparison):
                                            height = bar.get_height()
                                            ax1.text(bar.get_x() + bar.get_width()/2., height + max(calories_comparison)*0.02,
                                                   f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
                                        
                                        # Chart 2: Information Sources
                                        info_categories = ['Nutrition', 'Cultural', 'Health', 'Recipes']
                                        info_availability = [
                                            1 if 'nutrition' in web_info and web_info['nutrition'] else 0,
                                            1 if 'cultural' in web_info and web_info['cultural'] else 0,
                                            1 if 'health' in web_info and web_info['health'] else 0,
                                            1 if 'recipes' in web_info and web_info['recipes'] else 0
                                        ]
                                        
                                        bars2 = ax2.bar(info_categories, info_availability, 
                                                       color=['#FF6B6B', '#45B7D1', '#4ECDC4', '#FFD93D'], alpha=0.8)
                                        ax2.set_title('Information Availability', fontweight='bold')
                                        ax2.set_ylabel('Available (1) / Not Available (0)')
                                        ax2.set_ylim(0, 1.2)
                                        
                                        # Add value labels
                                        for bar, value in zip(bars2, info_availability):
                                            height = bar.get_height()
                                            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                                                   '✓' if value else '✗', ha='center', va='bottom', fontweight='bold', fontsize=16)
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig_web)
                                        
                                        st.markdown("""
                                        <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; 
                                                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 10px 0;">
                                            <p style="margin: 0; color: #333; font-size: 14px;">
                                                <strong>Chart Explanation:</strong> Left chart compares AI analysis vs web-sourced nutrition data. Right chart shows what types of information were found from web sources.
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                except Exception as e:
                                    st.info("Web analytics charts not available")
                                
                                # Quick web insights
                                st.markdown("### 💡 Web Insights")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    nutrition_info = web_info.get('nutrition', {})
                                    st.markdown(f"""
                                    <div style="background: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 15px; 
                                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                                        <h6 style="color: #FF6B6B; margin-bottom: 10px;">🔥 Nutrition</h6>
                                        <p style="margin: 2px 0; font-size: 14px;"><strong>Calories:</strong> {nutrition_info.get('calories', 'Variable')}</p>
                                        <p style="margin: 2px 0; font-size: 14px;"><strong>Protein:</strong> {nutrition_info.get('protein', 'Variable')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    cultural_info = web_info.get('cultural', {})
                                    st.markdown(f"""
                                    <div style="background: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 15px; 
                                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                                        <h6 style="color: #45B7D1; margin-bottom: 10px;">🌍 Culture</h6>
                                        <p style="margin: 2px 0; font-size: 14px;"><strong>Origin:</strong> {cultural_info.get('origin', 'Various')}</p>
                                        <p style="margin: 2px 0; font-size: 14px;"><strong>History:</strong> Rich heritage</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    health_info = web_info.get('health', {})
                                    st.markdown(f"""
                                    <div style="background: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 15px; 
                                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                                        <h6 style="color: #4ECDC4; margin-bottom: 10px;">💊 Health</h6>
                                        <p style="margin: 2px 0; font-size: 14px;"><strong>Benefits:</strong> Multiple</p>
                                        <p style="margin: 2px 0; font-size: 14px;"><strong>Allergens:</strong> Check ingredients</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Ask follow-up questions for enhanced results
                                st.markdown("### ❓ Ask Enhanced Questions")
                                user_question = st.text_input(
                                    "Ask about this food (nutrition, cooking, culture, health):",
                                    placeholder="e.g., How should I cook this? What are the health benefits?",
                                    key="comprehensive_question"
                                )
                                
                                if user_question and st.button("🚀 Ask AI", key="comprehensive_ask"):
                                    with st.spinner("🤖 AI is thinking..."):
                                        answer = agent.answer_user_questions(user_question, enhanced_result)
                                    
                                    st.markdown("### 💬 AI Response")
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                                                padding: 20px; border-radius: 15px; color: white; margin: 20px 0;">
                                        <h5 style="color: white; margin-bottom: 15px;">🤖 AI Assistant</h5>
                                        <div style="background: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 10px; color: #333;">
                                            <p style="margin: 0; line-height: 1.6;">{answer}</p>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Save to history
                            entry = {
                                "timestamp": datetime.now(),
                                "type": "standard_analysis",
                                "description": analysis_result.get('description', 'Food analysis'),
                                "analysis": analysis_result["analysis"],
                                "nutritional_data": analysis_result["nutritional_data"],
                                "context": context
                            }
                            st.session_state.history.append(entry)
                            
                            # Update daily calories
                            today = date.today().isoformat()
                            if today not in st.session_state.daily_calories:
                                st.session_state.daily_calories[today] = 0
                            st.session_state.daily_calories[today] += analysis_result["nutritional_data"]["total_calories"]
                            
                        else:
                            st.warning("Analysis had issues. Try adding more context or describing the meal manually.")
                            
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        st.info("Try uploading a clearer image or add more context description.")
                else:
                    st.error("Please upload an image and ensure models are loaded properly.")
        
        # Enhanced agent features info
        st.markdown("### 🤖 Enhanced Features Included")
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 5px solid #4ECDC4;">
            <h5 style="color: #4ECDC4; margin-bottom: 15px;">🚀 What You'll Get:</h5>
            <ul style="color: #333; line-height: 1.6;">
                <li><strong>📊 Standard Analysis:</strong> Detailed nutritional breakdown with calories, protein, carbs, and fats</li>
                <li><strong>🌐 Web Search:</strong> Comprehensive information from the web about your food</li>
                <li><strong>🌍 Cultural Background:</strong> Historical and cultural significance</li>
                <li><strong>📖 Recipe Suggestions:</strong> Cooking methods and preparation tips</li>
                <li><strong>💊 Health Information:</strong> Benefits, allergens, and dietary considerations</li>
                <li><strong>❓ AI Q&A:</strong> Ask follow-up questions about your food</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Analysis History</h3>
            <p>View your previous food analyses and track your nutrition over time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.history:
            st.info("No analysis history yet. Try analyzing a meal in the Food Analysis tab!")
        else:
            # Add complex trend charts
            st.markdown("### 🎨 Advanced Nutrition Trends")
            trend_charts = create_complex_history_trends(st.session_state.history)
            
            if trend_charts:
                trend_tab1, trend_tab2, trend_tab3 = st.tabs([
                    "🔥 Advanced Calorie Trend", "📊 Advanced Macro Trend", "🎯 Advanced Progress Trend"
                ])
                
                with trend_tab1:
                    st.markdown("#### 🔥 Advanced Calorie Intake Trend Analysis")
                    st.pyplot(trend_charts['complex_calorie_trend'])
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%); 
                                padding: 20px; border-radius: 15px; color: white; margin: 15px 0;">
                        <h5 style="color: white; margin-bottom: 10px;">🎨 Advanced Features:</h5>
                        <ul style="color: rgba(255,255,255,0.9); margin: 0; padding-left: 20px;">
                            <li>3D-style line with gradient area fill</li>
                            <li>Shadow effects for depth</li>
                            <li>Enhanced value labels with rounded boxes</li>
                            <li>Professional styling with background gradients</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with trend_tab2:
                    st.markdown("#### 📊 Advanced Macronutrient Intake Trend Analysis")
                    st.pyplot(trend_charts['complex_macro_trend'])
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                                padding: 20px; border-radius: 15px; color: white; margin: 15px 0;">
                        <h5 style="color: white; margin-bottom: 10px;">🎨 Advanced Features:</h5>
                        <ul style="color: rgba(255,255,255,0.9); margin: 0; padding-left: 20px;">
                            <li>3D-style grouped bars with shadows</li>
                            <li>Enhanced value labels with styling</li>
                            <li>Professional grid and background</li>
                            <li>Color-coded macronutrients</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with trend_tab3:
                    st.markdown("#### 🎯 Advanced Daily Target Progress Analysis")
                    st.pyplot(trend_charts['complex_progress_trend'])
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #FFD93D 0%, #FFB347 100%); 
                                padding: 20px; border-radius: 15px; color: white; margin: 15px 0;">
                        <h5 style="color: white; margin-bottom: 10px;">🎨 Advanced Features:</h5>
                        <ul style="color: rgba(255,255,255,0.9); margin: 0; padding-left: 20px;">
                            <li>Gradient bars with shadow effects</li>
                            <li>Enhanced target line styling</li>
                            <li>Progress percentage labels</li>
                            <li>Color-coded progress indicators</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("### 📋 Detailed History")
            for i, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"🍽️ {entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - {entry.get('description', 'Meal Analysis')}", expanded=False):
                    st.write(f"**Calories:** {entry.get('nutritional_data', {}).get('total_calories', 0)} kcal")
                    st.write(f"**Protein:** {entry.get('nutritional_data', {}).get('total_protein', 0):.1f}g")
                    st.write(f"**Carbs:** {entry.get('nutritional_data', {}).get('total_carbs', 0):.1f}g")
                    st.write(f"**Fats:** {entry.get('nutritional_data', {}).get('total_fats', 0):.1f}g")
                    st.write("---")
                    st.write(entry.get('analysis', 'No analysis available'))
    
    with tab3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Analytics Dashboard</h3>
            <p>Comprehensive nutrition analytics and insights from your food tracking data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.history:
            st.info("Analyze some meals first to see analytics!")
        else:
            # Summary statistics
            total_meals = len(st.session_state.history)
            total_calories = sum(entry.get('nutritional_data', {}).get('total_calories', 0) for entry in st.session_state.history)
            avg_calories = total_calories / total_meals if total_meals > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Meals", total_meals)
            with col2:
                st.metric("Total Calories", f"{total_calories} kcal")
            with col3:
                st.metric("Avg Calories/Meal", f"{avg_calories:.0f} kcal")
            
            # Weekly chart
            if st.session_state.daily_calories:
                st.markdown("### 📊 Weekly Calorie Intake")
                
                try:
                    import matplotlib.pyplot as plt
                    
                    dates = sorted(st.session_state.daily_calories.keys())[-7:]
                    cals = [st.session_state.daily_calories.get(d, 0) for d in dates]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(dates, cals, color='#667eea')
                    ax.set_ylabel('Calories')
                    ax.set_title('Daily Calorie Intake (Last 7 Days)')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating chart: {e}")
    
    # Footer
    from utils.ui import create_footer
    create_footer()

if __name__ == "__main__":
    main()
