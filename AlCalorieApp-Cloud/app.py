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

# Function to get fresh model status
def get_fresh_model_status():
    """Get fresh model status without caching"""
    if UTILS_AVAILABLE:
        from utils.models import load_models, get_model_status
        fresh_models = load_models()
        return get_model_status(fresh_models)
    return {}

# Load models - Force refresh to ensure BLIP is loaded
try:
    # Clear any cached models to force reload
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()
    models = initialize_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    models = {}

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
        # Add refresh button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 🤖 AI Models Status")
        with col2:
            if st.button("🔄", help="Refresh model status and clear cache"):
                with st.spinner("Refreshing models..."):
                    st.cache_resource.clear()
                    # Force reload
                    import time
                    time.sleep(1)
                st.success("Models refreshed!")
                st.rerun()
        
        if UTILS_AVAILABLE and "error" not in models:
            # Get fresh model status to avoid caching issues
            try:
                fresh_status = get_fresh_model_status()
                model_status = fresh_status if fresh_status else get_model_status(models)
            except:
                model_status = get_model_status(models)
            
            for model, status in model_status.items():
                status_class = "status-success" if status else "status-error"
                status_icon = "✅" if status else "❌"
                status_text = "Available" if status else "Not Available"
                # Display model names as-is (they are pre-trained, not fine-tuned)
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
        
        # Enhanced tips for ultra-comprehensive detection
        with st.expander("💡 Tips for Ultra-Enhanced Food Detection"):
            st.markdown("""
            ### 📸 **Image Quality Tips:**
            - Take clear, well-lit photos with good contrast
            - Ensure all food items are visible and not obscured
            - Use natural lighting when possible for best color accuracy
            - Avoid shadows that might hide food details
            
            ### 🍽️ **Food Arrangement Tips:**
            - Spread food items out when possible for better individual detection
            - Include all components of your meal in the frame
            - Show different angles of complex dishes if needed
            - Include garnishes, sauces, and condiments in the shot
            
            ### 📝 **Context Enhancement:**
            - Add descriptions for unusual or regional foods
            - Mention cooking methods if not visually obvious
            - Include ingredient details for complex dishes
            - Specify portion sizes if significantly different from standard
            
            ### 🎯 **Ultra-Enhanced Features:**
            - **Multi-Pass Detection:** Uses 8+ different AI prompts for comprehensive coverage
            - **YOLO Integration:** Multiple confidence levels for maximum food item detection
            - **Web Intelligence:** Searches for nutritional and cultural information
            - **Quality Assessment:** Evaluates food safety and freshness indicators
            - **Category Analysis:** Specialized detection for fruits, vegetables, proteins, etc.
            
            ### 🌐 **Enhanced Agent Benefits:**
            - Comprehensive web search for detailed food information
            - Cultural and historical context for dishes
            - Recipe suggestions and cooking tips
            - Health benefits and dietary considerations
            - Food safety and storage recommendations
            """)
        
        # Enhanced detection status
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                    padding: 15px; border-radius: 10px; color: white; margin: 10px 0;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 20px; margin-right: 10px;">⚡</span>
                <div>
                    <strong>Ultra-Enhanced Detection Active</strong><br>
                    <small>Using 8+ AI prompts + YOLO + Web Intelligence for maximum accuracy</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Single file upload for both analysis types
        uploaded_file = st.file_uploader(
            "📸 Upload a food image",
            type=['jpg', 'jpeg', 'png', 'webp'],
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
        
        # Comprehensive analysis
        st.markdown("### 🚀 Comprehensive Analysis")
        
        if st.button("🔍🧠 Run Comprehensive Analysis (Standard + Expert)", disabled=not uploaded_file, type="primary"):
            if uploaded_file and UTILS_AVAILABLE and "error" not in models:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📷 Loading image...")
                    progress_bar.progress(5)
                    
                    image = Image.open(uploaded_file)
                    
                    # Step 1: Standard Analysis
                    status_text.text("🔍 Running standard food detection...")
                    progress_bar.progress(20)
                    
                    analysis_result = analyze_food_image(image, context, models)
                    
                    # Step 2: Expert Analysis
                    status_text.text("🧠 Running expert multi-model analysis...")
                    progress_bar.progress(50)
                    
                    expert_result = None
                    try:
                        from utils.expert_food_recognition import ExpertFoodRecognitionSystem
                        expert_system = ExpertFoodRecognitionSystem(models)
                        detections = expert_system.recognize_food(image)
                        expert_summary = expert_system.get_detection_summary(detections)
                        expert_result = {"detections": detections, "summary": expert_summary}
                    except Exception as e:
                        st.warning(f"Expert system not available: {str(e)}")
                    
                    status_text.text("📊 Finalizing comprehensive analysis...")
                    progress_bar.progress(95)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Comprehensive analysis complete!")
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    if analysis_result["success"]:
                        st.success("✅ Comprehensive analysis completed!")
                        
                        # Display all results in tabs
                        tab1, tab2 = st.tabs(["🔍 Standard Analysis", "🧠 Expert Analysis"])
                        
                        with tab1:
                            display_analysis_results(analysis_result)
                        
                        with tab2:
                            if expert_result and expert_result["summary"]["success"]:
                                display_expert_results(expert_result["detections"], expert_result["summary"])
                            else:
                                st.info("Expert analysis not available or no detections found")
                        
                        # Save to history
                        history_entry = {
                            'timestamp': datetime.now(),
                            'image_name': uploaded_file.name,
                            'description': analysis_result.get('description', 'Comprehensive food analysis'),
                            'analysis': analysis_result["analysis"],
                            'nutritional_data': analysis_result["nutritional_data"],
                            'context': context,
                            'expert_detections': expert_result["detections"] if expert_result else []
                        }
                        
                        st.session_state.history.append(history_entry)
                        
                        # Update daily calories
                        today = date.today().isoformat()
                        if today not in st.session_state.daily_calories:
                            st.session_state.daily_calories[today] = 0
                        st.session_state.daily_calories[today] += analysis_result["nutritional_data"]["total_calories"]
                        
                        st.success(f"📝 Added {analysis_result['nutritional_data']['total_calories']:.0f} calories to today's total!")
                    
                    else:
                        st.error("❌ Analysis failed")
                
                except Exception as e:
                    st.error(f"Error during comprehensive analysis: {str(e)}")
            else:
                st.error("❌ AI models not available. Please check the configuration.")

def display_analysis_results(analysis_result):
    """Display standard analysis results with classification report"""
    if analysis_result["success"]:
        description = analysis_result.get('description', 'Food items detected')
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 20px; color: white; margin: 20px 0; 
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 50%; margin-right: 15px;">
                    <span style="font-size: 24px;">🎯</span>
                </div>
                <div>
                    <h3 style="color: white; margin: 0; font-size: 24px;">Standard Analysis Results</h3>
                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Multi-Model Food Detection</p>
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px;">
                <p style="font-size: 16px; margin: 0; line-height: 1.6; color: #333; font-weight: 500;">
                    <strong style="color: #667eea;">Detected Items:</strong> {description}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show detailed analysis only
        if analysis_result.get("analysis"):
            st.markdown("#### 📝 Detailed Analysis")
            with st.expander("🔍 Complete Analysis Report", expanded=True):
                st.markdown(analysis_result["analysis"])
        
        # Show detailed analysis
        if analysis_result.get("analysis"):
            st.markdown("#### 📝 Detailed Analysis")
            with st.expander("🔍 Complete Analysis Report", expanded=False):
                st.markdown(analysis_result["analysis"])

def display_expert_results(detections, summary):
    """Display expert analysis results with detailed analysis only"""
    st.markdown("### 🧠 Expert Multi-Model Analysis Results")
    
    # Filter out non-food items and generic detections
    valid_detections = []
    for detection in detections:
        # Skip generic/non-food items
        label_lower = detection.final_label.lower()
        if any(skip_word in label_lower for skip_word in ['what', 'how', 'when', 'where', 'why', 'food_item', 'unknown', 'other']):
            continue
        # Skip items that are clearly not food
        if any(non_food in label_lower for non_food in ['bottle', 'cup', 'plate', 'utensil', 'container']):
            continue
        valid_detections.append(detection)
    
    if valid_detections:
        st.success(f"✅ Expert analysis detected {len(valid_detections)} valid food items")
        
        # Show detailed detection information for valid foods only
        st.markdown("#### 🔍 Detailed Food Analysis")
        for detection in valid_detections:
            with st.expander(f"📋 {detection.final_label.replace('_', ' ').title()} - Detailed Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Detection Parameters:**")
                    st.write(f"• **Final Label:** {detection.final_label.replace('_', ' ').title()}")
                    st.write(f"• **Detection Method:** {detection.detection_method}")
                    st.write(f"• **Bounding Box:** {detection.bounding_box}")
                    st.write(f"• **Final Confidence:** {detection.confidence_score:.3f}")
                    st.write(f"• **Classifier Probability:** {detection.classifier_probability:.3f}")
                    st.write(f"• **CLIP Similarity:** {detection.clip_similarity:.3f}")
                
                with col2:
                    st.markdown("**Top Alternatives:**")
                    for i, (label, score) in enumerate(detection.top_3_alternatives, 1):
                        st.write(f"• **{i}.** {label.replace('_', ' ').title()}: {score:.3f}")
                    
                    if detection.blip_description:
                        st.markdown("**BLIP Description:**")
                        st.write(detection.blip_description)
                
                # Nutrition estimation
                nutrition = estimate_basic_nutrition(detection.final_label)
                st.markdown("**Estimated Nutrition (per 100g):**")
                col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                with col_n1:
                    st.metric("Calories", f"{nutrition['calories']} kcal")
                with col_n2:
                    st.metric("Protein", f"{nutrition['protein']}g")
                with col_n3:
                    st.metric("Carbs", f"{nutrition['carbs']}g")
                with col_n4:
                    st.metric("Fat", f"{nutrition['fat']}g")
    else:
        st.info("No valid food items detected in expert analysis")
    
    # Show detection method
    st.markdown("### 🔬 Detection Method")
    st.info("Expert Multi-Model System: YOLO + ViT-B/16 + Swin + CLIP + BLIP")
    
    # Model performance summary
    if summary.get("detection_method"):
        st.markdown("#### 🤖 Model Performance Summary")
        st.write(f"**Detection Method:** {summary['detection_method']}")
        st.write(f"**Total Detections:** {summary.get('total_detections', 0)}")
        st.write(f"**Success Rate:** {summary.get('success', False)}")

def categorize_food(food_name):
    """Categorize food into main categories"""
    food_lower = food_name.lower()
    
    # Protein sources
    if any(protein in food_lower for protein in ['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'egg', 'meat', 'steak', 'bacon', 'ham', 'turkey', 'duck', 'lamb', 'veal', 'sausage', 'hot dog', 'burger', 'meatball']):
        return "Protein"
    
    # Vegetables
    elif any(veg in food_lower for veg in ['tomato', 'potato', 'carrot', 'broccoli', 'spinach', 'lettuce', 'onion', 'garlic', 'pepper', 'cucumber', 'celery', 'mushroom', 'corn', 'pea', 'bean', 'cabbage', 'cauliflower', 'asparagus', 'zucchini', 'eggplant']):
        return "Vegetable"
    
    # Fruits
    elif any(fruit in food_lower for fruit in ['apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'raspberry', 'peach', 'pear', 'pineapple', 'mango', 'kiwi', 'lemon', 'lime', 'cherry', 'plum', 'apricot', 'fig', 'date']):
        return "Fruit"
    
    # Grains/Carbs
    elif any(grain in food_lower for grain in ['rice', 'bread', 'pasta', 'noodle', 'quinoa', 'oat', 'cereal', 'wheat', 'flour', 'pizza', 'sandwich', 'burger bun', 'tortilla', 'wrap', 'bagel', 'muffin', 'cake', 'cookie', 'biscuit']):
        return "Grain/Carb"
    
    # Dairy
    elif any(dairy in food_lower for dairy in ['cheese', 'milk', 'yogurt', 'butter', 'cream', 'ice cream', 'pudding', 'custard', 'sour cream', 'whipping cream']):
        return "Dairy"
    
    # Nuts/Seeds
    elif any(nut in food_lower for nut in ['almond', 'walnut', 'peanut', 'cashew', 'pistachio', 'pecan', 'hazelnut', 'macadamia', 'sunflower seed', 'pumpkin seed', 'chia seed', 'flax seed', 'sesame seed']):
        return "Nuts/Seeds"
    
    # Beverages
    elif any(beverage in food_lower for beverage in ['coffee', 'tea', 'juice', 'soda', 'water', 'milk', 'smoothie', 'shake', 'beer', 'wine', 'cocktail', 'lemonade']):
        return "Beverage"
    
    # Desserts/Sweets
    elif any(sweet in food_lower for sweet in ['cake', 'cookie', 'brownie', 'pie', 'donut', 'muffin', 'cupcake', 'chocolate', 'candy', 'ice cream', 'pudding', 'tiramisu', 'cheesecake', 'churro', 'baklava']):
        return "Dessert/Sweet"
    
    # Sauces/Condiments
    elif any(sauce in food_lower for sauce in ['ketchup', 'mustard', 'mayo', 'sauce', 'dressing', 'vinegar', 'oil', 'butter', 'jam', 'jelly', 'syrup', 'honey']):
        return "Sauce/Condiment"
    
    # Spices/Herbs
    elif any(spice in food_lower for spice in ['salt', 'pepper', 'garlic', 'onion', 'basil', 'oregano', 'thyme', 'rosemary', 'cinnamon', 'nutmeg', 'ginger', 'turmeric', 'paprika', 'cumin', 'coriander']):
        return "Spice/Herb"
    
    else:
        return "Other"

def estimate_basic_nutrition(food_name):
    """Estimate basic nutrition for a food item (per 100g)"""
    food_lower = food_name.lower()
    
    # Nutrition database (per 100g)
    nutrition_db = {
        # Proteins
        'chicken': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
        'beef': {'calories': 250, 'protein': 26, 'carbs': 0, 'fat': 15},
        'pork': {'calories': 242, 'protein': 27, 'carbs': 0, 'fat': 14},
        'fish': {'calories': 206, 'protein': 22, 'carbs': 0, 'fat': 12},
        'salmon': {'calories': 208, 'protein': 20, 'carbs': 0, 'fat': 13},
        'tuna': {'calories': 144, 'protein': 30, 'carbs': 0, 'fat': 1},
        'egg': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11},
        
        # Vegetables
        'tomato': {'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2},
        'potato': {'calories': 77, 'protein': 2, 'carbs': 17, 'fat': 0.1},
        'carrot': {'calories': 41, 'protein': 0.9, 'carbs': 10, 'fat': 0.2},
        'broccoli': {'calories': 34, 'protein': 2.8, 'carbs': 7, 'fat': 0.4},
        'spinach': {'calories': 23, 'protein': 2.9, 'carbs': 3.6, 'fat': 0.4},
        'lettuce': {'calories': 15, 'protein': 1.4, 'carbs': 2.9, 'fat': 0.1},
        'onion': {'calories': 40, 'protein': 1.1, 'carbs': 9, 'fat': 0.1},
        
        # Fruits
        'apple': {'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2},
        'banana': {'calories': 89, 'protein': 1.1, 'carbs': 23, 'fat': 0.3},
        'orange': {'calories': 47, 'protein': 0.9, 'carbs': 12, 'fat': 0.1},
        'grape': {'calories': 62, 'protein': 0.6, 'carbs': 16, 'fat': 0.2},
        'strawberry': {'calories': 32, 'protein': 0.7, 'carbs': 8, 'fat': 0.3},
        
        # Grains/Carbs
        'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3},
        'bread': {'calories': 265, 'protein': 9, 'carbs': 49, 'fat': 3.2},
        'pasta': {'calories': 131, 'protein': 5, 'carbs': 25, 'fat': 1.1},
        'pizza': {'calories': 266, 'protein': 11, 'carbs': 33, 'fat': 10},
        
        # Dairy
        'cheese': {'calories': 402, 'protein': 25, 'carbs': 1.3, 'fat': 33},
        'milk': {'calories': 42, 'protein': 3.4, 'carbs': 5, 'fat': 1},
        'yogurt': {'calories': 59, 'protein': 10, 'carbs': 3.6, 'fat': 0.4},
        
        # Nuts/Seeds
        'almond': {'calories': 579, 'protein': 21, 'carbs': 22, 'fat': 50},
        'walnut': {'calories': 654, 'protein': 15, 'carbs': 14, 'fat': 65},
        'peanut': {'calories': 567, 'protein': 26, 'carbs': 16, 'fat': 49},
        
        # Desserts
        'cake': {'calories': 257, 'protein': 5, 'carbs': 45, 'fat': 6},
        'cookie': {'calories': 502, 'protein': 6, 'carbs': 65, 'fat': 24},
        'ice cream': {'calories': 207, 'protein': 4, 'carbs': 24, 'fat': 11},
        'chocolate': {'calories': 545, 'protein': 4.9, 'carbs': 61, 'fat': 31},
    }
    
    # Try to find exact match
    for key, nutrition in nutrition_db.items():
        if key in food_lower:
            return nutrition
    
    # Try partial matches
    for key, nutrition in nutrition_db.items():
        if any(word in food_lower for word in key.split()):
            return nutrition
    
    # Default nutrition for unknown foods
    return {'calories': 100, 'protein': 5, 'carbs': 15, 'fat': 2}

def create_complex_history_trends(history):
    """Create complex trend charts from history data"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime, timedelta
        
        if not history:
            return None
        
        charts = {}
        
        # Prepare data
        dates = []
        calories = []
        proteins = []
        carbs = []
        fats = []
        
        for entry in history:
            if 'timestamp' in entry and 'nutritional_data' in entry:
                dates.append(entry['timestamp'])
                calories.append(entry['nutritional_data'].get('total_calories', 0))
                proteins.append(entry['nutritional_data'].get('total_protein', 0))
                carbs.append(entry['nutritional_data'].get('total_carbs', 0))
                fats.append(entry['nutritional_data'].get('total_fats', 0))
        
        if not dates:
            return None
        
        # Convert to datetime if needed
        if isinstance(dates[0], str):
            dates = [datetime.fromisoformat(d) for d in dates]
        
        # Sort by date
        sorted_data = sorted(zip(dates, calories, proteins, carbs, fats))
        dates, calories, proteins, carbs, fats = zip(*sorted_data)
        
        # 1. Complex Calorie Trend
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(dates, calories, 'o-', linewidth=3, markersize=8, color='#FF6B6B', alpha=0.8)
        ax1.fill_between(dates, calories, alpha=0.3, color='#FF6B6B')
        ax1.set_title('🔥 Advanced Calorie Trend Analysis', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Calories', fontsize=12)
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        charts['complex_calorie_trend'] = fig1
        
        # 2. Macro Trend
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(dates, proteins, 'o-', label='Protein', linewidth=2, markersize=6, color='#4ECDC4')
        ax2.plot(dates, carbs, 's-', label='Carbs', linewidth=2, markersize=6, color='#45B7D1')
        ax2.plot(dates, fats, '^-', label='Fats', linewidth=2, markersize=6, color='#FFD93D')
        ax2.set_title('📊 Advanced Macro Trend Analysis', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Grams', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        charts['complex_macro_trend'] = fig2
        
        # 3. Progress Trend
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        cumulative_calories = np.cumsum(calories)
        ax3.plot(dates, cumulative_calories, 'o-', linewidth=3, markersize=8, color='#9B59B6', alpha=0.8)
        ax3.fill_between(dates, cumulative_calories, alpha=0.3, color='#9B59B6')
        ax3.set_title('🎯 Advanced Progress Trend Analysis', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Cumulative Calories', fontsize=12)
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        charts['complex_progress_trend'] = fig3
        
        return charts
        
    except Exception as e:
        st.error(f"Error creating trend charts: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="header-card">
        <h1>🍱 AI Calorie Tracker</h1>
        <p>Advanced AI-powered food analysis and nutrition tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model status
        st.markdown("#### 🤖 AI Model Status")
        model_status = get_fresh_model_status()
        
        for model_name, is_available in model_status.items():
            if is_available:
                st.markdown(f"✅ {model_name}")
            else:
                st.markdown(f"❌ {model_name}")
        
        # Calorie target
        st.markdown("#### 🎯 Daily Calorie Target")
        calorie_target = st.number_input(
            "Target Calories", 
            min_value=1000, 
            max_value=5000, 
            value=st.session_state.calorie_target,
            step=100
        )
        st.session_state.calorie_target = calorie_target
        
        # Today's progress
        today = date.today().isoformat()
        today_calories = st.session_state.daily_calories.get(today, 0)
        progress = min(today_calories / calorie_target, 1.0)
        
        st.markdown("#### 📊 Today's Progress")
        st.progress(progress)
        st.metric("Calories", f"{today_calories:.0f} / {calorie_target}")
        
        # Clear data button
        if st.button("🗑️ Clear All Data"):
            st.session_state.history = []
            st.session_state.daily_calories = {}
            st.success("Data cleared!")
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Food Analysis", "📊 History", "📈 Analytics"])
    
    with tab1:
        st.markdown("""
        <div class="metric-card">
            <h3>📸 Upload Food Image</h3>
            <p>Upload a clear image of your food for AI-powered analysis and nutrition tracking.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a food image...",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload a clear image of your food"
        )
        
        # Context input
        context = st.text_area(
            "Additional Context (Optional)",
            placeholder="Describe the food, portion size, or any special preparation methods...",
            help="Provide additional context to improve analysis accuracy"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Food Image", use_column_width=True)
        
        # Comprehensive analysis
        st.markdown("### 🚀 Comprehensive Analysis")
        
        if st.button("🔍🧠 Run Comprehensive Analysis (Standard + Expert)", disabled=not uploaded_file, type="primary"):
            if uploaded_file and UTILS_AVAILABLE and "error" not in models:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📷 Loading image...")
                    progress_bar.progress(5)
                    
                    image = Image.open(uploaded_file)
                    
                    # Step 1: Standard Analysis
                    status_text.text("🔍 Running standard food detection...")
                    progress_bar.progress(20)
                    
                    analysis_result = analyze_food_image(image, context, models)
                    
                    # Step 2: Expert Analysis
                    status_text.text("🧠 Running expert multi-model analysis...")
                    progress_bar.progress(50)
                    
                    expert_result = None
                    try:
                        from utils.expert_food_recognition import ExpertFoodRecognitionSystem
                        expert_system = ExpertFoodRecognitionSystem(models)
                        detections = expert_system.recognize_food(image)
                        expert_summary = expert_system.get_detection_summary(detections)
                        expert_result = {"detections": detections, "summary": expert_summary}
                    except Exception as e:
                        st.warning(f"Expert system not available: {str(e)}")
                    
                    status_text.text("📊 Finalizing comprehensive analysis...")
                    progress_bar.progress(95)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Comprehensive analysis complete!")
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    if analysis_result["success"]:
                        st.success("✅ Comprehensive analysis completed!")
                        
                        # Display all results in tabs
                        tab1, tab2 = st.tabs(["🔍 Standard Analysis", "🧠 Expert Analysis"])
                        
                        with tab1:
                            display_analysis_results(analysis_result)
                        
                        with tab2:
                            if expert_result and expert_result["summary"]["success"]:
                                display_expert_results(expert_result["detections"], expert_result["summary"])
                            else:
                                st.info("Expert analysis not available or no detections found")
                        
                        # Save to history
                        history_entry = {
                            'timestamp': datetime.now(),
                            'image_name': uploaded_file.name,
                            'description': analysis_result.get('description', 'Comprehensive food analysis'),
                            'analysis': analysis_result["analysis"],
                            'nutritional_data': analysis_result["nutritional_data"],
                            'context': context,
                            'expert_detections': expert_result["detections"] if expert_result else []
                        }
                        
                        st.session_state.history.append(history_entry)
                        
                        # Update daily calories
                        today = date.today().isoformat()
                        if today not in st.session_state.daily_calories:
                            st.session_state.daily_calories[today] = 0
                        st.session_state.daily_calories[today] += analysis_result["nutritional_data"]["total_calories"]
                        
                        st.success(f"📝 Added {analysis_result['nutritional_data']['total_calories']:.0f} calories to today's total!")
                    
                    else:
                        st.error("❌ Analysis failed")
                
                except Exception as e:
                    st.error(f"Error during comprehensive analysis: {str(e)}")
            else:
                st.error("❌ AI models not available. Please check the configuration.")
    
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
                    if 'complex_calorie_trend' in trend_charts:
                        st.pyplot(trend_charts['complex_calorie_trend'])
                
                with trend_tab2:
                    if 'complex_macro_trend' in trend_charts:
                        st.pyplot(trend_charts['complex_macro_trend'])
                
                with trend_tab3:
                    if 'complex_progress_trend' in trend_charts:
                        st.pyplot(trend_charts['complex_progress_trend'])
            
            st.markdown("### 📋 Detailed History")
            for i, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"🍽️ {entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - {entry.get('description', 'Meal Analysis')}", expanded=False):
                    st.write(f"**Calories:** {entry.get('nutritional_data', {}).get('total_calories', 0)} kcal")
                    st.write(f"**Protein:** {entry.get('nutritional_data', {}).get('total_protein', 0)} g")
                    st.write(f"**Carbs:** {entry.get('nutritional_data', {}).get('total_carbs', 0)} g")
                    st.write(f"**Fats:** {entry.get('nutritional_data', {}).get('total_fats', 0)} g")
                    if entry.get('analysis'):
                        st.write("**Analysis:**")
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
                st.metric("Total Calories", f"{total_calories:.0f}")
            with col3:
                st.metric("Avg Calories/Meal", f"{avg_calories:.0f}")
    
    # Footer
    st.markdown("""
    <div class="modern-footer">
        <h3>🍱 AI Calorie Tracker</h3>
        <p>Powered by advanced AI models for accurate food analysis and nutrition tracking</p>
        <p>Built with Streamlit, PyTorch, and Transformers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
