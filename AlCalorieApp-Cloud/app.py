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
</style>
""", unsafe_allow_html=True)

def create_simple_chart(nutrition_data):
    """Create a simple nutrition chart"""
    try:
        import matplotlib.pyplot as plt
        
        categories = ['Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)']
        values = [
            nutrition_data.get('total_calories', 0),
            nutrition_data.get('total_protein', 0),
            nutrition_data.get('total_carbs', 0),
            nutrition_data.get('total_fats', 0)
        ]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Nutritional Breakdown', fontsize=14, fontweight='bold')
        ax.set_ylabel('Amount')
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

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
    tab1, tab2, tab3 = st.tabs(["📷 Food Analysis", "📊 History", "📈 Analytics"])
    
    with tab1:
        st.markdown("""
        <div class="metric-card">
            <h3>📷 Food Analysis</h3>
            <p>Upload a food image for AI-powered calorie and nutrition analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tips
        with st.expander("💡 Tips for Better Results"):
            st.markdown("""
            - 📸 Take clear photos in good lighting
            - 🍽️ Include all food items in the frame
            - 📝 Add context description if needed
            - 🔄 Try different angles if detection is incomplete
            """)
        
        # Upload section
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of your food"
        )
        
        # Context input
        context = st.text_area(
            "Additional Context (Optional)", 
            placeholder="Describe the meal if needed (e.g., 'chicken curry with rice')", 
            height=80
        )
        
        # Analyze button
        if st.button("🔍 Analyze Food", disabled=not uploaded_file):
            if uploaded_file and UTILS_AVAILABLE and "error" not in models:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📷 Loading image...")
                    progress_bar.progress(20)
                    
                    image = Image.open(uploaded_file)
                    
                    # Display image
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    status_text.text("🔍 Detecting food items...")
                    progress_bar.progress(50)
                    
                    # Analyze the food
                    analysis_result = analyze_food_image(image, context, models)
                    
                    status_text.text("📊 Processing results...")
                    progress_bar.progress(80)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    if analysis_result["success"]:
                        st.success("✅ Food analysis completed!")
                        
                        # Show detected food items
                        st.markdown(f"""
                        <div class="analysis-card">
                            <h4>🍽️ Detected Food Items</h4>
                            <p><strong>Foods found:</strong> {analysis_result.get('description', 'Food items detected')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show nutrition summary
                        nutrition = analysis_result["nutritional_data"]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Calories", f"{nutrition['total_calories']} kcal")
                        with col2:
                            st.metric("Protein", f"{nutrition['total_protein']}g")
                        with col3:
                            st.metric("Carbs", f"{nutrition['total_carbs']}g")
                        with col4:
                            st.metric("Fats", f"{nutrition['total_fats']}g")
                        
                        # Show chart
                        chart = create_simple_chart(nutrition)
                        if chart:
                            st.pyplot(chart)
                        
                        # Show analysis
                        st.markdown("""
                        <div class="analysis-card">
                            <h4>📝 Detailed Analysis</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(analysis_result["analysis"])
                        
                        # Save to history
                        entry = {
                            "timestamp": datetime.now(),
                            "type": "image",
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
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🍱 AI Calorie Tracker | Built with Streamlit and AI</p>
        <p>For accurate nutrition tracking, consider consulting with a registered dietitian.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
