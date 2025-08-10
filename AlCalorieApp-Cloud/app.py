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
                            
                            # Show chart with better styling
                            st.markdown("### 📈 Nutrition Chart")
                            chart = create_simple_chart(nutrition)
                            if chart:
                                st.pyplot(chart)
                            
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
                                
                                # Quick web insights
                                st.markdown("### 💡 Web Insights")
                                web_info = enhanced_result['web_information']
                                
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
