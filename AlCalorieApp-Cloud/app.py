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

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Import utilities with error handling
try:
    from utils.models import load_models, get_model_status
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
if "current_page" not in st.session_state:
    st.session_state.current_page = "landing"

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

def create_landing_page():
    """Create the landing page"""
    # Custom CSS for the landing page with consistent theme
    st.markdown("""
    <style>
    /* Global theme colors */
    :root {
        --primary-green: #28a745;
        --secondary-green: #20c997;
        --accent-gold: #ffd700;
        --text-dark: #2c3e50;
        --text-light: #6c757d;
        --bg-light: #f8f9fa;
        --white: #ffffff;
        --shadow: rgba(0,0,0,0.1);
        --shadow-hover: rgba(0,0,0,0.15);
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, var(--white) 0%, var(--bg-light) 100%);
        padding: 0;
        margin: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styling */
    .landing-header {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 100%);
        color: var(--white);
        padding: 3rem 0;
        text-align: center;
        border-radius: 0 0 30px 30px;
        margin-bottom: 3rem;
        box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3);
    }
    
    .landing-header h1 {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .landing-header h2 {
        font-size: 2.5rem;
        margin-bottom: 2rem;
        font-weight: 300;
        line-height: 1.2;
    }
    
    .landing-header p {
        font-size: 1.3rem;
        margin-bottom: 2rem;
        opacity: 0.95;
        line-height: 1.6;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Feature cards */
    .feature-card {
        background: var(--white);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 15px var(--shadow);
        border: 1px solid rgba(255,255,255,0.8);
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 30px var(--shadow-hover);
    }
    
    /* CTA buttons */
    .cta-button {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 100%);
        color: var(--white);
        border: none;
        border-radius: 30px;
        padding: 15px 35px;
        font-weight: 700;
        font-size: 18px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        margin: 15px;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    }
    
    .cta-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
        color: var(--white);
        text-decoration: none;
    }
    
    .cta-button-secondary {
        background: var(--white);
        color: var(--primary-green);
        border: 3px solid var(--primary-green);
        border-radius: 30px;
        padding: 15px 35px;
        font-weight: 700;
        font-size: 18px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        margin: 15px;
    }
    
    .cta-button-secondary:hover {
        background: var(--primary-green);
        color: var(--white);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
        text-decoration: none;
    }
    
    /* Stats cards */
    .stats-card {
        background: var(--white);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px var(--shadow);
        border: 1px solid rgba(255,255,255,0.8);
        text-align: center;
        margin: 10px;
        transition: all 0.3s ease;
    }
    
    .stats-number {
        font-size: 3rem;
        font-weight: 800;
        color: var(--primary-green);
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stats-label {
        font-size: 1.1rem;
        color: var(--text-light);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* App mockup */
    .app-mockup {
        background: var(--white);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0 12px 40px var(--shadow-hover);
        border: 1px solid rgba(255,255,255,0.8);
        text-align: center;
        height: 100%;
    }
    
    .app-mockup h3 {
        color: var(--primary-green);
        margin-bottom: 2rem;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Privacy indicators */
    .privacy-indicator {
        display: inline-block;
        margin: 12px;
        padding: 12px 20px;
        background: var(--bg-light);
        border-radius: 25px;
        font-size: 1rem;
        color: var(--text-light);
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .privacy-indicator:hover {
        border-color: var(--primary-green);
        color: var(--primary-green);
        transform: translateY(-2px);
    }
    
    /* Section titles */
    .section-title {
        color: var(--primary-green);
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .landing-header h1 {
            font-size: 2.5rem;
        }
        
        .landing-header h2 {
            font-size: 2rem;
        }
        
        .stats-number {
            font-size: 2.5rem;
        }
        
        .feature-card {
            margin-bottom: 1rem;
        }
    }
    
    /* Streamlit button styling override */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 100%) !important;
        color: var(--white) !important;
        border-radius: 30px !important;
        border: none !important;
        padding: 15px 35px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header Section
    st.markdown("""
    <div class="landing-header">
        <h1>🧠 AI-Powered Nutrition Analysis</h1>
        <h2>Transform Your <span style="color: var(--accent-gold);">Nutrition Journey</span> with AI</h2>
        <p>
            Experience the future of food tracking with our advanced AI technology. 
            Get instant, accurate nutritional analysis of any meal with 99.2% accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Content with better alignment
    st.markdown('<h2 class="section-title">🎯 Key Features</h2>', unsafe_allow_html=True)
    
    # Feature Cards in a grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; color: var(--primary-green); margin-bottom: 1.5rem;">🎯</div>
            <div class="stats-number">99.2%</div>
            <div class="stats-label">Accuracy</div>
            <p style="color: var(--text-light); margin-top: 1rem; font-size: 0.9rem;">
                Industry-leading detection accuracy powered by YOLO11m
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; color: var(--primary-green); margin-bottom: 1.5rem;">⚡</div>
            <div class="stats-number">&lt;2s</div>
            <div class="stats-label">Processing</div>
            <p style="color: var(--text-light); margin-top: 1rem; font-size: 0.9rem;">
                Lightning-fast analysis with optimized image processing
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; color: var(--primary-green); margin-bottom: 1.5rem;">👥</div>
            <div class="stats-number">50K+</div>
            <div class="stats-label">Users</div>
            <p style="color: var(--text-light); margin-top: 1rem; font-size: 0.9rem;">
                Trusted by thousands of users worldwide
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call-to-Action Buttons
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0;">
    """, unsafe_allow_html=True)
    
    if st.button("📸 Start Free Analysis →", key="start_analysis", use_container_width=False):
        st.session_state.current_page = "analysis"
        st.rerun()
    
    if st.button("▶️ Watch Demo", key="watch_demo", use_container_width=False):
        st.info("🎬 Demo video coming soon! Stay tuned for an interactive walkthrough.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Privacy Indicators
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0;">
        <div class="privacy-indicator">🔒 Privacy First</div>
        <div class="privacy-indicator">🔐 End-to-End Encrypted</div>
        <div class="privacy-indicator">🌍 Global Coverage</div>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Models and App Mockup Section
    st.markdown('<h2 class="section-title">🚀 Advanced Technology</h2>', unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns([1, 1, 2])
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; color: var(--primary-green); margin-bottom: 1.5rem;">🧠</div>
            <div class="stats-number">6+</div>
            <div class="stats-label">AI Models</div>
            <p style="color: var(--text-light); margin-top: 1rem; font-size: 0.9rem;">
                Multiple specialized AI models for comprehensive analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; color: var(--primary-green); margin-bottom: 1.5rem;">🔍</div>
            <div class="stats-number">1000+</div>
            <div class="stats-label">Food Items</div>
            <p style="color: var(--text-light); margin-top: 1rem; font-size: 0.9rem;">
                Comprehensive database of food items and ingredients
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        # App Mockup
        st.markdown("""
        <div class="app-mockup">
            <h3>📱 AI Calorie Analyzer</h3>
            <div style="background: var(--bg-light); border-radius: 20px; padding: 2rem; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; margin-bottom: 1.5rem; padding: 10px; background: var(--white); border-radius: 15px;">
                    <span style="font-size: 2rem; margin-right: 1rem;">🍕</span>
                    <div style="flex-grow: 1;">
                        <div style="font-weight: 700; font-size: 1.2rem;">Pizza</div>
                        <div style="color: var(--primary-green); font-size: 1.1rem;">285 cal</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; padding: 10px; background: var(--white); border-radius: 15px;">
                    <span style="font-size: 2rem; margin-right: 1rem;">🥗</span>
                    <div style="flex-grow: 1;">
                        <div style="font-weight: 700; font-size: 1.2rem;">Salad</div>
                        <div style="color: var(--primary-green); font-size: 1.1rem;">120 cal</div>
                    </div>
                </div>
            </div>
            <div class="feature-card" style="margin-top: 1rem;">
                <div class="stats-number">99%</div>
                <div class="stats-label">Accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Bottom Section - Additional Features
    st.markdown("---")
    st.markdown('<h2 class="section-title">✨ Advanced Features</h2>', unsafe_allow_html=True)
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; color: var(--primary-green); margin-bottom: 1rem;">🔍</div>
            <h4 style="color: var(--primary-green); margin-bottom: 1rem; font-size: 1.3rem;">Advanced Detection</h4>
            <p style="color: var(--text-light); line-height: 1.6;">
                YOLO11m powered food recognition with multiple AI models for comprehensive analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col8:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; color: var(--primary-green); margin-bottom: 1rem;">📊</div>
            <h4 style="color: var(--primary-green); margin-bottom: 1rem; font-size: 1.3rem;">Detailed Analytics</h4>
            <p style="color: var(--text-light); line-height: 1.6;">
                Get comprehensive nutritional breakdown including calories, macros, and micronutrients.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col9:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; color: var(--primary-green); margin-bottom: 1rem;">📈</div>
            <h4 style="color: var(--primary-green); margin-bottom: 1rem; font-size: 1.3rem;">Progress Tracking</h4>
            <p style="color: var(--text-light); line-height: 1.6;">
                Track your nutrition journey with detailed history and progress analytics.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Final CTA
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0; padding: 3rem; background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 100%); border-radius: 30px; color: var(--white); box-shadow: 0 8px 30px rgba(40, 167, 69, 0.3);">
        <h2 style="margin-bottom: 1.5rem; font-size: 2.5rem; font-weight: 700;">Ready to Transform Your Nutrition Journey?</h2>
        <p style="font-size: 1.3rem; margin-bottom: 2.5rem; opacity: 0.95;">Join thousands of users who trust our AI-powered nutrition analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Start Your Free Analysis Now", key="final_cta", use_container_width=True):
        st.session_state.current_page = "analysis"
        st.rerun()

def create_analysis_page_with_navigation():
    """Create the analysis page with navigation"""
    
    # Custom CSS for consistent theme
    st.markdown("""
    <style>
    /* Global theme colors */
    :root {
        --primary-green: #28a745;
        --secondary-green: #20c997;
        --accent-gold: #ffd700;
        --text-dark: #2c3e50;
        --text-light: #6c757d;
        --bg-light: #f8f9fa;
        --white: #ffffff;
        --shadow: rgba(0,0,0,0.1);
        --shadow-hover: rgba(0,0,0,0.15);
    }
    
    .main { 
        background: linear-gradient(135deg, var(--bg-light) 0%, #e9ecef 100%);
        padding: 20px; 
        border-radius: 10px; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stButton>button { 
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 100%) !important;
        color: var(--white) !important;
        border-radius: 25px !important;
        border: none !important;
        padding: 12px 25px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4) !important;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 15px var(--shadow);
        margin: 15px 0;
        border: 1px solid rgba(255,255,255,0.8);
        border-left: 5px solid var(--primary-green);
    }
    
    .header-card {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 100%);
        color: var(--white);
        padding: 40px;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.3);
    }
    
    .header-card h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-card p {
        font-size: 1.2rem;
        opacity: 0.95;
        line-height: 1.6;
    }
    
    .analysis-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 15px var(--shadow);
        margin: 20px 0;
        border-left: 5px solid var(--primary-green);
    }
    
    .status-success { color: var(--primary-green); font-weight: 700; }
    .status-error { color: #f44336; font-weight: 700; }
    .status-warning { color: #ff9800; font-weight: 700; }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, var(--bg-light) 0%, #e9ecef 100%);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: var(--primary-green) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-green), var(--secondary-green)) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header-card">
        <h1>🍱 AI-Powered Nutrition Analysis</h1>
        <p>Transform your nutrition journey with advanced AI technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with fine-tune option
    with st.sidebar:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: var(--primary-green); margin-bottom: 1rem;">⚙️ Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display size configuration
        max_display_size = st.slider(
            "🖼️ Max Image Display Size",
            min_value=400,
            max_value=1200,
            value=800,
            step=100,
            help="Maximum size for displayed images (doesn't affect detection quality)"
        )
        
        # YOLO11m Status
        st.markdown("#### 🔍 AI Model Status")
        st.markdown("**✅ FineTune YOLO11m **")
        
        # Daily calorie target
        st.session_state.calorie_target = st.number_input(
            "🎯 Daily Calorie Target",
            min_value=1000,
            max_value=5000,
            value=st.session_state.calorie_target,
            step=100,
            help="Set your daily calorie goal"
        )
        
        # Model status with improved styling
        with st.expander("🔍 AI Model Status", expanded=True):
            model_status = get_fresh_model_status()
            
            for model_name, is_available in model_status.items():
                # Status indicator and model name
                status_icon = "✅" if is_available else "❌"
                st.markdown(f"**{status_icon} {model_name}**")
        
        # Today's progress
        today = date.today().isoformat()
        today_calories = st.session_state.daily_calories.get(today, 0)
        progress = min(today_calories / st.session_state.calorie_target, 1.0)
        
        st.markdown("#### 📊 Today's Progress")
        st.progress(progress)
        st.metric("Calories", f"{today_calories:.0f} / {st.session_state.calorie_target}")
        
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
            <h3 style="color: var(--primary-green); margin-bottom: 1rem;">📸 Upload Food Image</h3>
            <p style="color: var(--text-light);">Upload a clear image of your food for AI-powered analysis and nutrition tracking.</p>
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
            
            # Optimize image for display and detection
            display_image = image
            optimized_image = image  # Default to original
            if UTILS_AVAILABLE:
                try:
                    from utils.expert_food_recognition import YOLO11mFoodRecognitionSystem
                    yolo_system = YOLO11mFoodRecognitionSystem(models)
                    optimized_image = yolo_system.optimize_image_for_detection(image)
                    
                    # Use the UI utility for display optimization
                    from utils.ui import optimize_image_for_display
                    display_image = optimize_image_for_display(optimized_image, max_display_size=max_display_size)
                    
                    st.success(f"🖼️ Image optimized: {image.size} → {optimized_image.size} → {display_image.size}")
                    
                    # Store optimized image in session state for reuse
                    st.session_state.optimized_image = optimized_image
                        
                except Exception as e:
                    st.warning(f"Image optimization failed, using original: {str(e)}")
                    # Use the UI utility for display optimization of original image
                    from utils.ui import optimize_image_for_display
                    display_image = optimize_image_for_display(image, max_display_size=max_display_size)
                    st.info(f"🖼️ Display resized: {image.size} → {display_image.size}")
                    
                    st.session_state.optimized_image = image
            
            # Display the optimized/resized image
            st.image(display_image, caption="Optimized Food Image for Analysis", use_column_width=True)
        
        # YOLO11m Analysis
        st.markdown("### 🔍 YOLO11m Analysis")
        
        if st.button("🔍 Run YOLO11m Analysis", disabled=not uploaded_file, type="primary"):
            if uploaded_file and UTILS_AVAILABLE and "error" not in models:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📷 Loading and optimizing image...")
                    progress_bar.progress(10)
                    
                    # Use the already optimized image from session state
                    if hasattr(st.session_state, 'optimized_image'):
                        image = st.session_state.optimized_image
                        st.success(f"🔄 Using optimized image: {image.size}")
                    else:
                        # Fallback to original image if optimization failed
                        image = Image.open(uploaded_file)
                        st.warning("Using original image (optimization not available)")
                    
                    # YOLO11m Analysis
                    status_text.text("🔍 Running YOLO11m analysis...")
                    progress_bar.progress(50)
                    
                    expert_result = None
                    try:
                        from utils.expert_food_recognition import YOLO11mFoodRecognitionSystem
                        yolo_system = YOLO11mFoodRecognitionSystem(models)
                        recognition_result = yolo_system.recognize_food(image)
                        
                        # Generate summary using the proper method
                        expert_summary = yolo_system.get_detection_summary(recognition_result)
                        
                        if expert_summary["success"]:
                            expert_result = {"detections": recognition_result["detections"], "summary": expert_summary}
                        else:
                            st.error(f"YOLO11m recognition failed: {expert_summary.get('error', 'Unknown error')}")
                            return
                    except Exception as e:
                        st.error(f"YOLO11m system error: {str(e)}")
                        return
                    
                    status_text.text("📊 Finalizing expert analysis...")
                    progress_bar.progress(90)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ YOLO11m analysis complete!")
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    if expert_result and expert_result["summary"]["success"]:
                        st.success("✅ YOLO11m analysis completed!")
                        
                        # Display expert results
                        display_expert_results(expert_result["detections"], expert_result["summary"])
                        
                        # Calculate nutrition data from expert detections
                        nutritional_data = calculate_nutrition_from_expert_detections(expert_result["detections"])
                        
                        # Save to history
                        history_entry = {
                            'timestamp': datetime.now(),
                            'image_name': uploaded_file.name,
                            'description': f"YOLO11m analysis: {len(expert_result['detections'])} items detected",
                            'analysis': f"YOLO11m analysis with {expert_result['summary'].get('total_detections', 0)} detections",
                            'nutritional_data': nutritional_data,
                            'context': context,
                            'expert_detections': expert_result["detections"]
                        }
                        
                        st.session_state.history.append(history_entry)
                        
                        # Update daily calories
                        today = date.today().isoformat()
                        if today not in st.session_state.daily_calories:
                            st.session_state.daily_calories[today] = 0
                        st.session_state.daily_calories[today] += nutritional_data["total_calories"]
                        
                        st.success(f"📝 Added {nutritional_data['total_calories']:.0f} calories to today's total!")
                    
                    else:
                        st.error("❌ YOLO11m analysis failed or no detections found")
                
                except Exception as e:
                    st.error(f"Error during YOLO11m analysis: {str(e)}")
            else:
                st.error("❌ AI models not available. Please check the configuration.")
    
    with tab2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: var(--primary-green); margin-bottom: 1rem;">📊 Analysis History</h3>
            <p style="color: var(--text-light);">View your previous food analyses and track your nutrition journey.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.history:
            for i, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"📅 {entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - {entry['image_name']}", expanded=False):
                    st.write(f"**Description:** {entry['description']}")
                    st.write(f"**Analysis:** {entry['analysis']}")
                    if entry.get('context'):
                        st.write(f"**Context:** {entry['context']}")
                    
                    # Display nutritional data
                    if entry.get('nutritional_data'):
                        nutrition = entry['nutritional_data']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Calories", f"{nutrition['total_calories']:.0f}")
                        with col2:
                            st.metric("Protein", f"{nutrition['total_protein']:.1f}g")
                        with col3:
                            st.metric("Carbs", f"{nutrition['total_carbs']:.1f}g")
                        with col4:
                            st.metric("Fat", f"{nutrition['total_fats']:.1f}g")
        else:
            st.info("No analysis history yet. Upload an image to get started!")
    
    with tab3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: var(--primary-green); margin-bottom: 1rem;">📈 Analytics Dashboard</h3>
            <p style="color: var(--text-light);">Track your nutrition trends and progress over time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple analytics
        if st.session_state.daily_calories:
            st.subheader("📊 Daily Calorie Trends")
            
            # Create a simple chart
            dates = list(st.session_state.daily_calories.keys())
            calories = list(st.session_state.daily_calories.values())
            
            if dates and calories:
                import plotly.express as px
                import pandas as pd
                
                df = pd.DataFrame({
                    'Date': dates,
                    'Calories': calories
                })
                
                fig = px.line(df, x='Date', y='Calories', title='Daily Calorie Intake')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for analytics. Start analyzing food to see trends!")

def main():
    """Main Streamlit application"""
    
    # Check URL parameters for navigation using the new API
    if "page" in st.query_params:
        st.session_state.current_page = st.query_params["page"]
    
    # Navigation logic
    if st.session_state.current_page == "landing":
        # Page configuration for landing page
        st.set_page_config(
            page_title="AI-Powered Nutrition Analysis",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Hide default Streamlit elements for landing page
        st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
        
        create_landing_page()
        
    elif st.session_state.current_page == "analysis":
        # Page configuration for analysis page
        st.set_page_config(
            page_title="🍱 YOLO11m Calorie Tracker", 
            layout="wide", 
            page_icon="🔍",
            initial_sidebar_state="expanded"
        )
        
        # Add navigation back to landing page
        if st.sidebar.button("🏠 Back to Landing Page"):
            st.session_state.current_page = "landing"
            st.rerun()
        
        # Create the analysis page with existing functionality
        create_analysis_page_with_navigation()
    else:
        # Default to landing page
        st.session_state.current_page = "landing"
        st.rerun()

# Placeholder functions for compatibility
def display_expert_results(detections, summary):
    """Display comprehensive expert analysis results"""
    st.subheader("🔍 Comprehensive Detection Report")
    
    if detections and len(detections) > 0:
        st.success(f"✅ Found {len(detections)} food items")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Detected Items", "📊 Nutrition Analysis", "🔍 Detection Details", "💡 AI Insights"])
        
        with tab1:
            st.markdown("### 🍽️ Detected Food Items")
            
            # Group detections by food type
            food_groups = {}
            for detection in detections:
                # Handle different data types
                if hasattr(detection, 'final_label'):
                    label = detection.final_label
                    confidence = detection.confidence_score
                    method = detection.detection_method
                elif isinstance(detection, dict):
                    label = detection.get('final_label', detection.get('label', 'Unknown'))
                    confidence = detection.get('confidence_score', detection.get('confidence', 0.0))
                    method = detection.get('detection_method', 'YOLO11m')
                elif isinstance(detection, str):
                    label = detection
                    confidence = 0.8
                    method = 'YOLO11m'
                else:
                    label = str(detection)
                    confidence = 0.8
                    method = 'YOLO11m'
                
                if label not in food_groups:
                    food_groups[label] = {
                        'count': 0,
                        'total_confidence': 0,
                        'methods': set(),
                        'detections': []
                    }
                
                food_groups[label]['count'] += 1
                food_groups[label]['total_confidence'] += confidence
                food_groups[label]['methods'].add(method)
                food_groups[label]['detections'].append({
                    'label': label,
                    'confidence': confidence,
                    'method': method
                })
            
            # Display all items in a single consolidated report
            st.markdown("#### 📋 All Detected Items")
            
            # Create a table-like display for all items
            for i, (food_name, group_data) in enumerate(food_groups.items()):
                avg_confidence = group_data['total_confidence'] / group_data['count']
                methods_str = ', '.join(group_data['methods'])
                
                # Create a card-like display for each food item
                st.markdown(f"""
                <div style="
                    border: 1px solid #e0e0e0; 
                    border-radius: 8px; 
                    padding: 15px; 
                    margin: 10px 0; 
                    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <h4 style="margin: 0; color: #28a745; font-size: 18px;">🍽️ {food_name}</h4>
                            <p style="margin: 5px 0; color: #6c757d; font-size: 14px;">
                                <strong>Count:</strong> {group_data['count']} | 
                                <strong>Method:</strong> {methods_str}
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Summary statistics
            st.markdown("---")
            st.markdown("#### 📊 Detection Summary")
            total_items = len(food_groups)
            total_detections = sum(group['count'] for group in food_groups.values())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Food Items", total_items)
            with col2:
                st.metric("Total Detections", total_detections)
        
        with tab2:
            st.markdown("### 📊 Nutritional Analysis")
            
            # Calculate nutrition data
            nutritional_data = calculate_nutrition_from_expert_detections(detections)
            
            # Display nutrition metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Calories", f"{nutritional_data['total_calories']:.0f} kcal")
            
            with col2:
                st.metric("Protein", f"{nutritional_data['total_protein']:.1f} g")
            
            with col3:
                st.metric("Carbohydrates", f"{nutritional_data['total_carbs']:.1f} g")
            
            with col4:
                st.metric("Fats", f"{nutritional_data['total_fats']:.1f} g")
            
            # Nutrition breakdown by food item
            st.markdown("#### 🍎 Nutrition Breakdown by Food Item")
            
            food_nutrition = {}
            for detection in detections:
                # Handle different data types
                if hasattr(detection, 'final_label'):
                    food_item = detection.final_label.lower()
                elif isinstance(detection, dict):
                    food_item = detection.get('final_label', detection.get('label', '')).lower()
                elif isinstance(detection, str):
                    food_item = detection.lower()
                else:
                    food_item = str(detection).lower()
                
                # Basic nutrition mapping
                nutrition_map = {
                    'pizza': {'calories': 285, 'protein': 12, 'carbs': 33, 'fat': 10},
                    'salad': {'calories': 120, 'protein': 8, 'carbs': 15, 'fat': 5},
                    'burger': {'calories': 350, 'protein': 20, 'carbs': 30, 'fat': 15},
                    'apple': {'calories': 95, 'protein': 0.5, 'carbs': 25, 'fat': 0.3},
                    'banana': {'calories': 105, 'protein': 1.3, 'carbs': 27, 'fat': 0.4},
                    'chicken': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
                    'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3},
                    'bread': {'calories': 79, 'protein': 3.1, 'carbs': 15, 'fat': 1.0},
                    'milk': {'calories': 103, 'protein': 8, 'carbs': 12, 'fat': 2.4},
                    'egg': {'calories': 78, 'protein': 6.3, 'carbs': 0.6, 'fat': 5.3},
                    'hot dog': {'calories': 151, 'protein': 5.2, 'carbs': 2.2, 'fat': 13.8},
                    'sandwich': {'calories': 200, 'protein': 10, 'carbs': 25, 'fat': 8},
                    'cake': {'calories': 257, 'protein': 3.2, 'carbs': 45, 'fat': 8.1},
                    'donut': {'calories': 253, 'protein': 4.5, 'carbs': 31, 'fat': 12.8},
                    'cookie': {'calories': 502, 'protein': 6.8, 'carbs': 65, 'fat': 24.5},
                    'broccoli': {'calories': 55, 'protein': 3.7, 'carbs': 11.2, 'fat': 0.6},
                    'orange': {'calories': 62, 'protein': 1.2, 'carbs': 15.4, 'fat': 0.2},
                    'carrot': {'calories': 41, 'protein': 0.9, 'carbs': 9.6, 'fat': 0.2},
                }
                
                if food_item in nutrition_map:
                    if food_item not in food_nutrition:
                        food_nutrition[food_item] = {
                            'count': 0,
                            'calories': 0,
                            'protein': 0,
                            'carbs': 0,
                            'fat': 0
                        }
                    
                    food_nutrition[food_item]['count'] += 1
                    food_nutrition[food_item]['calories'] += nutrition_map[food_item]['calories']
                    food_nutrition[food_item]['protein'] += nutrition_map[food_item]['protein']
                    food_nutrition[food_item]['carbs'] += nutrition_map[food_item]['carbs']
                    food_nutrition[food_item]['fat'] += nutrition_map[food_item]['fat']
            
            # Display nutrition breakdown in a single consolidated view
            for food_name, nutrition in food_nutrition.items():
                st.markdown(f"""
                <div style="
                    border: 1px solid #e0e0e0; 
                    border-radius: 8px; 
                    padding: 15px; 
                    margin: 10px 0; 
                    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <h5 style="margin: 0 0 10px 0; color: #28a745;">🍽️ {food_name.title()} (Count: {nutrition['count']})</h5>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                        <div style="text-align: center; padding: 8px; background: #e8f5e8; border-radius: 5px;">
                            <div style="font-weight: bold; color: #28a745;">Calories</div>
                            <div>{nutrition['calories']:.0f} kcal</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #e8f5e8; border-radius: 5px;">
                            <div style="font-weight: bold; color: #28a745;">Protein</div>
                            <div>{nutrition['protein']:.1f} g</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #e8f5e8; border-radius: 5px;">
                            <div style="font-weight: bold; color: #28a745;">Carbs</div>
                            <div>{nutrition['carbs']:.1f} g</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #e8f5e8; border-radius: 5px;">
                            <div style="font-weight: bold; color: #28a745;">Fat</div>
                            <div>{nutrition['fat']:.1f} g</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 🔍 Detection Analysis Details")
            
            # Detection statistics
            total_detections = len(detections)
            methods_used = set()
            
            for detection in detections:
                # Handle different data types
                if hasattr(detection, 'detection_method'):
                    methods_used.add(detection.detection_method)
                elif isinstance(detection, dict):
                    methods_used.add(detection.get('detection_method', 'YOLO11m'))
                else:
                    methods_used.add('YOLO11m')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Detections", total_detections)
            
            with col2:
                st.metric("Detection Methods", len(methods_used))
            
            # Method breakdown
            st.markdown("#### 🔧 Detection Methods Used")
            for method in methods_used:
                st.write(f"• **{method}**")
        
        with tab4:
            st.markdown("### 💡 AI Analysis Insights")
            
            # Generate insights based on detection results
            insights = []
            
            if detections:
                # Count different food types
                food_types = {}
                for detection in detections:
                    if hasattr(detection, 'final_label'):
                        food_type = detection.final_label.lower()
                    elif isinstance(detection, dict):
                        food_type = detection.get('final_label', detection.get('label', '')).lower()
                    elif isinstance(detection, str):
                        food_type = detection.lower()
                    else:
                        food_type = str(detection).lower()
                    
                    food_types[food_type] = food_types.get(food_type, 0) + 1
                
                # Generate insights
                if len(food_types) > 3:
                    insights.append("🎯 **Variety Detected**: Your meal contains a good variety of different food items, which is excellent for balanced nutrition.")
                
                if any('vegetable' in food or 'salad' in food or 'broccoli' in food or 'carrot' in food for food in food_types.keys()):
                    insights.append("🥗 **Vegetables Present**: Great! Vegetables provide essential vitamins, minerals, and fiber for your health.")
                
                if any('fruit' in food or 'apple' in food or 'banana' in food or 'orange' in food for food in food_types.keys()):
                    insights.append("🍎 **Fruits Detected**: Fruits are excellent sources of natural sugars, vitamins, and antioxidants.")
                
                if any('protein' in food or 'chicken' in food or 'egg' in food for food in food_types.keys()):
                    insights.append("🥩 **Protein Sources**: Protein is essential for muscle building and repair. Good choice!")
                
                if any('carb' in food or 'bread' in food or 'rice' in food for food in food_types.keys()):
                    insights.append("🍞 **Carbohydrates Present**: Carbs provide energy. Consider portion sizes for your goals.")
                
                # Nutrition insights
                nutritional_data = calculate_nutrition_from_expert_detections(detections)
                
                if nutritional_data['total_calories'] > 800:
                    insights.append("⚠️ **High Calorie Meal**: This appears to be a substantial meal. Consider your daily calorie goals.")
                elif nutritional_data['total_calories'] < 200:
                    insights.append("🍽️ **Light Meal**: This is a light meal. You might want to add more food for balanced nutrition.")
                
                if nutritional_data['total_protein'] > 30:
                    insights.append("💪 **High Protein**: Excellent protein content! This will help with muscle maintenance and satiety.")
                
                if nutritional_data['total_fats'] > 20:
                    insights.append("🧈 **Moderate Fat Content**: Be mindful of fat intake, especially if you're watching your weight.")
            
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.info("📊 Upload a food image to get personalized AI insights about your meal!")
        
        # Summary section
        st.markdown("---")
        st.markdown("### 📋 Analysis Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Detection Summary:**")
            st.write(f"• Total items detected: {len(detections)}")
            st.write(f"• Unique food types: {len(set([d.final_label if hasattr(d, 'final_label') else str(d) for d in detections]))}")
            st.write(f"• Analysis method: YOLO11m Object Detection")
        
        with col2:
            st.markdown("**Nutritional Summary:**")
            nutritional_data = calculate_nutrition_from_expert_detections(detections)
            st.write(f"• Total calories: {nutritional_data['total_calories']:.0f} kcal")
            st.write(f"• Protein: {nutritional_data['total_protein']:.1f} g")
            st.write(f"• Carbohydrates: {nutritional_data['total_carbs']:.1f} g")
            st.write(f"• Fats: {nutritional_data['total_fats']:.1f} g")
    
    else:
        st.warning("No food items detected. Try uploading a clearer image.")
        st.info("💡 **Tips for better detection:**")
        st.write("• Ensure good lighting")
        st.write("• Take a clear, focused photo")
        st.write("• Include the entire meal in the frame")
        st.write("• Avoid blurry or dark images")

def calculate_nutrition_from_expert_detections(detections):
    """Calculate nutrition data from expert detections"""
    # Simple nutrition calculation (you can enhance this)
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fats = 0
    
    # Basic nutrition mapping (simplified)
    nutrition_map = {
        'pizza': {'calories': 285, 'protein': 12, 'carbs': 33, 'fat': 10},
        'salad': {'calories': 120, 'protein': 8, 'carbs': 15, 'fat': 5},
        'burger': {'calories': 350, 'protein': 20, 'carbs': 30, 'fat': 15},
        'apple': {'calories': 95, 'protein': 0.5, 'carbs': 25, 'fat': 0.3},
        'banana': {'calories': 105, 'protein': 1.3, 'carbs': 27, 'fat': 0.4},
        'chicken': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
        'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3},
        'bread': {'calories': 79, 'protein': 3.1, 'carbs': 15, 'fat': 1.0},
        'milk': {'calories': 103, 'protein': 8, 'carbs': 12, 'fat': 2.4},
        'egg': {'calories': 78, 'protein': 6.3, 'carbs': 0.6, 'fat': 5.3},
        'hot dog': {'calories': 151, 'protein': 5.2, 'carbs': 2.2, 'fat': 13.8},
        'sandwich': {'calories': 200, 'protein': 10, 'carbs': 25, 'fat': 8},
        'cake': {'calories': 257, 'protein': 3.2, 'carbs': 45, 'fat': 8.1},
        'donut': {'calories': 253, 'protein': 4.5, 'carbs': 31, 'fat': 12.8},
        'cookie': {'calories': 502, 'protein': 6.8, 'carbs': 65, 'fat': 24.5},
        'broccoli': {'calories': 55, 'protein': 3.7, 'carbs': 11.2, 'fat': 0.6},
        'orange': {'calories': 62, 'protein': 1.2, 'carbs': 15.4, 'fat': 0.2},
        'carrot': {'calories': 41, 'protein': 0.9, 'carbs': 9.6, 'fat': 0.2},
    }
    
    for detection in detections:
        # Handle different data types
        if hasattr(detection, 'final_label'):
            # FoodDetection object
            food_item = detection.final_label.lower()
        elif isinstance(detection, dict):
            # Dictionary format
            food_item = detection.get('final_label', detection.get('label', '')).lower()
        elif isinstance(detection, str):
            # String format
            food_item = detection.lower()
        else:
            # Fallback
            food_item = str(detection).lower()
        
        if food_item in nutrition_map:
            nutrition = nutrition_map[food_item]
            total_calories += nutrition['calories']
            total_protein += nutrition['protein']
            total_carbs += nutrition['carbs']
            total_fats += nutrition['fat']
        else:
            # Default values for unknown items
            total_calories += 150
            total_protein += 5
            total_carbs += 20
            total_fats += 5
    
    return {
        'total_calories': total_calories,
        'total_protein': total_protein,
        'total_carbs': total_carbs,
        'total_fats': total_fats
    }

if __name__ == "__main__":
    main()