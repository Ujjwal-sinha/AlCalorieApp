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
        st.markdown("**✅ YOLO11m (Object Detection)**")
        
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
                        detections = yolo_system.recognize_food(image)
                        expert_summary = yolo_system.get_detection_summary(detections)
                        expert_result = {"detections": detections, "summary": expert_summary}
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
    """Display expert analysis results"""
    st.subheader("🔍 Detection Results")
    
    if detections and len(detections) > 0:
        st.success(f"✅ Found {len(detections)} food items")
        
        for i, detection in enumerate(detections):
            with st.expander(f"🍽️ {detection.final_label} (Confidence: {detection.confidence_score:.2f})"):
                st.write(f"**Label:** {detection.final_label}")
                st.write(f"**Confidence:** {detection.confidence_score:.2f}")
                st.write(f"**Method:** {detection.detection_method}")
    else:
        st.warning("No food items detected. Try uploading a clearer image.")

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
    }
    
    for detection in detections:
        food_item = detection.final_label.lower()
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