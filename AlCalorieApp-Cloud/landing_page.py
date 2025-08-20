#!/usr/bin/env python3
"""
AI-Powered Nutrition Analysis Landing Page
Beautiful landing page matching the promotional design
"""

import streamlit as st
from PIL import Image
import base64
from io import BytesIO

def get_base64_of_bin_file(bin_file):
    """Convert binary file to base64 for embedding in HTML"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    """Set background image"""
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)

def create_landing_page():
    """Create the main landing page"""
    
    # Custom CSS for the landing page
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 0;
        margin: 0;
    }
    
    /* Header styling */
    .landing-header {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 2rem 0;
        text-align: center;
        border-radius: 0 0 30px 30px;
        margin-bottom: 3rem;
        box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3);
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* CTA buttons */
    .cta-button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        margin: 10px;
    }
    
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
        color: white;
        text-decoration: none;
    }
    
    .cta-button-secondary {
        background: white;
        color: #28a745;
        border: 2px solid #28a745;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        margin: 10px;
    }
    
    .cta-button-secondary:hover {
        background: #28a745;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
        text-decoration: none;
    }
    
    /* Stats cards */
    .stats-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        text-align: center;
        margin: 10px;
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #28a745;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        font-size: 1rem;
        color: #6c757d;
        font-weight: 500;
    }
    
    /* App mockup */
    .app-mockup {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    
    /* Privacy indicators */
    .privacy-indicator {
        display: inline-block;
        margin: 10px;
        padding: 8px 15px;
        background: #f8f9fa;
        border-radius: 20px;
        font-size: 0.9rem;
        color: #6c757d;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .landing-header {
            padding: 1rem 0;
        }
        
        .stats-number {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header Section
    st.markdown("""
    <div class="landing-header">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; font-weight: bold;">
            🧠 AI-Powered Nutrition Analysis
        </h1>
        <h2 style="font-size: 2.5rem; margin-bottom: 2rem; font-weight: 300;">
            Transform Your <span style="color: #ffd700;">Nutrition Journey</span> with AI
        </h2>
        <p style="font-size: 1.2rem; margin-bottom: 2rem; opacity: 0.9;">
            Experience the future of food tracking with our advanced AI technology. 
            Get instant, accurate nutritional analysis of any meal with 99.2% accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Content
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        # Left Panel - Service Description
        st.markdown("""
        <div style="padding: 2rem;">
            <h3 style="color: #28a745; font-size: 1.5rem; margin-bottom: 1rem;">
                🎯 Key Features
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Cards
        col1_1, col1_2, col1_3 = st.columns(3)
        
        with col1_1:
            st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2rem; color: #28a745; margin-bottom: 1rem;">🎯</div>
                <div class="stats-number">99.2%</div>
                <div class="stats-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col1_2:
            st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2rem; color: #28a745; margin-bottom: 1rem;">⚡</div>
                <div class="stats-number">&lt;2s</div>
                <div class="stats-label">Processing</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col1_3:
            st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2rem; color: #28a745; margin-bottom: 1rem;">👥</div>
                <div class="stats-number">50K+</div>
                <div class="stats-label">Users</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Call-to-Action Buttons
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <a href="?page=analysis" class="cta-button">
                📸 Start Free Analysis →
            </a>
            <a href="#demo" class="cta-button-secondary">
                ▶️ Watch Demo
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # Privacy Indicators
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <div class="privacy-indicator">🔒 Privacy First</div>
            <div class="privacy-indicator">🔐 End-to-End Encrypted</div>
            <div class="privacy-indicator">🌍 Global Coverage</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Middle Section - AI Models Highlight
        st.markdown("""
        <div class="feature-card" style="margin-top: 2rem;">
            <div style="font-size: 2.5rem; color: #28a745; margin-bottom: 1rem;">🧠</div>
            <div class="stats-number">6+</div>
            <div class="stats-label">AI Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Right Panel - App Mockup
        st.markdown("""
        <div class="app-mockup">
            <h3 style="color: #28a745; margin-bottom: 1.5rem;">📱 AI Calorie Analyzer</h3>
            <div style="background: #f8f9fa; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">🍕</span>
                    <div>
                        <div style="font-weight: bold;">Pizza</div>
                        <div style="color: #28a745; font-size: 1.2rem;">285 cal</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">🥗</span>
                    <div>
                        <div style="font-weight: bold;">Salad</div>
                        <div style="color: #28a745; font-size: 1.2rem;">120 cal</div>
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
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2rem; color: #28a745; margin-bottom: 1rem;">🔍</div>
            <h4 style="color: #28a745; margin-bottom: 1rem;">Advanced Detection</h4>
            <p style="color: #6c757d;">YOLO11m powered food recognition with multiple AI models for comprehensive analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2rem; color: #28a745; margin-bottom: 1rem;">📊</div>
            <h4 style="color: #28a745; margin-bottom: 1rem;">Detailed Analytics</h4>
            <p style="color: #6c757d;">Get comprehensive nutritional breakdown including calories, macros, and micronutrients.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2rem; color: #28a745; margin-bottom: 1rem;">📈</div>
            <h4 style="color: #28a745; margin-bottom: 1rem;">Progress Tracking</h4>
            <p style="color: #6c757d;">Track your nutrition journey with detailed history and progress analytics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Final CTA
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0; padding: 2rem; background: linear-gradient(135deg, #28a745 0%, #20c997 100%); border-radius: 20px; color: white;">
        <h2 style="margin-bottom: 1rem;">Ready to Transform Your Nutrition Journey?</h2>
        <p style="font-size: 1.2rem; margin-bottom: 2rem;">Join thousands of users who trust our AI-powered nutrition analysis.</p>
        <a href="?page=analysis" class="cta-button" style="background: white; color: #28a745;">
            🚀 Start Your Free Analysis Now
        </a>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main function to run the landing page"""
    # Page configuration
    st.set_page_config(
        page_title="AI-Powered Nutrition Analysis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Hide default Streamlit elements
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Create the landing page
    create_landing_page()

if __name__ == "__main__":
    main()
