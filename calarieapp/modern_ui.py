import streamlit as st
import base64

# Try to import streamlit-option-menu, fallback to simple tabs if not available
try:
    from streamlit_option_menu import option_menu
    OPTION_MENU_AVAILABLE = True
except ImportError:
    OPTION_MENU_AVAILABLE = False

def load_css():
    """Load modern CSS styling"""
    st.markdown("""
    <style>
    /* Ultra Modern CSS for AI Calorie Tracker */
    
    /* Global Styles with Glassmorphism */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Enhanced Header with 3D Effects */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 30px 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        text-align: center;
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
        position: relative;
        z-index: 1;
        font-weight: 300;
    }
    
    /* Enhanced Card Styling with Glassmorphism */
    .modern-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .modern-card:hover::before {
        left: 100%;
    }
    
    .modern-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }
    
    /* Enhanced Metric Cards with Gradients */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .metric-card:hover::before {
        transform: translateX(100%);
    }
    
    .metric-card:hover {
        transform: scale(1.08) rotate(1deg);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.95;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Enhanced File Uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 3rem;
        border: 3px dashed #667eea;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .stFileUploader::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(102, 126, 234, 0.05) 50%, transparent 70%);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .stFileUploader:hover::before {
        transform: translateX(100%);
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.08);
        transform: scale(1.02);
    }
    
    /* Enhanced Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Enhanced Sidebar with Black Font Color */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        backdrop-filter: blur(20px);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        color: #333;
        backdrop-filter: blur(20px);
    }
    
    /* Sidebar Text Visibility Fixes - Black Font */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: #333 !important;
        text-shadow: none;
    }
    
    .css-1d391kg p, .css-1d391kg div, .css-1d391kg span, .css-1d391kg label {
        color: #333 !important;
    }
    
    .css-1d391kg .stMarkdown {
        color: #333 !important;
    }
    
    .css-1d391kg .stMarkdown h1, .css-1d391kg .stMarkdown h2, .css-1d391kg .stMarkdown h3, 
    .css-1d391kg .stMarkdown h4, .css-1d391kg .stMarkdown h5, .css-1d391kg .stMarkdown h6 {
        color: #333 !important;
        text-shadow: none;
    }
    
    .css-1d391kg .stMarkdown p {
        color: #333 !important;
    }
    
    /* Sidebar Input Elements */
    .css-1d391kg .stNumberInput > div > div > input {
        background: white !important;
        color: #333 !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 10px !important;
    }
    
    .css-1d391kg .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    .css-1d391kg .stNumberInput > div > div > label {
        color: #333 !important;
        font-weight: 600 !important;
        text-shadow: none !important;
    }
    
    /* Sidebar Selectbox */
    .css-1d391kg .stSelectbox > div > div > div {
        background: white !important;
        color: #333 !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 10px !important;
    }
    
    .css-1d391kg .stSelectbox > div > div > label {
        color: #333 !important;
        font-weight: 600 !important;
        text-shadow: none !important;
    }
    
    /* Sidebar Button */
    .css-1d391kg .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .css-1d391kg .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    
    /* Sidebar Progress Bar */
    .css-1d391kg .stProgress > div > div > div {
        background: #e9ecef !important;
        border-radius: 10px !important;
    }
    
    .css-1d391kg .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar Caption */
    .css-1d391kg .stCaption {
        color: #6c757d !important;
        font-size: 0.9rem !important;
    }
    
    /* Sidebar Metric */
    .css-1d391kg .stMetric {
        background: white !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        border: 1px solid #dee2e6 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    .css-1d391kg .stMetric > div > div > div {
        color: #333 !important;
        font-weight: 600 !important;
    }
    
    .css-1d391kg .stMetric > div > div > div > div {
        color: #6c757d !important;
    }
    
    /* Additional Sidebar Text Visibility Fixes - Black Font */
    .css-1d391kg .stText {
        color: #333 !important;
    }
    
    .css-1d391kg .stTextInput > div > div > input {
        background: white !important;
        color: #333 !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 10px !important;
    }
    
    .css-1d391kg .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    .css-1d391kg .stTextInput > div > div > label {
        color: #333 !important;
        font-weight: 600 !important;
        text-shadow: none !important;
    }
    
    /* Sidebar Divider */
    .css-1d391kg hr {
        border-color: #dee2e6 !important;
        margin: 1rem 0 !important;
    }
    
    /* Sidebar Links */
    .css-1d391kg a {
        color: #667eea !important;
        text-decoration: none !important;
        text-shadow: none !important;
    }
    
    .css-1d391kg a:hover {
        color: #5a6fd8 !important;
        text-decoration: underline !important;
    }
    
    /* Sidebar Code Blocks */
    .css-1d391kg code {
        background: #f8f9fa !important;
        color: #333 !important;
        border-radius: 5px !important;
        padding: 2px 6px !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* Sidebar Lists */
    .css-1d391kg ul, .css-1d391kg ol {
        color: #333 !important;
    }
    
    .css-1d391kg li {
        color: #333 !important;
        text-shadow: none !important;
    }
    
    /* Sidebar Blockquotes */
    .css-1d391kg blockquote {
        border-left: 4px solid #667eea !important;
        color: #333 !important;
        background: #f8f9fa !important;
        padding: 1rem !important;
        border-radius: 0 10px 10px 0 !important;
    }
    
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 8px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 15px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.9);
        color: #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Enhanced Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Enhanced Message Styling */
    .stSuccess {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        border: none;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
    }
    
    .stError {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        border: none;
        box-shadow: 0 6px 20px rgba(244, 67, 54, 0.3);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        border: none;
        box-shadow: 0 6px 20px rgba(255, 152, 0, 0.3);
    }
    
    .stInfo {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        border: none;
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.3);
    }
    
    /* Enhanced Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
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
    
    /* Enhanced Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .main-header p {
            font-size: 1.1rem;
        }
        
        .metric-value {
            font-size: 2.5rem;
        }
        
        .modern-card {
            padding: 2rem;
            margin: 1rem 0;
        }
        
        .stButton > button {
            padding: 0.8rem 2rem;
            font-size: 1rem;
        }
    }
    
    /* Enhanced Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(241, 241, 241, 0.8);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        border: 2px solid rgba(241, 241, 241, 0.8);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Enhanced Animations */
    .fade-in {
        animation: fadeIn 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(30px) scale(0.95); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0) scale(1); 
        }
    }
    
    .slide-in {
        animation: slideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes slideIn {
        from { 
            transform: translateX(-100%) scale(0.9); 
            opacity: 0;
        }
        to { 
            transform: translateX(0) scale(1); 
            opacity: 1;
        }
    }
    
    .bounce-in {
        animation: bounceIn 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Enhanced Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 30px;
        height: 30px;
        border: 4px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Enhanced Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 3rem;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255,255,255,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(102, 126, 234, 0.05) 50%, transparent 70%);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .feature-card:hover::before {
        transform: translateX(100%);
    }
    
    .feature-card:hover {
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.1));
    }
    
    /* Enhanced Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    /* Enhanced Timeline */
    .timeline-item {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        position: relative;
        transition: all 0.3s ease;
    }
    
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -10px;
        top: 50%;
        transform: translateY(-50%);
        width: 15px;
        height: 15px;
        background: #667eea;
        border-radius: 50%;
        border: 4px solid white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .timeline-item:hover {
        transform: translateX(5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    /* Enhanced Nutrition Badges */
    .nutrition-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.7rem 1.5rem;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0.3rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .nutrition-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Enhanced AI Analysis Box */
    .ai-analysis-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 3rem;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .ai-analysis-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="ai-pattern" width="50" height="50" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="10" cy="10" r="0.5" fill="rgba(255,255,255,0.05)"/><circle cx="40" cy="40" r="0.5" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="100" height="100" fill="url(%23ai-pattern)"/></svg>');
        opacity: 0.3;
    }
    
    .ai-analysis-box h3 {
        color: white;
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 1;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .ai-analysis-box > div {
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced Food Item Cards */
    .food-item-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .food-item-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(76, 175, 80, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .food-item-card:hover::before {
        left: 100%;
    }
    
    .food-item-card:hover {
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    /* Model Status Cards */
    .model-status-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .model-status-card:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
    }
    
    .fine-tuned-label {
        color: #FFD700;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Floating Action Button */
    .fab {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        cursor: pointer;
        z-index: 1000;
    }
    
    .fab:hover {
        transform: scale(1.1);
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
    }
    
    /* Glassmorphism Input Fields */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 0.8rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.95);
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Enhanced Number Input */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 0.8rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.95);
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Enhanced Selectbox */
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div > div:hover {
        background: rgba(255, 255, 255, 0.95);
        border-color: #667eea;
    }
    
    </style>
    """, unsafe_allow_html=True)

def create_modern_header():
    """Create a modern header with gradient background"""
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🍱 AI Calorie Tracker</h1>
        <p>Track your nutrition with advanced AI-powered food analysis</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(value, label, icon="📊"):
    """Create a modern metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def create_feature_card(icon, title, description):
    """Create a feature showcase card"""
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

def create_ai_analysis_box(title, content):
    """Create an AI analysis display box"""
    st.markdown(f"""
    <div class="ai-analysis-box">
        <h3>🤖 {title}</h3>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def create_food_item_card(item_name, calories, protein, carbs, fats):
    """Create a food item display card"""
    st.markdown(f"""
    <div class="food-item-card">
        <h4>🍽️ {item_name}</h4>
        <div class="stats-grid">
            <span class="nutrition-badge">🔥 {calories} cal</span>
            <span class="nutrition-badge">💪 {protein}g protein</span>
            <span class="nutrition-badge">🌾 {carbs}g carbs</span>
            <span class="nutrition-badge">🥑 {fats}g fats</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_timeline_item(timestamp, title, content):
    """Create a timeline item for history"""
    st.markdown(f"""
    <div class="timeline-item">
        <small style="color: #666;">{timestamp}</small>
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def create_modern_footer():
    """Create a modern footer"""
    st.markdown("""
    <div class="modern-footer">
        <h3>🍱 AI Calorie Tracker</h3>
        <p>🔬 AI Visualizations • 📊 Nutrition Analysis • 🚀 Modern Interface</p>
        <p><strong>Developed by Ujjwal Sinha</strong></p>
        <div style="margin-top: 1rem;">
            <a href="https://github.com/Ujjwal-sinha" target="_blank" style="color: white; margin: 0 1rem; text-decoration: none;">📱 GitHub</a>
            <a href="https://www.linkedin.com/in/sinhaujjwal01/" target="_blank" style="color: white; margin: 0 1rem; text-decoration: none;">💼 LinkedIn</a>
        </div>
        <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
            © 2025 Ujjwal Sinha • Built with ❤️ using Streamlit & Advanced AI
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_modern_sidebar():
    """Create a modern sidebar with navigation"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #333; font-weight: 700;">🍎 Nutrition Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if OPTION_MENU_AVAILABLE:
            # Navigation menu with option_menu
            selected = option_menu(
                menu_title=None,
                options=["📷 Food Analysis", "📊 History", "📅 Daily Summary", "📈 Analytics"],
                icons=["camera", "clock-history", "calendar", "graph-up"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "#667eea", "font-size": "18px"},
                    "nav-link": {
                        "color": "#333",
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#f8f9fa",
                        "border-radius": "10px",
                        "padding": "10px 15px",
                        "font-weight": "500",
                    },
                    "nav-link-selected": {
                        "background-color": "rgba(102, 126, 234, 0.1)",
                        "color": "#667eea",
                        "border-radius": "10px",
                        "font-weight": "600",
                    },
                }
            )
        else:
            # Fallback to simple selectbox
            st.markdown("""
            <h4 style="color: #333; font-weight: 600;">📱 Navigation</h4>
            """, unsafe_allow_html=True)
            selected = st.selectbox(
                "Choose a section:",
                ["📷 Food Analysis", "📊 History", "📅 Daily Summary", "📈 Analytics"],
                index=0
            )
        
        st.markdown("---")
        
        # Model status with fine-tuned labels
        st.markdown("""
        <h4 style="color: #333; font-weight: 600;">🤖 AI Models Status</h4>
        """, unsafe_allow_html=True)
        
        # Progress section
        st.markdown("""
        <h4 style="color: #333; font-weight: 600;">📊 Today's Progress</h4>
        """, unsafe_allow_html=True)
        
        return selected

def create_model_status_display(model_status):
    """Create a modern model status display with fine-tuned labels"""
    st.markdown("""
    <h4 style="color: #333; font-weight: 600;">🤖 AI Models Status</h4>
    """, unsafe_allow_html=True)
    
    for model, status in model_status.items():
        status_icon = "✅" if status else "❌"
        status_text = "Available" if status else "Not Available"
        
        # Add fine-tuned label for all models
        st.markdown(f"""
        <div style="margin: 0.5rem 0; padding: 0.5rem; border-radius: 8px; background: white; border: 1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="color: #333;">
                    <span style="color: #667eea; font-weight: bold;">🔧 Fine-tuned</span> 
                    <strong style="color: #333;">{model}</strong>
                </div>
                <div style="color: {'#4CAF50' if status else '#f44336'}; font-weight: bold;">
                    {status_icon} {status_text}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_upload_section():
    """Create a modern file upload section"""
    st.markdown("""
    <div class="modern-card">
        <h3>📷 Upload Food Image</h3>
        <p>Take a clear photo of your meal for AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader with custom styling
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of your food for analysis"
    )
    
    if uploaded_file is not None:
        st.markdown("""
        <div class="chart-container">
            <h4>📸 Uploaded Image</h4>
        </div>
        """, unsafe_allow_html=True)
    
    return uploaded_file

def create_analysis_results(nutrition_data, analysis_text):
    """Display analysis results in modern format"""
    st.markdown("""
    <div class="modern-card">
        <h3>📊 Analysis Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Nutrition metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card(f"{nutrition_data.get('total_calories', 0)}", "Calories", "🔥")
    with col2:
        create_metric_card(f"{nutrition_data.get('total_protein', 0):.1f}g", "Protein", "💪")
    with col3:
        create_metric_card(f"{nutrition_data.get('total_carbs', 0):.1f}g", "Carbs", "🌾")
    with col4:
        create_metric_card(f"{nutrition_data.get('total_fats', 0):.1f}g", "Fats", "🥑")
    
    # AI Analysis (only this section, no food items breakdown)
    if analysis_text:
        create_ai_analysis_box("AI Analysis Report", analysis_text)

def create_modern_chart_container(title):
    """Create a modern container for charts"""
    st.markdown(f"""
    <div class="chart-container">
        <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)

def create_loading_animation():
    """Create a modern loading animation"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <div class="loading-spinner"></div>
        <p style="margin-top: 1rem; color: #667eea; font-weight: 600;">AI is analyzing your food...</p>
    </div>
    """, unsafe_allow_html=True)

def create_empty_state(icon, title, description, action_text=None):
    """Create a modern empty state"""
    st.markdown(f"""
    <div class="modern-card" style="text-align: center;">
        <div class="feature-icon">{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
        {f'<p style="color: #667eea; font-weight: 600;">{action_text}</p>' if action_text else ''}
    </div>
    """, unsafe_allow_html=True)

# Export functions
__all__ = [
    'load_css',
    'create_modern_header',
    'create_metric_card',
    'create_feature_card',
    'create_ai_analysis_box',
    'create_timeline_item',
    'create_modern_footer',
    'create_modern_sidebar',
    'create_model_status_display',
    'create_upload_section',
    'create_analysis_results',
    'create_modern_chart_container',
    'create_loading_animation',
    'create_empty_state'
]
