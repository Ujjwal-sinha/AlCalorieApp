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
    /* Modern CSS for AI Calorie Tracker */
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card Styling */
    .modern-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* File Uploader Styling */
    .stFileUploader {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        border: 2px dashed #667eea;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: white;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #667eea;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Footer Styling */
    .modern-footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        margin: 2rem -1rem -1rem -1rem;
        border-radius: 20px 20px 0 0;
        text-align: center;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
        
        .modern-card {
            padding: 1.5rem;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Custom Components */
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Timeline Styling */
    .timeline-item {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        position: relative;
    }
    
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 50%;
        transform: translateY(-50%);
        width: 12px;
        height: 12px;
        background: #667eea;
        border-radius: 50%;
        border: 3px solid white;
    }
    
    /* Nutrition Badge */
    .nutrition-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    /* AI Analysis Box */
    .ai-analysis-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .ai-analysis-box h3 {
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Food Item Card */
    .food-item-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
        transition: all 0.3s ease;
    }
    
    .food-item-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
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
            © 2024 Ujjwal Sinha • Built with ❤️ using Streamlit & Advanced AI
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_modern_sidebar():
    """Create a modern sidebar with navigation"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🍎 Nutrition Dashboard</h2>
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
                    "icon": {"color": "white", "font-size": "18px"},
                    "nav-link": {
                        "color": "white",
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#eee",
                        "border-radius": "10px",
                        "padding": "10px 15px",
                    },
                    "nav-link-selected": {
                        "background-color": "rgba(255,255,255,0.2)",
                        "color": "white",
                        "border-radius": "10px",
                    },
                }
            )
        else:
            # Fallback to simple selectbox
            st.markdown("""
            <h4>📱 Navigation</h4>
            """, unsafe_allow_html=True)
            selected = st.selectbox(
                "Choose a section:",
                ["📷 Food Analysis", "📊 History", "📅 Daily Summary", "📈 Analytics"],
                index=0
            )
        
        st.markdown("---")
        
        # Model status
        st.markdown("""
        <h4>🤖 AI Models Status</h4>
        """, unsafe_allow_html=True)
        
        # Progress section
        st.markdown("""
        <h4>📊 Today's Progress</h4>
        """, unsafe_allow_html=True)
        
        return selected

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

def create_analysis_results(nutrition_data, food_items, analysis_text):
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
    
    # Food items breakdown
    if food_items:
        st.markdown("""
        <div class="modern-card">
            <h3>🍽️ Food Items Breakdown</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for item in food_items:
            create_food_item_card(
                item.get('item', 'Unknown'),
                item.get('calories', 0),
                item.get('protein', 0),
                item.get('carbs', 0),
                item.get('fats', 0)
            )
    
    # AI Analysis
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
    'create_food_item_card',
    'create_timeline_item',
    'create_modern_footer',
    'create_modern_sidebar',
    'create_upload_section',
    'create_analysis_results',
    'create_modern_chart_container',
    'create_loading_animation',
    'create_empty_state'
]
