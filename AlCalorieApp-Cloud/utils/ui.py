import streamlit as st
from typing import Dict, Any

def create_header():
    """Create the main header for the app"""
    st.markdown("""
    <div class="header-card">
        <h1>🍱 AI Calorie Tracker</h1>
        <p>Track your nutrition with AI-powered food analysis</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar(models: Dict[str, Any]):
    """Create the sidebar with model status and user settings"""
    with st.sidebar:
        st.markdown("### 🤖 AI Models Status")
        
        model_status = {
            'BLIP (Image Analysis)': models.get('blip_model') is not None,
            'LLM (Nutrition Analysis)': models.get('llm') is not None,
            'YOLO (Object Detection)': models.get('yolo_model') is not None,
            'CNN (Visualizations)': models.get('cnn_model') is not None
        }
        
        for model, status in model_status.items():
            status_class = "status-success" if status else "status-error"
            status_icon = "✅" if status else "❌"
            status_text = "Available" if status else "Not Available"
            st.markdown(f'<span class="{status_class}">{status_icon} **{model}**: {status_text}</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # User settings
        st.markdown("### 👤 Settings")
        st.number_input("Daily Calorie Target (kcal)", min_value=1000, max_value=5000, 
                       value=st.session_state.calorie_target, step=100, key="calorie_target")
        
        # Today's progress
        st.markdown("### 📊 Today's Progress")
        from datetime import date
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

def create_upload_section():
    """Create the file upload section"""
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of your food"
    )
    return uploaded_file

def create_metric_card(value, label, icon="📊"):
    """Create a metric card with custom styling"""
    st.markdown(f"""
    <div class="metric-card">
        <div style="text-align: center;">
            <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{icon}</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin: 10px 0;">{value}</div>
            <div style="color: #666; font-size: 0.9rem;">{label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_analysis_card(title, content):
    """Create an analysis card with custom styling"""
    st.markdown(f"""
    <div class="analysis-card">
        <h4>{title}</h4>
        <div style="margin-top: 15px;">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_progress_section():
    """Create a progress tracking section"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    return progress_bar, status_text

def create_nutrition_summary(nutrition_data):
    """Create nutrition summary with metrics"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Calories", f"{nutrition_data['total_calories']} kcal")
    with col2:
        st.metric("Protein", f"{nutrition_data['total_protein']}g")
    with col3:
        st.metric("Carbs", f"{nutrition_data['total_carbs']}g")
    with col4:
        st.metric("Fats", f"{nutrition_data['total_fats']}g")

def create_tips_section():
    """Create tips section"""
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        - 📸 Take clear photos in good lighting
        - 🍽️ Include all food items in the frame
        - 📝 Add context description if needed
        - 🔄 Try different angles if detection is incomplete
        """)

def create_footer():
    """Create the modern footer"""
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
