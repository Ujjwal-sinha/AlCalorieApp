import streamlit as st
from typing import Dict, Any

def create_header():
    """Create the main header for the app"""
    st.markdown("""
    <div class="header-card">
        <h1>🍱 YOLO11m Calorie Tracker</h1>
        <p>Track your nutrition with YOLO11m-powered food analysis</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar(models: Dict[str, Any]):
    """Create the sidebar with model status and user settings"""
    with st.sidebar:
        st.markdown("### 🔍 YOLO11m Model Status")
        
        model_status = {
            'YOLO11m (Object Detection)': models.get('yolo_model') is not None,
            'LLM (Nutrition Analysis)': models.get('llm') is not None,
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

def create_detection_results(detections):
    """Create detection results display"""
    if not detections:
        st.warning("No food items detected in the image.")
        return
    
    st.markdown("### 🔍 Detected Food Items")
    for i, detection in enumerate(detections):
        confidence = detection.confidence_score if hasattr(detection, 'confidence_score') else 0.0
        label = detection.final_label if hasattr(detection, 'final_label') else str(detection)
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{label}</strong>
                    <br>
                    <small style="color: #666;">Confidence: {confidence:.2f}</small>
                </div>
                <div style="font-size: 1.5rem;">🍽️</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_error_message(error_msg):
    """Create an error message with styling"""
    st.markdown(f"""
    <div style="background: #ffebee; border: 1px solid #f44336; border-radius: 8px; padding: 15px; margin: 10px 0;">
        <div style="color: #d32f2f; font-weight: bold;">❌ Error</div>
        <div style="color: #666; margin-top: 5px;">{error_msg}</div>
    </div>
    """, unsafe_allow_html=True)

def create_success_message(success_msg):
    """Create a success message with styling"""
    st.markdown(f"""
    <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 8px; padding: 15px; margin: 10px 0;">
        <div style="color: #2e7d32; font-weight: bold;">✅ Success</div>
        <div style="color: #666; margin-top: 5px;">{success_msg}</div>
    </div>
    """, unsafe_allow_html=True)
