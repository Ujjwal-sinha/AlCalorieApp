#!/usr/bin/env python3
"""
Test script for fine-tuned model labels in modern UI
"""

import streamlit as st
from modern_ui import (
    load_css, create_modern_header, create_model_status_display
)

def main():
    # Page config
    st.set_page_config(
        page_title="🔧 Fine-tuned Models Test", 
        layout="wide", 
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )
    
    # Load modern CSS
    load_css()
    
    # Modern header
    create_modern_header()
    
    st.markdown("## 🔧 Fine-tuned Models Test")
    
    # Test model status display
    st.markdown("### 🤖 Model Status with Fine-tuned Labels")
    
    # Sample model status
    model_status = {
        'BLIP': True,
        'YOLO': True,
        'LLM': True,
        'CNN': False,
        'Transformer': True
    }
    
    # Display in sidebar
    with st.sidebar:
        create_model_status_display(model_status)
    
    # Display in main area for testing
    st.markdown("### 📊 Model Status Display")
    create_model_status_display(model_status)
    
    # Show the model status data
    st.markdown("### 📋 Raw Model Status Data")
    for model, status in model_status.items():
        st.write(f"**{model}**: {status}")
    
    st.success("✅ Fine-tuned model labels test completed!")

if __name__ == "__main__":
    main()
