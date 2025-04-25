import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os
from datetime import datetime, date
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from transformers import BlipForConditionalGeneration, BlipProcessor
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import re
import torch

# ------------------------ Setup ------------------------ #
st.set_page_config(page_title="🍱 AI Calorie Tracker", layout="centered")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ------------------------ Model Initialization ------------------------ #
@st.cache_resource
def load_models():
    """Initialize all AI models with proper device handling"""
    models = {}
    
    # Initialize LangChain ChatGroq LLM
    try:
        models['llm'] = ChatGroq(
            model_name="llama3-8b-8192",
            api_key=groq_api_key
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        models['llm'] = None

    # Initialize BLIP image captioning
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load processor
        models['processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Load model without moving to device immediately
        models['blip_model'] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=dtype,
            low_cpu_mem_usage=True  # Optimize memory usage
        )
        
        # Move model to device after loading
        if device == "cuda":
            models['blip_model'] = models['blip_model'].to(device)
        else:
            models['blip_model'] = models['blip_model'].to(device)
        
        models['blip_model'].eval()
    except Exception as e:
        st.error(f"Failed to load image captioner: {e}")
        models['processor'] = None
        models['blip_model'] = None

    return models

models = load_models()

# ------------------------ Session State ------------------------ #
if "history" not in st.session_state:
    st.session_state.history = []

if "daily_calories" not in st.session_state:
    st.session_state.daily_calories = {}

if "last_results" not in st.session_state:
    st.session_state.last_results = {}

# ------------------------ Helper Functions ------------------------ #
def query_langchain(prompt):
    """Query the Groq LLM with error handling"""
    if not models['llm']:
        return "LLM service unavailable"
    try:
        response = models['llm']([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

def describe_image(image: Image.Image) -> str:
    """Generate caption for food image with robust error handling"""
    if not models['processor'] or not models['blip_model']:
        return "Image analysis unavailable"
    
    try:
        # Convert image to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Get device from model
        device = next(models['blip_model'].parameters()).device
        
        # Process image
        inputs = models['processor'](image, return_tensors="pt").to(device)
        
        # Generate caption
        with torch.no_grad():
            outputs = models['blip_model'].generate(**inputs, max_new_tokens=50)
        
        return models['processor'].decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Image analysis error: {str(e)}"

def extract_items_and_calories(text):
    """Extract food items and calorie counts from text"""
    pattern = r'(\b[\w\s]+\b)[^\d]*(\d{2,4})\s*cal'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [(item.strip(), int(cal)) for item, cal in matches if cal.isdigit()]

def plot_chart(food_data):
    """Create horizontal bar chart of food calories"""
    if not food_data:
        return None
    
    items, calories = zip(*food_data)
    fig, ax = plt.subplots(figsize=(8, len(items)*0.5))
    ax.barh(items, calories, color='#4CAF50')
    ax.set_xlabel("Calories", fontsize=10)
    ax.set_title("Calorie Breakdown", fontsize=12)
    plt.tight_layout()
    return fig

def generate_pdf_report(image, analysis, chart):
    """Generate PDF report with image, analysis, and chart"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "AI Nutrition Report", ln=1, align="C")
        pdf.set_font("Arial", "", 12)
        
        # Add image if available
        if image:
            img_path = "temp_img.jpg"
            image.save(img_path, quality=90)
            pdf.image(img_path, w=180, h=120)
            os.remove(img_path)
            pdf.ln(10)
        
        # Add analysis text
        pdf.multi_cell(0, 8, analysis)
        pdf.ln(10)
        
        # Add chart if available
        if chart:
            chart_path = "temp_chart.png"
            chart.savefig(chart_path, bbox_inches='tight', dpi=100)
            pdf.image(chart_path, w=180)
            os.remove(chart_path)
        
        # Add footer
        pdf.set_y(-15)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 0, 'C')
        
        pdf_path = "nutrition_report.pdf"
        pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None

# ------------------------ Streamlit UI ------------------------ #
st.title("🍽️ AI-Powered Calorie Tracker")
st.caption("Upload food photos or describe meals to track your nutrition using AI")

# Initialize tabs
tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "📝 Text Input", "📊 History"])

# Image Analysis Tab
with tab1:
    st.subheader("Analyze Food Photos")
    img_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])
    context = st.text_area("Additional context", placeholder="E.g. This was my breakfast after workout")
    
    if st.button("Analyze Meal", disabled=not img_file):
        with st.spinner("Analyzing..."):
            try:
                image = Image.open(img_file)
                st.image(image, use_column_width=True)
                
                # Get image description
                description = describe_image(image)
                
                # Get nutritional analysis
                prompt = f"""You are a nutrition expert analyzing a meal based on its description and additional context provided by the user. Provide a detailed analysis that incorporates the context (e.g., meal timing, dietary preferences, activity level) to tailor your response. Follow this exact format:

**Food Items and Calories**:
- Item: [Food Name], Calories: [X] cal
- Item: [Food Name], Calories: [X] cal
**Total Calories**: [X] cal
**Nutritional Assessment**: [Detailed assessment of macronutrients, vitamins, and suitability for the user's context]
**Health Suggestions**: [2-3 tailored suggestions based on the meal and context]

Meal description: {description}
Additional context: {context or 'No additional context provided'}

If the user provides specific context (e.g., dietary goals, meal timing, or health conditions), emphasize relevant nutritional aspects (e.g., protein for post-workout, low-carb for keto diet) and adjust suggestions accordingly."""
                
                analysis = query_langchain(prompt)
                
                # Display results
                st.subheader("Nutritional Analysis")
                st.markdown(analysis)
                
                # Extract and display calories
                food_data = extract_items_and_calories(analysis)
                if food_data:
                    total_calories = sum(cal for _, cal in food_data)
                    st.metric("Total Estimated Calories", f"{total_calories} cal")
                    
                    # Plot chart
                    chart = plot_chart(food_data)
                    if chart:
                        st.pyplot(chart)
                
                # Save results
                st.session_state.last_results = {
                    "type": "image",
                    "image": image,
                    "description": description,  # Store description for follow-up
                    "context": context or "None",  # Store context for follow-up
                    "analysis": analysis,
                    "chart": chart if 'chart' in locals() else None,
                    "calories": total_calories if 'total_calories' in locals() else 0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                # Update history
                st.session_state.history.append(st.session_state.last_results)
                
                # Update daily calories
                today = date.today().isoformat()
                st.session_state.daily_calories[today] = st.session_state.daily_calories.get(today, 0) + st.session_state.last_results.get("calories", 0)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

    # Follow-up question section
    if st.session_state.last_results.get("type") == "image":
        st.subheader("Ask for More Details")
        follow_up_question = st.text_input("Ask a specific question about this meal", placeholder="E.g. How much protein is in this meal?", key="image_follow_up")
        if st.button("Get More Details", disabled=not follow_up_question, key="image_follow_up_button"):
            with st.spinner("Fetching details..."):
                follow_up_prompt = f"""Based on the following meal analysis, answer the user's specific question in detail. Provide a clear and concise response, focusing on the requested information.

Previous meal description: {st.session_state.last_results.get('description')}
Previous context: {st.session_state.last_results.get('context')}
Previous analysis: {st.session_state.last_results.get('analysis')}

User's question: {follow_up_question}"""
                follow_up_answer = query_langchain(follow_up_prompt)
                st.markdown("**Additional Details**:")
                st.markdown(follow_up_answer)

# Text Input Tab
with tab2:
    st.subheader("Describe Your Meal")
    meal_desc = st.text_area("Describe what you ate", placeholder="E.g. A large pepperoni pizza and soda")
    
    if st.button("Analyze Description"):
        with st.spinner("Analyzing..."):
            try:
                prompt = f"""You are a nutrition expert analyzing a meal based on the user's description. If the description includes additional details (e.g., portion sizes, meal timing, dietary preferences, or activity level), incorporate them into your analysis and provide tailored advice. Follow this exact format:

**Food Items and Calories**:
- Item: [Food Name], Calories: [X] cal
- Item: [Food Name], Calories: [X] cal
**Total Calories**: [X] cal
**Nutritional Assessment**: [Detailed assessment of macronutrients, vitamins, and suitability for the user's context]
**Health Suggestions**: [2-3 tailored suggestions based on the meal and any provided details]

Meal description: {meal_desc}

If the description includes specific details (e.g., 'post-workout meal' or 'large portion'), emphasize relevant nutritional aspects (e.g., protein for recovery, portion control) and adjust suggestions accordingly. If no specific details are provided, give a general but informative analysis."""
                
                analysis = query_langchain(prompt)
                
                # Display results
                st.subheader("Nutritional Analysis")
                st.markdown(analysis)
                
                # Extract and display calories
                food_data = extract_items_and_calories(analysis)
                if food_data:
                    total_calories = sum(cal for _, cal in food_data)
                    st.metric("Total Estimated Calories", f"{total_calories} cal")
                    
                    # Plot chart
                    chart = plot_chart(food_data)
                    if chart:
                        st.pyplot(chart)
                
                # Save results
                st.session_state.last_results = {
                    "type": "text",
                    "description": meal_desc,  # Store description for follow-up
                    "analysis": analysis,
                    "chart": chart if 'chart' in locals() else None,
                    "calories": total_calories if 'total_calories' in locals() else 0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                # Update history
                st.session_state.history.append(st.session_state.last_results)
                
                # Update daily calories
                today = date.today().isoformat()
                st.session_state.daily_calories[today] = st.session_state.daily_calories.get(today, 0) + st.session_state.last_results.get("calories", 0)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

    # Follow-up question section
    if st.session_state.last_results.get("type") == "text":
        st.subheader("Ask for More Details")
        follow_up_question = st.text_input("Ask a specific question about this meal", placeholder="E.g. Is this meal good for weight loss?", key="text_follow_up")
        if st.button("Get More Details", disabled=not follow_up_question, key="text_follow_up_button"):
            with st.spinner("Fetching details..."):
                follow_up_prompt = f"""Based on the following meal analysis, answer the user's specific question in detail. Provide a clear and concise response, focusing on the requested information.

Previous meal description: {st.session_state.last_results.get('description')}
Previous analysis: {st.session_state.last_results.get('analysis')}

User's question: {follow_up_question}"""
                follow_up_answer = query_langchain(follow_up_prompt)
                st.markdown("**Additional Details**:")
                st.markdown(follow_up_answer)

# History Tab
with tab3:
    st.subheader("Your Nutrition History")
    
    # Daily summary
    today = date.today().isoformat()
    today_cals = st.session_state.daily_calories.get(today, 0)
    st.metric("Today's Total Calories", f"{today_cals} cal")
    
    # Export button
    if st.session_state.last_results:
        if st.button("📄 Generate PDF Report"):
            pdf_path = generate_pdf_report(
                st.session_state.last_results.get("image"),
                st.session_state.last_results.get("analysis"),
                st.session_state.last_results.get("chart")
            )
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download Report",
                        f,
                        file_name="nutrition_report.pdf",
                        mime="application/pdf"
                    )
                os.remove(pdf_path)
    
    # History entries
    for i, entry in enumerate(reversed(st.session_state.history)):
        with st.expander(f"{entry['timestamp']} - {entry['type'].title()} Analysis"):
            if entry['type'] == "image" and entry.get("image"):
                st.image(entry["image"], width=200)
            
            st.markdown(entry["analysis"])
            
            if entry.get("calories", 0) > 0:
                st.metric("Estimated Calories", f"{entry['calories']} cal")
            
            if entry.get("chart"):
                st.pyplot(entry["chart"])

# Sidebar
with st.sidebar:
    st.header("Nutrition Dashboard")
    st.subheader("Weekly Summary")
    
    # Weekly calories chart
    if st.session_state.daily_calories:
        dates = sorted(st.session_state.daily_calories.keys())[-7:]  # Last 7 days
        cals = [st.session_state.daily_calories[d] for d in dates]
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(dates, cals, color='#4CAF50')
        ax.set_title("Daily Calories")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Clear history button
    if st.button("Clear All History"):
        st.session_state.history.clear()
        st.session_state.daily_calories.clear()
        st.session_state.last_results = {}
        st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ by <b>Ujjwal Sinha</b> • "
    "<a href='https://github.com/Ujjwal-sinha' target='_blank'>GitHub</a></p>",
    unsafe_allow_html=True
)