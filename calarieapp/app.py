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
st.set_page_config(
    page_title="🍱 AI Calorie Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🍽️"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* General styling */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f5f7fa;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #4CAF50;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 8px;
        padding: 10px;
    }
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
        padding: 20px;
        border-radius: 10px;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .footer {
        text-align: center;
        padding: 20px;
        background-color: #e9ecef;
        border-radius: 8px;
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTextInput input, .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ced4da;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Activity calorie burn rates (kcal/hour for average adult)
ACTIVITY_BURN_RATES = {
    "Brisk Walking": 300,
    "Running": 600,
    "Cycling": 500,
    "Swimming": 550,
    "Strength Training": 400
}

# ------------------------ Model Initialization ------------------------ #
@st.cache_resource
def load_models():
    """Initialize all AI models with proper device handling"""
    models = {}
    try:
        models['llm'] = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        models['llm'] = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        models['processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        models['blip_model'] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device).eval()
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
if "calorie_target" not in st.session_state:
    st.session_state.calorie_target = 2000
if "activity_preference" not in st.session_state:
    st.session_state.activity_preference = ["Brisk Walking"]

# ------------------------ Helper Functions ------------------------ #
def query_langchain(prompt):
    if not models['llm']:
        return "LLM service unavailable"
    try:
        response = models['llm']([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

def describe_image(image: Image.Image) -> str:
    if not models['processor'] or not models['blip_model']:
        return "Image analysis unavailable"
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        device = next(models['blip_model'].parameters()).device
        inputs = models['processor'](image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = models['blip_model'].generate(**inputs, max_new_tokens=50)
        caption = models['processor'].decode(outputs[0], skip_special_tokens=True)
        if any(phrase in caption.lower() for phrase in ["plate of food", "meal", "food item"]):
            return f"Vague caption detected: '{caption}'. Please provide more context or a clearer image."
        return caption
    except Exception as e:
        return f"Image analysis error: {str(e)}"

def extract_items_and_nutrients(text):
    items = []
    pattern = r'Item:\s*([^,]+),\s*Calories:\s*(\d{1,4})\s*(?:cal|kcal|calories)?(?:,\s*Protein:\s*(\d+\.?\d*)\s*g)?(?:,\s*Carbs:\s*(\d+\.?\d*)\s*g)?(?:,\s*Fats:\s*(\d+\.?\d*)\s*g)?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        item = match[0].strip()
        calories = int(match[1]) if match[1] else 0
        protein = float(match[2]) if match[2] else None
        carbs = float(match[3]) if match[3] else None
        fats = float(match[4]) if match[4] else None
        items.append({
            "item": item,
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fats": fats
        })
    totals = {
        "calories": sum(item["calories"] for item in items),
        "protein": sum(item["protein"] for item in items if item["protein"] is not None),
        "carbs": sum(item["carbs"] for item in items if item["carbs"] is not None),
        "fats": sum(item["fats"] for item in items if item["fats"] is not None)
    }
    return items, totals

def plot_chart(food_data):
    if not food_data:
        return None
    items = [item["item"] for item in food_data]
    calories = [item["calories"] for item in food_data]
    proteins = [item["protein"] if item["protein"] is not None else 0 for item in food_data]
    carbs = [item["carbs"] if item["carbs"] is not None else 0 for item in food_data]
    fats = [item["fats"] if item["fats"] is not None else 0 for item in food_data]
    
    fig, ax = plt.subplots(figsize=(8, len(items) * 0.6))
    bar_width = 0.2
    indices = range(len(items))
    
    ax.barh([i - bar_width*1.5 for i in indices], calories, bar_width, label="Calories (kcal)", color="#4CAF50")
    ax.barh([i - bar_width*0.5 for i in indices], proteins, bar_width, label="Protein (g)", color="#2196F3")
    ax.barh([i + bar_width*0.5 for i in indices], carbs, bar_width, label="Carbs (g)", color="#FF9800")
    ax.barh([i + bar_width*1.5 for i in indices], fats, bar_width, label="Fats (g)", color="#F44336")
    
    ax.set_yticks(indices)
    ax.set_yticklabels(items, fontsize=10)
    ax.set_xlabel("Amount", fontsize=12)
    ax.set_title("Nutritional Breakdown", fontsize=14, pad=15)
    ax.legend(fontsize=10)
    plt.style.use('seaborn')
    plt.tight_layout()
    return fig

def generate_daily_summary(calorie_target, activity_preferences):
    today = date.today().isoformat()
    total_calories = st.session_state.daily_calories.get(today, 0)
    daily_nutrients = {"protein": 0, "carbs": 0, "fats": 0}
    for entry in st.session_state.history:
        entry_date = entry["timestamp"].split()[0]
        if entry_date == today and entry.get("totals"):
            daily_nutrients["protein"] += entry["totals"].get("protein", 0)
            daily_nutrients["carbs"] += entry["totals"].get("carbs", 0)
            daily_nutrients["fats"] += entry["totals"].get("fats", 0)
    
    calorie_diff = total_calories - calorie_target
    status = "surplus" if calorie_diff > 0 else "deficit" if calorie_diff < 0 else "balanced"
    
    summary = f"**Daily Nutritional Summary ({today})**\n"
    summary += f"- **Total Calories**: {total_calories} kcal (Target: {calorie_target} kcal)\n"
    summary += f"- **Total Protein**: {daily_nutrients['protein']:.1f} g\n"
    summary += f"- **Total Carbs**: {daily_nutrients['carbs']:.1f} g\n"
    summary += f"- **Total Fats**: {daily_nutrients['fats']:.1f} g\n"
    summary += f"- **Calorie Status**: {'Surplus' if calorie_diff > 0 else 'Deficit' if calorie_diff < 0 else 'Balanced'} ({abs(calorie_diff)} kcal)\n\n"
    
    advice = "**Personalized Fitness Advice**\n"
    if status == "surplus":
        advice += f"You consumed {calorie_diff} kcal above your target. To balance this, consider:\n"
        for activity in activity_preferences:
            burn_rate = ACTIVITY_BURN_RATES.get(activity, 300)
            duration = (calorie_diff / burn_rate) * 60
            advice += f"- **{activity}**: {duration:.0f} minutes\n"
        advice += "\n**Motivation**: Great job tracking your intake! A short workout can help you stay on track!"
    elif status == "deficit":
        advice += f"You consumed {abs(calorie_diff)} kcal below your target. To avoid excessive deficit:\n"
        advice += "- Consider a nutrient-dense snack (e.g., banana with peanut butter, ~200-300 kcal).\n"
        advice += "- Ensure adequate hydration and rest.\n"
        advice += "\n**Motivation**: You're doing awesome! Fuel your body for your goals!"
    else:
        advice += "Your calorie intake is perfectly balanced! Keep it up:\n"
        advice += "- Maintain a mix of activities.\n"
        advice += "- Stay consistent with nutrition.\n"
        advice += "\n**Motivation**: You're in the zone! Keep making mindful choices!"
    
    return summary + advice

def generate_pdf_report(image, analysis, chart, nutrients, daily_summary=None):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "AI Nutrition Report", ln=1, align="C")
        pdf.set_font("Arial", "", 12)
        
        if image:
            img_path = "temp_img.jpg"
            image.save(img_path, quality=90)
            pdf.image(img_path, w=180, h=120)
            os.remove(img_path)
            pdf.ln(10)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Nutritional Summary", ln=1)
        pdf.set_font("Arial", "", 10)
        pdf.cell(50, 8, "Food Item", border=1)
        pdf.cell(30, 8, "Calories", border=1)
        pdf.cell(30, 8, "Protein (g)", border=1)
        pdf.cell(30, 8, "Carbs (g)", border=1)
        pdf.cell(30, 8, "Fats (g)", border=1)
        pdf.ln()
        
        for item in nutrients:
            pdf.cell(50, 8, item["item"], border=1)
            pdf.cell(30, 8, str(item["calories"]), border=1)
            pdf.cell(30, 8, str(item["protein"] or "-"), border=1)
            pdf.cell(30, 8, str(item["carbs"] or "-"), border=1)
            pdf.cell(30, 8, str(item["fats"] or "-"), border=1)
            pdf.ln()
        
        pdf.ln(10)
        pdf.multi_cell(0, 8, analysis)
        pdf.ln(10)
        
        if daily_summary:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Daily Summary", ln=1)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 8, daily_summary)
            pdf.ln(10)
        
        if chart:
            chart_path = "temp_chart.png"
            chart.savefig(chart_path, bbox_inches="tight", dpi=100)
            pdf.image(chart_path, w=180)
            os.remove(chart_path)
        
        pdf.set_y(-15)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 0, "C")
        
        pdf_path = "nutrition_report.pdf"
        pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None

# ------------------------ Streamlit UI ------------------------ #
with st.container():
    st.title("🍽️ AI-Powered Calorie Tracker")
    st.caption("Track your nutrition with AI-powered image analysis or manual input")

    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "📝 Manual Input", "📊 History"])

    # Image Analysis Tab
    with tab1:
        st.subheader("📷 Analyze Food Photos")
        with st.container():
            img_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"], key="img_uploader")
            context = st.text_area(
                "Additional Context (Optional)",
                placeholder="E.g., Identify each item in my meal or specify dietary preferences",
                height=100
            )
            
            if st.button("🔍 Analyze Meal", disabled=not img_file, key="analyze_image"):
                with st.spinner("Analyzing your meal..."):
                    try:
                        if img_file is None or getattr(img_file, 'size', 1) == 0:
                            st.error("File upload failed. Please refresh and re-upload your image.")
                            st.stop()
                        image = Image.open(img_file)
                        st.image(image, caption="Uploaded Meal", use_column_width=True, clamp=True)
                        
                        description = describe_image(image)
                        if "Vague caption detected" in description:
                            st.warning(description)
                            st.stop()
                        
                        prompt = f"""You are a nutrition expert analyzing a meal based on its description and additional context provided by the user. Provide a detailed analysis that incorporates the context (e.g., meal timing, dietary preferences, activity level, or specific requests like identifying all items). Follow this exact format:

**Food Items and Nutrients**:
- Item: [Food Name], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- Item: [Food Name], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
**Total Calories**: [X] cal
**Nutritional Assessment**: [Detailed assessment of macronutrients, vitamins, and suitability for the user's context]
**Health Suggestions**: [2-3 tailored suggestions based on the meal and context]

Meal description: {description}
Additional context: {context or 'No additional context provided'}

Instructions:
1. If the context includes a request to "identify each and every item," list all visible food items in the description individually, specifying estimated portion sizes (e.g., "Grilled Chicken Breast (200g)").
2. For each food item, provide estimated calories, protein, carbs, and fats based on typical nutritional values, even if exact data is unavailable. Do not omit macronutrients.
3. If the meal description is vague (e.g., "a plate of food"), make reasonable assumptions about common food items and their portions, and list them explicitly.
4. Incorporate the context to emphasize relevant nutritional aspects (e.g., high protein for post-workout, low-carb for keto diet) and tailor health suggestions accordingly.
5. Ensure the total calories match the sum of individual item calories.
6. Strictly adhere to the specified format to ensure compatibility with parsing logic."""
                        
                        analysis = query_langchain(prompt)
                        food_data, totals = extract_items_and_nutrients(analysis)
                        
                        if not food_data or any(item["protein"] is None or item["carbs"] is None or item["fats"] is None for item in food_data):
                            st.warning("Incomplete data detected. Retrying with stricter instructions...")
                            prompt += "\nPlease strictly follow the format, listing all food items individually with estimated portion sizes and complete macronutrient data (calories, protein, carbs, fats)."
                            analysis = query_langchain(prompt)
                            food_data, totals = extract_items_and_nutrients(analysis)
                        
                        st.subheader("🍴 Nutritional Analysis")
                        st.markdown(analysis, unsafe_allow_html=True)
                        
                        if food_data:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Calories", f"{totals['calories']} kcal", delta=f"{totals['calories']-st.session_state.calorie_target} kcal")
                            col2.metric("Protein", f"{totals['protein']:.1f} g" if totals['protein'] else "-")
                            col3.metric("Carbs", f"{totals['carbs']:.1f} g" if totals['carbs'] else "-")
                            col4.metric("Fats", f"{totals['fats']:.1f} g" if totals['fats'] else "-")
                            
                            chart = plot_chart(food_data)
                            if chart:
                                st.pyplot(chart)
                        else:
                            st.error("Failed to extract food items. Try a clearer image or more specific context.")
                        
                        st.session_state.last_results = {
                            "type": "image",
                            "image": image,
                            "description": description,
                            "context": context or "None",
                            "analysis": analysis,
                            "chart": chart if 'chart' in locals() else None,
                            "nutrients": food_data,
                            "totals": totals,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        st.session_state.history.append(st.session_state.last_results)
                        today = date.today().isoformat()
                        st.session_state.daily_calories[today] = st.session_state.daily_calories.get(today, 0) + totals["calories"]
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            
            # Follow-up question section
            if st.session_state.last_results.get("type") == "image":
                st.subheader("❓ Ask for More Details")
                follow_up_question = st.text_input(
                    "Ask about this meal",
                    placeholder="E.g., How much protein is in this meal?",
                    key="image_follow_up"
                )
                if st.button("🔎 Get Details", disabled=not follow_up_question, key="image_follow_up_button"):
                    with st.spinner("Fetching details..."):
                        follow_up_prompt = f"""Based on the following meal analysis, answer the user's specific question in detail. Provide a clear and concise response, focusing on the requested information.

Previous meal description: {st.session_state.last_results.get('description')}
Previous context: {st.session_state.last_results.get('context')}
Previous analysis: {st.session_state.last_results.get('analysis')}

User's question: {follow_up_question}"""
                        follow_up_answer = query_langchain(follow_up_prompt)
                        st.markdown(f"**Additional Details**:\n{follow_up_answer}")

    # Text Input Tab
    with tab2:
        st.subheader("📝 Describe Your Meal")
        with st.container():
            meal_desc = st.text_area(
                "Describe what you ate",
                placeholder="E.g., A large pepperoni pizza and soda",
                height=100
            )
            
            if st.button("🔍 Analyze Description", key="analyze_text"):
                with st.spinner("Analyzing your description..."):
                    try:
                        prompt = f"""You are a nutrition expert analyzing a meal based on the user's description. If the description includes additional details (e.g., portion sizes, meal timing, dietary preferences, or activity level), incorporate them into your analysis and provide tailored advice. Follow this exact format:

**Food Items and Nutrients**:
- Item: [Food Name], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats

: [X] g
- Item: [Food Name], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
**Total Calories**: [X] cal
**Nutritional Assessment**: [Detailed assessment of macronutrients, vitamins, and suitability for the user's context]
**Health Suggestions**: [2-3 tailored suggestions based on the meal and any provided details]

Meal description: {meal_desc}

If the description includes specific details (e.g., 'post-workout meal' or 'large portion'), emphasize relevant nutritional aspects (e.g., protein for recovery, portion control) and adjust suggestions accordingly. If no specific details are provided, give a general but informative analysis. Estimate macronutrients based on typical values if not specified."""
                        
                        analysis = query_langchain(prompt)
                        st.subheader("🍴 Nutritional Analysis")
                        st.markdown(analysis, unsafe_allow_html=True)
                        
                        food_data, totals = extract_items_and_nutrients(analysis)
                        if food_data:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Calories", f"{totals['calories']} kcal", delta=f"{totals['calories']-st.session_state.calorie_target} kcal")
                            col2.metric("Protein", f"{totals['protein']:.1f} g" if totals['protein'] else "-")
                            col3.metric("Carbs", f"{totals['carbs']:.1f} g" if totals['carbs'] else "-")
                            col4.metric("Fats", f"{totals['fats']:.1f} g" if totals['fats'] else "-")
                            
                            chart = plot_chart(food_data)
                            if chart:
                                st.pyplot(chart)
                        
                        st.session_state.last_results = {
                            "type": "text",
                            "description": meal_desc,
                            "analysis": analysis,
                            "chart": chart if 'chart' in locals() else None,
                            "nutrients": food_data,
                            "totals": totals,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        st.session_state.history.append(st.session_state.last_results)
                        today = date.today().isoformat()
                        st.session_state.daily_calories[today] = st.session_state.daily_calories.get(today, 0) + totals["calories"]
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            
            if st.session_state.last_results.get("type") == "text":
                st.subheader("❓ Ask for More Details")
                follow_up_question = st.text_input(
                    "Ask about this meal",
                    placeholder="E.g., Is this meal good for weight loss?",
                    key="text_follow_up"
                )
                if st.button("🔎 Get Details", disabled=not follow_up_question, key="text_follow_up_button"):
                    with st.spinner("Fetching details..."):
                        follow_up_prompt = f"""Based on the following meal analysis, answer the user's specific question in detail. Provide a clear and concise response, focusing on the requested information.

Previous meal description: {st.session_state.last_results.get('description')}
Previous analysis: {st.session_state.last_results.get('analysis')}

User's question: {follow_up_question}"""
                        follow_up_answer = query_langchain(follow_up_prompt)
                        st.markdown(f"**Additional Details**:\n{follow_up_answer}")

    # History Tab
    with tab3:
        st.subheader("📊 Your Nutrition History")
        with st.container():
            calorie_target = st.session_state.calorie_target
            activity_preference = st.session_state.activity_preference
            if st.button("📅 Generate Daily Summary", key="daily_summary"):
                daily_summary = generate_daily_summary(calorie_target, activity_preference)
                st.markdown(daily_summary, unsafe_allow_html=True)
                if st.session_state.last_results:
                    include_summary_in_pdf = st.checkbox("Include Daily Summary in PDF Report")
                    if include_summary_in_pdf:
                        st.session_state.last_results["daily_summary"] = daily_summary
            
            today = date.today().isoformat()
            today_cals = st.session_state.daily_calories.get(today, 0)
            st.metric("Today's Total Calories", f"{today_cals} kcal", delta=f"{today_cals - calorie_target} kcal")
            
            if st.session_state.last_results:
                if st.button("📄 Export PDF Report", key="export_pdf"):
                    pdf_path = generate_pdf_report(
                        st.session_state.last_results.get("image"),
                        st.session_state.last_results.get("analysis"),
                        st.session_state.last_results.get("chart"),
                        st.session_state.last_results.get("nutrients", []),
                        st.session_state.last_results.get("daily_summary")
                    )
                    if pdf_path:
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "Download PDF Report",
                                f,
                                file_name="nutrition_report.pdf",
                                mime="application/pdf",
                                key="download_pdf"
                            )
                        os.remove(pdf_path)
            
            for i, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"📅 {entry['timestamp']} - {entry['type'].title()} Analysis"):
                    if entry['type'] == "image" and entry.get("image"):
                        st.image(entry["image"], caption="Meal Image", width=300)
                    
                    st.markdown(entry["analysis"], unsafe_allow_html=True)
                    
                    if entry.get("totals", {}):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Calories", f"{entry['totals']['calories']} kcal")
                        col2.metric("Protein", f"{entry['totals']['protein']:.1f} g" if entry['totals']['protein'] else "-")
                        col3.metric("Carbs", f"{entry['totals']['carbs']:.1f} g" if entry['totals']['carbs'] else "-")
                        col4.metric("Fats", f"{entry['totals']['fats']:.1f} g" if entry['totals']['fats'] else "-")
                    
                    if entry.get("chart"):
                        st.pyplot(entry["chart"])

# Sidebar
with st.sidebar:
    st.header("🍎 Nutrition Dashboard")
    
    st.subheader("User Profile")
    calorie_target = st.number_input(
        "Daily Calorie Target (kcal)",
        min_value=1000,
        max_value=5000,
        value=st.session_state.calorie_target,
        step=100,
        key="calorie_target"
    )
    activity_preference = st.multiselect(
        "Preferred Activities",
        options=["Brisk Walking", "Running", "Cycling", "Swimming", "Strength Training"],
        default=st.session_state.activity_preference,
        key="activity_preference"
    )
    
    # Calorie Progress Bar
    today = date.today().isoformat()
    today_cals = st.session_state.daily_calories.get(today, 0)
    progress = min(today_cals / calorie_target, 1.0) if calorie_target > 0 else 0
    st.progress(progress)
    st.caption(f"Progress: {today_cals}/{calorie_target} kcal ({progress*100:.1f}%)")
    
    st.subheader("📈 Weekly Summary")
    if st.session_state.daily_calories:
        dates = sorted(st.session_state.daily_calories.keys())[-7:]
        cals = [st.session_state.daily_calories.get(d, 0) for d in dates]
        daily_nutrients = {d: {"protein": 0, "carbs": 0, "fats": 0} for d in dates}
        for entry in st.session_state.history:
            entry_date = entry["timestamp"].split()[0]
            if entry_date in dates and entry.get("totals"):
                daily_nutrients[entry_date]["protein"] += entry["totals"].get("protein", 0)
                daily_nutrients[entry_date]["carbs"] += entry["totals"].get("carbs", 0)
                daily_nutrients[entry_date]["fats"] += entry["totals"].get("fats", 0)
        
        proteins = [daily_nutrients[d]["protein"] for d in dates]
        carbs = [daily_nutrients[d]["carbs"] for d in dates]
        fats = [daily_nutrients[d]["fats"] for d in dates]
        
        fig, ax = plt.subplots(figsize=(6, 3))
        bar_width = 0.2
        indices = range(len(dates))
        
        ax.bar([i - bar_width*1.5 for i in indices], cals, bar_width, label="Calories (kcal)", color="#4CAF50")
        ax.bar([i - bar_width*0.5 for i in indices], proteins, bar_width, label="Protein (g)", color="#2196F3")
        ax.bar([i + bar_width*0.5 for i in indices], carbs, bar_width, label="Carbs (g)", color="#FF9800")
        ax.bar([i + bar_width*1.5 for i in indices], fats, bar_width, label="Fats (g)", color="#F44336")
        
        ax.set_xticks(indices)
        ax.set_xticklabels(dates, rotation=45, fontsize=8)
        ax.set_ylabel("Amount", fontsize=10)
        ax.set_title("Weekly Nutrition", fontsize=12)
        ax.legend(fontsize=8)
        plt.style.use('seaborn')
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Last 7 days' nutrition trends")
    
    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.history.clear()
        st.session_state.daily_calories.clear()
        st.session_state.last_results = {}
        st.rerun()

# Footer
st.markdown("""
<div class='footer'>
    <p>Built with ❤️ by <b>Ujjwal Sinha</b> • 
    <a href='https://github.com/Ujjwal-sinha' target='_blank'>GitHub</a> • 
</div>
""", unsafe_allow_html=True)