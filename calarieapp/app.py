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
            low_cpu_mem_usage=True
        )
        
        # Move model to device after loading
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

def extract_items_and_nutrients(text):
    """Extract food items, calories, and macronutrients from text"""
    items = []
    # Capture food items and their nutritional data
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
    
    # Calculate totals
    totals = {
        "calories": sum(item["calories"] for item in items),
        "protein": sum(item["protein"] for item in items if item["protein"] is not None),
        "carbs": sum(item["carbs"] for item in items if item["carbs"] is not None),
        "fats": sum(item["fats"] for item in items if item["fats"] is not None)
    }
    
    return items, totals

def plot_chart(food_data):
    """Create a side-by-side bar chart for calories and macronutrients"""
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
    
    # Plot bars
    ax.barh([i - bar_width*1.5 for i in indices], calories, bar_width, label="Calories (kcal)", color="#4CAF50")
    ax.barh([i - bar_width*0.5 for i in indices], proteins, bar_width, label="Protein (g)", color="#2196F3")
    ax.barh([i + bar_width*0.5 for i in indices], carbs, bar_width, label="Carbs (g)", color="#FF9800")
    ax.barh([i + bar_width*1.5 for i in indices], fats, bar_width, label="Fats (g)", color="#F44336")
    
    ax.set_yticks(indices)
    ax.set_yticklabels(items)
    ax.set_xlabel("Amount")
    ax.set_title("Nutritional Breakdown")
    ax.legend()
    plt.tight_layout()
    return fig

def generate_pdf_report(image, analysis, chart, nutrients):
    """Generate PDF report with image, analysis, chart, and nutrient table"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "AI Nutrition Report", ln=1, align="C")
        pdf.set_font("Arial", "", 12)
        
        # Add image
        if image:
            img_path = "temp_img.jpg"
            image.save(img_path, quality=90)
            pdf.image(img_path, w=180, h=120)
            os.remove(img_path)
            pdf.ln(10)
        
        # Add nutrient table
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
        
        # Add analysis
        pdf.multi_cell(0, 8, analysis)
        pdf.ln(10)
        
        # Add chart
        if chart:
            chart_path = "temp_chart.png"
            chart.savefig(chart_path, bbox_inches="tight", dpi=100)
            pdf.image(chart_path, w=180)
            os.remove(chart_path)
        
        # Add footer
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
st.title("🍽️ AI-Powered Calorie Tracker")
st.caption("Upload food photos or describe meals to track your nutrition using AI")

# Initialize tabs
tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "📝 Text Input", "📊 History"])

# Image Analysis Tab
# Image Analysis Tab
with tab1:
    st.subheader("Analyze Food Photos")
    img_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])
    context = st.text_area("Additional context", placeholder="E.g. Identify each and every item in my food and give total calorie")
    
    if st.button("Analyze Meal", disabled=not img_file):
        with st.spinner("Analyzing..."):
            try:
                image = Image.open(img_file)
                st.image(image, use_column_width=True)
                
                # Get image description
                description = describe_image(image)
                
                # Updated prompt
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
                
                # Extract and validate nutrients
                food_data, totals = extract_items_and_nutrients(analysis)
                
                # Fallback if no items or incomplete nutrients
                if not food_data or any(item["protein"] is None or item["carbs"] is None or item["fats"] is None for item in food_data):
                    st.warning("Incomplete or no food items detected. Retrying with stricter instructions...")
                    prompt += "\nPlease strictly follow the format, listing all food items individually with estimated portion sizes and complete macronutrient data (calories, protein, carbs, fats)."
                    analysis = query_langchain(prompt)
                    food_data, totals = extract_items_and_nutrients(analysis)
                
                # Display results
                st.subheader("Nutritional Analysis")
                st.markdown(analysis)
                
                # Extract and display nutrients
                if food_data:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Calories", f"{totals['calories']} cal")
                    col2.metric("Total Protein", f"{totals['protein']:.1f} g" if totals['protein'] else "-")
                    col3.metric("Total Carbs", f"{totals['carbs']:.1f} g" if totals['carbs'] else "-")
                    col4.metric("Total Fats", f"{totals['fats']:.1f} g" if totals['fats'] else "-")
                    
                    # Plot chart
                    chart = plot_chart(food_data)
                    if chart:
                        st.pyplot(chart)
                else:
                    st.error("Failed to extract food items. Please try a different image or provide more specific context.")
                
                # Save results
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
                
                # Update history
                st.session_state.history.append(st.session_state.last_results)
                
                # Update daily calories
                today = date.today().isoformat()
                st.session_state.daily_calories[today] = st.session_state.daily_calories.get(today, 0) + totals["calories"]
                
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

**Food Items and Nutrients**:
- Item: [Food Name], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- Item: [Food Name], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
**Total Calories**: [X] cal
**Nutritional Assessment**: [Detailed assessment of macronutrients, vitamins, and suitability for the user's context]
**Health Suggestions**: [2-3 tailored suggestions based on the meal and any provided details]

Meal description: {meal_desc}

If the description includes specific details (e.g., 'post-workout meal' or 'large portion'), emphasize relevant nutritional aspects (e.g., protein for recovery, portion control) and adjust suggestions accordingly. If no specific details are provided, give a general but informative analysis. Estimate macronutrients based on typical values if not specified."""
                
                analysis = query_langchain(prompt)
                
                # Display results
                st.subheader("Nutritional Analysis")
                st.markdown(analysis)
                
                # Extract and display nutrients
                food_data, totals = extract_items_and_nutrients(analysis)
                if food_data:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Calories", f"{totals['calories']} cal")
                    col2.metric("Total Protein", f"{totals['protein']:.1f} g" if totals['protein'] else "-")
                    col3.metric("Total Carbs", f"{totals['carbs']:.1f} g" if totals['carbs'] else "-")
                    col4.metric("Total Fats", f"{totals['fats']:.1f} g" if totals['fats'] else "-")
                    
                    # Plot chart
                    chart = plot_chart(food_data)
                    if chart:
                        st.pyplot(chart)
                
                # Save results
                st.session_state.last_results = {
                    "type": "text",
                    "description": meal_desc,
                    "analysis": analysis,
                    "chart": chart if 'chart' in locals() else None,
                    "nutrients": food_data,
                    "totals": totals,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                # Update history
                st.session_state.history.append(st.session_state.last_results)
                
                # Update daily calories
                today = date.today().isoformat()
                st.session_state.daily_calories[today] = st.session_state.daily_calories.get(today, 0) + totals["calories"]
                
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
                st.session_state.last_results.get("chart"),
                st.session_state.last_results.get("nutrients", [])
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
            
            if entry.get("totals", {}):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Calories", f"{entry['totals']['calories']} cal")
                col2.metric("Protein", f"{entry['totals']['protein']:.1f} g" if entry['totals']['protein'] else "-")
                col3.metric("Carbs", f"{entry['totals']['carbs']:.1f} g" if entry['totals']['carbs'] else "-")
                col4.metric("Fats", f"{entry['totals']['fats']:.1f} g" if entry['totals']['fats'] else "-")
            
            if entry.get("chart"):
                st.pyplot(entry["chart"])

with st.sidebar:
    st.header("Nutrition Dashboard")
    st.subheader("Weekly Summary")
    
    # Weekly nutrients bar graph
    if st.session_state.daily_calories:
        dates = sorted(st.session_state.daily_calories.keys())[-7:]  # Last 7 days
        cals = [st.session_state.daily_calories.get(d, 0) for d in dates]
        
        # Aggregate macronutrients from history
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
        
        # Create bar graph
        fig, ax = plt.subplots(figsize=(8, 4))
        bar_width = 0.2
        indices = range(len(dates))
        
        # Plot bars
        ax.bar([i - bar_width*1.5 for i in indices], cals, bar_width, label="Calories (kcal)", color="#4CAF50")
        ax.bar([i - bar_width*0.5 for i in indices], proteins, bar_width, label="Protein (g)", color="#2196F3")
        ax.bar([i + bar_width*0.5 for i in indices], carbs, bar_width, label="Carbs (g)", color="#FF9800")
        ax.bar([i + bar_width*1.5 for i in indices], fats, bar_width, label="Fats (g)", color="#F44336")
        
        ax.set_xticks(indices)
        ax.set_xticklabels(dates, rotation=45)
        ax.set_ylabel("Amount")
        ax.set_title("Daily Nutrition Trends")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)  # Removed caption parameter
        st.caption("Weekly nutrition summary (last 7 days)")  # Added caption separately
    
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