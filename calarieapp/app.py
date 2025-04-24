import streamlit as st
from PIL import Image
import base64
from dotenv import load_dotenv
import os
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from transformers import pipeline
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import re


# ------------------------ Setup ------------------------ #
st.set_page_config(page_title="🍱 Calorie Tracker with LangChain", layout="centered")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=groq_api_key
)

captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------ Title ------------------------ #
st.title("🍽️ AI-Powered Calorie Tracker")
st.caption("Upload a food photo, scan a barcode, or log exercise to track your calories using AI!")

# ------------------------ Helper Functions ------------------------ #
def query_langchain(prompt):
    response = llm([HumanMessage(content=prompt)])
    return response.content

def describe_image(image: Image.Image) -> str:
    results = captioner(image)
    return results[0]['generated_text']

def extract_items_and_calories(text):
    pattern = r'(\b[\w\s]+\b)[^\d]*(\d{2,4})\s*cal'
    return re.findall(pattern, text, re.IGNORECASE)

def plot_chart(food_data):
    items, calories = zip(*[(item.strip(), int(cal)) for item, cal in food_data])
    fig, ax = plt.subplots()
    ax.barh(items, calories, color='lightgreen')
    ax.set_xlabel("Calories")
    ax.set_title("Calorie Breakdown")
    st.pyplot(fig)
    return fig

def generate_pdf(image, summary, chart_fig):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "AI-Powered Calorie Report", ln=1, align="C")

    if image:
        img_path = "temp_image.jpg"
        image.save(img_path)
        pdf.image(img_path, w=100)
    
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, summary)

    chart_buf = io.BytesIO()
    chart_fig.savefig(chart_buf, format='png')
    chart_buf.seek(0)
    chart_path = "chart.png"
    with open(chart_path, "wb") as f:
        f.write(chart_buf.getvalue())
    pdf.image(chart_path, w=100)

    output_path = "Calorie_Report.pdf"
    pdf.output(output_path)
    return output_path

# ------------------------ Tabs ------------------------ #
tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "📦 Barcode Scan", "🏃 Exercise Log"])

# ------------------------ Tab 1: Image Analysis ------------------------ #
with tab1:
    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
    user_input = st.text_input("Add extra context (optional)", placeholder="e.g. I ate this after gym")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("📊 Analyze Calories"):
        if uploaded_file:
            try:
                description = describe_image(image)
                prompt = f"""
You are a nutritionist AI. Given the following meal description, estimate:
1. Food items with estimated calories
2. Total calorie count
3. Health suggestions

Meal: {description}
Extra context: {user_input}
"""
                with st.spinner("Analyzing with LangChain..."):
                    output_text = query_langchain(prompt)
                    st.success("✅ Analysis Complete!")
                    st.markdown("### 🍱 Nutritional Breakdown")
                    st.markdown(output_text)

                    st.session_state.last_image = image
                    st.session_state.last_summary = output_text

                    food_data = extract_items_and_calories(output_text)
                    if food_data:
                        chart_fig = plot_chart(food_data)
                        st.session_state.last_chart = chart_fig
                    else:
                        st.warning("⚠️ Couldn’t extract individual food items for the chart.")

                    st.session_state.history.append({
                        "timestamp": str(datetime.now()),
                        "type": "Image Analysis",
                        "data": output_text
                    })

            except Exception as e:
                st.error(f"Error: {e}")

    # PDF Export
    if "last_summary" in st.session_state and st.button("📤 Export PDF Report"):
        try:
            pdf_path = generate_pdf(
                st.session_state.last_image,
                st.session_state.last_summary,
                st.session_state.get("last_chart", None)
            )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Download PDF",
                    data=f,
                    file_name="Calorie_Report.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

# ------------------------ Tab 2: Barcode Scan ------------------------ #
with tab2:
    barcode = st.text_input("Enter barcode or product code")
    if st.button("🔍 Get Nutrition Info"):
        if barcode:
            prompt = f"You are a barcode nutrition assistant. Provide the nutritional value and calorie breakdown of the product with barcode: {barcode}."
            with st.spinner("Querying LangChain..."):
                try:
                    output_text = query_langchain(prompt)
                    st.success("✅ Retrieved Info")
                    st.markdown(output_text)
                    st.session_state.history.append({
                        "timestamp": str(datetime.now()),
                        "type": "Barcode Scan",
                        "data": output_text
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

# ------------------------ Tab 3: Exercise Log ------------------------ #
with tab3:
    activity = st.text_input("Describe your activity", placeholder="e.g., 30 minutes of jogging")
    if st.button("🔥 Estimate Calories Burned"):
        if activity:
            prompt = f"Estimate the calories burned during the following activity: {activity}"
            with st.spinner("Estimating..."):
                try:
                    output_text = query_langchain(prompt)
                    st.success("✅ Estimated Successfully")
                    st.markdown(output_text)
                    st.session_state.history.append({
                        "timestamp": str(datetime.now()),
                        "type": "Exercise Log",
                        "data": output_text
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

# ------------------------ History in Sidebar ------------------------ #
st.sidebar.title("📅 Daily Log")
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history.clear()

for entry in reversed(st.session_state.history):
    st.sidebar.markdown(f"**[{entry['timestamp']}] {entry['type']}**\n{entry['data'][:100]}...")


# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ by <b>Ujjwal Sinha</b> • "
    "<a href='https://github.com/Ujjwal-sinha' target='_blank'>GitHub</a></p>",
    unsafe_allow_html=True
)