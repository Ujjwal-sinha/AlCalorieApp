# 🍱 AI Calorie Tracker

A Streamlit-based web application that leverages AI to analyze food images or text descriptions, providing detailed nutritional breakdowns, calorie tracking, and personalized health suggestions. Powered by the BLIP model for image captioning and Groq's LLaMA3-8b for nutritional analysis, this app helps users monitor their diet with ease.

---

## Features

- **Image Analysis:** Upload food photos to receive a nutritional breakdown including calories, protein, carbs, and fats.
- **Text Input:** Describe meals manually for a detailed nutritional analysis.
- **History Tracking:** View past meal analyses and monitor daily calorie intake.
- **PDF Reports:** Generate downloadable reports with nutritional summaries, charts, and images.
- **Weekly Dashboard:** Visualize daily trends for calories and macronutrients.
- **Follow-Up Questions:** Ask specific questions about meals for deeper insights.
- **Responsive UI:** Intuitive tabs, charts, and metrics for a seamless user experience.

---

## Tech Stack

- **Frontend:** Streamlit
- **AI Models:**
  - BLIP (Salesforce/blip-image-captioning-base) for image captioning
  - Groq (LLaMA3-8b-8192) for nutritional analysis
- **Libraries:**
  - `langchain-groq` for LLM integration
  - `transformers` for BLIP model
  - `PIL` for image processing
  - `matplotlib` for data visualization
  - `fpdf` for PDF generation
  - `torch` for deep learning framework
  - `python-dotenv` for environment variable management
- **Hardware:** Supports CPU and GPU (CUDA) for model inference

---

## Prerequisites

- Python 3.8 or higher
- A Groq API key (sign up at Groq to obtain one)
- Optional: NVIDIA GPU with CUDA for faster image processing

---

## Installation

1. **Clone the Repository:** git clone https://github.com/Ujjwal-sinha/ai-calorie-tracker.git
cd ai-calorie-tracker


2. **Create a Virtual Environment:**
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. **Install Dependencies:**
pip install -r requirements.txt

4. **Set Up Environment Variables:**

Create a `.env` file in the project root and add your Groq API key:

echo "GROQ_API_KEY=your_groq_api_key" > .env

5. **Download Pretrained Models:**

The BLIP model will be automatically downloaded from Hugging Face on the first run. Ensure you have an internet connection and sufficient disk space (~1GB).

---
# Architecture
![alt text](<ChatGPT Image Apr 26, 2025, 02_37_12 PM.png>)

## Usage

1. **Run the Application:**

streamlit run app.py

This will launch the app in your default browser at `http://localhost:8501`.

2. **Analyze Meals:**

- **Image Analysis Tab:** Upload a food image (JPG/PNG) and optionally add context (e.g., "Identify each item"). The app generates a nutritional breakdown and visualizes macronutrients.
- **Text Input Tab:** Describe a meal in natural language (e.g., "Grilled chicken with rice and broccoli") for nutritional analysis.
- **Follow-Up Questions:** Ask specific questions about the meal (e.g., "Is this good for weight loss?") for tailored insights.

3. **Track and Export:**

- **History Tab:** View past analyses and today’s calorie total. Generate a PDF report for the latest analysis.
- **Sidebar:** Monitor weekly nutrient trends and clear history if needed.

---

## Example

### Image Analysis

Upload a photo of a plate with grilled salmon, quinoa, and avocado. Add context: "Identify each item and estimate portion sizes."

**Output:**

- **Food Items and Nutrients:**
- Grilled Salmon (150g): 250 cal, Protein: 25g, Carbs: 0g, Fats: 15g
- Quinoa (100g): 120 cal, Protein: 4g, Carbs: 21g, Fats: 2g
- Avocado (70g): 160 cal, Protein: 1g, Carbs: 6g, Fats: 15g

- **Total Calories:** 530 cal

- **Nutritional Assessment:** Balanced meal with high protein and healthy fats.

- **Health Suggestions:** Consider adding leafy greens for micronutrients.

View a bar chart and ask follow-up questions like "How much omega-3 is in the salmon?"

### Text Input

Enter: "Large pepperoni pizza slice and a cola."

Output: Similar structured analysis with calories, macronutrients, and suggestions.

---


---

## Dependencies

Key packages listed in `requirements.txt`:


---

## Troubleshooting

- **Groq API Key Error:** Ensure `GROQ_API_KEY` is set correctly in `.env`. Verify your key at Groq Console.
- **BLIP Model Fails to Load:** Check internet connectivity and disk space. Try running on CPU by setting `device="cpu"` in model loading.
- **Low GPU Memory:** If using CUDA, ensure your GPU has at least 4GB VRAM. The app falls back to CPU if CUDA is unavailable.
- **PDF Generation Fails:** Ensure write permissions in the project directory and sufficient disk space.
- **Vague Analysis:** Provide detailed context with image uploads (e.g., "List all items") to improve LLM accuracy.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature/new-feature`
5. Open a pull request.

Please include tests and update documentation as needed.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

Built with ❤️ by Ujjwal Sinha.

---

## Notes

- You can customize the repository URL, contact details, or license as needed.
- The BLIP model is downloaded automatically on first run.
- For advanced deployment or additional features like database support or user authentication, the README can be extended accordingly.

---

If you want me to add demo GIFs, deployment instructions (e.g., Heroku, Docker), or sections like "Future Improvements," just let me know!






