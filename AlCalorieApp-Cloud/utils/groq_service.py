#!/usr/bin/env python3
"""
GROQ LLM Service for Diet Report Generation
Generates comprehensive diet analysis and recommendations using GROQ API
"""

import os
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class GroqDietReportService:
    """Service for generating diet reports using GROQ LLM"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the GROQ service"""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"  # Using Llama3-70B for better analysis
        
    def is_available(self) -> bool:
        """Check if GROQ API is available"""
        return bool(self.api_key)
    
    def generate_diet_report(self, detected_foods: List[str], nutritional_data: Dict[str, Any], 
                           context: str = "", meal_time: str = "lunch") -> Dict[str, Any]:
        """
        Generate comprehensive diet report using GROQ LLM
        
        Args:
            detected_foods: List of detected food items
            nutritional_data: Nutritional information (calories, protein, carbs, fats)
            context: Additional context about the meal
            meal_time: Time of meal (breakfast, lunch, dinner, snack)
            
        Returns:
            Dictionary containing the generated report
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "GROQ API key not configured",
                "report": "GROQ analysis not available. Please configure GROQ_API_KEY environment variable."
            }
        
        try:
            # Create comprehensive prompt for diet analysis
            prompt = self._create_diet_analysis_prompt(detected_foods, nutritional_data, context, meal_time)
            
            # Generate report using GROQ
            response = self._call_groq_api(prompt)
            
            if response.get("success"):
                return {
                    "success": True,
                    "report": response["content"],
                    "generated_at": datetime.now().isoformat(),
                    "model_used": self.model
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error"),
                    "report": "Failed to generate diet report."
                }
                
        except Exception as e:
            logger.error(f"Error generating diet report: {e}")
            return {
                "success": False,
                "error": str(e),
                "report": "An error occurred while generating the diet report."
            }
    
    def _create_diet_analysis_prompt(self, detected_foods: List[str], nutritional_data: Dict[str, Any], 
                                   context: str, meal_time: str) -> str:
        """Create a comprehensive prompt for diet analysis"""
        
        # Format nutritional data
        calories = nutritional_data.get("total_calories", 0)
        protein = nutritional_data.get("total_protein", 0)
        carbs = nutritional_data.get("total_carbs", 0)
        fats = nutritional_data.get("total_fats", 0)
        
        # Create food list
        food_list = ", ".join(detected_foods) if detected_foods else "No foods detected"
        
        prompt = f"""
You are an expert nutritionist and dietitian with 20+ years of experience. Analyze the following meal and provide a comprehensive diet report.

MEAL INFORMATION:
- Detected Foods: {food_list}
- Meal Time: {meal_time.title()}
- Additional Context: {context if context else "No additional context provided"}

NUTRITIONAL DATA:
- Total Calories: {calories:.0f} kcal
- Protein: {protein:.1f} g
- Carbohydrates: {carbs:.1f} g
- Fats: {fats:.1f} g

Please provide a comprehensive analysis in the following format:

## 🍽️ MEAL ANALYSIS REPORT

### 📊 Nutritional Overview
[Provide a 2-3 sentence summary of the overall nutritional profile]

### 🎯 Meal Assessment
- **Meal Type**: [Breakfast/Lunch/Dinner/Snack]
- **Portion Size**: [Small/Medium/Large/Extra Large]
- **Cooking Methods**: [Identify cooking methods if visible]
- **Main Macronutrient**: [Carb-heavy/Protein-rich/Fat-dense/Balanced]

### 🏆 Nutritional Quality Score: [1-10]
**Score**: [X]/10
**Justification**: [Explain the score based on nutritional balance, variety, and health factors]

### ✅ STRENGTHS
[What's nutritionally good about this meal - 2-3 specific points]

### ⚠️ AREAS FOR IMPROVEMENT
[What could be better - 2-3 specific suggestions]

### 💡 HEALTH RECOMMENDATIONS
1. **Immediate Suggestions**: [2-3 specific tips for this meal]
2. **Portion Adjustments**: [If needed]
3. **Complementary Foods**: [What to add for better nutrition]
4. **Timing Considerations**: [Best time to eat this meal]

### 🥗 DIETARY CONSIDERATIONS
- **Allergen Information**: [Common allergens present]
- **Dietary Restrictions**: [Vegan/Vegetarian/Gluten-free compatibility]
- **Blood Sugar Impact**: [High/Medium/Low glycemic impact]
- **Special Considerations**: [Any other important dietary notes]

### 🍎 FOOD-SPECIFIC INSIGHTS
[Provide insights for each detected food item - 1-2 sentences each]

### 📈 LONG-TERM RECOMMENDATIONS
[2-3 suggestions for improving overall diet based on this meal]

Be specific, practical, and evidence-based. Focus on actionable insights that the user can implement immediately. Use a friendly, encouraging tone while being honest about nutritional facts.
"""
        
        return prompt
    
    def _call_groq_api(self, prompt: str) -> Dict[str, Any]:
        """Make API call to GROQ"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert nutritionist and dietitian. Provide accurate, helpful, and actionable dietary advice."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.9
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "content": content
                }
            else:
                logger.error(f"GROQ API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            logger.error("GROQ API timeout")
            return {
                "success": False,
                "error": "API timeout"
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"GROQ API request error: {e}")
            return {
                "success": False,
                "error": f"Request error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error in GROQ API call: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def generate_quick_insights(self, detected_foods: List[str], nutritional_data: Dict[str, Any]) -> str:
        """Generate quick insights for the sidebar"""
        if not self.is_available():
            return "GROQ analysis not available. Configure GROQ_API_KEY for AI insights."
        
        try:
            prompt = f"""
Generate 2-3 quick, actionable insights about this meal in 1-2 sentences each:

Foods: {', '.join(detected_foods) if detected_foods else 'No foods detected'}
Calories: {nutritional_data.get('total_calories', 0):.0f} kcal
Protein: {nutritional_data.get('total_protein', 0):.1f} g
Carbs: {nutritional_data.get('total_carbs', 0):.1f} g
Fats: {nutritional_data.get('total_fats', 0):.1f} g

Provide insights in this format:
• [Insight 1]
• [Insight 2]
• [Insight 3]

Keep it concise and practical.
"""
            
            response = self._call_groq_api(prompt)
            if response.get("success"):
                return response["content"]
            else:
                return "Unable to generate insights at this time."
                
        except Exception as e:
            logger.error(f"Error generating quick insights: {e}")
            return "Unable to generate insights at this time."

# Global instance
groq_service = GroqDietReportService()
