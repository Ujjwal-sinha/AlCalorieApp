import os
import logging
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st

# Setup logging
logger = logging.getLogger(__name__)

class FoodDetectionAgent:
    """Simplified agent for detecting and identifying food items in images using global search and AI analysis."""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",
            api_key=groq_api_key,
            temperature=0.1
        )
        self.search_tool = DuckDuckGoSearchRun()
    
    def search_food_information(self, query: str) -> str:
        """Search for food information using DuckDuckGo."""
        try:
            search_results = self.search_tool.run(query)
            return search_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search failed: {str(e)}"
    
    def detect_food_from_image_description(self, image_description: str, context: str = "") -> Dict[str, Any]:
        """Main method to detect food items from image description using enhanced analysis."""
        try:
            # Create a comprehensive prompt for food analysis
            prompt = f"""You are an expert food identification and nutrition analysis specialist. Analyze the following food description and provide detailed information.

Image Description: {image_description}
Additional Context: {context if context else "No additional context provided"}

Please provide a comprehensive analysis including:

1. **Food Items Identified**: List all food items, dishes, ingredients, sauces, and components you can identify
2. **Detailed Descriptions**: Describe each food item in detail (what it is, how it's prepared, typical ingredients)
3. **Portion Sizes**: Provide estimated portion sizes for each item (e.g., 100g chicken, 1 cup rice, 2 rotis)
4. **Nutritional Information**: For each item, provide:
   - Calories
   - Protein (grams)
   - Carbohydrates (grams)
   - Fats (grams)
5. **Total Meal Analysis**: Sum up the total calories and macronutrients
6. **Health Assessment**: Provide 2-3 health suggestions based on the meal

**Instructions:**
- Be comprehensive and identify every food item mentioned or implied
- If you encounter unknown items, make educated guesses based on the description
- Provide realistic portion sizes and nutritional data
- Support all cuisines (Indian, international, etc.)
- Break down complex dishes into individual components

**Output Format:**
**Food Items and Nutrients:**
- Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g
- Item: [Food Name with portion size], Calories: [X] cal, Protein: [X] g, Carbs: [X] g, Fats: [X] g

**Total Calories**: [X] cal
**Nutritional Assessment**: [Detailed assessment of the meal]
**Health Suggestions**: [2-3 tailored suggestions]

Please provide a complete and detailed analysis."""

            # Get analysis from LLM
            response = self.llm.invoke(prompt)
            analysis = response.content
            
            # Extract food items and nutritional data
            food_items = self.extract_food_items(analysis)
            nutritional_data = self.extract_nutritional_data(analysis)
            
            return {
                "success": True,
                "analysis": analysis,
                "food_items": food_items,
                "nutritional_data": nutritional_data
            }
            
        except Exception as e:
            logger.error(f"Error in food detection agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": "Failed to analyze food items",
                "food_items": [],
                "nutritional_data": {}
            }
    
    def extract_food_items(self, analysis: str) -> List[Dict[str, str]]:
        """Extract food items from analysis."""
        try:
            lines = analysis.split('\n')
            food_items = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    # Extract food item from bullet point
                    item = line[1:].strip()
                    if item and len(item) > 3:
                        food_items.append({
                            "item": item,
                            "description": item
                        })
            
            return food_items
        except Exception as e:
            logger.error(f"Error extracting food items: {e}")
            return []
    
    def extract_nutritional_data(self, analysis: str) -> Dict[str, Any]:
        """Extract nutritional data from analysis."""
        try:
            lines = analysis.split('\n')
            nutritional_data = {
                "total_calories": 0,
                "protein": 0,
                "carbs": 0,
                "fats": 0,
                "items": []
            }
            
            for line in lines:
                line = line.strip().lower()
                if "calorie" in line and any(char.isdigit() for char in line):
                    # Extract calories
                    import re
                    calories = re.findall(r'\d+', line)
                    if calories:
                        nutritional_data["total_calories"] = int(calories[0])
                elif "protein" in line and any(char.isdigit() for char in line):
                    # Extract protein
                    import re
                    protein = re.findall(r'\d+', line)
                    if protein:
                        nutritional_data["protein"] = float(protein[0])
                elif "carb" in line and any(char.isdigit() for char in line):
                    # Extract carbs
                    import re
                    carbs = re.findall(r'\d+', line)
                    if carbs:
                        nutritional_data["carbs"] = float(carbs[0])
                elif "fat" in line and any(char.isdigit() for char in line):
                    # Extract fats
                    import re
                    fats = re.findall(r'\d+', line)
                    if fats:
                        nutritional_data["fats"] = float(fats[0])
            
            return nutritional_data
        except Exception as e:
            logger.error(f"Error extracting nutritional data: {e}")
            return {"total_calories": 0, "protein": 0, "carbs": 0, "fats": 0, "items": []}

class FoodSearchAgent:
    """Specialized agent for searching food information globally."""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",
            api_key=groq_api_key,
            temperature=0.1
        )
        self.search_tool = DuckDuckGoSearchRun()
    
    def search_food_information(self, query: str) -> str:
        """Search for food information globally."""
        try:
            # Search for food information
            search_results = self.search_tool.run(query)
            
            # Process search results
            prompt = f"""
Search Query: {query}
Search Results: {search_results}

Please analyze the search results and provide:
1. Accurate information about the food item/dish
2. Common ingredients and preparation methods
3. Typical nutritional information
4. Cultural context (if applicable)
5. Alternative names or variations

Provide a comprehensive summary of the food information found.
"""

            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in food search: {e}")
            return f"Error searching for food information: {str(e)}"
    
    def identify_unknown_food(self, description: str) -> str:
        """Identify unknown food items using global search."""
        try:
            # Create search query
            search_query = f"food dish {description} ingredients preparation"
            
            # Search for information
            search_results = self.search_tool.run(search_query)
            
            # Analyze results
            prompt = f"""
Unknown Food Description: {description}
Search Results: {search_results}

Please identify what this food item/dish is:
1. What is the name of this food/dish?
2. What are the main ingredients?
3. How is it typically prepared?
4. What cuisine does it belong to?
5. What are the typical nutritional values?

Provide a detailed identification and description.
"""

            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error identifying unknown food: {e}")
            return f"Error identifying food: {str(e)}"

# Initialize agents
def initialize_agents(groq_api_key: str):
    """Initialize food detection agents."""
    try:
        food_agent = FoodDetectionAgent(groq_api_key)
        search_agent = FoodSearchAgent(groq_api_key)
        return food_agent, search_agent
    except Exception as e:
        logger.error(f"Error initializing agents: {e}")
        return None, None
