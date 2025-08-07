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
        """Enhanced food detection from image description with comprehensive analysis."""
        try:
            # Enhanced prompt for better food detection and analysis
            prompt = f"""You are an expert nutritionist and food identification specialist. Analyze this food description comprehensively:

FOOD DESCRIPTION: {image_description}
ADDITIONAL CONTEXT: {context if context else "None provided"}

TASK: Provide a detailed analysis following this exact format:

## IDENTIFIED FOOD ITEMS:
[List every food item you can identify, even if mentioned briefly. Include:]
- Main dishes
- Side dishes  
- Beverages
- Condiments/sauces
- Garnishes
- Individual ingredients visible

## DETAILED NUTRITIONAL BREAKDOWN:
For each identified item, provide:
- Item: [Name with estimated portion size]
- Calories: [Amount]
- Protein: [Amount in grams]  
- Carbohydrates: [Amount in grams]
- Fats: [Amount in grams]
- Key nutrients: [Vitamins, minerals, fiber]

## MEAL TOTALS:
- Total Calories: [Sum]
- Total Protein: [Sum in grams]
- Total Carbohydrates: [Sum in grams] 
- Total Fats: [Sum in grams]

## MEAL ASSESSMENT:
- Meal type: [Breakfast/Lunch/Dinner/Snack]
- Cuisine style: [If identifiable]
- Nutritional balance: [Assessment of macro balance]
- Portion size: [Small/Medium/Large/Extra Large]

## HEALTH INSIGHTS:
- Positive aspects: [What's nutritionally good]
- Areas for improvement: [Suggestions]
- Missing nutrients: [What could be added]

## RECOMMENDATIONS:
- Healthier alternatives: [If applicable]
- Portion adjustments: [If needed]
- Complementary foods: [What would complete the meal]

IMPORTANT: Be thorough in identifying ALL food items, even small garnishes or condiments. Provide realistic portion estimates and accurate nutritional values."""

            # Get comprehensive analysis from LLM
            response = self.llm.invoke(prompt)
            analysis = response.content
            
            # Enhanced extraction of food items and nutritional data
            food_items = self.extract_food_items_enhanced(analysis)
            nutritional_data = self.extract_nutritional_data_enhanced(analysis)
            
            # Search for additional information on unidentified items
            if len(food_items) < 3:  # If we didn't identify many items, search for more
                search_results = self.search_additional_food_info(image_description)
                if search_results:
                    analysis += f"\n\nADDITIONAL RESEARCH:\n{search_results}"
            
            return {
                "success": True,
                "analysis": analysis,
                "food_items": food_items,
                "nutritional_data": nutritional_data,
                "comprehensive": True
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced food detection agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": "Failed to analyze food items comprehensively",
                "food_items": [],
                "nutritional_data": {},
                "comprehensive": False
            }
    
    def extract_food_items_enhanced(self, analysis: str) -> List[Dict[str, str]]:
        """Enhanced extraction of food items from analysis."""
        try:
            lines = analysis.split('\n')
            food_items = []
            in_food_section = False
            
            for line in lines:
                line = line.strip()
                
                # Check if we're in the food items section
                if "IDENTIFIED FOOD ITEMS" in line.upper() or "FOOD ITEMS" in line.upper():
                    in_food_section = True
                    continue
                elif line.startswith("##") and in_food_section:
                    in_food_section = False
                    continue
                
                # Extract food items
                if in_food_section and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    item = line[1:].strip()
                    if item and len(item) > 2:
                        food_items.append({
                            "item": item,
                            "description": item,
                            "category": self.categorize_food_item(item)
                        })
                
                # Also look for "Item:" format in nutritional breakdown
                elif line.lower().startswith("- item:") or line.lower().startswith("item:"):
                    item = line.split(":", 1)[1].strip()
                    if item and len(item) > 2:
                        food_items.append({
                            "item": item,
                            "description": item,
                            "category": self.categorize_food_item(item)
                        })
            
            # If no items found in structured format, try to extract from text
            if not food_items:
                food_items = self.extract_food_items(analysis)
            
            return food_items
            
        except Exception as e:
            logger.error(f"Error in enhanced food item extraction: {e}")
            return self.extract_food_items(analysis)  # Fallback to original method
    
    def extract_nutritional_data_enhanced(self, analysis: str) -> Dict[str, Any]:
        """Enhanced extraction of nutritional data from analysis."""
        try:
            import re
            
            nutritional_data = {
                "total_calories": 0,
                "total_protein": 0,
                "total_carbs": 0,
                "total_fats": 0,
                "items": [],
                "meal_assessment": "",
                "health_insights": ""
            }
            
            lines = analysis.split('\n')
            
            # Extract totals
            for line in lines:
                line = line.strip().lower()
                
                # Extract total calories
                if "total calories" in line or "total calorie" in line:
                    calories = re.findall(r'\d+', line)
                    if calories:
                        nutritional_data["total_calories"] = int(calories[0])
                
                # Extract total protein
                elif "total protein" in line:
                    protein = re.findall(r'\d+\.?\d*', line)
                    if protein:
                        nutritional_data["total_protein"] = float(protein[0])
                
                # Extract total carbs
                elif "total carbohydrate" in line or "total carbs" in line:
                    carbs = re.findall(r'\d+\.?\d*', line)
                    if carbs:
                        nutritional_data["total_carbs"] = float(carbs[0])
                
                # Extract total fats
                elif "total fats" in line or "total fat" in line:
                    fats = re.findall(r'\d+\.?\d*', line)
                    if fats:
                        nutritional_data["total_fats"] = float(fats[0])
            
            # Extract meal assessment
            assessment_section = ""
            in_assessment = False
            for line in lines:
                if "MEAL ASSESSMENT" in line.upper():
                    in_assessment = True
                    continue
                elif line.startswith("##") and in_assessment:
                    break
                elif in_assessment:
                    assessment_section += line + " "
            
            nutritional_data["meal_assessment"] = assessment_section.strip()
            
            # Extract health insights
            insights_section = ""
            in_insights = False
            for line in lines:
                if "HEALTH INSIGHTS" in line.upper():
                    in_insights = True
                    continue
                elif line.startswith("##") and in_insights:
                    break
                elif in_insights:
                    insights_section += line + " "
            
            nutritional_data["health_insights"] = insights_section.strip()
            
            return nutritional_data
            
        except Exception as e:
            logger.error(f"Error in enhanced nutritional data extraction: {e}")
            return self.extract_nutritional_data(analysis)  # Fallback to original method
    
    def categorize_food_item(self, item: str) -> str:
        """Categorize food items for better organization."""
        item_lower = item.lower()
        
        # Protein sources
        if any(protein in item_lower for protein in ['chicken', 'beef', 'pork', 'fish', 'egg', 'tofu', 'beans', 'lentils', 'turkey', 'lamb', 'shrimp', 'salmon']):
            return "protein"
        
        # Vegetables
        elif any(veg in item_lower for veg in ['lettuce', 'tomato', 'onion', 'carrot', 'broccoli', 'spinach', 'pepper', 'cucumber', 'cabbage']):
            return "vegetable"
        
        # Fruits
        elif any(fruit in item_lower for fruit in ['apple', 'banana', 'orange', 'berry', 'grape', 'lemon', 'lime', 'mango', 'pineapple']):
            return "fruit"
        
        # Grains/Carbs
        elif any(grain in item_lower for grain in ['rice', 'bread', 'pasta', 'noodle', 'potato', 'quinoa', 'oats', 'cereal']):
            return "grain/carb"
        
        # Dairy
        elif any(dairy in item_lower for dairy in ['milk', 'cheese', 'yogurt', 'butter', 'cream']):
            return "dairy"
        
        # Beverages
        elif any(drink in item_lower for drink in ['water', 'juice', 'coffee', 'tea', 'soda', 'beer', 'wine', 'smoothie']):
            return "beverage"
        
        else:
            return "other"
    
    def search_additional_food_info(self, description: str) -> str:
        """Search for additional information about food items that might have been missed."""
        try:
            # Create a focused search query
            search_query = f"food ingredients nutrition {description}"
            search_results = self.search_tool.run(search_query)
            
            # Process search results to extract additional food information
            prompt = f"""
Based on this food description: "{description}"
And these search results: {search_results}

Identify any additional food items, ingredients, or components that might have been missed in the initial analysis. Focus on:
1. Hidden ingredients or components
2. Common accompaniments for this type of food
3. Typical garnishes or sides
4. Cooking methods that might add calories (oils, butter, etc.)

Provide a brief summary of additional items to consider.
"""
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error searching for additional food info: {e}")
            return ""
    
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
