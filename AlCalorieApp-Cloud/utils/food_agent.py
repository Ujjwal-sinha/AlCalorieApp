import streamlit as st
import logging
import requests
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import re
from datetime import datetime
import hashlib
import os

# Setup logging
logger = logging.getLogger(__name__)

class FoodAgent:
    """
    Enhanced Food Analysis Agent with Web Search and Context Management
    
    This agent provides comprehensive food analysis by:
    1. Analyzing food images
    2. Searching web for detailed information
    3. Storing context for follow-up questions
    4. Providing enhanced nutritional insights
    """
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.context_cache = {}
        self.search_cache = {}
        self.session_id = None
        
    def generate_session_id(self, image: Image.Image) -> str:
        """Generate unique session ID for image analysis"""
        # Create hash from image data and timestamp
        img_bytes = image.tobytes()
        timestamp = datetime.now().isoformat()
        combined = f"{img_bytes[:1000]}{timestamp}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def analyze_food_image(self, image: Image.Image) -> Dict[str, Any]:
        """Step 1: Analyze food image and generate description"""
        try:
            # Generate session ID for this analysis
            self.session_id = self.generate_session_id(image)
            
            # Use existing enhanced analysis
            from .analysis import describe_image_enhanced
            food_description = describe_image_enhanced(image, self.models)
            
            # Enhanced description with more context
            enhanced_prompt = f"""
            Analyze this food image comprehensively and provide:
            1. Main food items identified
            2. Cooking methods visible
            3. Portion sizes estimated
            4. Cuisine style if identifiable
            5. Preparation techniques
            6. Any special ingredients or garnishes
            
            Current description: {food_description}
            
            Provide a detailed, structured analysis.
            """
            
            # Get enhanced description from LLM
            enhanced_description = self._query_llm(enhanced_prompt)
            
            return {
                "session_id": self.session_id,
                "original_description": food_description,
                "enhanced_description": enhanced_description,
                "timestamp": datetime.now().isoformat(),
                "image_hash": self.session_id
            }
            
        except Exception as e:
            logger.error(f"Error analyzing food image: {e}")
            return {"error": str(e)}
    
    def search_web_information(self, food_description: str) -> Dict[str, Any]:
        """Step 2: Search web for comprehensive food information"""
        try:
            # Check cache first
            cache_key = hashlib.md5(food_description.encode()).hexdigest()
            if cache_key in self.search_cache:
                return self.search_cache[cache_key]
            
            # Prepare search queries
            search_queries = self._generate_search_queries(food_description)
            
            # Perform web searches
            search_results = {}
            for query_type, query in search_queries.items():
                try:
                    results = self._perform_web_search(query)
                    search_results[query_type] = results
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Search failed for {query_type}: {e}")
                    search_results[query_type] = []
            
            # Extract and structure information
            structured_info = self._extract_structured_info(search_results, food_description)
            
            # Cache results
            self.search_cache[cache_key] = structured_info
            
            return structured_info
            
        except Exception as e:
            logger.error(f"Error searching web information: {e}")
            return {"error": str(e)}
    
    def _generate_search_queries(self, food_description: str) -> Dict[str, str]:
        """Generate targeted search queries for different information types"""
        # Extract key food terms
        food_terms = self._extract_food_terms(food_description)
        
        queries = {
            "nutrition": f"{food_terms} nutrition facts calories protein carbs fats",
            "recipes": f"{food_terms} recipe cooking instructions ingredients",
            "cultural": f"{food_terms} cultural background history origin traditional",
            "health": f"{food_terms} health benefits nutritional value dietary information",
            "variations": f"{food_terms} variations types different ways to prepare"
        }
        
        return queries
    
    def _extract_food_terms(self, description: str) -> str:
        """Extract key food terms for search queries"""
        # Remove common words and keep food-specific terms
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = description.lower().split()
        food_terms = [word for word in words if word not in common_words and len(word) > 2]
        return ' '.join(food_terms[:5])  # Limit to 5 key terms
    
    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search using available APIs or fallback methods"""
        try:
            # Try DuckDuckGo Instant Answer API (free, no API key required)
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._parse_duckduckgo_results(data)
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        # Fallback: Return structured mock data based on query
        return self._generate_mock_search_results(query)
    
    def _parse_duckduckgo_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse DuckDuckGo API results"""
        results = []
        
        # Extract abstract
        if data.get('Abstract'):
            results.append({
                'title': data.get('Heading', 'Food Information'),
                'snippet': data.get('Abstract'),
                'url': data.get('AbstractURL', ''),
                'type': 'abstract'
            })
        
        # Extract related topics
        for topic in data.get('RelatedTopics', [])[:3]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append({
                    'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                    'snippet': topic.get('Text'),
                    'url': topic.get('FirstURL', ''),
                    'type': 'related'
                })
        
        return results
    
    def _generate_mock_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Generate mock search results for fallback"""
        # This is a fallback when web search is not available
        # In production, you might want to use a different search API
        
        mock_data = {
            'nutrition': [
                {
                    'title': 'Nutritional Information',
                    'snippet': f'Comprehensive nutritional data for {query}. Includes calories, protein, carbohydrates, fats, vitamins, and minerals.',
                    'url': 'https://nutrition-database.com',
                    'type': 'nutrition'
                }
            ],
            'recipes': [
                {
                    'title': 'Cooking Instructions',
                    'snippet': f'Traditional and modern recipes for preparing {query}. Step-by-step instructions with ingredient lists.',
                    'url': 'https://recipe-collection.com',
                    'type': 'recipe'
                }
            ],
            'cultural': [
                {
                    'title': 'Cultural Background',
                    'snippet': f'Historical and cultural significance of {query}. Origin, traditional preparation methods, and cultural importance.',
                    'url': 'https://food-culture.com',
                    'type': 'cultural'
                }
            ]
        }
        
        # Return relevant mock data based on query type
        for key, data in mock_data.items():
            if key in query.lower():
                return data
        
        return [{
            'title': 'General Information',
            'snippet': f'General information about {query} including preparation, nutrition, and cultural aspects.',
            'url': 'https://food-info.com',
            'type': 'general'
        }]
    
    def _extract_structured_info(self, search_results: Dict[str, List], food_description: str) -> Dict[str, Any]:
        """Extract and structure information from search results"""
        structured_info = {
            "food_name": self._extract_food_name(food_description),
            "nutrition": self._extract_nutrition_info(search_results.get("nutrition", [])),
            "recipes": self._extract_recipe_info(search_results.get("recipes", [])),
            "cultural": self._extract_cultural_info(search_results.get("cultural", [])),
            "health": self._extract_health_info(search_results.get("health", [])),
            "variations": self._extract_variation_info(search_results.get("variations", [])),
            "summary": self._generate_summary(search_results, food_description)
        }
        
        return structured_info
    
    def _extract_food_name(self, description: str) -> str:
        """Extract primary food name from description"""
        # Simple extraction - in production, use more sophisticated NLP
        words = description.split(',')[0].strip()
        return words.title()
    
    def _extract_nutrition_info(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract nutritional information from search results"""
        nutrition_info = {
            "calories": "Variable",
            "protein": "Variable",
            "carbs": "Variable", 
            "fats": "Variable",
            "vitamins": [],
            "minerals": [],
            "notes": ""
        }
        
        for result in results:
            snippet = result.get('snippet', '').lower()
            if 'calorie' in snippet:
                nutrition_info["notes"] += f" {result.get('snippet', '')}"
        
        return nutrition_info
    
    def _extract_recipe_info(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract recipe information from search results"""
        recipes = []
        for result in results:
            recipes.append({
                "title": result.get('title', 'Recipe'),
                "description": result.get('snippet', ''),
                "source": result.get('url', '')
            })
        return recipes
    
    def _extract_cultural_info(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract cultural and historical information"""
        cultural_info = {
            "origin": "Various regions",
            "history": "",
            "cultural_significance": "",
            "traditional_uses": []
        }
        
        for result in results:
            cultural_info["history"] += f" {result.get('snippet', '')}"
        
        return cultural_info
    
    def _extract_health_info(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract health and dietary information"""
        health_info = {
            "benefits": [],
            "precautions": [],
            "dietary_considerations": [],
            "allergen_info": "Check ingredients"
        }
        
        for result in results:
            snippet = result.get('snippet', '')
            if 'benefit' in snippet.lower():
                health_info["benefits"].append(snippet)
        
        return health_info
    
    def _extract_variation_info(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract variation and preparation information"""
        variations = []
        for result in results:
            variations.append(result.get('snippet', ''))
        return variations
    
    def _generate_summary(self, search_results: Dict[str, List], food_description: str) -> str:
        """Generate comprehensive summary from all search results"""
        summary_parts = [f"Food Analysis: {food_description}"]
        
        for category, results in search_results.items():
            if results:
                summary_parts.append(f"\n{category.title()} Information:")
                for result in results[:2]:  # Limit to 2 results per category
                    summary_parts.append(f"- {result.get('snippet', '')}")
        
        return "\n".join(summary_parts)
    
    def _query_llm(self, prompt: str) -> str:
        """Query the language model for enhanced analysis"""
        try:
            if self.models.get('llm'):
                from langchain.schema import HumanMessage
                response = self.models['llm']([HumanMessage(content=prompt)])
                return response.content
            else:
                return "LLM not available for enhanced analysis."
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return "Enhanced analysis not available."
    
    def answer_user_questions(self, question: str, context: Dict[str, Any]) -> str:
        """Step 5: Answer user questions using stored context"""
        try:
            # Create context-aware prompt
            context_prompt = f"""
            Context Information:
            {json.dumps(context, indent=2)}
            
            User Question: {question}
            
            Please provide a comprehensive answer based on the context information above.
            Include relevant details about nutrition, recipes, cultural background, and health information.
            """
            
            return self._query_llm(context_prompt)
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Unable to answer question: {str(e)}"
    
    def store_context(self, session_id: str, context: Dict[str, Any]):
        """Step 6: Store extracted context for follow-up questions"""
        self.context_cache[session_id] = {
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }
    
    def get_stored_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored context for follow-up questions"""
        if session_id in self.context_cache:
            self.context_cache[session_id]["access_count"] += 1
            return self.context_cache[session_id]["context"]
        return None
    
    def process_food_image_complete(self, image: Image.Image, user_question: str = None) -> Dict[str, Any]:
        """Complete food processing pipeline"""
        try:
            # Step 1: Analyze image
            analysis_result = self.analyze_food_image(image)
            if "error" in analysis_result:
                return analysis_result
            
            # Step 2: Search web information
            search_result = self.search_web_information(analysis_result["enhanced_description"])
            
            # Step 3: Combine results
            combined_context = {
                "image_analysis": analysis_result,
                "web_information": search_result,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            # Step 4: Store context
            self.store_context(analysis_result["session_id"], combined_context)
            
            # Step 5: Answer user question if provided
            if user_question:
                answer = self.answer_user_questions(user_question, combined_context)
                combined_context["user_answer"] = answer
            
            return combined_context
            
        except Exception as e:
            logger.error(f"Error in complete processing: {e}")
            return {"error": str(e)}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and statistics"""
        return {
            "context_cache_size": len(self.context_cache),
            "search_cache_size": len(self.search_cache),
            "models_available": {
                "blip": self.models.get('blip_model') is not None,
                "llm": self.models.get('llm') is not None,
                "yolo": self.models.get('yolo_model') is not None
            },
            "session_id": self.session_id
        }

# Example usage function
def example_usage():
    """Example usage of the FoodAgent"""
    st.markdown("## 🍽️ Food Agent Example Usage")
    
    st.markdown("""
    ### How to use the Food Agent:
    
    1. **Upload a food image**
    2. **The agent will:**
       - Analyze the image and generate descriptions
       - Search the web for comprehensive information
       - Store context for follow-up questions
       - Provide detailed nutritional and cultural insights
    
    3. **Ask follow-up questions** about the food without re-uploading
    4. **Get enhanced analysis** with web-sourced information
    """)
    
    # Example with sample data
    st.markdown("### Example Output:")
    
    example_result = {
        "image_analysis": {
            "session_id": "abc123def456",
            "original_description": "Grilled chicken with vegetables",
            "enhanced_description": "Grilled chicken breast with mixed vegetables including broccoli, carrots, and bell peppers. Served with a light sauce.",
            "timestamp": "2025-01-15T10:30:00"
        },
        "web_information": {
            "food_name": "Grilled Chicken with Vegetables",
            "nutrition": {
                "calories": "350-450 kcal",
                "protein": "35-45g",
                "carbs": "15-25g",
                "fats": "8-12g"
            },
            "recipes": [
                {
                    "title": "Healthy Grilled Chicken Recipe",
                    "description": "Marinate chicken in herbs and grill with vegetables",
                    "source": "https://healthy-recipes.com"
                }
            ],
            "cultural": {
                "origin": "Mediterranean cuisine",
                "history": "Grilled chicken has been a staple in Mediterranean diets for centuries"
            }
        }
    }
    
    st.json(example_result)

# Integration function for Streamlit app
def create_food_agent_interface(models: Dict[str, Any]):
    """Create Streamlit interface for the Food Agent"""
    
    # Initialize agent
    agent = FoodAgent(models)
    
    st.markdown("## 🤖 Enhanced Food Agent")
    st.markdown("Get comprehensive food analysis with web-sourced information!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a food image for enhanced analysis",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of your food for comprehensive analysis"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Food Image", use_column_width=True)
        
        # Process image
        if st.button("🔍 Analyze with Enhanced Agent"):
            with st.spinner("Processing image with enhanced agent..."):
                result = agent.process_food_image_complete(image)
                
                if "error" not in result:
                    st.success("Enhanced analysis complete!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📸 Image Analysis")
                        st.write(f"**Session ID:** {result['image_analysis']['session_id']}")
                        st.write(f"**Description:** {result['image_analysis']['enhanced_description']}")
                    
                    with col2:
                        st.markdown("### 🌐 Web Information")
                        web_info = result['web_information']
                        st.write(f"**Food Name:** {web_info.get('food_name', 'Unknown')}")
                        st.write(f"**Nutrition:** {web_info.get('nutrition', {}).get('calories', 'Unknown')}")
                    
                    # Ask follow-up questions
                    st.markdown("### ❓ Ask Follow-up Questions")
                    user_question = st.text_input("Ask a question about this food:")
                    
                    if user_question and st.button("Ask"):
                        answer = agent.answer_user_questions(user_question, result)
                        st.markdown("### 💬 Answer:")
                        st.write(answer)
                    
                    # Show agent status
                    with st.expander("🤖 Agent Status"):
                        status = agent.get_agent_status()
                        st.json(status)
                
                else:
                    st.error(f"Analysis failed: {result['error']}")
    
    # Show example usage
    with st.expander("📖 Example Usage"):
        example_usage()

if __name__ == "__main__":
    # This would be used for testing the agent independently
    print("Food Agent Module - Import and use in Streamlit app")
