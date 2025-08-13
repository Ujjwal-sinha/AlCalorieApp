#!/usr/bin/env python3
"""
Color-based food analysis script
This script analyzes color data and returns potential food matches
"""

import sys
import json
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_colors(color_data):
    """Analyze colors and return potential food matches"""
    detected_foods = []
    
    try:
        mean_color = np.array(color_data['mean_color'])
        r, g, b = mean_color
        
        # Color-based food inference (similar to Python implementation)
        if r > 150 and g < 100 and b < 100:  # Red dominant
            detected_foods.extend(['tomato', 'strawberry', 'apple'])
        elif g > 150 and r < 120 and b < 120:  # Green dominant
            detected_foods.extend(['broccoli', 'lettuce', 'spinach'])
        elif r > 200 and g > 180 and b < 100:  # Yellow dominant
            detected_foods.extend(['banana', 'corn', 'cheese'])
        elif r > 150 and g > 100 and b < 80:  # Orange dominant
            detected_foods.extend(['carrot', 'orange', 'sweet potato'])
        elif r > 100 and g > 80 and b > 60:  # Brown dominant
            detected_foods.extend(['bread', 'chicken', 'potato'])
        else:
            # Fallback based on brightness
            brightness = (r + g + b) / 3
            if brightness > 200:
                detected_foods.append('rice')
            elif brightness > 100:
                detected_foods.append('mixed food')
            else:
                detected_foods.append('dark food')
        
        # Limit to top 2 foods
        detected_foods = detected_foods[:2]
        
        logger.info(f"Color analysis detected: {detected_foods}")
        
    except Exception as e:
        logger.error(f"Color analysis failed: {e}")
        detected_foods = ['food item']  # Fallback
    
    return detected_foods

def main():
    """Main function to process color analysis request"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        color_data = json.loads(input_data)
        
        # Analyze colors
        detected_foods = analyze_colors(color_data)
        
        # Return results
        result = {
            'success': True,
            'detected_foods': detected_foods,
            'method': 'color_analysis'
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'detected_foods': []
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == '__main__':
    main()