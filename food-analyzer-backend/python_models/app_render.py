#!/usr/bin/env python3
"""
Render-Optimized Flask App for Food Detection
Uses optimized models without BLIP for 512MB RAM limit
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import json
import time
import os
import sys
from models_render_optimized import (
    load_model, detect_with_yolo, detect_with_vit, detect_with_swin, detect_with_blip, detect_with_clip,
    MODEL_AVAILABILITY
)

app = Flask(__name__)
CORS(app)

# Global model cache
MODEL_CACHE = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'food-detection-api',
        'version': 'render-optimized',
        'loaded_models': list(MODEL_CACHE.keys()),
        'available_models': [k for k, v in MODEL_AVAILABILITY.items() if v],
        'memory_optimized': True,
        'blip_disabled': True
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint"""
    return jsonify({
        'message': 'Food Detection API is running!',
        'version': 'render-optimized',
        'models_available': [k for k, v in MODEL_AVAILABILITY.items() if v],
        'note': 'BLIP model disabled for memory optimization'
    })

@app.route('/detect', methods=['POST'])
def detect_food():
    """Detect food in uploaded image"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        model_type = data.get('model_type', 'yolo')
        image_data = data.get('image_data')
        width = data.get('width', 1024)
        height = data.get('height', 1024)
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Decode image
        try:
            image_bytes = base64.b64decode(image_data)
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('RGB')
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid image data: {str(e)}'}), 400
        
        # Load model if not cached
        if model_type not in MODEL_CACHE:
            model = load_model(model_type)
            if model is None:
                return jsonify({
                    'success': False, 
                    'error': f'Failed to load {model_type} model. Model may not be available.',
                    'available_models': [k for k, v in MODEL_AVAILABILITY.items() if v],
                    'note': 'BLIP model is disabled in this version'
                }), 500
            MODEL_CACHE[model_type] = model
        
        # Perform detection
        start_time = time.time()
        
        if model_type == 'yolo':
            result = detect_with_yolo(image, MODEL_CACHE[model_type])
        elif model_type == 'vit':
            result = detect_with_vit(image, MODEL_CACHE[model_type])
        elif model_type == 'swin':
            result = detect_with_swin(image, MODEL_CACHE[model_type])
        elif model_type == 'blip':
            result = detect_with_blip(image, MODEL_CACHE[model_type])
        elif model_type == 'clip':
            result = detect_with_clip(image, MODEL_CACHE[model_type])
        else:
            return jsonify({'success': False, 'error': f'Unsupported model type: {model_type}'}), 400
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Format response
        response = {
            'success': True,
            'detected_foods': result.get('detected_foods', []),
            'confidence_scores': result.get('confidence_scores', {}),
            'processing_time': processing_time,
            'model_info': {
                'model_type': model_type,
                'detection_count': len(result.get('detected_foods', [])),
                'confidence_threshold': 0.3,
                'version': 'render-optimized'
            }
        }
        
        # Add caption for BLIP (will be disabled message)
        if model_type == 'blip' and 'caption' in result:
            response['model_info']['caption'] = result['caption']
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Detection error: {str(e)}", file=sys.stderr)
        return jsonify({
            'success': False,
            'error': f'Detection failed: {str(e)}',
            'detected_foods': [],
            'confidence_scores': {},
            'processing_time': 0
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'available_models': ['yolo', 'vit', 'swin', 'clip'],  # BLIP excluded
        'loaded_models': list(MODEL_CACHE.keys()),
        'model_availability': MODEL_AVAILABILITY,
        'note': 'Models will be loaded on first use. BLIP model disabled for memory optimization.',
        'version': 'render-optimized'
    })

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        print(f"Starting Render-Optimized Food Detection API on port {port}")
        print(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")
        print(f"Python version: {sys.version}")
        print(f"Available models: {[k for k, v in MODEL_AVAILABILITY.items() if v]}")
        print(f"BLIP model: Disabled for memory optimization")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"Failed to start server: {str(e)}", file=sys.stderr)
        sys.exit(1)
