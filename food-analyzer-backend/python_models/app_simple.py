#!/usr/bin/env python3
"""
Simple Flask API Server for Food Detection Models
Simplified version for debugging deployment issues
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import json
import time
import os
import sys

app = Flask(__name__)
CORS(app)

# Global model cache
MODEL_CACHE = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'food-detection-api-simple',
        'models_loaded': list(MODEL_CACHE.keys()),
        'python_version': sys.version,
        'environment': os.environ.get('FLASK_ENV', 'production')
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify the service is working"""
    return jsonify({
        'message': 'Food Detection API is working!',
        'timestamp': time.time(),
        'port': os.environ.get('PORT', 5000)
    })

@app.route('/detect', methods=['POST'])
def detect_food():
    """Main detection endpoint - simplified version"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        model_type = data.get('model_type', 'yolo')
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # For now, just return a mock response
        response = {
            'success': True,
            'detected_foods': ['apple', 'banana', 'orange'],
            'confidence_scores': {'apple': 0.95, 'banana': 0.87, 'orange': 0.92},
            'processing_time': 100,
            'model_info': {
                'model_type': model_type,
                'detection_count': 3,
                'confidence_threshold': 0.3,
                'note': 'This is a mock response for testing'
            }
        }
        
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
        'available_models': ['yolo', 'vit', 'swin', 'blip', 'clip'],
        'loaded_models': list(MODEL_CACHE.keys()),
        'note': 'Models will be loaded on first use'
    })

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        print(f"Starting Simple Food Detection API on port {port}")
        print(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")
        print(f"Python version: {sys.version}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"Failed to start server: {str(e)}", file=sys.stderr)
        sys.exit(1)
