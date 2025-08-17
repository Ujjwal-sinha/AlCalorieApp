#!/usr/bin/env python3
"""
Simple Flask API Server for Food Detection Models
YOLO-only version optimized for 512MB memory limit
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import json
import time
import os
import sys
import gc
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Global YOLO model cache
YOLO_MODEL = None

def load_yolo_model():
    """Load YOLO model with memory optimization"""
    global YOLO_MODEL
    try:
        if YOLO_MODEL is None:
            print("Loading YOLO model...", file=sys.stderr)
            from ultralytics import YOLO
            # Use the smallest YOLO model
            YOLO_MODEL = YOLO('yolov8n.pt')  # nano version (smallest)
            print("✅ YOLO model loaded successfully", file=sys.stderr)
        return YOLO_MODEL
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {str(e)}", file=sys.stderr)
        return None

def detect_with_yolo(image):
    """Detect food items using YOLO"""
    try:
        model = load_yolo_model()
        if model is None:
            return {
                'success': False,
                'error': 'YOLO model not available',
                'detected_foods': [],
                'confidence_scores': {}
            }
        
        # Run detection with single confidence level to save memory
        results = model(image, conf=0.3, verbose=False)
        
        detected_foods = []
        confidence_scores = {}
        
        # Common food classes in COCO dataset
        food_classes = {
            'apple', 'banana', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'dining table', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class name and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get class name from model
                    class_name = model.names[cls].lower()
                    
                    # Filter for food-related classes and confidence threshold
                    if conf > 0.3 and class_name in food_classes:
                        detected_foods.append(class_name)
                        confidence_scores[class_name] = conf
        
        return {
            'success': True,
            'detected_foods': detected_foods,
            'confidence_scores': confidence_scores,
            'model_type': 'yolo',
            'detection_count': len(detected_foods)
        }
        
    except Exception as e:
        print(f"YOLO detection error: {str(e)}", file=sys.stderr)
        return {
            'success': False,
            'error': str(e),
            'detected_foods': [],
            'confidence_scores': {}
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if YOLO_MODEL is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'service': 'food-detection-api-simple',
        'model_status': model_status,
        'models_loaded': ['yolo'] if YOLO_MODEL is not None else [],
        'python_version': sys.version,
        'environment': os.environ.get('FLASK_ENV', 'production'),
        'memory_optimized': True
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify the service is working"""
    return jsonify({
        'message': 'YOLO Food Detection API is working!',
        'timestamp': time.time(),
        'port': os.environ.get('PORT', 5000),
        'model': 'yolo'
    })

@app.route('/detect', methods=['POST'])
def detect_food():
    """Main detection endpoint - YOLO only"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        model_type = data.get('model_type', 'yolo')
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Only support YOLO for now
        if model_type != 'yolo':
            return jsonify({
                'success': False, 
                'error': f'Only YOLO model is supported in this deployment. Requested: {model_type}',
                'supported_models': ['yolo']
            }), 400
        
        # Decode and process image
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('RGB')
            
            # Resize image to save memory (max 512px)
            max_size = 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid image data: {str(e)}'}), 400
        
        # Perform YOLO detection
        start_time = time.time()
        result = detect_with_yolo(image)
        processing_time = int((time.time() - start_time) * 1000)
        
        # Add processing time to result
        result['processing_time'] = processing_time
        
        # Clear memory
        gc.collect()
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Detection error: {str(e)}", file=sys.stderr)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'available_models': ['yolo'],
        'loaded_models': ['yolo'] if YOLO_MODEL is not None else [],
        'note': 'YOLO-only deployment for memory optimization'
    })

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        print(f"Starting YOLO-Only Food Detection API on port {port}")
        print(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")
        print(f"Python version: {sys.version}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"Failed to start server: {str(e)}", file=sys.stderr)
        sys.exit(1)