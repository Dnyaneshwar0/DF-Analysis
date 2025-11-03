"""
DEEPFAKE DETECTION API ROUTES
=============================
"""

import os
import tempfile
import uuid
import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from src.inference.deepfake.df_detect import analyze_media_with_top_frames_and_images

deepfake_bp = Blueprint("deepfake", __name__)
logger = logging.getLogger(__name__)

ALLOWED_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg', '.jpg', '.jpeg', '.png', '.bmp'}

def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

# ============================================
# ENDPOINT 1: /predict
# ============================================
@deepfake_bp.route('/predict', methods=['POST'])
def deepfake_predict():
    """
    POST /predict - Analyze video or image
    
    Returns:
    {
        "status": "success",
        "deepfake": {
            "label": "FAKE",
            "confidence": 0.92,
            "metadata": {
                "duration_sec": 13.4,
                "frames_analyzed": 30
            },
            "top_frames": [
                {
                    "timestamp_sec": 2.3,
                    "frame_index": 1,
                    "fake_confidence": 0.97,
                    "url": "/assets/frames/frame1.png"
                },
                {
                    "timestamp_sec": 4.6,
                    "frame_index": 2,
                    "fake_confidence": 0.95,
                    "url": "/assets/frames/frame2.png"
                }
            ]
        }
    }
    """
    logger.info("=== /predict endpoint called ===")
    
    if "video" not in request.files and "image" not in request.files:
        return jsonify({"error": "Missing 'video' or 'image' file"}), 400
    
    file_storage = request.files.get("video") or request.files.get("image")
    
    if not file_storage or file_storage.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    if not allowed_file(file_storage.filename):
        return jsonify({"error": f"Allowed types: {', '.join(ALLOWED_EXT)}"}), 400
    
    suffix = os.path.splitext(file_storage.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_storage.read())
        tmp_path = tmp.name
    
    logger.info(f"File saved: {tmp_path}")
    
    try:
        # Inference
        result = analyze_media_with_top_frames_and_images(tmp_path, top_k=5)
        
        # Check for errors
        if isinstance(result, dict) and result.get("error"):
            error_type = result.get("error")
            http_status = 422 if error_type == "no_faces_detected" else 500
            return jsonify({"status": "error", "error": result}), http_status
        
        # Success
        return jsonify({
            "status": "success",
            "deepfake": result
        }), 200
    
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500
    
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass

# ============================================
# ENDPOINT 2: /analyze
# ============================================
@deepfake_bp.route('/analyze', methods=['POST'])
def analyze_route():
    """Alternative endpoint"""
    logger.info("=== /analyze endpoint called ===")
    
    if 'file' not in request.files:
        return jsonify({"error": "Missing 'file'"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "Empty file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"Allowed: {', '.join(ALLOWED_EXT)}"}), 400
    
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"df_{uuid.uuid4().hex}_{secure_filename(file.filename)}")
    
    try:
        file.save(tmp_path)
        result = analyze_media_with_top_frames_and_images(tmp_path, top_k=5)
        
        if isinstance(result, dict) and result.get("error"):
            http_status = 422 if result.get("error") == "no_faces_detected" else 500
            return jsonify({"status": "error", "error": result}), http_status
        
        return jsonify({
            "status": "success",
            "deepfake": result
        }), 200
    
    except Exception as e:
        logger.exception("Analysis failed")
        return jsonify({"error": str(e)}), 500
    
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass

# ============================================
# ENDPOINT 3: /health
# ============================================
@deepfake_bp.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    logger.info("=== /health endpoint called ===")
    
    try:
        from src.inference.deepfake.df_detect import _load_model
        _load_model()
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
