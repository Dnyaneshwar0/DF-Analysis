"""
DEEPFAKE DETECTION API ROUTES
=============================
"""

import os
import tempfile
import logging
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from src.inference.detection.df_detect import analyze_media_with_top_frames_and_images

deepfake_bp = Blueprint("deepfake", __name__)
logger = logging.getLogger(__name__)

ALLOWED_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg', '.jpg', '.jpeg', '.png', '.bmp'}


def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT


@deepfake_bp.route('/analyze', methods=['POST'])
def deepfake_predict():
    """
    POST /analyze - Analyze video or image
    """
    logger.info("=== /analyze endpoint called ===")

    if "file" not in request.files:
        return jsonify({"error": "Missing video or image file"}), 400

    file_storage = request.files.get("file")

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
        # --- Run deepfake inference ---
        result = analyze_media_with_top_frames_and_images(tmp_path, top_k=5)

        # --- Handle any error return ---
        if isinstance(result, dict) and result.get("error"):
            error_type = result.get("error")
            http_status = 422 if error_type == "no_faces_detected" else 500
            return jsonify({"status": "error", "error": result}), http_status

        # --- Make frame URLs absolute ---
        base = request.url_root.rstrip('/')  # e.g. http://localhost:5000
        if isinstance(result, dict) and result.get("top_frames"):
            for f in result["top_frames"]:
                url = f.get("url")
                if url and url.startswith("/"):
                    if not url.startswith(base):
                        f["url"] = base + url  # now absolute URL

        # --- Success ---
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
        except Exception:
            pass


# ============================================
# STATIC FILE SERVE: /data/detection/frames/*
# ============================================
@deepfake_bp.route('/data/detection/frames/<path:filename>')
def serve_detection_frame(filename):
    """
    Serve saved detection frame images from backend/data/detection/frames.
    Example: /detect/data/detection/frames/frame1.png
    """
    # Go up 3 levels from /src/routes/ → /backend/data/detection/frames
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'detection', 'frames')
    )
    return send_from_directory(base_dir, filename)


# ============================================
# ENDPOINT : /health
# ============================================
@deepfake_bp.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    logger.info("=== /health endpoint called ===")

    try:
        from src.inference.detection.df_detect import _load_model
        _load_model()
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
