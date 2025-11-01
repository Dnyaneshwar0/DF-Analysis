import os
import tempfile
import uuid
import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from src.inference.deepfake.df_detect import analyze_media

deepfake_bp = Blueprint("deepfake", __name__)
logger = logging.getLogger(__name__)

ALLOWED_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg', '.jpg', '.jpeg', '.png', '.bmp'}

def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

@deepfake_bp.route('/analyze', methods=['POST'])
def analyze_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    tmp_dir = tempfile.gettempdir()
    unique_name = f"deepfake_{uuid.uuid4().hex}_{filename}"
    tmp_path = os.path.join(tmp_dir, unique_name)

    try:
        file.save(tmp_path)
        result = analyze_media(tmp_path)

        if isinstance(result, dict) and result.get("error"):
            err = result.get("error")
            if err == "no_faces_detected":
                status = 422
            elif err in ("model_load_failed", "inference_failed"):
                status = 500
            else:
                status = 400
            return jsonify({"status": "error", "result": result}), status

        return jsonify({"status": "success", "result": result}), 200

    except Exception as e:
        logger.exception("Failed to analyze media")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            logger.exception("Failed to remove temp file")
