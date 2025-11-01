import os
import tempfile
import uuid
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from src.inference.revEng.df_revEng import analyze_video

reverse_bp = Blueprint("reverse", __name__)
logger = logging.getLogger(__name__)

ALLOWED_EXT = {'.mp4', '.avi', '.mov', '.mkv'}

def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

@reverse_bp.route('/analyze', methods=['POST'])
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
    unique_name = f"reveng_{uuid.uuid4().hex}_{filename}"
    tmp_path = os.path.join(tmp_dir, unique_name)

    try:
        file.save(tmp_path)
        result = analyze_video(tmp_path)

        # If analyze_video returned an error dict, propagate appropriate HTTP code
        if isinstance(result, dict) and result.get("error"):
            err = result.get("error")
            # map some known errors to status codes
            if err == "no_faces_detected":
                status = 422
            elif err in ("model_load_failed", "inference_failed"):
                status = 500
            else:
                status = 400
            return jsonify({"status": "error", "result": result}), status

        return jsonify({"status": "success", "result": result}), 200

    except Exception as e:
        logger.exception("Failed to analyze video")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            logger.exception("Failed to remove temp file")
