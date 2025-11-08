import os
import tempfile
import uuid
import logging
from pathlib import Path

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

# Try to import the orchestrator main function from your df_emo module.
# The file you posted exposes `main(argv=None)` and returns the merged dict.
# If your file uses a different name/path, change the import accordingly.
try:
    from src.inference.emotion.df_emo import main as run_emotion
except Exception:
    # fallback: some repos call it orchestrator.py — try that name too
    try:
        from src.inference.emotion.df_emo import main as run_emotion
    except Exception as e:
        run_emotion = None
        # we'll raise a useful error at runtime if route is called


emotion_bp = Blueprint("emotion", __name__)
logger = logging.getLogger(__name__)

ALLOWED_EXT = {'.mp4', '.avi', '.mov', '.mkv'}

def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

@emotion_bp.route('/analyze', methods=['POST'])
def analyze_route():
    if run_emotion is None:
        return jsonify({"error": "Emotion inference module not available (import failed)"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    tmp_dir = tempfile.gettempdir()
    unique_name = f"emo_{uuid.uuid4().hex}_{filename}"
    tmp_path = os.path.join(tmp_dir, unique_name)

    try:
        file.save(tmp_path)

        # df_emo.main expects an argv-like list; pass the uploaded file path
        # It returns a merged JSON dict (as in the file you posted).
        result = run_emotion([tmp_path])

        # If the orchestrator returns an error-style payload, map to status codes.
        if isinstance(result, dict) and result.get("error"):
            err = result.get("error")
            # map some common errors if you want
            if err == "video_not_found":
                status = 422
            else:
                status = 400
            return jsonify({"status": "error", "result": result}), status

        # Success: return the merged JSON produced by the orchestrator
        return jsonify({"status": "success", "result": result}), 200

    except Exception as e:
        logger.exception("Emotion analysis failed")
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            logger.exception("Failed to remove temp file")