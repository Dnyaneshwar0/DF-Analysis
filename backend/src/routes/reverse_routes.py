# backend/src/routes/reverse_routes.py
import os
import tempfile
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from src.utils.preprocessing import save_uploaded_file, extract_faces_from_video
import requests
import cv2

reverse_bp = Blueprint("reverse", __name__)

# URL of the revEng model service (expected to expose POST /predict)
REV_SERVICE_URL = os.environ.get("REV_SERVICE_URL", "http://localhost:5200")
PREDICT_ENDPOINT = REV_SERVICE_URL.rstrip("/") + "/predict"

@reverse_bp.route("/analyze", methods=["POST"])
def reverse_analyze():
    """
    Accepts multipart/form-data with key 'video' (or 'image').
    If 'video' is provided, extracts faces from video and sends the first face to the revEng service.
    If 'image' is provided, sends that image directly.
    Returns the revEng service response.
    """
    # Accept either a single image or a video file (frontend sends video)
    tmpdir = tempfile.mkdtemp(prefix="reveng_upload_")

    try:
        # prefer 'image' if provided
        if "image" in request.files and request.files["image"].filename != "":
            img_file = request.files["image"]
            image_path = save_uploaded_file(img_file, tmpdir)
            # read image and send to model service
            img = cv2.imread(str(image_path))
            if img is None:
                return jsonify({"error": "bad_image", "details": "Could not read uploaded image"}), 400

            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                return jsonify({"error": "image_encoding_failed"}), 500

            files = [("images", ("frame.jpg", buf.tobytes(), "image/jpeg"))]

        elif "video" in request.files and request.files["video"].filename != "":
            video_file = request.files["video"]
            video_path = save_uploaded_file(video_file, tmpdir)

            faces, meta = extract_faces_from_video(str(video_path), sample_rate=1)
            if len(faces) == 0:
                return jsonify({"error": "no_faces_found"}), 400

            # choose the first face (or you can send all faces)
            face = faces[0]
            ok, buf = cv2.imencode(".jpg", face)
            if not ok:
                return jsonify({"error": "face_encoding_failed"}), 500

            files = [("images", ("face_0.jpg", buf.tobytes(), "image/jpeg"))]

        else:
            return jsonify({"error": "missing_file", "details": "Provide 'video' or 'image' file"}), 400

        # send to revEng model service
        try:
            resp = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
        except requests.exceptions.RequestException as re:
            current_app.logger.exception("Error calling revEng model service")
            return jsonify({"error": "model_service_unreachable", "details": str(re)}), 502

        if resp.status_code != 200:
            return jsonify({"error": "model_service_error", "status_code": resp.status_code, "details": resp.text}), 502

        model_json = resp.json()
        # Normalize response if service returns different keys
        preds = model_json.get("predictions", [])
        # If predictions contain "score" or "mock_score" or "raw", just return the service payload
        return jsonify({"service_response": model_json})

    except Exception as e:
        current_app.logger.exception("Error in reverse_analyze")
        return jsonify({"error": "internal_server_error", "details": str(e)}), 500
    finally:
        # cleanup
        try:
            import shutil
            shutil.rmtree(tmpdir)
        except Exception:
            pass
