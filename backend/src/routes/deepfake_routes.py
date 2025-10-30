# backend/src/routes/deepfake_routes.py
import os
import tempfile
import time
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from src.utils.preprocessing import save_uploaded_file, extract_faces_from_video
from src.utils.postprocessing import combine_detections
import requests
import cv2
import numpy as np
import shutil

deepfake_bp = Blueprint("deepfake", __name__)

MODEL_SERVICE_URL = os.environ.get("DF_SERVICE_URL", "http://localhost:5100")
PREDICT_ENDPOINT = MODEL_SERVICE_URL.rstrip("/") + "/predict"

logger = logging.getLogger("deepfake_routes")


@deepfake_bp.route("/analyze", methods=["POST"])
def analyze_video():
    """
    Accepts multipart/form-data with key 'video'.
    - extracts faces (faces, meta)
    - sends faces as images[] to the model service
    - receives per-face predictions (expects list of {"index":..,"score":..} or {"score":..,"frame_index":..,"bbox":..})
    - combines detections into a single video-level result via combine_detections()
    """
    if "video" not in request.files:
        return jsonify({"error": "Missing 'video' file in request"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    tmpdir = tempfile.mkdtemp(prefix="deepfake_upload_")
    video_path = save_uploaded_file(video_file, tmpdir)

    start_ts = time.time()
    try:
        # extract faces and metadata (list of faces as BGR numpy arrays, meta aligned by index)
        faces, meta = extract_faces_from_video(str(video_path), sample_rate=1)
        if len(faces) == 0:
            return jsonify({"error": "No faces found in the provided video"}), 400

        # build multipart files for model service
        files = []
        for i, face in enumerate(faces):
            ok, buf = cv2.imencode(".jpg", face)
            if not ok:
                continue
            files.append(("images", (f"face_{i}.jpg", buf.tobytes(), "image/jpeg")))

        if len(files) == 0:
            return jsonify({"error": "Failed to encode any faces"}), 500

        # call model service
        try:
            resp = requests.post(PREDICT_ENDPOINT, files=files, timeout=60)
        except requests.exceptions.RequestException as re:
            logger.exception("Error calling model service")
            return jsonify({"error": "model_service_unreachable", "details": str(re)}), 502

        if resp.status_code != 200:
            return jsonify({
                "error": "model_service_error",
                "status_code": resp.status_code,
                "details": resp.text
            }), 502

        model_json = resp.json()
        preds = model_json.get("predictions", [])

        # Build unified detections list: each detection must have frame_index, bbox, score
        detections = []

        # Case A: predictions already include frame_index and bbox (service did the mapping)
        service_has_meta = False
        if preds and isinstance(preds, list):
            first = preds[0]
            if isinstance(first, dict) and ("frame_index" in first or "bbox" in first):
                service_has_meta = True

        if service_has_meta:
            # Normalize service-returned predictions
            for p in preds:
                try:
                    frame_idx = int(p.get("frame_index", 0))
                    bbox = p.get("bbox", [0, 0, 0, 0])
                    score = float(p.get("score", p.get("mock_score", p.get("raw", 0.0))))
                    detections.append({"frame_index": frame_idx, "bbox": bbox, "score": score})
                except Exception:
                    # skip malformed entry
                    continue
        else:
            # Service only returned scores (or unspecified). We need to align with local meta.
            # If lengths match, align by index. Otherwise align up to min length.
            n_meta = len(meta)
            n_preds = len(preds)
            n = min(n_meta, n_preds)

            if n == 0:
                # nothing to combine
                return jsonify({"error": "no_predictions_returned"}), 500

            # If preds are simple values (numbers), extract score appropriately
            for i in range(n):
                p = preds[i]
                # score could be a dict or scalar
                if isinstance(p, dict):
                    score = float(p.get("score", p.get("mock_score", p.get("raw", 0.0))))
                else:
                    # scalar like 0.123
                    try:
                        score = float(p)
                    except Exception:
                        score = 0.0
                # meta holds bbox & frame
                mm = meta[i]
                detections.append({
                    "frame_index": int(mm["frame_idx"]),
                    "bbox": mm["bbox"],
                    "score": score
                })

            # If there are more preds than meta (rare), append remaining preds without bbox/frame
            if n_preds > n_meta:
                for i in range(n_meta, n_preds):
                    p = preds[i]
                    if isinstance(p, dict):
                        score = float(p.get("score", p.get("mock_score", p.get("raw", 0.0))))
                        frame_idx = int(p.get("frame_index", 0))
                        bbox = p.get("bbox", [0, 0, 0, 0])
                    else:
                        try:
                            score = float(p)
                        except Exception:
                            score = 0.0
                        frame_idx = 0
                        bbox = [0, 0, 0, 0]
                    detections.append({"frame_index": frame_idx, "bbox": bbox, "score": score})

        # Combine into video-level result
        combined = combine_detections(
            detections,
            iou_threshold=float(os.environ.get("DF_IOU_THRESHOLD", 0.45)),
            max_frame_gap=int(os.environ.get("DF_MAX_FRAME_GAP", 4)),
            decision_threshold=float(os.environ.get("DF_DECISION_THRESHOLD", 0.5)),
            top_k_for_decision=int(os.environ.get("DF_TOP_K", 5))
        )

        elapsed = time.time() - start_ts

        response = {
            "label": combined["label"],
            "confidence": combined["confidence"],
            "metrics": combined["metrics"],
            "tracks": combined["tracks"],
            "top_detections": combined.get("top_detections", []),
            "n_input_faces": len(faces),
            "n_detections": combined["metrics"].get("n_detections"),
            "model_version": model_json.get("model_version"),
            "elapsed_seconds": elapsed,
            "raw_per_detection": combined["per_detection"]
        }

        return jsonify(response)

    except Exception as e:
        logger.exception("Error analyzing video")
        return jsonify({"error": "internal_server_error", "details": str(e)}), 500
    finally:
        # cleanup
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
