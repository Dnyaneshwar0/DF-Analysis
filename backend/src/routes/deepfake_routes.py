from flask import Blueprint, request, jsonify
from ..utils.preprocessing import predict_video

deepfake_routes = Blueprint("deepfake_routes", __name__)

@deepfake_routes.route("/deepfake", methods=["POST"])
def deepfake():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video uploaded"}), 400

    file_path = f"temp_{file.filename}"
    file.save(file_path)

    result = predict_video(file_path)
    return jsonify({"deepfake_score": result})
