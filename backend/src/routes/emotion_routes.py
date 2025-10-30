from flask import Blueprint, jsonify

emotion_bp = Blueprint("emotion", __name__)

@emotion_bp.route("/analyze", methods=["POST"])
def analyze_emotion():
    """
    Emotion endpoint intentionally left as a stub (not implemented yet).
    """
    return jsonify({"error": "Emotion analysis not implemented yet"}), 501
