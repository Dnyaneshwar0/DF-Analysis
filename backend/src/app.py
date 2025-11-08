import os
import sys
import logging
from flask import Flask, jsonify, send_from_directory  # ✅ added send_from_directory
from flask_cors import CORS

# --- ensure project root (parent of src/) is in sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- import blueprints from routes ---
from src.routes.reverse_routes import reverse_bp
from src.routes.deepfake_routes import deepfake_bp
from src.routes.emotion_routes import emotion_bp 

# Try to import optional preload helpers (non-fatal if they don't exist)
try:
    from src.inference.detection.df_detect import _load_model as _preload_deepfake_model
except Exception:
    _preload_deepfake_model = None

try:
    from src.inference.revEng.df_revEng import _load_model as _preload_reveng_model
except Exception:
    _preload_reveng_model = None

# Emotion module preload (optional — df_emo may not provide _load_model)
try:
    from src.inference.emotion.df_emo import _load_model as _preload_emotion_model
except Exception:
    _preload_emotion_model = None

def create_app(preload_models: bool = True):
    """
    Flask app factory – scalable for multiple modules + serves React build in production.
    """
    # ✅ Point to your React build folder (adjust if frontend path differs)
    REACT_BUILD_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "frontend", "build"))
    print("***************************************************************************************************************************************************************************************")
    print(REACT_BUILD_DIR)
    # ✅ Add static serving for production build
    app = Flask(
        __name__,
        static_folder=REACT_BUILD_DIR,
        static_url_path="/"
    )

    CORS(app)  # ✅ Allow frontend (React) to make API requests

    # Register API blueprints
    app.register_blueprint(reverse_bp, url_prefix="/reveng")
    app.register_blueprint(deepfake_bp, url_prefix="/detect")
    app.register_blueprint(emotion_bp, url_prefix="/emotion")

    # ✅ Serve React frontend build (for production)
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_react(path):
        build_dir = app.static_folder
        target = os.path.join(build_dir, path)
        if path != "" and os.path.exists(target):
            return app.send_static_file(path)
        else:
            return app.send_static_file("index.html")
        
    # Health check route
    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "ok"}), 200

    # Configure logger for startup messages
    logger = logging.getLogger(__name__)

    # Optionally preload models (lazy by default)
    if preload_models:
        if _preload_deepfake_model is not None:
            try:
                _preload_deepfake_model()
                logger.info("Preloaded deepfake model at startup.")
            except Exception as e:
                logger.warning("Deepfake model preload failed (lazy-load on first request): %s", e)

        if _preload_reveng_model is not None:
            try:
                _preload_reveng_model()
                logger.info("Preloaded revEng model at startup.")
            except Exception as e:
                logger.warning("RevEng model preload failed (lazy-load on first request): %s", e)
                
        if _preload_emotion_model is not None:
            try:
                _preload_emotion_model()
                logger.info("Preloaded emotion model at startup.")
            except Exception as e:
                logger.warning("Emotion model preload failed (lazy-load on first request): %s", e)

    return app

# --- main entry point ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(preload_models=True)

    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"

    logging.getLogger(__name__).info(f"Starting Flask app on port {port} (debug={debug}) ...")
    app.run(host="0.0.0.0", port=port, debug=debug)
