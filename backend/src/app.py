import os
import sys
import logging
from flask import Flask, jsonify
from flask_cors import CORS

# --- ensure project root (parent of src/) is in sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- import blueprints from routes ---
from src.routes.reverse_routes import reverse_bp
from src.routes.deepfake_routes import deepfake_bp

# Try to import optional preload helpers (non-fatal if they don't exist)
try:
    from src.inference.deepfake.df_detect import _load_model as _preload_deepfake_model
except Exception:
    _preload_deepfake_model = None

try:
    from src.inference.revEng.df_revEng import _load_model as _preload_reveng_model
except Exception:
    _preload_reveng_model = None

def create_app(preload_models: bool = True):
    """
    Flask app factory – scalable for multiple modules.
    Set preload_models=False if you don't want model loading attempted at startup.
    """
    # app = Flask(__name__, static_url_path='/data', static_folder='data')
    app = Flask(__name__)
    CORS(app)  # ✅ Allow frontend (React) to make API requests
    

    # Register blueprints here
    # Reverse engineering endpoints
    app.register_blueprint(reverse_bp, url_prefix="/reveng")
    # Deepfake detection endpoints
    app.register_blueprint(deepfake_bp, url_prefix="/detect")

    # Health check route
    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "ok"}), 200

    # Configure logger for startup messages
    logger = logging.getLogger(__name__)

    # Optionally preload models (lazy by default). Wrap in try/except so app still starts if models fail.
    if preload_models:
        # Preload deepfake model
        if _preload_deepfake_model is not None:
            try:
                _preload_deepfake_model()
                logger.info("Preloaded deepfake model at startup.")
            except Exception as e:
                logger.warning("Deepfake model preload failed (will lazy-load on first request): %s", e)
        # Preload revEng model
        if _preload_reveng_model is not None:
            try:
                _preload_reveng_model()
                logger.info("Preloaded revEng model at startup.")
            except Exception as e:
                logger.warning("RevEng model preload failed (will lazy-load on first request): %s", e)
    return app

# --- main entry point ---
if __name__ == "__main__":
    # Basic logging configuration for dev
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = create_app(preload_models=True)

    # Host on all interfaces for dev/testing
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"

    logging.getLogger(__name__).info(f"Starting Flask app on port {port} (debug={debug}) ...")
    app.run(host="0.0.0.0", port=port, debug=debug)
