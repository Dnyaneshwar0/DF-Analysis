import sys
from pathlib import Path

# Point Python to the backend root (one level above /src)
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from flask import Flask

def create_app():
    app = Flask(__name__)

    # Register blueprints
    from src.routes.deepfake_routes import deepfake_bp
    from src.routes.reverse_routes import reverse_bp
    from src.routes.emotion_routes import emotion_bp

    app.register_blueprint(deepfake_bp, url_prefix="/deepfake")
    app.register_blueprint(reverse_bp, url_prefix="/reverse")
    app.register_blueprint(emotion_bp, url_prefix="/emotion")

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok", "root": str(Path(__file__).resolve().parents[2])}

    return app

if __name__ == "__main__":
    app = create_app()
    # For local dev only. Use gunicorn for production.
    app.run(host="0.0.0.0", port=5000, debug=True)
