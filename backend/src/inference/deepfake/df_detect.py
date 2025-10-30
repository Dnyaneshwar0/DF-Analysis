# backend/src/inference/deepfake/service.py
"""
Deepfake model service (runs from inside backend/src/inference/deepfake).
Endpoints:
  GET  /health
  POST /predict  (multipart images[] or JSON base64 list)
Run with:
  python service.py
Env:
  MOCK_MODE=1        -> run without TensorFlow and return deterministic mock scores
  DF_MODEL_PATH      -> path to Keras model (defaults to ../models/deepfake/df_detect.keras)
  DF_MODEL_PORT      -> port to listen on (default 5100)
"""
import os
import io
from pathlib import Path
from typing import List
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

APP = Flask(__name__)

ROOT = Path(__file__).resolve().parents[2]  # backend/src/inference
DEFAULT_MODEL_PATH = str(ROOT / "models" / "deepfake" / "df_detect.keras")
MODEL_PATH = os.environ.get("DF_MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_VERSION = os.environ.get("DF_MODEL_VERSION", "dev")
MOCK = os.environ.get("MOCK_MODE", "0") == "1"

_MODEL = None
_INPUT_SIZE = (224, 224)

def _load_model():
    global _MODEL, _INPUT_SIZE
    if _MODEL is not None:
        return
    if MOCK:
        _MODEL = "MOCK"
        return
    try:
        from tensorflow.keras.models import load_model
    except Exception as e:
        raise RuntimeError("TensorFlow required to load model: " + str(e))
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    _MODEL = load_model(MODEL_PATH)
    shape = getattr(_MODEL, "input_shape", None)
    if shape and len(shape) >= 3:
        if len(shape) == 4:
            _, h, w, c = shape
        else:
            h, w = 224, 224
        _INPUT_SIZE = (int(h) if h else 224, int(w) if w else 224)

def _prepare_image_bytes(img_bytes: bytes):
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    im = im.resize((_INPUT_SIZE[1], _INPUT_SIZE[0]))
    arr = np.asarray(im).astype("float32") / 255.0
    return arr

def _predict_batch(batch: List[np.ndarray]):
    if MOCK:
        # deterministic pseudo-score: sum % 1000 / 1000
        return [float(round((np.sum(img) % 1000) / 1000.0, 6)) for img in batch]
    X = np.stack(batch, axis=0)
    preds = _MODEL.predict(X, batch_size=16)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] == 2:
        scores = preds[:, 1]
    elif preds.ndim == 2 and preds.shape[1] == 1:
        scores = preds[:, 0]
    else:
        scores = preds.reshape((preds.shape[0], -1)).mean(axis=1)
    return [float(round(float(s), 6)) for s in scores]

@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _MODEL is not None, "mock": MOCK, "model_path": MODEL_PATH})

@APP.route("/predict", methods=["POST"])
def predict():
    try:
        _load_model()
    except Exception as e:
        return jsonify({"error": "model_load_failed", "details": str(e)}), 500

    files = request.files.getlist("images") or request.files.getlist("images[]")
    imgs = []
    if files:
        for f in files:
            try:
                b = f.read()
                imgs.append(_prepare_image_bytes(b))
            except Exception as ex:
                return jsonify({"error": "bad_image", "details": str(ex)}), 400
    elif request.is_json:
        data = request.get_json()
        imgs_b64 = data.get("images", [])
        if not imgs_b64:
            return jsonify({"error": "no_images"}), 400
        import base64
        for b64 in imgs_b64:
            imgs.append(_prepare_image_bytes(base64.b64decode(b64)))
    else:
        return jsonify({"error": "no_images"}), 400

    scores = _predict_batch(imgs)
    return jsonify({"predictions": [{"index": i, "score": s} for i, s in enumerate(scores)], "model_version": MODEL_VERSION})

if __name__ == "__main__":
    host = os.environ.get("DF_MODEL_HOST", "0.0.0.0")
    port = int(os.environ.get("DF_MODEL_PORT", 5100))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Starting deepfake model service on {host}:{port} (mock={MOCK})")
    APP.run(host=host, port=port, debug=debug)
