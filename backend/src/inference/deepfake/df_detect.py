import os
import logging
import numpy as np
import cv2
from mtcnn import MTCNN
import tensorflow as tf

logger = logging.getLogger(__name__)

# Config (can be overridden by env)
FRAMES_PER_CLIP = int(os.environ.get("DF_FRAMES", 15))
IMG_SIZE = int(os.environ.get("DF_IMG_SIZE", 96))
DEFAULT_MODEL_PATH = os.environ.get('DF_MODEL_PATH', 'models/deepfake/df_detect.keras')

# Globals
_MODEL = None
_DETECTOR = None


def _load_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = MTCNN()
    return _DETECTOR


def _load_model(model_path=DEFAULT_MODEL_PATH):
    """
    Load a .keras SavedModel or TF SavedModel dir. Raises on failure.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    try:
        logger.info(f"Loading deepfake model from {model_path}")
        _MODEL = tf.keras.models.load_model(model_path, compile=False)
        return _MODEL
    except Exception as e:
        logger.exception("Failed to load model")
        raise RuntimeError(f"Failed to load model: {e}")


def extract_largest_face(frame, target_size=IMG_SIZE):
    detector = _load_detector()
    try:
        res = detector.detect_faces(frame)
        if not res:
            return None
        best = max(res, key=lambda r: max(0, r['box'][2]) * max(0, r['box'][3]))
        x, y, w, h = best['box']
        x, y = max(0, x), max(0, y)
        x2, y2 = x + max(0, w), y + max(0, h)
        face = frame[y:y2, x:x2]
        if face.size == 0:
            return None
        face = cv2.resize(face, (target_size, target_size))
        return face
    except Exception as e:
        logger.debug("Face extraction error: %s", e)
        return None


def video_to_faces(video_path, max_frames=FRAMES_PER_CLIP, target_size=IMG_SIZE):
    """
    Sample frames uniformly (or pad if video short) and extract largest face per frame.
    Mirrors the approach in your `predict_video` snippet.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("cv2.VideoCapture couldn't open: %s", video_path)
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total > 0:
        # pick uniform indices across the video
        if total >= max_frames:
            indices = np.linspace(0, total - 1, max_frames, dtype=int)
        else:
            # repeat last frame index to pad if fewer frames than needed
            idxs = list(np.linspace(0, total - 1, total, dtype=int))
            while len(idxs) < max_frames:
                idxs.append(idxs[-1])
            indices = np.array(idxs[:max_frames], dtype=int)
    else:
        # unknown total frames (some containers) -> read sequentially until we get enough or EOF
        faces = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while len(faces) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            f = extract_largest_face(frame, target_size)
            if f is not None:
                faces.append(f)
        cap.release()
        return faces

    faces = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        f = extract_largest_face(frame, target_size)
        if f is not None:
            faces.append(f)

    cap.release()
    return faces


def image_to_faces(image_path, max_frames=FRAMES_PER_CLIP, target_size=IMG_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("cv2.imread couldn't read image: %s", image_path)
        return []
    face = extract_largest_face(img, target_size)
    if face is None:
        return []
    # replicate same face to create the frames sequence expected by the model
    return [face.copy() for _ in range(max_frames)]


def preprocess_faces_list(faces, frames=FRAMES_PER_CLIP, img_size=IMG_SIZE):
    while len(faces) < frames:
        faces.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    arr = np.array([faces[:frames]], dtype=np.float32) / 255.0
    return arr


def analyze_media(media_path, model_path=None):
    """
    Returns dict:
      - success: {"predicted_label", "confidence", "all_probs"}
      - or failure: {"error", "message"}
    Interprets single-output models as sigmoid binary (REAL/FAKE) per your predict_video.
    """
    model_path = model_path or DEFAULT_MODEL_PATH

    # load model
    try:
        model = _load_model(model_path)
    except Exception as e:
        logger.exception("Model loading failed")
        return {"error": "model_load_failed", "message": str(e)}

    _, ext = os.path.splitext(media_path.lower())
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    if ext in image_exts:
        faces = image_to_faces(media_path, max_frames=FRAMES_PER_CLIP, target_size=IMG_SIZE)
    else:
        faces = video_to_faces(media_path, max_frames=FRAMES_PER_CLIP, target_size=IMG_SIZE)

    if len(faces) == 0:
        return {"error": "no_faces_detected", "message": "No faces found in input."}

    inp = preprocess_faces_list(faces, frames=FRAMES_PER_CLIP, img_size=IMG_SIZE)

    try:
        preds = model.predict(inp, verbose=0)
    except Exception as e:
        logger.exception("Inference failed")
        return {"error": "inference_failed", "message": str(e)}

    # Normalize prediction array shape
    preds_arr = np.asarray(preds).reshape(-1)
    # Case A: single probability output (binary sigmoid) -> use your label logic
    if preds_arr.size == 1:
        prob = float(preds_arr[0])
        label = 'FAKE' if prob > 0.5 else 'REAL'
        confidence = prob if prob > 0.5 else (1.0 - prob)
        all_probs = {"REAL": float(1.0 - prob), "FAKE": float(prob)}
        return {"predicted_label": label, "confidence": float(confidence), "all_probs": all_probs}

    # Case B: multiclass softmax-like output
    idx = int(np.argmax(preds_arr))
    # try to derive placeholder labels
    labels = [f"class_{i}" for i in range(preds_arr.size)]
    predicted = labels[idx]
    all_probs = {labels[i]: float(preds_arr[i]) for i in range(preds_arr.size)}
    return {"predicted_label": predicted, "confidence": float(preds_arr[idx]), "all_probs": all_probs}
