import os
import json
import logging
import numpy as np
import cv2
from mtcnn import MTCNN
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input, TimeDistributed, GlobalAveragePooling2D,
    LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)

# Config (match training)
FRAMES_PER_CLIP = int(os.environ.get("REVENG_FRAMES", 15))
IMG_SIZE = int(os.environ.get("REVENG_IMG_SIZE", 96))
DEFAULT_MODEL_PATH = os.environ.get('REVENG_MODEL_PATH', 'models/revEng/df_revEng.h5')
DEFAULT_LABELS_PATH = os.environ.get('REVENG_LABELS_PATH', 'models/revEng/labels.json')

# Globals
_MODEL = None
_LABELS = None
_DETECTOR = None


def _load_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = MTCNN()
    return _DETECTOR


def build_multiclass_model(frames=FRAMES_PER_CLIP, img_size=IMG_SIZE, num_classes=2):
    """
    Recreates the model architecture used in training (MobileNetV2 as feature extractor
    wrapped in a TimeDistributed + LSTM head). Names align with training for clarity.
    """
    # Build a CNN feature extractor model that expects single frame input
    cnn_input = Input(shape=(img_size, img_size, 3), name="cnn_input")
    base = MobileNetV2(include_top=False, weights='imagenet', input_tensor=cnn_input)
    cnn_out = GlobalAveragePooling2D(name="cnn_gap")(base.output)
    cnn_model = Model(inputs=cnn_input, outputs=cnn_out, name="cnn_feature_extractor")
    cnn_model.trainable = False

    # Video-level model
    video_input = Input(shape=(frames, img_size, img_size, 3), name="video_input")
    x = TimeDistributed(cnn_model, name="time_distributed_cnn")(video_input)
    x = LSTM(64, return_sequences=False, name="lstm")(x)
    x = BatchNormalization(name="bn")(x)
    x = Dropout(0.4, name="drop1")(x)
    x = Dense(64, activation='relu', name="dense1")(x)
    x = Dropout(0.3, name="drop2")(x)
    output = Dense(num_classes, activation='softmax', name="predictions")(x)

    model = Model(inputs=video_input, outputs=output, name="reveng_model")
    return model


def _load_labels(labels_path=DEFAULT_LABELS_PATH):
    global _LABELS
    if _LABELS is not None:
        return _LABELS
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            _LABELS = json.load(f)
        logger.info(f"Loaded labels ({len(_LABELS)}) from {labels_path}")
    else:
        _LABELS = None
        logger.warning(f"No labels file found at {labels_path}. You should provide labels.json.")
    return _LABELS


def _load_model(model_path=DEFAULT_MODEL_PATH, labels_path=DEFAULT_LABELS_PATH):
    """
    Try to load the model directly. If TF loading fails due to TimeDistributed
    serialization issues, rebuild architecture and load weights (if possible).
    Returns (model, labels)
    """
    global _MODEL, _LABELS
    if _MODEL is not None and _LABELS is not None:
        return _MODEL, _LABELS

    labels = _load_labels(labels_path)

    # First try: direct load (SavedModel / Keras .keras / .h5 might work)
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        logger.info(f"Attempting to load model via tf.keras.models.load_model('{model_path}')")
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded with tf.keras.models.load_model.")
        _MODEL = model
        _LABELS = labels if labels is not None else _infer_labels_from_model(model)
        return _MODEL, _LABELS
    except Exception as e:
        logger.warning(f"Direct load_model() failed: {e}. Will attempt rebuild+load_weights fallback.")

    # Fallback: need labels to know num_classes
    if labels is None:
        raise RuntimeError("Cannot rebuild model without labels.json (needed to know num_classes).")

    num_classes = len(labels)
    logger.info(f"Rebuilding architecture with num_classes={num_classes}, frames={FRAMES_PER_CLIP}, img_size={IMG_SIZE}.")
    model = build_multiclass_model(frames=FRAMES_PER_CLIP, img_size=IMG_SIZE, num_classes=num_classes)

    # Load weights if model_path points to HDF5 / .h5 or .keras weight file
    if os.path.isfile(model_path) and model_path.endswith(('.h5', '.keras')):
        logger.info(f"Loading weights from {model_path} into rebuilt model.")
        model.load_weights(model_path)
        _MODEL = model
        _LABELS = labels
        return _MODEL, _LABELS
    else:
        raise RuntimeError(
            "Rebuild fallback requires an H5/keras weights file at model_path (endswith .h5 or .keras). "
            f"Found {model_path}. Either save weights as .h5 or ensure tf.keras.load_model() works."
        )


def _infer_labels_from_model(model):
    """
    If labels.json missing, attempt to infer number of classes from model output shape.
    Returns placeholder labels list.
    """
    try:
        out_shape = model.output_shape
        num = out_shape[-1]
        labels = [f"class_{i}" for i in range(num)]
        logger.warning("Labels not provided; using fallback labels: %s", labels)
        return labels
    except Exception:
        raise RuntimeError("Unable to infer number of classes from model. Provide labels.json.")


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


def video_to_faces_uniform(video_path, frames_needed=FRAMES_PER_CLIP, target_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("cv2.VideoCapture couldn't open: %s", video_path)
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total >= frames_needed and total > 0:
        indices = np.linspace(0, total - 1, frames_needed, dtype=int)
    elif 0 < total < frames_needed:
        idxs = list(np.linspace(0, total - 1, total, dtype=int))
        while len(idxs) < frames_needed:
            idxs.append(idxs[-1])
        indices = np.array(idxs[:frames_needed], dtype=int)
    else:
        faces = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while len(faces) < frames_needed:
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


def preprocess_faces_list(faces, frames=FRAMES_PER_CLIP, img_size=IMG_SIZE):
    while len(faces) < frames:
        faces.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    arr = np.array([faces[:frames]], dtype=np.float32) / 255.0
    return arr


def analyze_video(video_path, model_path=None, labels_path=None):
    """
    Returns dict:
      - success: {"predicted_label", "confidence", "all_probs"}
      - or failure: {"error", "message"}
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    labels_path = labels_path or DEFAULT_LABELS_PATH

    try:
        model, labels = _load_model(model_path, labels_path)
    except Exception as e:
        logger.exception("Model loading failed")
        return {"error": "model_load_failed", "message": str(e)}

    faces = video_to_faces_uniform(video_path, frames_needed=FRAMES_PER_CLIP, target_size=IMG_SIZE)
    if len(faces) == 0:
        return {"error": "no_faces_detected", "message": "No faces found in video frames."}

    inp = preprocess_faces_list(faces, frames=FRAMES_PER_CLIP, img_size=IMG_SIZE)
    try:
        probs = model.predict(inp, verbose=0)[0]
    except Exception as e:
        logger.exception("Inference failed")
        return {"error": "inference_failed", "message": str(e)}

    idx = int(np.argmax(probs))
    predicted = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
    all_probs = {(labels[i] if labels and i < len(labels) else f"class_{i}"): float(probs[i]) for i in range(len(probs))}

    return {"predicted_label": predicted, "confidence": float(probs[idx]), "all_probs": all_probs}
