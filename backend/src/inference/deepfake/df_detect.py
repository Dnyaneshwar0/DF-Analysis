import os
import logging
import numpy as np
import cv2
import base64
import csv
import subprocess
from mtcnn import MTCNN
import tensorflow as tf

logger = logging.getLogger(__name__)

FRAMES_PER_CLIP = int(os.environ.get("DF_FRAMES", 15))
IMG_SIZE = int(os.environ.get("DF_IMG_SIZE", 96))

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'deepfake'))
MODEL_H5_PATH = os.path.join(MODEL_DIR, 'deepfake_efficient_model.h5')
DEFAULT_MODEL_PATH = os.environ.get('DF_MODEL_PATH', MODEL_H5_PATH)

# Output directories
OUTPUT_FRAMES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'frames'))
OUTPUT_CSV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'csv'))

# Create directories if they don't exist
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

_MODEL = None
_DETECTOR = None

def _load_detector():
    global _DETECTOR
    if _DETECTOR is None:
        logger.info("Initializing MTCNN face detector...")
        _DETECTOR = MTCNN()
    return _DETECTOR

def build_model():
    logger.info("Building model from scratch...")
    input_layer = tf.keras.layers.Input(shape=(FRAMES_PER_CLIP, IMG_SIZE, IMG_SIZE, 3))
    base_cnn = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_cnn.trainable = True
    for layer in base_cnn.layers[:-10]:
        layer.trainable = False
    
    x = tf.keras.layers.TimeDistributed(base_cnn)(input_layer)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss='binary_crossentropy', metrics=['accuracy'])
    logger.info("✓ Model built successfully")
    return model

def _load_model(model_path=DEFAULT_MODEL_PATH):
    global _MODEL
    if _MODEL is not None:
        logger.info("Model already loaded from cache")
        return _MODEL
    
    logger.info(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        logger.info(f"Loading .h5 model...")
        _MODEL = tf.keras.models.load_model(model_path, compile=False)
        logger.info("✓ .h5 model loaded successfully!")
        return _MODEL
    except Exception as e:
        logger.warning(f".h5 load failed: {e}")
        logger.warning("Rebuilding model and attempting weight load...")
        try:
            _MODEL = build_model()
            _MODEL.load_weights(model_path)
            logger.info("✓ Model rebuilt and weights loaded")
            return _MODEL
        except Exception as e2:
            logger.error(f"Weight loading failed: {e2}")
            raise RuntimeError(f"Cannot load model: {e2}")

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
    logger.info(f"Extracting faces from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("cv2.VideoCapture couldn't open: %s", video_path)
        return [], [], 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30   
    duration = total / fps if fps > 0 else 0
    logger.info(f"Video has {total} frames ({duration:.1f}s), extracting {max_frames} face samples")
    faces = []
    metadata = []
    if total > 0:
        indices = (np.linspace(0, total - 1, max_frames, dtype=int)
                   if total >= max_frames
                   else np.array(list(np.linspace(0, total - 1, total, dtype=int)) + [total - 1] * (max_frames - total), dtype=int))
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            f = extract_largest_face(frame, target_size)
            if f is not None:
                faces.append(f)
                metadata.append((int(idx), round(idx / fps, 2)))
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while len(faces) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            f = extract_largest_face(frame, target_size)
            if f is not None:
                faces.append(f)
                metadata.append((len(faces) - 1, round((len(faces) - 1) / fps, 2)))
    cap.release()
    while len(faces) < max_frames:
        faces.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))
        metadata.append((-1, -1.0))
    logger.info(f"✓ Extracted {len(faces)} face frames")
    return faces, metadata, duration

def image_to_faces(image_path, max_frames=FRAMES_PER_CLIP, target_size=IMG_SIZE):
    logger.info(f"Extracting face from image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("cv2.imread couldn't read image: %s", image_path)
        return [], [], 0
    face = extract_largest_face(img, target_size)
    if face is None:
        return [], [], 0
    faces = [face.copy() for _ in range(max_frames)]
    metadata = [(-1, -1.0)] * max_frames
    return faces, metadata, 0

def preprocess_faces_list(faces, frames=FRAMES_PER_CLIP, img_size=IMG_SIZE):
    while len(faces) < frames:
        faces.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    arr = np.array([faces[:frames]], dtype=np.float32) / 255.0
    logger.info(f"Preprocessed input shape: {arr.shape}")
    return arr

def save_frames_to_disk(frames_data, media_basename):
    """Save top frames to disk as PNG files"""
    saved_paths = []
    for idx, (frame, timestamp, confidence) in enumerate(frames_data, 1):
        filename = f"{media_basename}_frame_{idx}_ts{timestamp}.png"
        filepath = os.path.join(OUTPUT_FRAMES_DIR, filename)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)
        saved_paths.append((filepath, f"/assets/frames/{filename}"))
        logger.info(f"Saved frame {idx} to {filepath}")
    return saved_paths

def save_metadata_to_csv(frames_data, media_basename):
    """Save frame metadata to CSV"""
    csv_filename = f"{media_basename}_metadata.csv"
    csv_filepath = os.path.join(OUTPUT_CSV_DIR, csv_filename)
    
    with open(csv_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_index', 'timestamp_sec', 'fake_confidence'])
        for idx, (frame, timestamp, confidence) in enumerate(frames_data, 1):
            writer.writerow([idx, timestamp, confidence])
    
    logger.info(f"Saved metadata to {csv_filepath}")
    return csv_filepath

def extract_frames_ffmpeg(video_path, timestamps, media_basename):
    """Use FFmpeg to extract frames at specific timestamps"""
    extracted_frames = []
    
    for idx, ts in enumerate(timestamps, 1):
        filename = f"{media_basename}_frame_{idx}_ts{ts}.png"
        filepath = os.path.join(OUTPUT_FRAMES_DIR, filename)
        
        # FFmpeg command to extract frame at timestamp
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(ts),
            '-vframes', '1',
            '-vf', 'scale=96:96',
            '-y',
            filepath
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=10)
            extracted_frames.append((filepath, f"/assets/frames/{filename}"))
            logger.info(f"Extracted frame {idx} at timestamp {ts}s")
        except Exception as e:
            logger.error(f"FFmpeg extraction failed for frame {idx}: {e}")
    
    return extracted_frames

def analyze_media_with_top_frames_and_images(media_path, model_path=None, top_k=5):
    logger.info(f"=== Starting analysis for: {media_path} ===")
    model_path = model_path or DEFAULT_MODEL_PATH
    
    try:
        model = _load_model(model_path)
        intermediate_model = tf.keras.Model(
            inputs=model.input, 
            outputs=model.get_layer(index=2).output
        )
    except Exception as e:
        logger.exception("Model loading failed")
        return {"error": "model_load_failed", "message": str(e)}
    
    _, ext = os.path.splitext(media_path.lower())
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    media_basename = os.path.splitext(os.path.basename(media_path))[0]
    
    if ext in image_exts:
        faces, metadata, duration = image_to_faces(media_path, max_frames=FRAMES_PER_CLIP, target_size=IMG_SIZE)
        is_video = False
    else:
        faces, metadata, duration = video_to_faces(media_path, max_frames=FRAMES_PER_CLIP, target_size=IMG_SIZE)
        is_video = True
    
    if len(faces) == 0:
        logger.warning("No faces detected in input media")
        return {"error": "no_faces_detected", "message": "No faces found in input."}
    
    inp = preprocess_faces_list(faces, frames=FRAMES_PER_CLIP, img_size=IMG_SIZE)
    
    try:
        logger.info("Running model prediction...")
        preds = model.predict(inp, verbose=0)
        cnn_features = intermediate_model.predict(inp, verbose=0)
    except Exception as e:
        logger.exception("Inference failed")
        return {"error": "inference_failed", "message": str(e)}
    
    prob = float(preds[0][0])
    label = 'FAKE' if prob > 0.5 else 'REAL'
    confidence = prob if prob > 0.5 else 1.0 - prob
    
    frame_scores = [(idx, float(np.sum(np.abs(cnn_features[0, idx])))) for idx in range(cnn_features.shape[1])]
    top_frames = sorted(frame_scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    max_score = max(score for _, score in top_frames) if top_frames else 1
    
    # Prepare top frames data
    top_frames_data = []
    timestamps_for_extraction = []
    
    for rank, (idx_frame, score) in enumerate(top_frames, 1):
        norm_score = score / max_score if max_score != 0 else 0
        frame_idx, timestamp = metadata[idx_frame] if idx_frame < len(metadata) else (-1, -1.0)
        top_frames_data.append((faces[idx_frame], timestamp, norm_score))
        timestamps_for_extraction.append(timestamp)
    
    # Save metadata to CSV
    csv_path = save_metadata_to_csv(top_frames_data, media_basename)
    
    # Extract frames using FFmpeg if it's a video
    if is_video:
        try:
            frame_paths = extract_frames_ffmpeg(media_path, timestamps_for_extraction, media_basename)
        except Exception as e:
            logger.warning(f"FFmpeg extraction failed: {e}. Saving frames to disk instead.")
            frame_paths = save_frames_to_disk(top_frames_data, media_basename)
    else:
        frame_paths = save_frames_to_disk(top_frames_data, media_basename)
    
    # Build final response
    top_frames_list = []
    for (filepath, web_url), (frame, timestamp, confidence) in zip(frame_paths, top_frames_data):
        top_frames_list.append({
            "timestamp_sec": float(timestamp),
            "frame_index": len(top_frames_list) + 1,
            "fake_confidence": round(confidence, 4),
            "url": web_url
        })
    
    logger.info(f"✓ Final Result: {label} (confidence: {confidence:.3f})")
    
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "metadata": {
            "duration_sec": round(duration, 2),
            "frames_analyzed": len(faces)
        },
        "top_frames": top_frames_list,
        "csv_path": csv_path
    }
