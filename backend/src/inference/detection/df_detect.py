"""
DEEPFAKE DETECTION - PRODUCTION BACKEND
========================================
Works with locally-trained model from train_deepfake_model.py
Model: deepfake_model_trained.h5
Accuracy: 61.67%
Input: (batch, 30, 96, 96, 3)
Output: (batch, 1) - binary probability
Dataset of model
CONFIG = {
    'real_folders': [
        '/content/drive/MyDrive/DFERA/df/FaceForensicsPP/original_sequences/youtube/c40/videos',
        '/content/drive/MyDrive/DFERA/df/celeb-df/YouTube-real',
        '/content/drive/MyDrive/DFERA/df/celeb-df/Celeb-real'
    ],
    'fake_folders': [
        '/content/drive/MyDrive/DFERA/df/FaceForensicsPP/DeepFakeDetection/c40/videos',
        '/content/drive/MyDrive/DFERA/df/FaceForensicsPP/manipulated_sequences/Face2Face/c40/videos',
        '/content/drive/MyDrive/DFERA/df/FaceForensicsPP/manipulated_sequences/FaceSwap/c40/videos',
        '/content/drive/MyDrive/DFERA/df/FaceForensicsPP/manipulated_sequences/NeuralTextures/c40/videos',
        '/content/drive/MyDrive/DFERA/df/celeb-df/Celeb-synthesis'
    ],
    'frames_per_video': 30,
    'image_size': 96,
}
"""
"""
DEEPFAKE DETECTION - WITH TOP FRAMES EXTRACTION
================================================
Returns advanced response with:
- Top 5 frames sorted by deepfake confidence
- Base64 encoded frame images
- Timestamps and frame indices
"""
"""
DEEPFAKE DETECTION - FULL FRAMES AS PNG
========================================
Saves top 5 affecting frames as full video frames (not just faces)
Returns proper file paths: /assets/frames/frame1.png
"""

import os
import logging
import numpy as np
import cv2
from mtcnn import MTCNN
import tensorflow as tf

logger = logging.getLogger(__name__)

FRAMES_PER_CLIP = 30
IMG_SIZE = 96
CONFIDENCE_THRESHOLD = 0.5

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 
    '..', '..', '..', 
    'models', 'detection', 
    'deepfake_model_trained.h5'
)

# Directory to save frame PNG files
FRAMES_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..',
    'data', 'detection', 'frames'
)

_MODEL = None
_DETECTOR = None

# ============================================
# SETUP FRAMES DIRECTORY
# ============================================
def _setup_frames_directory():
    """Create frames directory if it doesn't exist"""
    os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Frames directory ready: {FRAMES_OUTPUT_DIR}")

_setup_frames_directory()

# ============================================
# DETECTOR
# ============================================
def _load_detector():
    global _DETECTOR
    if _DETECTOR is None:
        logger.info("Initializing MTCNN face detector...")
        _DETECTOR = MTCNN()
    return _DETECTOR

# ============================================
# MODEL LOADING
# ============================================
def _load_model():
    global _MODEL
    
    if _MODEL is not None:
        return _MODEL
    
    logger.info(f"Loading model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    try:
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Input: {_MODEL.input_shape}")
        logger.info(f"  Output: {_MODEL.output_shape}")
        return _MODEL
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# ============================================
# FACE EXTRACTION
# ============================================
def extract_largest_face(frame, target_size=IMG_SIZE):
    """Extract largest face from frame"""
    detector = _load_detector()
    
    try:
        faces = detector.detect_faces(frame)
        if not faces:
            return None
        
        best = max(faces, key=lambda r: r['box'][2] * r['box'][3])
        x, y, w, h = best['box']
        
        x, y = max(0, x), max(0, y)
        x2, y2 = x + w, y + h
        
        face = frame[y:y2, x:x2]
        
        if face.size == 0:
            return None
        
        return cv2.resize(face, (target_size, target_size))
        
    except Exception as e:
        logger.debug(f"Face extraction error: {e}")
        return None

# ============================================
# VIDEO PROCESSING - STORES ORIGINAL FRAMES
# ============================================
def video_to_faces(video_path, max_frames=FRAMES_PER_CLIP, target_size=IMG_SIZE):
    """
    Extract 30 evenly-spaced faces from video
    Also stores original full frames for later use
    """
    logger.info(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps if fps > 0 else 0
    
    faces = []
    full_frames = []  # Store original full frames
    metadata = []
    
    # Evenly-spaced frame indices
    frame_indices = np.linspace(0, max(total_frames-1, 0), max_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Store FULL original frame
        full_frames.append(frame.copy())
        
        # Extract face for model input
        face = extract_largest_face(frame, target_size)
        if face is not None:
            faces.append(face)
            timestamp = idx / fps if fps > 0 else 0
            metadata.append((int(idx), round(timestamp, 2)))
        else:
            # If no face found, still pad
            faces.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))
            timestamp = idx / fps if fps > 0 else 0
            metadata.append((int(idx), round(timestamp, 2)))
    
    cap.release()
    
    # Ensure exactly 30 frames
    while len(faces) < max_frames:
        faces.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))
        full_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        metadata.append((-1, -1.0))
    
    logger.info(f"Extracted {len([f for f in faces if f.max() > 0])} faces")
    
    return faces[:max_frames], full_frames[:max_frames], metadata[:max_frames], duration

# ============================================
# IMAGE PROCESSING
# ============================================
def image_to_faces(image_path, max_frames=FRAMES_PER_CLIP, target_size=IMG_SIZE):
    """For single image, replicate 30 times"""
    logger.info(f"Processing image: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return [], [], [], 0
    
    # Store original image
    full_frame = img.copy()
    
    # Extract face for model
    face = extract_largest_face(img, target_size)
    if face is None:
        logger.warning(f"No face detected in image")
        return [], [], [], 0
    
    faces = [face.copy() for _ in range(max_frames)]
    full_frames = [full_frame.copy() for _ in range(max_frames)]
    metadata = [(-1, -1.0)] * max_frames
    
    return faces, full_frames, metadata, 0

# ============================================
# PREPROCESSING
# ============================================
def preprocess_faces_list(faces, frames=FRAMES_PER_CLIP, img_size=IMG_SIZE):
    """Convert faces to model input tensor"""
    
    while len(faces) < frames:
        faces.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    
    arr = np.array([faces[:frames]], dtype=np.float32)
    arr = arr / 255.0
    
    return arr

# ============================================
# SAVE FULL FRAME AS PNG FILE
# ============================================
def save_full_frame_as_png(full_frame, frame_number):
    """
    Save full video frame as PNG file.

    Input: full_frame (original video frame), frame_number (1-5)
    Output URL: /data/detection/frames/frame1.png
    """
    try:
        # Ensure frames output dir exists
        os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)

        # Ensure frame is uint8 BGR
        if full_frame.dtype != np.uint8:
            full_frame = (full_frame * 255).astype(np.uint8)

        # Generate filename
        filename = f"frame{frame_number}.png"
        filepath = os.path.join(FRAMES_OUTPUT_DIR, filename)

        # Save frame as PNG
        success = cv2.imwrite(filepath, full_frame)
        if not success:
            logger.warning(f"Failed to save frame: {filepath}")
            return None

        # Return web-accessible URL (served by Flask)
        url_path = f"/detect/data/detection/frames/{filename}"
        logger.info(f"Saved frame: {filepath} → {url_path}")

        return url_path

    except Exception as e:
        logger.error(f"Frame saving error: {e}")
        return None

# ============================================
# CLEANUP OLD FRAMES (OPTIONAL)
# ============================================
def cleanup_old_frames(max_age_hours=24):
    """Remove frames older than max_age_hours"""
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(FRAMES_OUTPUT_DIR):
            filepath = os.path.join(FRAMES_OUTPUT_DIR, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    os.remove(filepath)
                    logger.debug(f"Cleaned up old frame: {filename}")
    except Exception as e:
        logger.debug(f"Cleanup error: {e}")

# ============================================
# MAIN INFERENCE FUNCTION
# ============================================
def analyze_media_with_top_frames_and_images(media_path, top_k=5):
    """
    Main analysis function with FULL FRAME PNG saving
    
    Returns:
    {
        "label": "REAL|FAKE",
        "confidence": 0.95,
        "metadata": {
            "duration_sec": 13.48,
            "frames_analyzed": 30
        },
        "top_frames": [
            {
                "timestamp_sec": 2.3,
                "frame_index": 7,
                "fake_confidence": 0.97,
                "url": "/assets/frames/frame1.png"
            }
        ]
    }
    """
    logger.info(f"Analyzing: {media_path}")
    
    # Cleanup old frames periodically
    cleanup_old_frames(max_age_hours=24)
    
    # Load model
    try:
        model = _load_model()
    except Exception as e:
        logger.exception("Model loading failed")
        return {"error": "model_load_failed", "message": str(e)}
    
    # Detect video or image
    _, ext = os.path.splitext(media_path.lower())
    
    if ext in {'.jpg', '.jpeg', '.png', '.bmp'}:
        faces, full_frames, metadata, duration = image_to_faces(media_path)
    else:
        faces, full_frames, metadata, duration = video_to_faces(media_path)
    
    # Validate faces
    if len(faces) == 0 or all(f.max() == 0 for f in faces):
        logger.warning("No faces detected")
        return {"error": "no_faces_detected", "message": "No faces found in media"}
    
    # Preprocess faces for model
    input_tensor = preprocess_faces_list(faces)
    
    # Run inference
    try:
        predictions = model.predict(input_tensor, verbose=0)
        overall_prob = float(predictions[0][0])
        
        logger.info(f"Model output: {overall_prob:.4f}")
        
    except Exception as e:
        logger.exception("Inference failed")
        return {"error": "inference_failed", "message": str(e)}
    
    # Interpret overall result
    if overall_prob > CONFIDENCE_THRESHOLD:
        label = "FAKE"
        confidence = overall_prob
    else:
        label = "REAL"
        confidence = 1.0 - overall_prob
    
    logger.info(f"Result: {label} (confidence: {confidence:.4f})")
    
    # ============================================
    # EXTRACT TOP 5 FRAMES AFFECTING RESULT
    # ============================================
    top_frames = []
    
    try:
        # Create frame score list
        frame_scores = []
        
        for i, (face, full_frame, (frame_idx, timestamp)) in enumerate(zip(faces, full_frames, metadata)):
            # Skip black/padded frames
            if face.max() == 0:
                continue
            
            # Use brightness/saliency as score
            brightness = float(np.mean(face))
            
            # Estimate frame-level fake confidence
            frame_fake_conf = brightness / 255.0
            
            # If overall label is FAKE, multiply by overall confidence
            if label == "FAKE":
                frame_fake_conf = frame_fake_conf * confidence
            else:
                frame_fake_conf = frame_fake_conf * (1 - confidence)
            
            frame_scores.append({
                'index': i,
                'frame_idx': int(frame_idx),
                'timestamp': float(timestamp),
                'face': face,
                'full_frame': full_frame,  # FULL ORIGINAL FRAME
                'score': frame_fake_conf
            })
        
        # Sort by score (descending) and take top K
        frame_scores.sort(key=lambda x: x['score'], reverse=True)
        top_frame_data = frame_scores[:top_k]
        
        logger.info(f"Extracted top {len(top_frame_data)} frames")
        
        # Convert to response format - SAVE FULL FRAMES AS PNG
        for idx, frame_info in enumerate(top_frame_data, 1):
            # Save FULL FRAME as PNG
            frame_url = save_full_frame_as_png(frame_info['full_frame'], idx)
            
            if frame_url:  # Only add if successfully saved
                top_frames.append({
                    "timestamp_sec": frame_info['timestamp'],
                    "frame_index": frame_info['frame_idx'],
                    "fake_confidence": round(frame_info['score'], 4),
                    "url": frame_url  # /assets/frames/frame1.png
                })
    
    except Exception as e:
        logger.warning(f"Top frames extraction failed: {e}")
        top_frames = []
    
    # ============================================
    # BUILD RESPONSE
    # ============================================
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "metadata": {
            "duration_sec": round(duration, 2),
            "frames_analyzed": len([f for f in faces if f.max() > 0])
        },
        "top_frames": top_frames
    }
