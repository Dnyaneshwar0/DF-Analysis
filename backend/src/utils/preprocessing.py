import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

def save_uploaded_file(file_storage, dest_dir: str) -> Path:
    """
    Save Werkzeug FileStorage to dest_dir and return the Path object.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file_storage.filename).name
    dest_path = dest_dir / filename
    file_storage.save(str(dest_path))
    return dest_path

def extract_frames(video_path: str, sample_rate: int = 1):
    """
    Extract frames from video every `sample_rate` frames.
    Yields (frame_idx, frame_bgr)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            yield idx, frame
        idx += 1
    cap.release()

def detect_faces_in_frame(frame_bgr):
    """
    Uses Haar cascade to detect faces. Returns list of bboxes (x,y,w,h).
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # reasonable parameters for frontal faces
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def extract_faces_from_video(video_path: str, sample_rate: int = 1) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Extracts faces from a video. Returns:
      - faces: list of face images (BGR numpy arrays)
      - meta: list of metadata dicts with keys {"frame_idx": int, "bbox": [x,y,w,h]}
    """
    faces = []
    meta = []
    for frame_idx, frame in extract_frames(video_path, sample_rate=sample_rate):
        dets = detect_faces_in_frame(frame)
        for (x, y, w, h) in dets:
            # expand bbox slightly
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(frame.shape[1], x + w + pad_x)
            y1 = min(frame.shape[0], y + h + pad_y)
            face = frame[y0:y1, x0:x1].copy()
            if face.size == 0:
                continue
            faces.append(face)
            meta.append({"frame_idx": int(frame_idx), "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]})
    return faces, meta

def load_image_from_path(path: str):
    """
    Read image from disk (BGR) and return numpy array.
    """
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read image {path}")
    return img
