#!/usr/bin/env python3
"""
Unified RAF-DB video pipeline — single-file.

Combines:
 - model-backed per-frame prediction (ONNX preferred, TorchScript fallback)
 - face detection (MTCNN preferred, OpenCV Haar fallback)
 - video annotation (draw box + label)
 - per-frame CSV of probabilities
 - summary PNG and one-line summary

Usage:
  python -m src.inference.emotion.rafdb.video_pipeline path/to/video.mp4

Output (written to backend/data/emotion/output/):
  - annotated video (out_<video>.mp4)
  - per_frame_probs_<video>.csv
  - summary_<video>.png
  - printed one-line summary

This file is intended to replace the separate infer/video_infer/video_summary modules.
"""

from __future__ import annotations
import os
# avoid ORT GPU probing noise
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_LOG_VERBOSITY_LEVEL"] = "0"

import sys
import json
import warnings
from pathlib import Path
from collections import deque
import csv
import time
from typing import Tuple, Dict, Optional

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Repo-relative paths
# ---------------------------------------------------------------------
# file location: backend/src/inference/emotion/rafdb/video_pipeline.py
REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = REPO_ROOT / "models" / "emotion" / "rafdb"
ONNX_PATH = MODEL_DIR / "raf_model.onnx"
TS_PATH = MODEL_DIR / "raf_model_ts.pt"
LABEL_MAP = MODEL_DIR / "raf_label_map.json"

OUT_DIR = REPO_ROOT / "data" / "emotion" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Load label map
# ---------------------------------------------------------------------
if not LABEL_MAP.exists():
    raise FileNotFoundError(f"Label map not found at {LABEL_MAP}. Place raf_label_map.json in models/emotion/rafdb/")
with open(LABEL_MAP, "r", encoding="utf8") as fh:
    label_map = json.load(fh)
raw_idx2label = label_map.get("idx2label", label_map)
idx2label = {int(k): v for k, v in raw_idx2label.items()}
# canonical label ordering
COMMON = [idx2label[i] for i in sorted(idx2label.keys())]

# ---------------------------------------------------------------------
# Model backend: try ONNXRuntime (CPU), fallback TorchScript
# ---------------------------------------------------------------------
SESSION = None
INPUT_NAME = None
BACKEND = None
model_ts = None

try:
    import onnxruntime as ort
    warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
    try:
        ort.set_default_logger_severity(3)
    except Exception:
        pass
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found at {ONNX_PATH}")
    SESSION = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    INPUT_NAME = SESSION.get_inputs()[0].name
    BACKEND = "onnx"
except Exception:
    SESSION = None
    BACKEND = None

if BACKEND is None:
    try:
        import torch
        if not TS_PATH.exists():
            raise FileNotFoundError("TorchScript model not found at models/emotion/rafdb/raf_model_ts.pt")
        model_ts = torch.jit.load(str(TS_PATH), map_location="cpu")
        model_ts.eval()
        BACKEND = "torchscript"
    except Exception as e:
        raise RuntimeError("No valid RAF-DB model found (onnx or torchscript).") from e

# ---------------------------------------------------------------------
# Face detector: prefer MTCNN, fallback Haar
# ---------------------------------------------------------------------
MTCNN_DETECTOR = None
DETECTOR = "haar"
try:
    import torch as _torch
    from facenet_pytorch import MTCNN
    mtcnn_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    MTCNN_DETECTOR = MTCNN(keep_all=True, device=mtcnn_device)
    DETECTOR = "mtcnn"
except Exception:
    MTCNN_DETECTOR = None
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    DETECTOR = "haar"

# ---------------------------------------------------------------------
# Preprocessing constants (must match training)
# ---------------------------------------------------------------------
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# smoothing, skip, thresholds
SMOOTH_WINDOW = 7
FRAME_SKIP = 1
CONF_THRESHOLD = 0.25

# ---------------------------------------------------------------------
# Helpers: preprocess, detect, draw
# ---------------------------------------------------------------------
def preprocess_crop(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2,0,1)).astype(np.float32)  # CHW
    return arr

def detect_faces_opencv(frame_bgr: np.ndarray):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    boxes = []
    for (x,y,w,h) in faces:
        boxes.append([x, y, x+w, y+h])
    return boxes

def detect_faces(frame_bgr: np.ndarray):
    """Return list of boxes [x1,y1,x2,y2]"""
    if DETECTOR == "mtcnn" and MTCNN_DETECTOR is not None:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes, _ = MTCNN_DETECTOR.detect(img)
        if boxes is None:
            return []
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]
    else:
        return detect_faces_opencv(frame_bgr)

def draw_label(frame: np.ndarray, box, text: str, prob: float):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (8,150,255), 2)
    label = f"{text} {prob:.2f}"
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.rectangle(frame, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), (8,150,255), -1)
    cv2.putText(frame, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

# ---------------------------------------------------------------------
# Model prediction wrapper for a single cropped face (numpy CHW float32)
# ---------------------------------------------------------------------
def predict_logits_from_array(arr_chw: np.ndarray) -> np.ndarray:
    """arr_chw: C,H,W float32"""
    inp = arr_chw[None, :, :, :].astype(np.float32)  # 1,C,H,W
    if BACKEND == "onnx":
        out = SESSION.run(None, {INPUT_NAME: inp})
        logits = np.asarray(out[0][0], dtype=np.float32)
    else:
        import torch as _torch
        x = _torch.from_numpy(inp)
        with _torch.no_grad():
            logits = model_ts(x).cpu().numpy()[0].astype(np.float32)
    return logits

def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    a = logits - logits.max()
    e = np.exp(a)
    return e / e.sum()

# ---------------------------------------------------------------------
# Video processing + summary
# ---------------------------------------------------------------------
def extract_face_box_from_frame(frame_bgr: np.ndarray) -> Optional[list]:
    """Return largest face box or None."""
    boxes = detect_faces(frame_bgr)
    if not boxes:
        return None
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    return boxes[int(np.argmax(areas))]

def run_and_summarize(input_path: Path,
                      output_name: Optional[str] = None,
                      frame_skip: int = FRAME_SKIP,
                      smooth_k: int = SMOOTH_WINDOW,
                      conf_threshold: float = CONF_THRESHOLD):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = Path(input_path).name

    out_video_name = output_name if output_name else f"out_{video_name}"
    out_video_path = OUT_DIR / out_video_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w,h))

    frame_idx = 0
    probs_buf = deque(maxlen=smooth_k)
    csv_rows = []
    frames_with_face = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            writer.write(frame)
            continue

        box = extract_face_box_from_frame(frame)
        if box is None:
            probs_buf.clear()
            csv_rows.append([frame_idx, 0] + [0.0]*len(COMMON))
            writer.write(frame)
            continue

        x1,y1,x2,y2 = map(int, box)
        pad = int(0.15 * max(x2-x1, y2-y1))
        x1 = max(0, x1-pad); y1 = max(0, y1-pad)
        x2 = min(w, x2+pad); y2 = min(h, y2+pad)
        if x2 <= x1 or y2 <= y1:
            csv_rows.append([frame_idx, 0] + [0.0]*len(COMMON))
            writer.write(frame)
            continue

        face = frame[y1:y2, x1:x2]
        pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        arr = preprocess_crop(pil)  # CHW
        logits = predict_logits_from_array(arr)
        probs = logits_to_probs(logits)

        probs_buf.append(probs)
        avg_probs = np.mean(np.stack(list(probs_buf)), axis=0)

        top_idx = int(np.argmax(avg_probs))
        top_prob = float(avg_probs[top_idx])
        label = COMMON[top_idx] if top_prob >= conf_threshold else "Uncertain"

        draw_label(frame, (x1,y1,x2,y2), label, top_prob)
        writer.write(frame)

        csv_rows.append([frame_idx, 1] + [float(x) for x in avg_probs.tolist()])
        frames_with_face += 1

    cap.release()
    writer.release()

    # write CSV
    csv_path = OUT_DIR / f"per_frame_probs_{video_name}.csv"
    header = ["frame_idx", "has_face"] + COMMON
    with open(csv_path, "w", newline="") as cf:
        wcsv = csv.writer(cf)
        wcsv.writerow(header)
        wcsv.writerows(csv_rows)

    # compute global averages
    arr = np.array([row[2:] for row in csv_rows if row[1] == 1], dtype=np.float32)
    if arr.size == 0:
        avg = np.zeros(len(COMMON), dtype=np.float32)
    else:
        avg = arr.mean(axis=0)

    # summary plot
    summary_png = OUT_DIR / f"summary_{video_name}.png"
    plt.figure(figsize=(8,4))
    ax = plt.gca()
    x = np.arange(len(COMMON))
    ax.bar(x, avg)
    ax.set_xticks(x)
    ax.set_xticklabels(COMMON, rotation=30, ha='right')
    ax.set_ylim(0,1)
    ax.set_ylabel("Average probability")
    ax.set_title(f"Video emotion summary — {video_name}")
    plt.tight_layout()
    plt.savefig(summary_png, dpi=150)
    plt.close()

    pred_idx = int(np.argmax(avg))
    pred_label = COMMON[pred_idx]
    pred_prob = float(avg[pred_idx])
    one_liner = f"Video summary: Predominant emotion — {pred_label} (avg p = {pred_prob:.2f}), frames_with_face = {frames_with_face}"

    # return artifacts
    return {
        "out_video": str(out_video_path),
        "csv": str(csv_path),
        "summary_png": str(summary_png),
        "one_liner": one_liner
    }

# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        print("Usage: python -m src.inference.emotion.rafdb.video_pipeline path/to/video.mp4")
        sys.exit(1)
    inp = Path(argv[0])
    if not inp.exists():
        alt = REPO_ROOT / inp
        if alt.exists():
            inp = alt
        else:
            raise FileNotFoundError(f"Input not found: {argv[0]}")
    res = run_and_summarize(inp)
    print("Artifacts written:", res.get("out_video"), res.get("csv"), res.get("summary_png"))
    print(res.get("one_liner"))

if __name__ == "__main__":
    main()
