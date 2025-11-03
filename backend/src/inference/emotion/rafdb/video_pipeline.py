#!/usr/bin/env python3
# RAF-DB Video Inference (Silent + Compact JSON + Annotated Output)

from __future__ import annotations
import sys, json
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort
from PIL import Image

# Paths
REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = REPO_ROOT / "models" / "emotion" / "rafdb"
ONNX_PATH = MODEL_DIR / "raf_model.onnx"
LABEL_MAP_PATH = MODEL_DIR / "raf_label_map.json"
OUT_DIR = REPO_ROOT / "data" / "emotion" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Labels
if not LABEL_MAP_PATH.exists():
    raise FileNotFoundError(f"Missing label map: {LABEL_MAP_PATH}")
label_map = json.loads(LABEL_MAP_PATH.read_text())["idx2label"]
LABELS = [label_map[str(i)] for i in sorted(map(int, label_map.keys()))]

# Model
sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

# Detection and preprocessing
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
IMG_SIZE, CONF_THRESHOLD = 224, 0.25
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
FRAME_SKIP = 5

def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1))[None, :, :, :]

def draw(frame, box, label, conf):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (8, 150, 255), 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), (8, 150, 255), -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

# Main inference
def run(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    duration = total / fps

    out_vid = OUT_DIR / f"out_{video_path.stem}.mp4"
    out_json = OUT_DIR / f"summary_{video_path.stem}.json"
    writer = cv2.VideoWriter(str(out_vid), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_data, frame_idx = [], 0
    prev_label, prev_conf, prev_box = "Uncertain", 0.0, (50, 50, 150, 150)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx % FRAME_SKIP == 0:
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces):
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                prev_box = (x, y, x + w, y + h)
                face = frame[y:y + h, x:x + w]
                pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                inp = preprocess(pil)
                probs = sess.run(None, {input_name: inp.astype(np.float32)})[0][0]
                probs = np.exp(probs - probs.max())
                probs /= probs.sum()
                top = int(np.argmax(probs))
                prev_label, prev_conf = LABELS[top], float(probs[top])

        draw(frame, prev_box, prev_label, prev_conf)
        writer.write(frame)

        frame_data.append({
            "t": round(frame_idx / fps, 2),
            "emo": prev_label,
            "conf": round(prev_conf, 3)
        })

    cap.release()
    writer.release()

    # JSON summary
    NUM_POINTS = 20
    if len(frame_data) > NUM_POINTS:
        step = max(1, len(frame_data) // NUM_POINTS)
        timeline = frame_data[::step][:NUM_POINTS]
    else:
        timeline = frame_data

    labels = [d["emo"] for d in frame_data]
    dominant = max(set(labels), key=labels.count) if labels else "None"

    summary = {
        "video": video_path.name,
        "duration_s": round(duration, 2),
        "dominant_emotion": dominant,
        "timeline": timeline
    }
    out_json.write_text(json.dumps(summary, indent=2))

# CLI
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        print("Usage: python -m src.inference.emotion.rafdb.video_pipeline path/to/video.mp4")
        sys.exit(1)
    inp = Path(argv[0])
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")
    run(inp)

if __name__ == "__main__":
    main()
