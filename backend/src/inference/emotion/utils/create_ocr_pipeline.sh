#!/usr/bin/env bash
set -e

mkdir -p ocr/frames
mkdir -p ocr/output
chmod -R 755 ocr

echo "Creating ocr scripts..."

# 1) extract_frames.sh
cat > ocr/extract_frames.sh <<'SH'
#!/usr/bin/env bash
# Usage: ./extract_frames.sh path/to/video.mp4 0.5
VIDEO="$1"
INTERVAL="${2:-0.5}"   # seconds between samples
OUTDIR="${3:-ocr/frames/$(basename "${VIDEO%.*}")}"
mkdir -p "$OUTDIR"
# fps = 1/interval
FPS=$(python3 - <<PY
i = $INTERVAL
print(1.0 / i)
PY
)
ffmpeg -hide_banner -loglevel error -i "$VIDEO" -vf "scale=-2:720,fps=${FPS}" "$OUTDIR/frame_%05d.jpg"
echo "Frames written to $OUTDIR (interval=${INTERVAL}s)"
SH
chmod +x ocr/extract_frames.sh

# 2) ocr_frame.py
cat > ocr/ocr_frame.py <<'PY'
#!/usr/bin/env python3
"""
ocr_frame.py
Given an image path, crop the bottom region (default 25%) and run OCR.
Returns JSON: {"text": "...", "conf": 0.XX}
"""
import sys, json
from PIL import Image, ImageOps
import numpy as np

def try_pytesseract(img):
    try:
        import pytesseract
    except Exception:
        return None
    txt = pytesseract.image_to_string(img, lang='eng')
    # pytesseract doesn't always give confidences easily; return text only
    return {"text": txt.strip(), "conf": None}

def try_easyocr(img):
    try:
        import easyocr
    except Exception:
        return None
    reader = easyocr.Reader(['en'], gpu=False)
    res = reader.readtext(np.array(img))
    if not res:
        return {"text": "", "conf": 0.0}
    texts = []
    confs = []
    for bbox, text, conf in res:
        texts.append(text)
        confs.append(conf)
    combined = " ".join(texts).strip()
    return {"text": combined, "conf": float(sum(confs)/len(confs)) if confs else 0.0}

def main():
    if len(sys.argv) < 2:
        print("Usage: ocr_frame.py path/to/image.jpg [crop_ratio_bottom]")
        sys.exit(2)
    p = sys.argv[1]
    crop_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    im = Image.open(p).convert("RGB")
    w,h = im.size
    top = int(h * (1.0 - crop_ratio))
    crop = im.crop((0, top, w, h))
    # optional preproc
    crop = ImageOps.autocontrast(crop)
    # Try pytesseract first, then easyocr
    out = try_pytesseract(crop)
    if out is None or (out.get("text","").strip()==""):
        out = try_easyocr(crop) or {"text":"", "conf":0.0}
    print(json.dumps({"image": p, "text": out["text"].strip(), "conf": out.get("conf")}))
if __name__ == "__main__":
    main()
PY
chmod +x ocr/ocr_frame.py

# 3) ocr_batch.py
cat > ocr/ocr_batch.py <<'PY'
#!/usr/bin/env python3
"""
ocr_batch.py
Walks a frames directory, runs ocr_frame.py on each image (in order),
and writes raw JSON detections to outputs/<video>_raw_captions.json

Usage:
  python ocr/ocr_batch.py ocr/frames/<video_dir> outputs/<video>_raw_captions.json --interval 0.5
"""
import os, sys, json, subprocess, argparse
from pathlib import Path
def run_ocr_on_frame(frame, crop_ratio):
    cmd = [sys.executable, "ocr/ocr_frame.py", frame, str(crop_ratio)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"image": frame, "text":"", "conf":0.0}
    try:
        return json.loads(proc.stdout.strip())
    except Exception:
        return {"image": frame, "text":proc.stdout.strip(), "conf":0.0}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames_dir")
    p.add_argument("out_json")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between frames (used to compute timestamps)")
    p.add_argument("--crop", type=float, default=0.25, help="bottom crop fraction")
    args = p.parse_args()
    frames = sorted([str(x) for x in Path(args.frames_dir).glob("*.jpg")])
    results=[]
    for idx,f in enumerate(frames):
        ts = round(idx * args.interval, 3)
        res = run_ocr_on_frame(f, args.crop)
        res["timestamp"] = ts
        results.append(res)
        print(f"[{idx}] {ts}s -> {res.get('text')[:80]}")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json,"w",encoding="utf8") as fh:
        json.dump({"frames": results, "interval": args.interval, "source_dir": args.frames_dir}, fh, indent=2, ensure_ascii=False)
    print("Wrote", args.out_json)

if __name__ == "__main__":
    main()
PY
chmod +x ocr/ocr_batch.py

# 4) postprocess_captions.py
cat > ocr/postprocess_captions.py <<'PY'
#!/usr/bin/env python3
"""
postprocess_captions.py
Takes raw frame-level OCR JSON and groups consecutive identical (or near-identical)
texts into caption segments with start/end timestamps.
"""
import json, argparse, difflib
from pathlib import Path

def normalize(s):
    return " ".join(s.strip().split()).lower()

def similar(a,b):
    if not a or not b: return False
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.85

def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_json")
    p.add_argument("out_json")
    p.add_argument("--min_len", type=int, default=3)
    p.add_argument("--merge_gap", type=float, default=0.25)
    args = p.parse_args()
    data = json.load(open(args.raw_json, encoding="utf8"))
    frames = data.get("frames", [])
    segments=[]
    cur=None
    for fr in frames:
        txt = normalize(fr.get("text",""))
        ts = fr.get("timestamp",0.0)
        if len(txt) < args.min_len:
            # treat as empty
            txt=""
        if cur is None:
            if txt:
                cur={"text":txt,"start":ts,"end":ts,"conf":fr.get("conf") or 0.0}
        else:
            if txt==cur["text"] or similar(txt, cur["text"]):
                cur["end"]=ts
                # update conf as average
                cur["conf"] = (cur.get("conf",0.0)+ (fr.get("conf") or 0.0))/2.0
            else:
                # finalize
                segments.append(cur)
                if txt:
                    cur={"text":txt,"start":ts,"end":ts,"conf":fr.get("conf") or 0.0}
                else:
                    cur=None
    if cur:
        segments.append(cur)
    # Merge very short segments into neighbors if needed
    merged=[]
    for seg in segments:
        if merged and seg["start"] - merged[-1]["end"] <= args.merge_gap and seg["text"]==merged[-1]["text"]:
            merged[-1]["end"]=seg["end"]
            merged[-1]["conf"] = (merged[-1]["conf"] + seg["conf"])/2.0
        else:
            merged.append(seg)
    out = {"video": Path(args.raw_json).stem, "segments": merged}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json,"w",encoding="utf8"), indent=2, ensure_ascii=False)
    print("Wrote", args.out_json, "segments:", len(merged))

if __name__ == "__main__":
    main()
PY
chmod +x ocr/postprocess_captions.py

# 5) captions_to_emotions.py
cat > ocr/captions_to_emotions.py <<'PY'
#!/usr/bin/env python3
"""
captions_to_emotions.py
Load segments JSON, run GoEmotions pipeline (joblib TF-IDF + classifier),
and append emotion probabilities to each segment.
"""
import argparse, json
from pathlib import Path
import joblib, numpy as np

def load_goemotions_models(base="models/goemotions_model"):
    vec = joblib.load(f"{base}/goemotions_tfidf.joblib")
    clf = joblib.load(f"{base}/goemotions_clf.joblib")
    try:
        labels = list(clf.classes_)
    except Exception:
        labels = None
    return vec, clf, labels

def predict_probs(clf, vec, texts):
    X = vec.transform(texts)
    # try predict_proba, if not available use decision_function or predict
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
        # shape (n_samples, n_classes)
        return probs
    elif hasattr(clf, "decision_function"):
        df = clf.decision_function(X)
        # attempt to convert to probabilities via softmax
        from scipy.special import softmax
        return softmax(df, axis=1)
    else:
        preds = clf.predict(X)
        # map to sparse one-hot
        out = []
        for p in preds:
            v = [1.0 if p==c else 0.0 for c in clf.classes_]
            out.append(v)
        return np.array(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("segments_json")
    p.add_argument("out_json")
    args = p.parse_args()
    segs = json.load(open(args.segments_json, encoding="utf8")).get("segments",[])
    if not segs:
        print("No segments found.")
        return
    vec, clf, labels = load_goemotions_models()
    texts = [s["text"] for s in segs]
    probs = predict_probs(clf, vec, texts)
    all_segments=[]
    for i,s in enumerate(segs):
        prob_vec = probs[i].tolist()
        # map to labels if available
        if labels:
            em = {labels[j]: float(prob_vec[j]) for j in range(len(labels))}
            dominant = labels[int(np.argmax(prob_vec))]
        else:
            em = {"class_"+str(j): float(prob_vec[j]) for j in range(len(prob_vec))}
            dominant = max(em, key=em.get)
        seg_out = dict(s)
        seg_out["emotions"]=em
        seg_out["dominant"]=dominant
        all_segments.append(seg_out)
    out = {"source":"ocr_captions","segments": all_segments}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json,"w",encoding="utf8"), indent=2, ensure_ascii=False)
    print("Wrote", args.out_json)
if __name__ == "__main__":
    main()
PY
chmod +x ocr/captions_to_emotions.py

# 6) small README for ocr
cat > ocr/README.md <<'MD'
OCR Caption Extraction (burned-in)
-------------------------------
1) Install system deps:
   sudo apt update
   sudo apt install -y ffmpeg tesseract-ocr libtesseract-dev

2) Python deps (add to requirements.txt):
   pytesseract
   easyocr   # optional
   python-Levenshtein  # optional

3) Quick run:
   ./ocr/extract_frames.sh data/your_video.mp4 0.5
   python3 ocr/ocr_batch.py ocr/frames/your_video outputs/your_video_raw_captions.json --interval 0.5
   python3 ocr/postprocess_captions.py outputs/your_video_raw_captions.json outputs/your_video_segments.json
   python3 ocr/captions_to_emotions.py outputs/your_video_segments.json outputs/your_video_captions_emotions.json
MD

echo "All files created under ocr/. Edit them if you want to tune parameters."
echo "Next: install tesseract (system) and python deps and then run the pipeline as in ocr/README.md"
