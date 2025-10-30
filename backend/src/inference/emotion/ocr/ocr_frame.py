#!/usr/bin/env python3
"""
backend/src/inference/emotion/ocr/ocr_frame.py

Given an image path, crops the bottom region (default 25%) and runs OCR.
Returns JSON: {"image": "...", "text": "...", "conf": 0.xx}

- Uses pytesseract primarily (requires system tesseract-ocr)
- Falls back to easyocr if pytesseract not found or empty
- Designed to be called by ocr_batch.py
"""

import sys
import json
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Helper: run pytesseract
# ---------------------------------------------------------------------
def try_pytesseract(img):
    try:
        import pytesseract
        # Optional: specify custom tesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    except Exception:
        return None

    try:
        txt = pytesseract.image_to_string(img, lang="eng")
        txt = txt.strip()
        if not txt:
            return {"text": "", "conf": 0.0}
        return {"text": txt, "conf": None}
    except Exception:
        return {"text": "", "conf": 0.0}

# ---------------------------------------------------------------------
# Helper: run easyocr (fallback)
# ---------------------------------------------------------------------
def try_easyocr(img):
    try:
        import easyocr
    except Exception:
        return None

    reader = easyocr.Reader(["en"], gpu=False)
    res = reader.readtext(np.array(img))
    if not res:
        return {"text": "", "conf": 0.0}

    texts, confs = [], []
    for _, text, conf in res:
        texts.append(text)
        confs.append(conf)
    combined = " ".join(texts).strip()
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return {"text": combined, "conf": avg_conf}

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.emotion.ocr.ocr_frame path/to/image.jpg [crop_ratio_bottom]")
        sys.exit(2)

    image_path = Path(sys.argv[1])
    crop_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

    if not image_path.exists():
        print(json.dumps({"image": str(image_path), "text": "", "conf": 0.0}))
        sys.exit(0)

    im = Image.open(image_path).convert("RGB")
    w, h = im.size

    # Crop bottom portion
    top = int(h * (1.0 - crop_ratio))
    crop = im.crop((0, top, w, h))

    # Slight preprocessing to improve OCR accuracy
    crop = ImageOps.autocontrast(crop)
    crop = ImageOps.grayscale(crop)
    crop = ImageOps.invert(crop)

    # Try pytesseract first, fallback to easyocr
    out = try_pytesseract(crop)
    if out is None or not out.get("text", "").strip():
        out = try_easyocr(crop) or {"text": "", "conf": 0.0}

    print(json.dumps({
        "image": str(image_path),
        "text": out.get("text", "").strip(),
        "conf": out.get("conf", 0.0)
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
