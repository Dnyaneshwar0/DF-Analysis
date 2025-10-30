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
