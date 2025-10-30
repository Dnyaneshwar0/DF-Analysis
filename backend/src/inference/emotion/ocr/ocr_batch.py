#!/usr/bin/env python3
"""
backend/src/inference/emotion/ocr/ocr_batch.py

Batch OCR runner.
- Scans a directory of extracted frames (e.g. src/inference/emotion/ocr/frames/eddiecaptions)
- Calls ocr_frame.py for each frame
- Aggregates results into JSON
- Writes output under backend/data/emotion/output/ocr/<video>_raw_captions.json

Usage:
  python3 -m src.inference.emotion.ocr.ocr_batch eddiecaptions
  (or)
  python3 -m src.inference.emotion.ocr.ocr_batch src/inference/emotion/ocr/frames/eddiecaptions
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

# ---------------------------------------------------------------------
# Repo-relative paths
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parents[3]
FRAMES_ROOT = SCRIPT_DIR / "frames"
OUTPUT_ROOT = BACKEND_DIR / "data" / "emotion" / "output" / "ocr"
OCR_FRAME = SCRIPT_DIR / "ocr_frame.py"

def run_ocr_on_frame(frame_path: Path, crop_ratio: float):
    """Call ocr_frame.py on a single frame and parse its JSON output."""
    cmd = [sys.executable, str(OCR_FRAME), str(frame_path), str(crop_ratio)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"image": str(frame_path), "text": "", "conf": 0.0}

    try:
        return json.loads(proc.stdout.strip())
    except Exception:
        return {"image": str(frame_path), "text": proc.stdout.strip(), "conf": 0.0}

def main():
    parser = argparse.ArgumentParser(description="Run OCR on all frames in a directory.")
    parser.add_argument("frames_dir", help="Directory containing extracted frames or its basename.")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between frames (for timestamps).")
    parser.add_argument("--crop", type=float, default=0.25, help="Bottom crop fraction.")
    args = parser.parse_args()

    # Resolve frames directory (allow bare name like 'eddiecaptions')
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        frames_dir = FRAMES_ROOT / args.frames_dir
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    video_name = frames_dir.name
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        print(f"No frames found in {frames_dir}")
        sys.exit(1)

    results = []
    for idx, f in enumerate(frames):
        ts = round(idx * args.interval, 3)
        res = run_ocr_on_frame(f, args.crop)
        res["timestamp"] = ts
        results.append(res)
        print(f"[{idx:03d}] {ts:6.2f}s → {res.get('text', '')[:80]}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_ROOT / f"{video_name}_raw_captions.json"

    with open(out_json, "w", encoding="utf8") as fh:
        json.dump(
            {"frames": results, "interval": args.interval, "source_dir": str(frames_dir)},
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n✅ Wrote: {out_json}")

if __name__ == "__main__":
    main()
