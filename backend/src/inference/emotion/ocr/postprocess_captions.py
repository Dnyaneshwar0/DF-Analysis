#!/usr/bin/env python3
"""
backend/src/inference/emotion/ocr/postprocess_captions.py

Cleans and merges frame-level OCR detections into caption segments with timestamps.

Usage:
  python3 -m src.inference.emotion.ocr.postprocess_captions data/emotion/output/ocr/eddiecaptions_raw_captions.json
"""

import json
import argparse
import difflib
from pathlib import Path


def normalize(s: str) -> str:
    return " ".join(s.strip().split()).lower()


def similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.85


def main():
    parser = argparse.ArgumentParser(description="Merge OCR captions into clean text segments.")
    parser.add_argument("raw_json", help="Path to *_raw_captions.json (relative to backend/ or absolute)")
    parser.add_argument(
        "--out_json",
        help="Optional explicit output path. Defaults to backend/data/emotion/output/ocr/<video>_segments_clean.json",
    )
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--merge_gap", type=float, default=0.25)
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Resolve repo root properly
    # -----------------------------------------------------------------
    BACKEND_DIR = Path(__file__).resolve().parents[4]  # not 3 — this points correctly to backend/
    raw_path = Path(args.raw_json)
    if not raw_path.is_absolute():
        raw_path = BACKEND_DIR / raw_path
    raw_path = raw_path.resolve()

    if not raw_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {raw_path}")

    # -----------------------------------------------------------------
    # Process frames
    # -----------------------------------------------------------------
    data = json.load(open(raw_path, encoding="utf8"))
    frames = data.get("frames", [])
    segments, cur = [], None

    for fr in frames:
        txt = normalize(fr.get("text", ""))
        ts = fr.get("timestamp", 0.0)
        if len(txt) < args.min_len:
            txt = ""

        if cur is None:
            if txt:
                cur = {"text": txt, "start": ts, "end": ts, "conf": fr.get("conf") or 0.0}
        else:
            if txt == cur["text"] or similar(txt, cur["text"]):
                cur["end"] = ts
                cur["conf"] = (cur.get("conf", 0.0) + (fr.get("conf") or 0.0)) / 2.0
            else:
                segments.append(cur)
                cur = {"text": txt, "start": ts, "end": ts, "conf": fr.get("conf") or 0.0} if txt else None

    if cur:
        segments.append(cur)

    merged = []
    for seg in segments:
        if (
            merged
            and seg["text"] == merged[-1]["text"]
            and seg["start"] - merged[-1]["end"] <= args.merge_gap
        ):
            merged[-1]["end"] = seg["end"]
            merged[-1]["conf"] = (merged[-1]["conf"] + seg["conf"]) / 2.0
        else:
            merged.append(seg)

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------
    video_name = raw_path.stem.replace("_raw_captions", "")
    out_json = (
        Path(args.out_json)
        if args.out_json
        else BACKEND_DIR / "data" / "emotion" / "output" / "ocr" / f"{video_name}_segments_clean.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    json.dump(
        {"video": video_name, "segments": merged},
        open(out_json, "w", encoding="utf8"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"✅ Wrote {out_json} (segments: {len(merged)})")


if __name__ == "__main__":
    main()
