#!/usr/bin/env python3
"""
emo_preprocessing.py

Single-file OCR preprocessing:
 - Extract frames from an input MP4 (ffmpeg)
 - Run per-frame OCR (pytesseract primary, easyocr fallback)
 - Merge & clean OCR results into unique caption segments
 - Write ONE final JSON:
     data/emotion/processed/<video_basename>_captions.json

Temporary frames are stored in:
  data/emotion/processed/tmp/<video_basename>/
and are deleted automatically at the end (no raw JSONs, no plots).
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Dict

try:
    from PIL import Image, ImageOps
except Exception as e:
    raise SystemExit("Pillow is required (pip install pillow).") from e

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "emotion" / "input" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "emotion" / "input" / "processed"
DEFAULT_TMP_ROOT = DEFAULT_PROCESSED_DIR / "tmp"


def run_cmd_quiet(cmd: List[str]):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout


def try_pytesseract(img: Image.Image) -> Optional[Dict]:
    try:
        import pytesseract
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


def try_easyocr(img: Image.Image) -> Optional[Dict]:
    try:
        import easyocr
        import numpy as np
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


TIMESTAMP_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?\)")
NON_PRINT_RE = re.compile(r"[^A-Za-z0-9.,!?;:'\"()\s-]")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    return " ".join(s.strip().split()).lower()


def similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def similar_enough(a: str, b: str, cutoff: float = 0.85) -> bool:
    if not a or not b:
        return False
    return similarity_ratio(a, b) > cutoff


@dataclass
class FrameResult:
    image: str
    timestamp: float
    raw_text: str
    text: str
    conf: float


def extract_frames_ffmpeg(video_path: Path, out_dir: Path, interval: float = 0.5, scale_h: Optional[int] = 720):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    fps = 1.0 / float(interval)
    scale = f"scale=-2:{scale_h}" if scale_h else "null"
    vf = f"{scale},fps={fps}"
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", vf, str(out_dir / "frame_%05d.jpg")]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr.strip()}") from e


def process_frames_dir(frames_dir: Path, interval: float, crop: float, min_len: int, engine_order: List[str]) -> List[FrameResult]:
    frames = sorted([p for p in frames_dir.glob("*.jpg")])
    results: List[FrameResult] = []
    for idx, fp in enumerate(frames):
        ts = round(idx * interval, 3)
        try:
            im = Image.open(fp).convert("RGB")
        except Exception:
            results.append(FrameResult(image=str(fp), timestamp=ts, raw_text="", text="", conf=0.0))
            continue
        w, h = im.size
        top = int(h * (1.0 - crop))
        crop_img = im.crop((0, top, w, h))
        crop_img = ImageOps.autocontrast(crop_img)
        crop_img = ImageOps.grayscale(crop_img)
        crop_img = ImageOps.invert(crop_img)

        text = ""
        conf = 0.0
        for eng in engine_order:
            if eng == "tesseract":
                out = try_pytesseract(crop_img)
            elif eng == "easyocr":
                out = try_easyocr(crop_img)
            else:
                continue
            if out and out.get("text", "").strip():
                text = out["text"].strip()
                conf = out.get("conf", 0.0) or 0.0
                break

        cleaned = TIMESTAMP_BRACKET_RE.sub("", text)
        cleaned = NON_PRINT_RE.sub("", cleaned)
        cleaned = MULTISPACE_RE.sub(" ", cleaned).strip().lower()
        if len(cleaned) < min_len:
            cleaned = ""

        results.append(FrameResult(image=str(fp), timestamp=ts, raw_text=text, text=cleaned, conf=float(conf)))
    return results


def merge_frame_results(frames: List[FrameResult], sim_cutoff: float = 0.85, merge_gap: float = 0.25):
    segments = []
    cur = None
    for fr in frames:
        txt = fr.text
        ts = fr.timestamp
        if cur is None:
            if txt:
                cur = {"text": txt, "start": ts, "end": ts, "conf": fr.conf}
        else:
            if txt == cur["text"] or similar_enough(txt, cur["text"], cutoff=sim_cutoff):
                cur["end"] = ts
                cur["conf"] = (cur.get("conf", 0.0) + fr.conf) / 2.0
            else:
                segments.append(cur)
                cur = {"text": txt, "start": ts, "end": ts, "conf": fr.conf} if txt else None
    if cur:
        segments.append(cur)

    merged = []
    for seg in segments:
        if merged and seg["text"] == merged[-1]["text"] and seg["start"] - merged[-1]["end"] <= merge_gap:
            merged[-1]["end"] = seg["end"]
            merged[-1]["conf"] = (merged[-1]["conf"] + seg["conf"]) / 2.0
        else:
            merged.append(seg)
    return merged


def main(argv=None):
    p = argparse.ArgumentParser(description="Unified OCR preprocessing -> single cleaned captions JSON.")
    p.add_argument("--video", help="Path to MP4 video (defaults to data/emotion/input/raw/).")
    p.add_argument("--frames-dir", help="Use existing frames dir instead of extracting.")
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--crop", type=float, default=0.25)
    p.add_argument("--scale-h", type=int, default=720)
    p.add_argument("--out-root", help="Output root (defaults to data/emotion/processed/).")
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--sim-cutoff", type=float, default=0.85)
    p.add_argument("--merge-gap", type=float, default=0.25)
    p.add_argument("--engine-order", nargs="+", default=["tesseract", "easyocr"])
    args = p.parse_args(argv)

    input_video = None
    frames_dir = None

    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        if not frames_dir.exists():
            alt = REPO_ROOT / args.frames_dir
            if alt.exists():
                frames_dir = alt
            else:
                raise FileNotFoundError(f"Frames dir not found: {args.frames_dir}")
        video_basename = frames_dir.name
    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            alt = DEFAULT_INPUT_DIR / Path(args.video).name
            if alt.exists():
                video_path = alt
            else:
                raise FileNotFoundError(f"Video not found: {args.video}")
        input_video = video_path
        video_basename = video_path.stem
    else:
        raise ValueError("Either --video or --frames-dir must be provided.")

    out_root = Path(args.out_root) if args.out_root else DEFAULT_PROCESSED_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    tmp_frames_dir = DEFAULT_TMP_ROOT / video_basename
    if tmp_frames_dir.exists():
        shutil.rmtree(tmp_frames_dir)
    tmp_frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        if input_video:
            extract_frames_ffmpeg(input_video, tmp_frames_dir, interval=args.interval, scale_h=args.scale_h)
            frames_dir = tmp_frames_dir

        frames_results = process_frames_dir(frames_dir, args.interval, args.crop, args.min_len, args.engine_order)
        segments = merge_frame_results(frames_results, args.sim_cutoff, args.merge_gap)

        out_json = out_root / f"{video_basename}_captions.json"
        with open(out_json, "w", encoding="utf8") as fh:
            json.dump({"video": video_basename, "segments": segments}, fh, ensure_ascii=False, indent=2)

    finally:
        try:
            if tmp_frames_dir.exists():
                shutil.rmtree(tmp_frames_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
