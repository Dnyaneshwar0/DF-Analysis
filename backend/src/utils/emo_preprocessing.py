#!/usr/bin/env python3
"""
emo_preprocessing.py

Unified preprocessing pipeline for emotion analysis:
- Extracts frames from videos and runs OCR on captions (burned-in text)
- Extracts audio from MP4 videos and generates Mel-spectrograms for RAVDESS
- Supports both individual and batch (--all) processing
- Writes processed outputs to:
    data/emotion/input/processed/
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

# === Path setup ===
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "emotion" / "input" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "emotion" / "input" / "processed"
DEFAULT_TMP_ROOT = DEFAULT_PROCESSED_DIR / "tmp"

# ===============================================================
# === UTILS =====================================================
# ===============================================================

def run_cmd_quiet(cmd: List[str]):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ===============================================================
# === OCR HELPERS ===============================================
# ===============================================================

def try_pytesseract(img: Image.Image) -> Optional[Dict]:
    try:
        import pytesseract
    except Exception:
        return None
    try:
        txt = pytesseract.image_to_string(img, lang="eng").strip()
        return {"text": txt, "conf": None if not txt else 1.0}
    except Exception:
        return {"text": "", "conf": 0.0}


def try_easyocr(img: Image.Image) -> Optional[Dict]:
    try:
        import easyocr, numpy as np
    except Exception:
        return None
    reader = easyocr.Reader(["en"], gpu=False)
    res = reader.readtext(np.array(img))
    if not res:
        return {"text": "", "conf": 0.0}
    texts, confs = zip(*[(t, c) for _, t, c in res])
    combined = " ".join(texts).strip()
    return {"text": combined, "conf": float(sum(confs) / len(confs)) if confs else 0.0}


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
    return similarity_ratio(a, b) > cutoff


@dataclass
class FrameResult:
    image: str
    timestamp: float
    raw_text: str
    text: str
    conf: float


def extract_frames_ffmpeg(video_path: Path, out_dir: Path, interval: float = 0.5, scale_h: int = 720):
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / float(interval)
    vf = f"scale=-2:{scale_h},fps={fps}"
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", vf, str(out_dir / "frame_%05d.jpg")]
    run_cmd_quiet(cmd)


def process_frames_dir(frames_dir: Path, interval: float, crop: float, min_len: int, engine_order: List[str]) -> List[FrameResult]:
    frames = sorted(frames_dir.glob("*.jpg"))
    results = []
    for idx, fp in enumerate(frames):
        ts = round(idx * interval, 3)
        try:
            im = Image.open(fp).convert("RGB")
        except Exception:
            results.append(FrameResult(str(fp), ts, "", "", 0.0))
            continue

        w, h = im.size
        top = int(h * (1.0 - crop))
        crop_img = im.crop((0, top, w, h))
        crop_img = ImageOps.invert(ImageOps.grayscale(ImageOps.autocontrast(crop_img)))

        text, conf = "", 0.0
        for eng in engine_order:
            out = try_pytesseract(crop_img) if eng == "tesseract" else try_easyocr(crop_img)
            if out and out.get("text", "").strip():
                text, conf = out["text"].strip(), out.get("conf", 0.0)
                break

        cleaned = MULTISPACE_RE.sub(" ", NON_PRINT_RE.sub("", TIMESTAMP_BRACKET_RE.sub("", text))).strip().lower()
        if len(cleaned) < min_len:
            cleaned = ""
        results.append(FrameResult(str(fp), ts, text, cleaned, float(conf)))
    return results


def merge_frame_results(frames: List[FrameResult], sim_cutoff: float = 0.85, merge_gap: float = 0.25):
    segments, cur = [], None
    for fr in frames:
        txt, ts = fr.text, fr.timestamp
        if not txt:
            continue
        if cur and (txt == cur["text"] or similar_enough(txt, cur["text"], sim_cutoff)):
            cur["end"] = ts
            cur["conf"] = (cur["conf"] + fr.conf) / 2.0
        else:
            if cur:
                segments.append(cur)
            cur = {"text": txt, "start": ts, "end": ts, "conf": fr.conf}
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

# ===============================================================
# === AUDIO EXTRACTION + RAVDESS PREPROCESSING ==================
# ===============================================================

def extract_audio_ffmpeg(video_path: Path, out_wav: Path, target_sr: int = 16000):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(target_sr), "-ac", "1",
        str(out_wav)
    ]
    run_cmd_quiet(cmd)


def preprocess_ravdess_audio(audio_path: Path, out_dir: Path, target_sr: int = 16000, n_mels: int = 64):
    import torch, torchaudio, librosa, numpy as np
    waveform, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_mels=n_mels)(torch.tensor(waveform).unsqueeze(0))
    mel = (mel - mel.mean()) / mel.std()
    np.save(out_dir / f"{audio_path.stem}_mel.npy", mel.squeeze().cpu().numpy())

# ===============================================================
# === ALL-IN-ONE PIPELINE ======================================
# ===============================================================

def run_all_preprocessings():
    raw_dir, out_dir = DEFAULT_INPUT_DIR, DEFAULT_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for mp4 in raw_dir.glob("*.mp4"):
        tmp = DEFAULT_TMP_ROOT / mp4.stem
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            extract_frames_ffmpeg(mp4, tmp)
            frames = process_frames_dir(tmp, 0.5, 0.25, 3, ["tesseract", "easyocr"])
            segs = merge_frame_results(frames)
            with open(out_dir / f"{mp4.stem}_captions.json", "w") as f:
                json.dump({"video": mp4.stem, "segments": segs}, f, ensure_ascii=False, indent=2)

            wav_path = out_dir / f"{mp4.stem}.wav"
            extract_audio_ffmpeg(mp4, wav_path)
            preprocess_ravdess_audio(wav_path, out_dir)
            wav_path.unlink(missing_ok=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

# ===============================================================
# === CLI ENTRY ================================================
# ===============================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Unified OCR + RAVDESS preprocessing pipeline.")
    p.add_argument("--video", help="Path to MP4 video for OCR/audio preprocessing.")
    p.add_argument("--audio", help="Path to WAV file for RAVDESS preprocessing.")
    p.add_argument("--all", action="store_true", help="Run all available preprocessings automatically.")
    args = p.parse_args(argv)

    if args.all:
        run_all_preprocessings()
        return

    out_dir = DEFAULT_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            alt = DEFAULT_INPUT_DIR / Path(args.video).name
            if not alt.exists():
                raise FileNotFoundError(f"Video not found: {args.video}")
            video_path = alt

        tmp = DEFAULT_TMP_ROOT / video_path.stem
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            extract_frames_ffmpeg(video_path, tmp)
            frames = process_frames_dir(tmp, 0.5, 0.25, 3, ["tesseract", "easyocr"])
            segs = merge_frame_results(frames)
            with open(out_dir / f"{video_path.stem}_captions.json", "w") as f:
                json.dump({"video": video_path.stem, "segments": segs}, f, ensure_ascii=False, indent=2)

            wav_path = out_dir / f"{video_path.stem}.wav"
            extract_audio_ffmpeg(video_path, wav_path)
            preprocess_ravdess_audio(wav_path, out_dir)
            wav_path.unlink(missing_ok=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            alt = DEFAULT_INPUT_DIR / Path(args.audio).name
            if not alt.exists():
                raise FileNotFoundError(f"Audio file not found: {args.audio}")
            audio_path = alt
        preprocess_ravdess_audio(audio_path, out_dir)
        return


if __name__ == "__main__":
    main()
