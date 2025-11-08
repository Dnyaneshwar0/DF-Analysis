# Unified preprocessing pipeline for emotion analysis (OCR improvements + stable config)
# Changes made:
# - Force CPU-only environment for PyTorch/EasyOCR
# - Show stage-level progress for subprocess calls
# - Replace slow spell-correction with a fast conservative version
# - Reduced fallback wordlist sizes for speed
# NOTE: Save this as src/utils/emo_preprocessing_fixed.py and run from your repo root.
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
# Optional: also patch sys.stdout to ignore encoding errors
sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import List, Optional, Dict

# Make sure processes use CPU only (avoid accidental CUDA probes)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TORCH_DEVICE", "cpu")

import warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")
warnings.filterwarnings("ignore", message="Using CPU. Note:")

try:
    from PIL import Image, ImageOps
except Exception as e:
    raise SystemExit("Pillow is required (pip install pillow).") from e

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "emotion" / "input" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "emotion" / "input" / "processed"
DEFAULT_TMP_ROOT = DEFAULT_PROCESSED_DIR / "tmp"
FFMPEG_PATH = REPO_ROOT / "ffmpeg" / "bin" / "ffmpeg.exe"

# Utils
def run_cmd_quiet(cmd: List[str]):
    # Print the command so user sees progress; keep stdout/stderr visible to aid debugging
    print(f"[RUNNING] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[DONE] {' '.join(cmd)}")

# OCR helpers
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
    # force CPU
    reader = easyocr.Reader(["en"], gpu=False)
    res = reader.readtext(np.array(img))
    if not res:
        return {"text": "", "conf": 0.0}
    texts, confs = zip(*[(t, c) for _, t, c in res])
    combined = " ".join(texts).strip()
    return {"text": combined, "conf": float(sum(confs) / len(confs)) if confs else 0.0}

# Text cleaning and similarity
TIMESTAMP_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?\)")
PARENTHETICAL_NOISE_RE = re.compile(
    r"\(\s*(?:cheer|cheers|applause|laughter|music|inaudible|audience|crowd|aud|applau|clapping)\b[^\)]*\)",
    flags=re.I,
)
NON_PRINT_RE = re.compile(r"[^\w\s\.,!\?;:'\"\-\|/]")
MULTISPACE_RE = re.compile(r"\s+")
PUNCT_RUN_RE = re.compile(r"([^\w\s])\1{1,}")
PUNCT_CLUSTER_RE = re.compile(r"(?<=\s)[^\w\s]{2,}(?=\s)")
TOKEN_STRIP_RE = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")

COMMON_OCR_CORRECTIONS: Dict[str, str] = {
    "eople": "people",
    "peop1e": "people",
    "mathmatcis": "mathematics",
    "mathmatcs": "mathematics",
    "mathmatics": "mathematics",
    "mathematics!\"": "mathematics!",
    "oyonev": "one",
    "ss": "",
    "sess": "",
    "applau": "applause",
    "im": "i'm",
    # small garbage removal
    "ap": "",
    "ma": "",
    "he": "",
}

_ALLOWED_TWO_LETTER = {
    "of", "to", "in", "on", "at", "by", "an", "as", "is", "it", "be", "we", "me",
    "my", "or", "do", "go", "so", "up", "no", "if", "ox", "us", "ok", "ya", "oh",
    "i'm", "i'd", "i'll", "i've", "u.s", "e.g", "id"
}

def is_allowed_two_letter(tok: str) -> bool:
    if not tok:
        return False
    t = tok.lower()
    if any(c.isdigit() for c in t):
        return True
    if t in _ALLOWED_TWO_LETTER:
        return True
    stripped = re.sub(r"[^a-z0-9]", "", t)
    if stripped in _ALLOWED_TWO_LETTER:
        return True
    return False

#-------------------------
# Dynamic spell-correction helpers (fast conservative)
#-------------------------
try:
    from wordfreq import zipf_frequency, top_n_list
    WORDFREQ_OK = True
except Exception:
    WORDFREQ_OK = False

_FALLBACK_WORDS: Optional[List[str]] = None

def _load_fallback_wordlist(limit: int = 5000) -> List[str]:
    global _FALLBACK_WORDS
    if _FALLBACK_WORDS is not None:
        return _FALLBACK_WORDS
    if WORDFREQ_OK:
        try:
            _FALLBACK_WORDS = top_n_list("en", n=limit)
            return _FALLBACK_WORDS
        except Exception:
            _FALLBACK_WORDS = []
    dict_paths = ["/usr/share/dict/words", "/usr/dict/words"]
    words = []
    for p in dict_paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    w = line.strip().lower()
                    if w and w.isalpha():
                        words.append(w)
                        if len(words) >= limit:
                            break
            if words:
                _FALLBACK_WORDS = words[:limit]
                return _FALLBACK_WORDS
        except Exception:
            continue
    _FALLBACK_WORDS = []
    return _FALLBACK_WORDS

def spell_correct_token(token: str, cutoff_ratio: float = 0.75) -> str:
    """
    Faster, conservative token correction:
      - exact mapping (COMMON_OCR_CORRECTIONS)
      - quick difflib.get_close_matches against a SMALL fallback list
      - final cleanup: strip non-alnum
    """
    if not token:
        return token
    t = token.lower()
    if t in COMMON_OCR_CORRECTIONS:
        return COMMON_OCR_CORRECTIONS[t] or ""
    if len(t) <= 2:
        cleaned = re.sub(r"[^a-z0-9]", "", t)
        return cleaned if cleaned else t
    try:
        words = _load_fallback_wordlist(limit=5000)
        if words:
            close = get_close_matches(t, words, n=3, cutoff=cutoff_ratio)
            if close:
                return close[0]
    except Exception:
        pass
    cleaned = re.sub(r"[^a-z0-9]", "", t)
    return cleaned if cleaned else t

# token-level corrections (uses the faster spell_correct_token)
def token_level_corrections(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        tt = TOKEN_STRIP_RE.sub("", t)
        if not tt:
            continue
        tt = tt.lower()
        if len(tt) == 1 and tt not in ("a", "i"):
            continue
        if len(tt) == 2:
            if not is_allowed_two_letter(tt):
                corrected_try = spell_correct_token(tt, cutoff_ratio=0.85)
                if corrected_try and corrected_try != tt:
                    if len(corrected_try) > 2 or is_allowed_two_letter(corrected_try):
                        tt = corrected_try
                    else:
                        continue
                else:
                    continue
        letter_count = sum(c.isalpha() for c in tt)
        if letter_count < max(1, len(tt) // 2) and not tt.isdigit():
            clipped = re.sub(r"[^A-Za-z0-9]", "", tt)
            if len(clipped) >= 2:
                tt = clipped
            else:
                continue
        corrected = spell_correct_token(tt)
        if not corrected:
            continue
        if len(corrected) == 1 and corrected not in ("a", "i"):
            continue
        if len(corrected) == 2 and not is_allowed_two_letter(corrected):
            continue
        out.append(corrected)
    return out

def normalize_text(s: str) -> str:
    return " ".join(s.strip().split()).lower()

def similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def similar_enough(a: str, b: str, cutoff: float = 0.85) -> bool:
    return similarity_ratio(a, b) > cutoff

# Clean OCR text
def clean_ocr_text(s: str) -> str:
    if not s:
        return ""
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
        .replace("—", "-")
        .replace("–", "-")
    )
    s = TIMESTAMP_BRACKET_RE.sub(" ", s)
    s = PARENTHETICAL_NOISE_RE.sub(" ", s)
    s = re.sub(r"\|{1,}", " ", s)
    s = NON_PRINT_RE.sub(" ", s)
    s = PUNCT_RUN_RE.sub(r"\1", s)
    s = PUNCT_CLUSTER_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    s = s.strip(" \"'.,:;!-/|")
    toks = s.split()
    toks = token_level_corrections(toks)
    if toks:
        corrected = []
        keys = list(COMMON_OCR_CORRECTIONS.keys())
        for t in toks:
            if t in COMMON_OCR_CORRECTIONS:
                mapped = COMMON_OCR_CORRECTIONS[t]
                if mapped:
                    corrected.append(mapped)
                continue
            close = get_close_matches(t, keys, n=1, cutoff=0.85)
            if close:
                mapped = COMMON_OCR_CORRECTIONS.get(close[0], t)
                if mapped:
                    corrected.append(mapped)
                    continue
            corrected.append(t)
        toks = corrected
    out = " ".join(toks).strip()
    return out.lower()

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
    cmd = [str(FFMPEG_PATH), "-y", "-i", str(video_path), "-vf", vf, str(out_dir / "frame_%05d.jpg")]
    run_cmd_quiet(cmd)

def process_frames_dir(frames_dir: Path, interval: float, crop: float, min_len: int, engine_order: List[str]) -> List[FrameResult]:
    frames = sorted(frames_dir.glob("*.jpg"))
    results: List[FrameResult] = []
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
        try:
            gray = ImageOps.grayscale(crop_img)
            ac = ImageOps.autocontrast(gray)
            thumb = ac.resize((10, 10))
            mean_pixel = sum(thumb.getdata()) / (10 * 10)
            if mean_pixel < 180:
                proc_img = ImageOps.invert(ac)
            else:
                proc_img = ac
        except Exception:
            proc_img = crop_img.convert("L")
        text, conf = "", 0.0
        for eng in engine_order:
            out = try_pytesseract(proc_img) if eng == "tesseract" else try_easyocr(proc_img)
            if out and out.get("text", "").strip():
                text, conf = out["text"].strip(), out.get("conf", 0.0)
                break
        cleaned = clean_ocr_text(text)
        if len(cleaned) < min_len:
            cleaned = ""
        results.append(FrameResult(str(fp), ts, text, cleaned, float(conf)))
    return results

def merge_frame_results(frames: List[FrameResult], sim_cutoff: float = 0.83, merge_gap: float = 0.5, frame_interval: float = 0.5):
    segments: List[Dict] = []
    cur: Optional[Dict] = None
    for fr in frames:
        txt, ts = fr.text, fr.timestamp
        if not txt:
            continue
        if cur and (txt == cur["text"] or similar_enough(txt, cur["text"], sim_cutoff)):
            cur["end"] = max(cur.get("end", ts), ts + frame_interval)
            cur["conf"] = (cur.get("conf", 0.0) + fr.conf) / 2.0
        else:
            if cur:
                segments.append(cur)
            cur = {"text": txt, "start": ts, "end": ts + frame_interval, "conf": fr.conf}
    if cur:
        segments.append(cur)
    merged: List[Dict] = []
    for seg in segments:
        if merged and (seg["text"] == merged[-1]["text"] or similar_enough(seg["text"], merged[-1]["text"], sim_cutoff)) and seg["start"] - merged[-1]["end"] <= merge_gap:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            merged[-1]["conf"] = (merged[-1].get("conf", 0.0) + seg.get("conf", 0.0)) / 2.0
            if "note" not in merged[-1]:
                merged[-1]["note"] = "merged/cleaned"
            else:
                merged[-1]["note"] += ";merged/cleaned"
        else:
            merged.append(seg)
    return merged

def finalize_ocr_segments(segments: List[Dict]) -> List[Dict]:
    if not segments:
        return []
    cleaned: List[Dict] = []
    for s in segments:
        txt = s.get("text", "")
        if not txt:
            continue
        txt = txt.strip()
        txt = re.sub(r'[_\u200b]+', " ", txt)
        txt = re.sub(r'\s{2,}', " ", txt)
        txt = txt.strip(" \"'.,:;!-/")
        txt = txt.lower()
        toks = token_level_corrections(txt.split())
        if not toks:
            continue
        txt2 = " ".join(toks).strip()
        if len(txt2) < 4 or not re.search(r"[a-z]", txt2):
            continue
        entry = {
            "text": txt2,
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "conf": float(s.get("conf", 0.0)),
            "note": s.get("note", "") or "cleaned",
        }
        cleaned.append(entry)
    out: List[Dict] = []
    for seg in cleaned:
        if not out:
            out.append(seg)
            continue
        prev = out[-1]
        if seg["text"] == prev["text"] or similar_enough(seg["text"], prev["text"], 0.83):
            prev["end"] = max(prev["end"], seg["end"])
            prev["conf"] = (prev.get("conf", 0.0) + seg.get("conf", 0.0)) / 2.0
            prev["note"] = (prev.get("note", "") + ";merged") if prev.get("note") else "merged"
        else:
            out.append(seg)
    final: List[Dict] = []
    for seg in out:
        if not final:
            final.append(seg)
            continue
        prev = final[-1]
        gap = seg["start"] - prev["end"]
        if gap <= 0.6 and len(prev["text"].split()) <= 6 and len(seg["text"].split()) <= 8:
            prev["end"] = seg["end"]
            prev["text"] = (prev["text"].rstrip(" .,") + " " + seg["text"]).strip()
            prev["conf"] = (prev.get("conf", 0.0) + seg.get("conf", 0.0)) / 2.0
            prev["note"] = (prev.get("note", "") + ";concatenated") if prev.get("note") else "concatenated"
        else:
            final.append(seg)
    return final

def extract_audio_ffmpeg(video_path: Path, out_wav: Path, target_sr: int = 16000):
    cmd = [
        str(FFMPEG_PATH), "-y", "-i", str(video_path),
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

def run_all_preprocessings():
    raw_dir, out_dir = DEFAULT_INPUT_DIR, DEFAULT_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for mp4 in raw_dir.glob("*.mp4"):
        tmp = DEFAULT_TMP_ROOT / mp4.stem
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            extract_frames_ffmpeg(mp4, tmp)
            frames = process_frames_dir(tmp, 0.5, 0.25, 4, ["tesseract", "easyocr"])
            segs = merge_frame_results(frames, sim_cutoff=0.83, merge_gap=0.5)
            segs = finalize_ocr_segments(segs)
            with open(out_dir / f"{mp4.stem}_captions.json", "w") as f:
                json.dump({"video": mp4.stem, "segments": segs}, f, ensure_ascii=False, indent=2)
            with open(out_dir / f"{mp4.stem}_captions_raw.json", "w") as f:
                raw_frames = [fr.__dict__ for fr in frames]
                json.dump({"video": mp4.stem, "frames": raw_frames}, f, ensure_ascii=False, indent=2)
            wav_path = out_dir / f"{mp4.stem}.wav"
            extract_audio_ffmpeg(mp4, wav_path)
            preprocess_ravdess_audio(wav_path, out_dir)
            wav_path.unlink(missing_ok=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

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
            frames = process_frames_dir(tmp, 0.5, 0.25, 4, ["tesseract", "easyocr"])
            segs = merge_frame_results(frames, sim_cutoff=0.83, merge_gap=0.5)
            segs = finalize_ocr_segments(segs)
            with open(out_dir / f"{video_path.stem}_captions.json", "w") as f:
                json.dump({"video": video_path.stem, "segments": segs}, f, ensure_ascii=False, indent=2)
            with open(out_dir / f"{video_path.stem}_captions_raw.json", "w") as f:
                raw_frames = [fr.__dict__ for fr in frames]
                json.dump({"video": video_path.stem, "frames": raw_frames}, f, ensure_ascii=False, indent=2)
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
