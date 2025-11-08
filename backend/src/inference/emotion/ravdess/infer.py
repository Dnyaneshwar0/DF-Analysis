#!/usr/bin/env python3
"""
RAVDESS inference (compact, UI-ready output)

Outputs:
  - <stem>_ravdess_results.json
  - <stem>_waveform.npy              (full-resolution amplitude envelope, binary)
  - <stem>_waveform.json             (adaptive downsampled envelope for UI)
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Repo paths
# -------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = REPO_ROOT / "models" / "emotion" / "ravdess"
MODEL_PATH = MODEL_DIR / "simple_audio_cnn_aug2_state_dict.pt"
OUT_DIR = REPO_ROOT / "data" / "emotion" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Make local model importable
sys.path.append(str(MODEL_DIR))
from simple_audio_cnn import SimpleAudioCNN  # noqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

SR = 16000
HOP_LENGTH = 512
SMOOTH_SEC = 0.06


# -------------------------------------------------------------------------
# Model Load
# -------------------------------------------------------------------------
model = SimpleAudioCNN(num_classes=len(LABELS)).to(device)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def load_mel(npy: Path) -> torch.Tensor:
    arr = np.load(npy)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim == 3 and arr.shape[0] != 1:
        arr = arr[:1]
    return torch.from_numpy(arr)[None].float().to(device)  # (1,1,n_mels,T)


def predict(mel):
    with torch.no_grad():
        logits = model(mel)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return LABELS[idx], probs


def amplitude_envelope(mel_np):
    env = np.mean(np.abs(mel_np), axis=0)
    mx = float(env.max())
    if mx > 0:
        env /= mx
    return env.astype(np.float32)


def smooth(env):
    win = max(1, int(SMOOTH_SEC * SR / HOP_LENGTH))
    kernel = np.ones(win, np.float32) / win
    return np.convolve(env, kernel, mode="same").astype(np.float32)


def adaptive_downsample(env: np.ndarray, sr=SR, hop=HOP_LENGTH):
    """
    Adaptive: choose number of points based on clip length, clamp range.
    """
    n = len(env)
    export_points = int(np.clip(n / 60, 24, 64))  # *key adaptive rule*
    idx = np.linspace(0, n - 1, export_points, dtype=int)

    frame_dur = hop / float(sr)
    times = (idx * frame_dur).tolist()
    env_ds = env[idx].tolist()

    return export_points, times, env_ds


# -------------------------------------------------------------------------
# Output Writers
# -------------------------------------------------------------------------
def write_results(npy, label, probs):
    out = {
        "file": str(npy),
        "predicted_emotion": label,
        "probabilities": {lbl: float(p) for lbl, p in zip(LABELS, probs)},
    }
    p = OUT_DIR / f"{npy.stem}_ravdess_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[OK] Saved {p.name}")


def write_waveform(env, stem, sr=SR, hop=HOP_LENGTH):
    # Full binary (optional but kept for future detail use)
    np.save(OUT_DIR / f"{stem}_waveform.npy", env.astype(np.float32))

    export_num_points, times, env_ds = adaptive_downsample(env, sr, hop)
    frame_dur = hop / float(sr)

    payload = {
        "meta": {
            "sr": sr,
            "hop_length": hop,
            "frame_duration": frame_dur,
            "original_num_frames": int(len(env)),
            "export_num_points": int(export_num_points),
        },
        "axes": {"x_label": "Time (s)", "y_label": "Normalized Amplitude"},
        "frames": {"time": times, "envelope": env_ds},
    }

    p = OUT_DIR / f"{stem}_waveform.json"
    p.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Saved {p.name}  (adaptive {export_num_points} points)")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        print("Usage: python -m src.inference.emotion.ravdess.infer path/to/<name>_mel.npy")
        sys.exit(1)

    inp = Path(argv[0])
    if not inp.exists():
        alt = REPO_ROOT / inp
        if alt.exists():
            inp = alt
        else:
            raise FileNotFoundError(inp)

    if inp.suffix.lower() != ".npy":
        raise RuntimeError("Expected .npy mel spectrogram")

    mel = load_mel(inp)
    label, probs = predict(mel)
    write_results(inp, label, probs)

    mel_np = mel.squeeze().cpu().numpy()
    mel_np = mel_np[0] if mel_np.ndim == 3 else mel_np
    if mel_np.ndim != 2:
        raise RuntimeError(f"Bad mel shape: {mel_np.shape}")

    env = smooth(amplitude_envelope(mel_np))
    write_waveform(env, inp.stem)


if __name__ == "__main__":
    main()
