#!/usr/bin/env python3
"""
RAVDESS inference (model-only)

This script **does not** perform audio preprocessing. It expects a
precomputed Mel-spectrogram saved as a NumPy .npy file.

Usage:
  # preferred: use preprocessor to create <name>_mel.npy first
  python -m src.inference.emotion.ravdess.infer path/to/<name>_mel.npy

If you only have a .wav or .mp4, run the preprocessing step first:
  python backend/src/utils/emo_preprocessing.py --audio path/to/file.wav
  or (if you want audio from video)
  python backend/src/utils/emo_preprocessing.py --video path/to/file.mp4
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Resolve repo paths
# -------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = REPO_ROOT / "models" / "emotion" / "ravdess"
MODEL_PATH = MODEL_DIR / "simple_audio_cnn_aug2_state_dict.pt"
OUT_DIR = REPO_ROOT / "data" / "emotion" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# make local model module importable
sys.path.append(str(MODEL_DIR))
from simple_audio_cnn import SimpleAudioCNN  # local model definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load model ----
model = SimpleAudioCNN(num_classes=8).to(device)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ---- Emotion labels (RAVDESS standard) ----
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# -------------------------------------------------------------------------
# Inference helpers (no preprocessing here)
# -------------------------------------------------------------------------
def load_mel_from_npy(npy_path: Path):
    """Load a mel spectrogram saved as NumPy array and convert to torch tensor.
    Expected shape on disk: (n_mels, time) or (1, n_mels, time).
    Returns a tensor of shape (1, 1, n_mels, time) on the model device.
    """
    arr = np.load(npy_path)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 0)   # (1, n_mels, time)
    if arr.ndim == 3:
        # if already (batch, n_mels, time), take first sample
        # convert to shape (1, 1, n_mels, time)
        arr = arr[0]
        arr = np.expand_dims(arr, 0)
    # now arr shape should be (1, n_mels, time)
    tensor = torch.from_numpy(arr).unsqueeze(0).float()  # (1, 1, n_mels, time)
    return tensor.to(device, non_blocking=True)

def predict_from_mel_tensor(mel_tensor: torch.Tensor):
    """Run the model and return (label, probs_numpy)."""
    mel_tensor = mel_tensor.to(device)
    if mel_tensor.ndim == 3:
        mel_tensor = mel_tensor.unsqueeze(0)  # ensure batch dim
    # Model expects (batch, 1, n_mels, time) — adapt if needed
    if mel_tensor.ndim == 4:
        inp = mel_tensor
    elif mel_tensor.ndim == 2:
        inp = mel_tensor.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected mel tensor shape: {mel_tensor.shape}")

    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        return LABELS[pred_idx], probs.squeeze().cpu().numpy()

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        print("Usage: python -m src.inference.emotion.ravdess.infer path/to/<name>_mel.npy")
        sys.exit(1)

    inp = Path(argv[0])
    if not inp.exists():
        # try resolving relative to repo root data folder
        alt = REPO_ROOT / inp
        if alt.exists():
            inp = alt
        else:
            raise FileNotFoundError(f"Input file not found: {argv[0]}")

    # refuse to load raw audio — preprocessing must be done separately
    if inp.suffix.lower() in {".wav", ".mp3", ".mp4", ".m4a", ".flac"}:
        raise RuntimeError(
            "This script no longer performs audio preprocessing. "
            "Please run the preprocessing utility to create a mel-spectrogram .npy first.\n"
            "Example:\n"
            "  python backend/src/utils/emo_preprocessing.py --audio ../data/emotion/input/raw/file.wav\n"
            "or\n"
            "  python backend/src/utils/emo_preprocessing.py --video ../data/emotion/input/raw/file.mp4\n"
        )

    if inp.suffix.lower() != ".npy":
        raise RuntimeError("Expected a .npy mel spectrogram as input (use the preprocessing script).")

    mel_tensor = load_mel_from_npy(inp)
    label, probs = predict_from_mel_tensor(mel_tensor)

    # write outputs (and print a minimal summary)
    out_json = OUT_DIR / f"{inp.stem}_ravdess_results.json"
    summary = {
        "file": str(inp),
        "predicted_emotion": label,
        "probabilities": {lbl: float(p) for lbl, p in zip(LABELS, probs)}
    }
    with open(out_json, "w", encoding="utf8") as fh:
        json.dump(summary, fh, indent=4)
    # Minimal CLI feedback (one line)
    print(f"Predicted: {label} — saved: {out_json}")

if __name__ == "__main__":
    main()
