# src/inference/emotion/ravdess/infer.py
"""
RAVDESS Audio Emotion Inference
- Loads SimpleAudioCNN from backend/models/emotion/ravdess/
- Uses librosa+soundfile for audio loading (no TorchCodec issues)
- Runs inference on a .wav file and prints predicted emotion + probabilities.
- Writes JSON output to backend/data/emotion/output/.
"""

import torch
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import sys, json

# -------------------------------------------------------------------------
# Resolve paths relative to repo structure
# -------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = REPO_ROOT / "models" / "emotion" / "ravdess"
MODEL_PATH = MODEL_DIR / "simple_audio_cnn_aug2_state_dict.pt"
OUT_DIR = REPO_ROOT / "data" / "emotion" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# add the model directory so `simple_audio_cnn` can be imported
sys.path.append(str(MODEL_DIR))

from simple_audio_cnn import SimpleAudioCNN  # local import works now

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

def load_audio(audio_path, target_sr=16000):
    """Robustly load audio using librosa (avoids TorchCodec)."""
    waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    waveform = torch.tensor(waveform).unsqueeze(0)  # shape: (1, N)
    return waveform, target_sr

def predict_emotion(audio_path: str):
    """Predict emotion for a given .wav file using RAVDESS model."""
    waveform, sr = load_audio(audio_path)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=64, n_fft=1024, hop_length=512
    )(waveform)
    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
    mel_spec = mel_spec.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(mel_spec)
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_label = LABELS[pred_idx]
        return pred_label, probs.squeeze().cpu().numpy()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.emotion.ravdess.infer path/to/audio.wav")
        sys.exit(1)

    inp = Path(sys.argv[1])
    if not inp.exists():
        p_repo = REPO_ROOT / inp
        if not p_repo.exists():
            raise FileNotFoundError(f"Audio file not found: {inp}")
        inp = p_repo

    # Run inference
    emotion, probs = predict_emotion(str(inp))
    print(f"Predicted emotion: {emotion}")
    print("Probabilities:")
    for lbl, p in zip(LABELS, probs):
        print(f"  {lbl:10s} : {p:.4f}")

    # ---- Write output JSON ----
    out_json = OUT_DIR / f"{inp.stem}_ravdess_results.json"
    summary = {
        "file": str(inp),
        "predicted_emotion": emotion,
        "probabilities": {lbl: float(p) for lbl, p in zip(LABELS, probs)}
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nResults saved to: {out_json}")
