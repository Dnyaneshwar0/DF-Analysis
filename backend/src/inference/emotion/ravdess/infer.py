#!/usr/bin/env python3
"""
RAVDESS inference (model-only) with JSON waveform + TOP-3 emotion timeseries output.
Both the waveform JSON and the top-3 timeseries JSON are forced to exactly 30 points
(evenly spaced across the original clip duration) regardless of the original frame count.

Produces:
  - <name>_ravdess_results.json            (emotion predictions)
  - <name>_waveform.json                   (30 time points + normalized amplitude envelope)
  - <name>_top3_emotions_timeseries.json   (30 time points + top 3 emotion intensities)

Usage:
  python -m src.inference.emotion.ravdess.infer path/to/<name>_mel.npy
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

# ---- Emotion labels ----
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# ---- Constants ----
SR = 16000
HOP_LENGTH = 512
SMOOTH_SEC = 0.06
OUTPUT_POINTS = 30  # force exactly 30 points for waveform & timeseries

# Inference helpers
def load_mel_from_npy(npy_path: Path):
    arr = np.load(npy_path)
    # Normalize expected shapes into (1,1,n_mels,T)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 0)  # (n_mels, T) -> (1, n_mels, T)
    if arr.ndim == 3 and arr.shape[0] != 1:
        # If there are multiple channels/batches, take the first
        arr = arr[0:1]
    # arr should now be (1, n_mels, T)
    tensor = torch.from_numpy(arr).unsqueeze(0).float()  # (1,1,n_mels,T)
    return tensor.to(device, non_blocking=True)


def predict_from_mel_tensor(mel_tensor: torch.Tensor):
    mel_tensor = mel_tensor.to(device)
    if mel_tensor.ndim == 3:
        mel_tensor = mel_tensor.unsqueeze(0)
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
# Envelope calculation and smoothing
# -------------------------------------------------------------------------
def amplitude_envelope_from_mel(mel: np.ndarray):
    """Compute normalized amplitude envelope from mel per frame.
    mel expected shape: (n_mels, n_frames)
    """
    env = np.mean(np.abs(mel), axis=0)
    if env.max() > 0:
        env = env / float(env.max())
    return env


def smooth_vec(x: np.ndarray, sr: int = SR, hop_length: int = HOP_LENGTH, win_sec: float = SMOOTH_SEC):
    if win_sec <= 0:
        return x
    win_frames = max(1, int(win_sec * sr / hop_length))
    kernel = np.ones(win_frames, dtype=float) / win_frames
    return np.convolve(x, kernel, mode="same")


# -------------------------------------------------------------------------
# Helpers to downsample/interpolate to fixed number of points
# -------------------------------------------------------------------------
def make_fixed_points(times_orig: np.ndarray, values_orig: np.ndarray, n_points: int):
    """
    Interpolate (or sample) the original values to produce exactly n_points
    at evenly spaced times across the original duration.

    times_orig: 1D array of original times (seconds)
    values_orig: 1D array of original values (same length as times_orig)
    returns:
      times_new: shape (n_points,)
      values_new: shape (n_points,)
    """
    if times_orig.ndim != 1:
        raise ValueError("times_orig must be 1D")
    if values_orig.ndim != 1:
        raise ValueError("values_orig must be 1D")

    if len(times_orig) == 0:
        # No frames -> return zeros
        times_new = np.linspace(0.0, 0.0, n_points)
        values_new = np.zeros(n_points, dtype=float)
        return times_new, values_new

    t_start = float(times_orig[0])
    t_end = float(times_orig[-1])
    times_new = np.linspace(t_start, t_end, n_points)
    # Use numpy.interp for 1D interpolation (handles duplicates/short sequences)
    values_new = np.interp(times_new, times_orig, values_orig)
    return times_new, values_new


# -------------------------------------------------------------------------
# JSON exporters (modified to produce exactly OUTPUT_POINTS points)
# -------------------------------------------------------------------------
def export_waveform_json(mel_np: np.ndarray, out_path: Path, sr: int = SR, hop_length: int = HOP_LENGTH):
    """Generate JSON containing exactly OUTPUT_POINTS time + normalized amplitude envelope + axis labels."""
    env = amplitude_envelope_from_mel(mel_np)         # shape (n_frames,)
    env_s = smooth_vec(env)
    n_frames = len(env_s)
    frame_dur = hop_length / sr
    times_orig = (np.arange(n_frames) * frame_dur).astype(float)

    times_new, env_new = make_fixed_points(times_orig, env_s, OUTPUT_POINTS)

    data = {
        "meta": {
            "sr": sr,
            "hop_length": hop_length,
            "frame_duration": frame_dur,
            "original_num_frames": n_frames,
            "export_num_points": OUTPUT_POINTS
        },
        "axes": {
            "x_label": "Time (s)",
            "y_label": "Normalized Amplitude"
        },
        "frames": {
            "time": times_new.tolist(),
            "envelope": env_new.tolist()
        }
    }

    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved waveform JSON: {out_path}")


def export_top3_emotions_timeseries_json(mel_np: np.ndarray, probs_vec: np.ndarray, out_path: Path,
                                         sr: int = SR, hop_length: int = HOP_LENGTH):
    """
    Export top-3 emotion intensities across time, but reduced to exactly OUTPUT_POINTS.
    Top-3 probs are taken from the clip-level probabilities and then broadcasted to OUTPUT_POINTS.
    """
    n_frames = mel_np.shape[1]
    frame_dur = hop_length / sr
    times_orig = (np.arange(n_frames) * frame_dur).astype(float)

    probs_vec = np.asarray(probs_vec).astype(float)
    top3_idx = np.argsort(-probs_vec)[:3]
    top3_labels = [LABELS[i] for i in top3_idx]
    top3_probs = probs_vec[top3_idx]  # shape (3,)

    # For timeseries we create exactly OUTPUT_POINTS time samples
    if n_frames > 0:
        times_new = np.linspace(times_orig[0], times_orig[-1], OUTPUT_POINTS)
    else:
        times_new = np.linspace(0.0, 0.0, OUTPUT_POINTS)

    # Since clip-level probs are constant across time, intensities are just tiled for OUTPUT_POINTS
    intensities = np.tile(top3_probs[np.newaxis, :], (OUTPUT_POINTS, 1))  # shape (OUTPUT_POINTS, 3)

    data = {
        "meta": {
            "sr": sr,
            "hop_length": hop_length,
            "frame_duration": frame_dur,
            "original_num_frames": n_frames,
            "export_num_points": OUTPUT_POINTS
        },
        "axes": {
            "x_label": "Time (s)",
            "y_label": "Intensity (0..1)"
        },
        "emotions": top3_labels,
        "frames": {
            "time": times_new.tolist(),
            "intensities": intensities.tolist()
        },
        "summary": {
            "predicted_emotion": top3_labels[0],
            "top3_clip_probs": [
                {"label": lbl, "prob": float(prob)} for lbl, prob in zip(top3_labels, top3_probs)
            ]
        }
    }

    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved top-3 emotions timeseries JSON: {out_path}")


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
        alt = REPO_ROOT / inp
        if alt.exists():
            inp = alt
        else:
            raise FileNotFoundError(f"Input file not found: {argv[0]}")

    if inp.suffix.lower() != ".npy":
        raise RuntimeError("Expected a .npy mel spectrogram as input.")

    mel_tensor = load_mel_from_npy(inp)
    label, probs = predict_from_mel_tensor(mel_tensor)

    # Save clip-level emotion prediction
    out_json = OUT_DIR / f"{inp.stem}_ravdess_results.json"
    summary = {
        "file": str(inp),
        "predicted_emotion": label,
        "probabilities": {lbl: float(p) for lbl, p in zip(LABELS, probs)}
    }
    out_json.write_text(json.dumps(summary, indent=4))
    print(f"Predicted: {label} — saved: {out_json}")

    # prepare mel for JSON export
    mel_np = mel_tensor.squeeze().cpu().numpy()
    # after squeeze expected shape: (n_mels, n_frames)
    if mel_np.ndim == 3:
        mel_np = mel_np[0]
    if mel_np.ndim != 2:
        raise RuntimeError(f"Unexpected mel numpy shape: {mel_np.shape}")

    # Save waveform JSON (exactly OUTPUT_POINTS)
    wave_json = OUT_DIR / f"{inp.stem}_waveform.json"
    export_waveform_json(mel_np, wave_json)

    # Save top-3 emotions timeseries JSON (exactly OUTPUT_POINTS)
    emo_json = OUT_DIR / f"{inp.stem}_top3_emotions_timeseries.json"
    export_top3_emotions_timeseries_json(mel_np, probs, emo_json)


if __name__ == "__main__":
    main()

