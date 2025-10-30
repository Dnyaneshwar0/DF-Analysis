#!/usr/bin/env python3
"""
backend/src/inference/emotion/ocr/captions_to_emotions.py

Runs GoEmotions model on OCR-cleaned caption segments and generates emotion
labels, probabilities, CSV summaries, and visualization plots.

This version resolves the backend root by searching ancestors for a folder named 'backend',
so it won't break depending on how you invoke the module.
"""

import json
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from src.inference.emotion.goemotions.infer import GoEmotionsPipeline

def find_backend_root() -> Path:
    p = Path(__file__).resolve()
    for a in p.parents:
        if a.name == "backend":
            return a
    # fallback: go 4 levels up (best-effort)
    return p.parents[4]

BACKEND_DIR = find_backend_root()

def resolve_input_path(inp: str) -> Path:
    p = Path(inp)
    if p.exists():
        return p.resolve()
    # try relative to backend root
    p2 = (BACKEND_DIR / inp)
    if p2.exists():
        return p2.resolve()
    # try in data/emotion/output/ocr/
    p3 = BACKEND_DIR / "data" / "emotion" / "output" / "ocr" / Path(inp).name
    if p3.exists():
        return p3.resolve()
    raise FileNotFoundError(f"Segments JSON not found (tried): {p}, {p2}, {p3}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.emotion.ocr.captions_to_emotions <path/to/segments_clean.json>")
        sys.exit(1)

    inp_json = resolve_input_path(sys.argv[1])
    print(f"Using segments JSON: {inp_json}")  # quick feedback for debugging

    video_name = inp_json.stem.replace("_segments_clean", "")
    out_dir = BACKEND_DIR / "data" / "emotion" / "output" / "ocr"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"{video_name}_captions_emotions.json"
    out_csv = out_dir / f"{video_name}_linewise_emotions.csv"

    data = json.load(open(inp_json, encoding="utf8"))
    segments = data.get("segments", [])
    texts = [seg["text"] for seg in segments]
    print(f"Loaded {len(texts)} caption segments for emotion analysis.")

    emo_model = GoEmotionsPipeline()
    results = emo_model.predict_batch(texts)

    for seg, res in zip(segments, results):
        seg.update({
            "labels": res.get("labels", []),
            "probs": res.get("probs", {})
        })

    with open(out_json, "w", encoding="utf8") as f:
        json.dump({"video": video_name, "segments": segments}, f, indent=2, ensure_ascii=False)

    with open(out_csv, "w", newline="", encoding="utf8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["start", "end", "text", "labels"])
        for seg in segments:
            writer.writerow([seg.get("start"), seg.get("end"), seg.get("text"), ",".join(seg.get("labels", []))])

    print(f"✅ Wrote {out_json}")
    print(f"✅ Wrote {out_csv}")

    top_labels = [seg.get("labels", ["none"])[0] for seg in segments]
    timeline = [seg.get("end", 0.0) for seg in segments]
    unique_labels = sorted(set(top_labels))

    if unique_labels:
        plt.figure(figsize=(10, 3))
        plt.plot(timeline, [unique_labels.index(l) for l in top_labels], "o-", alpha=0.7)
        plt.yticks(range(len(unique_labels)), unique_labels)
        plt.xlabel("Time (s)")
        plt.title(f"Dominant Emotion Timeline — {video_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"linewise_dominant_timeline.png", dpi=150)
        plt.close()

        # Top-3 emotion distribution (counts of top labels)
        label_counts = {}
        for lbl in top_labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        labels, counts = zip(*sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
        plt.figure(figsize=(8, 4))
        plt.bar(labels, counts)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Frequency")
        plt.title(f"Top Emotion Distribution — {video_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"linewise_top3_stacked.png", dpi=150)
        plt.close()

        print("✅ Plots saved to:", out_dir)
    else:
        print("No labels found — skipping plots.")

if __name__ == "__main__":
    main()
