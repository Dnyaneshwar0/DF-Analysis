#!/usr/bin/env python3
"""
Unified Emotion Orchestrator (quiet, pretty progress)
----------------------------------------------------
Runs:
  preprocessing -> ravdess -> rafdb -> goemotions
Merges everything into a single JSON file:
  <video>_multimodel_summary.json

Behavior differences vs previous:
 - External tools' stdout/stderr are suppressed; you only see concise [RUN]/[DONE] stage messages.
 - Final result block prints a compact, pretty summary (no long raw JSON dump).
 - Intermediate JSONs are inlined into the final merged JSON and then deleted, leaving only:
     out_<stem>.mp4
     <stem>_multimodel_summary.json
"""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "emotion" / "output"
PROCESSED_DIR = REPO_ROOT / "data" / "emotion" / "input" / "processed"

# Prefer fixed preprocessing script, else fallback
PREPROCESS_SCRIPT_CANDIDATES = [
    REPO_ROOT / "src" / "utils" / "emo_preprocessing_fixed.py",
    REPO_ROOT / "src" / "utils" / "emo_preprocessing.py",
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def find_preprocess_script() -> Path:
    for p in PREPROCESS_SCRIPT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Preprocessing script not found in utils/.")


def run_cmd_quiet(cmd):
    """
    Run a command but suppress its stdout/stderr. Show concise START/DONE messages.
    Raises CalledProcessError on non-zero exit (will propagate).
    """
    print(f"[RUN]  {' '.join(map(str, cmd))}")
    # capture output to avoid flooding user console
    res = subprocess.run(list(map(str, cmd)), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        # surface a terse error but include stderr for debugging
        print(f"[ERROR] Command failed: {' '.join(map(str, cmd))}")
        print(f"[ERROR] stderr: {res.stderr.strip() or '(no stderr)'}")
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)
    print(f"[DONE]  {' '.join(map(str, cmd))}")
    return res


def run_preprocessing(video_path: Path):
    """Run OCR + audio preprocessing to get captions.json + mel.npy."""
    pre_script = find_preprocess_script()
    run_cmd_quiet([sys.executable, str(pre_script), "--video", str(video_path)])
    captions = PROCESSED_DIR / f"{video_path.stem}_captions.json"
    mel = PROCESSED_DIR / f"{video_path.stem}_mel.npy"
    if not captions.exists():
        raise FileNotFoundError(f"Missing captions: {captions}")
    if not mel.exists():
        raise FileNotFoundError(f"Missing mel: {mel}")
    print(f"[OK] Preprocessing produced: {captions.name}, {mel.name}")
    return captions, mel


def run_ravdess(mel_path: Path):
    """Call ravdess infer module with suppressed output."""
    print(f"[STEP] RAVDESS inference (audio -> results + waveform + top3)")
    run_cmd_quiet([sys.executable, "-m", "src.inference.emotion.ravdess.infer", str(mel_path)])
    base = mel_path.stem
    results = OUT_DIR / f"{base}_ravdess_results.json"
    waveform = OUT_DIR / f"{base}_waveform.json"
    top3 = OUT_DIR / f"{base}_top3_emotions_timeseries.json"
    for p in (results, waveform, top3):
        if not p.exists():
            raise FileNotFoundError(f"Missing RAVDESS output: {p}")
    print(f"[OK] RAVDESS -> {results.name}, {waveform.name}, {top3.name}")
    return results, waveform, top3


def run_rafdb(video_path: Path):
    """Call RAF-DB pipeline script with suppressed output."""
    print(f"[STEP] RAF-DB inference (video -> annotated mp4 + summary)")
    run_cmd_quiet([sys.executable, "-m", "src.inference.emotion.rafdb.video_pipeline", str(video_path)])
    out_vid = OUT_DIR / f"out_{video_path.stem}.mp4"
    summary = OUT_DIR / f"summary_{video_path.stem}.json"
    if not out_vid.exists() or not summary.exists():
        raise FileNotFoundError("RAF-DB outputs missing.")
    print(f"[OK] RAF-DB -> {out_vid.name}, {summary.name}")
    return out_vid, summary


def run_goemotions(captions_json: Path):
    """
    Run the goemotions pipeline programmatically but suppress any prints it may emit.
    Returns dict of output path strings as the module does.
    """
    print(f"[STEP] GoEmotions inference (text segments -> annotated jsons)")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    # capture stdout/stderr during the call
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        from inference.emotion.goemotions.infer import run_inference as _run
        res = _run(captions_json)
    # If the module emitted errors to stderr, surface a short note (but don't spam)
    err_val = buf_err.getvalue().strip()
    if err_val:
        print(f"[WARN] GoEmotions emitted warnings (suppressed).")
    print(f"[OK] GoEmotions -> produced {len(res)} files")
    # Convert to Path objects and validate
    for p in res.values():
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing GoEmotions output: {p}")
    return {k: Path(p) for k, p in res.items()}


def merge_all(stem: str, raf_summary: Path, goem: dict, ravdess: tuple):
    """Combine all JSON content into one and delete extras."""
    print(f"[STEP] Merging outputs into {stem}_multimodel_summary.json")
    merged = {
        "file": f"{stem}.mp4",
        "models": {
            "rafdb": {},
            "goemotions": {},
            "ravdess": {}
        }
    }

    # RAF summary
    merged["models"]["rafdb"] = json.loads(raf_summary.read_text())

    # GoEmotions
    annotated = Path(goem["annotated_json"])
    summary = Path(goem["summary_json"])
    line = Path(goem["linegraph_json"])
    table = Path(goem["tabledata_json"])
    merged["models"]["goemotions"] = {
        "annotated": json.loads(annotated.read_text()),
        "summary": json.loads(summary.read_text()),
        "linegraph": json.loads(line.read_text()),
        "tabledata": json.loads(table.read_text())
    }

    # Ravdess
    results, waveform, top3 = ravdess
    merged["models"]["ravdess"] = {
        "results": json.loads(results.read_text()),
        "waveform": json.loads(waveform.read_text()),
        "top3_timeseries": json.loads(top3.read_text())
    }

    # Write unified JSON
    out_json = OUT_DIR / f"{stem}_multimodel_summary.json"
    out_json.write_text(json.dumps(merged, indent=2))
    print(f"[OK] Wrote merged summary: {out_json.name}")

    # Cleanup intermediate JSONs (keep out_<video>.mp4 + multimodel_summary)
    keep = {str(out_json), str(OUT_DIR / f"out_{stem}.mp4")}
    for f in OUT_DIR.glob(f"*{stem}*.json"):
        if str(f) not in keep:
            try:
                f.unlink()
                print(f"[CLEAN] Deleted {f.name}")
            except Exception as e:
                print(f"[WARN] Could not delete {f.name}: {e}")

    return out_json


def pretty_print_result(multimodel_json: Path):
    """Print a concise, human-friendly summary of the merged JSON (no raw files)."""
    print("\nResult Summary")
    data = json.loads(multimodel_json.read_text())
    print(f"File: {data.get('file')}")
    # RAF
    raf = data.get("models", {}).get("rafdb", {})
    dur = raf.get("duration_s") or data.get("duration_s")
    if dur:
        print(f"Duration: {dur} s")
    dom = raf.get("dominant_emotion") or raf.get("dominant", "N/A")
    print(f"RAF-DB dominant emotion: {dom}")
    timeline = raf.get("timeline", [])
    print(f"RAF-DB timeline points: {len(timeline)}")

    # RAVDESS
    rav = data.get("models", {}).get("ravdess", {})
    results = rav.get("results", {})
    pred = results.get("predicted_emotion")
    if pred:
        print(f"RAVDESS predicted emotion: {pred}")
        probs = results.get("probabilities", {})
        # top3 quick
        top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
        print("RAVDESS top3:", ", ".join([f"{k} ({v:.2f})" for k, v in top3]))

    # GoEmotions
    ge_sum = data.get("models", {}).get("goemotions", {}).get("summary", {})
    ge_dom = ge_sum.get("dominant_emotion") if ge_sum else None
    if ge_dom:
        print(f"GoEmotions dominant: {ge_dom}")

   
    print(f"\nFiles left in output folder: {OUT_DIR.name}/")
    for p in sorted(OUT_DIR.iterdir()):
        print(" -", p.name)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        print("Usage: python src/inference/emotion/orchestrator.py path/to/video.mp4")
        sys.exit(1)

    video = Path(argv[0])
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    stem = video.stem

    # 1) Preprocess
    captions_json, mel_npy = run_preprocessing(video)

    # 2) RAVDESS
    ravdess_results, ravdess_wave, ravdess_top3 = run_ravdess(mel_npy)

    # 3) RAF-DB
    raf_vid, raf_summary = run_rafdb(video)

    # 4) GoEmotions
    goem = run_goemotions(captions_json)

    # 5) Merge and cleanup
    merged_json = merge_all(stem, raf_summary, goem, (ravdess_results, ravdess_wave, ravdess_top3))

    # 6) Pretty-print a compact result block (no raw JSON contents spilled)
    pretty_print_result(merged_json)

    print("\nDone.")


if __name__ == "__main__":
    main()
