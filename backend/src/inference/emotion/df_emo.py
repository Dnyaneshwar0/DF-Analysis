#!/usr/bin/env python3
"""

Runs:
  preprocessing -> ravdess -> rafdb -> goemotions
Merges everything into a single JSON file:
  <video>_multimodel_summary.json

This version EXCLUDES from the merged JSON:
  - goemotions.annotated
  - ravdess.top3_timeseries

Behavior:
 - External tools' stdout/stderr are suppressed; you see concise [RUN]/[DONE].
 - Final result block prints a compact summary (no raw JSON dump).
 - Intermediate JSONs are deleted; only remain:
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
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
    res = subprocess.run(list(map(str, cmd)), check=False,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
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
    print(f"[STEP] RAVDESS inference (audio -> results + waveform)")
    script = REPO_ROOT / "src" / "inference" / "emotion" / "ravdess" / "infer.py"
    subprocess.run(
        [sys.executable, str(script), str(mel_path)],
        check=True,
        cwd=REPO_ROOT,
    )
    base = mel_path.stem
    results = OUT_DIR / f"{base}_ravdess_results.json"
    waveform = OUT_DIR / f"{base}_waveform.json"

    # We intentionally do NOT require or include *_top3_emotions_timeseries.json
    for p in (results, waveform):
        if not p.exists():
            raise FileNotFoundError(f"Missing RAVDESS output: {p}")
    print(f"[OK] RAVDESS -> {results.name}, {waveform.name}")
    return results, waveform


def run_rafdb(video_path: Path):
    """Call RAF-DB pipeline script with suppressed output."""
    print(f"[STEP] RAF-DB inference (video -> annotated mp4 + summary)")
    script = REPO_ROOT / "src" / "inference" / "emotion" / "rafdb" / "video_pipeline.py"
    run_cmd_quiet([sys.executable, str(script), str(video_path)])
    out_vid = OUT_DIR / f"out_{video_path.stem}.mp4"
    summary = OUT_DIR / f"summary_{video_path.stem}.json"
    if not out_vid.exists() or not summary.exists():
        raise FileNotFoundError("RAF-DB outputs missing.")
    print(f"[OK] RAF-DB -> {out_vid.name}, {summary.name}")
    return out_vid, summary


def run_goemotions(captions_json: Path):
    """
    Run the goemotions pipeline programmatically but suppress its prints.
    Returns dict of output path strings as the module does.
    """
    print(f"[STEP] GoEmotions inference (text segments -> summary/linegraph/tabledata)")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    # with redirect_stdout(buf_out), redirect_stderr(buf_err):
    from inference.emotion.goemotions.infer import run_inference as _run
    res = _run(captions_json)
    
    print("Raw result:", res)
    if (msg := buf_err.getvalue().strip()):
        print(f"[WARN] GoEmotions emitted warnings (suppressed).")
    # Validate existence
    for p in res.values():
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing GoEmotions output: {p}")
    return {k: Path(p) for k, p in res.items()}


def merge_all(stem: str, raf_summary: Path, goem: dict, ravdess: tuple[Path, Path]):
    print(f"[STEP] Merging outputs into {stem}_multimodel_summary.json")
    merged = {
        "file": f"{stem}.mp4",
        "models": {
            "rafdb": {},
            "goemotions": {},
            "ravdess": {}
        }
    }

    # RAF summary + annotated video URL
    raf_data = json.loads(raf_summary.read_text())
    raf_data["annotated_video_url"] = f"/emotion_media/out_{stem}.mp4"
    merged["models"]["rafdb"] = raf_data

    # GoEmotions
    ge_summary = Path(goem["summary_json"])
    ge_line = Path(goem["linegraph_json"])
    ge_table = Path(goem["tabledata_json"])
    merged["models"]["goemotions"] = {
        "summary": json.loads(ge_summary.read_text()),
        "linegraph": json.loads(ge_line.read_text()),
        "tabledata": json.loads(ge_table.read_text()),
    }

    # RAVDESS
    results, waveform = ravdess
    merged["models"]["ravdess"] = {
        "results": json.loads(results.read_text()),
        "waveform": json.loads(waveform.read_text()),
    }

    out_json = OUT_DIR / f"{stem}_multimodel_summary.json"
    out_json.write_text(json.dumps(merged, indent=2))
    print(f"[OK] Wrote merged summary: {out_json.name}")

    keep = {str(out_json), str(OUT_DIR / f"out_{stem}.mp4")}
    for f in OUT_DIR.glob(f"*{stem}*.json"):
        if str(f) not in keep:
            try:
                f.unlink()
                print(f"[CLEAN] Deleted {f.name}")
            except Exception as e:
                print(f"[WARN] Could not delete {f.name}: {e}")

    return out_json

    
    # RAF summary + expose annotated video URL for frontend
    raf_data = json.loads(raf_summary.read_text())
    raf_data["annotated_video_url"] = f"/emotion_media/out_{stem}.mp4"
    merged["models"]["rafdb"] = raf_data

    
    # GoEmotions: include only summary, linegraph, tabledata (no annotated)
    ge_summary = Path(goem["summary_json"])
    ge_line = Path(goem["linegraph_json"])
    ge_table = Path(goem["tabledata_json"])
    merged["models"]["goemotions"] = {
        "summary": json.loads(ge_summary.read_text()),
        "linegraph": json.loads(ge_line.read_text()),
        "tabledata": json.loads(ge_table.read_text()),
    }

    # Ravdess: include only results + waveform (no top3_timeseries)
    results, waveform = ravdess
    merged["models"]["ravdess"] = {
        "results": json.loads(results.read_text()),
        "waveform": json.loads(waveform.read_text()),
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
    """Print a concise, human-friendly summary of the merged JSON."""
    print("\nResult Summary")
    data = json.loads(multimodel_json.read_text())
    print(f"File: {data.get('file')}")

    # RAF-DB
    raf = data.get("models", {}).get("rafdb", {})
    dur = raf.get("duration_s") or data.get("duration_s")
    if dur:
        print(f"Duration: {dur} s")
    dom = raf.get("dominant_emotion") or raf.get("dominant", "N/A")
    print(f"RAF-DB dominant emotion: {dom}")
    print(f"RAF-DB timeline points: {len(raf.get('timeline', []))}")

    # RAVDESS
    rav = data.get("models", {}).get("ravdess", {})
    results = rav.get("results", {})
    pred = results.get("predicted_emotion")
    if pred:
        print(f"RAVDESS predicted emotion: {pred}")
        probs = results.get("probabilities", {})
        top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
        print("RAVDESS top3:", ", ".join([f"{k} ({v:.2f})" for k, v in top3]))

    # GoEmotions
    ge_sum = data.get("models", {}).get("goemotions", {}).get("summary", {})
    ge_dom = ge_sum.get("dominant_emotion")
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

    # 2) RAVDESS (returns only results + waveform)
    ravdess_results, ravdess_wave = run_ravdess(mel_npy)

    # 3) RAF-DB
    _, raf_summary = run_rafdb(video)

    # 4) GoEmotions
    goem = run_goemotions(captions_json)

    # 5) Merge (exclude annotated + top3) and cleanup
    merged_json = merge_all(stem, raf_summary, goem, (ravdess_results, ravdess_wave))

    # 6) Pretty-print a compact result block
    # pretty_print_result(merged_json)# Load the merged JSON content into a Python dict
    with open(merged_json, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    print(f"[OK] Emotion analysis complete: {merged_json}")
    return result_data

if __name__ == "__main__":
    main()