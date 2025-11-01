#!/usr/bin/env python3
"""
GoEmotions inference module — extended and writes outputs to backend/data/emotion/output/

Extended features:
 - annotate an OCR segments JSON with GoEmotions predictions (predict_json)
 - export annotated JSON to CSV (export_csv)
 - compute a summary distribution (summarize)
 - CLI: run any of the above from the command line

All outputs are written under: backend/data/emotion/output/
"""

import joblib
import numpy as np
from pathlib import Path
import json
import csv
import argparse
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "data" / "emotion" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

class GoEmotionsPipeline:
    def __init__(self, base_dir: Optional[Path] = None):
        # Default hard-coded base dir (update if you move the repo)
        if base_dir is None:
            BASE_DIR = Path("/home/ampm/projects/DF-Analysis/backend/models/emotion/goemotions_model")
        else:
            BASE_DIR = Path(base_dir)

        self.model_path = BASE_DIR / "goemotions_clf.joblib"
        self.tfidf_path = BASE_DIR / "goemotions_tfidf.joblib"
        self.labels_path = BASE_DIR / "labels.txt"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model at {self.model_path}")

        if not self.tfidf_path.exists():
            raise FileNotFoundError(f"Missing TF-IDF at {self.tfidf_path}")

        if not self.labels_path.exists():
            raise FileNotFoundError(f"Missing labels file at {self.labels_path}")

        # Load components
        self.vectorizer = joblib.load(self.tfidf_path)
        self.model = joblib.load(self.model_path)
        self.labels = [x.strip() for x in open(self.labels_path, "r", encoding="utf8").readlines() if x.strip()]

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict emotions for a single text."""
        if not text or not text.strip():
            return {"text": text, "labels": [], "probs": {}}

        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]

        probs_dict = {label: float(p) for label, p in zip(self.labels, probs)}
        top_labels = [label for label, _ in sorted(probs_dict.items(), key=lambda x: -x[1])[:4]]
        return {"text": text, "labels": top_labels, "probs": probs_dict}

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict emotions for multiple texts."""
        return [self.predict(t) for t in texts]

    # ------------------------------------------------------------------
    # High-level helpers: annotate JSON, export CSV, summarize
    # ------------------------------------------------------------------
    def predict_json(self, ocr_json_path: Path, out_path: Optional[Path] = None, overwrite: bool = True) -> Path:
        """
        Load an OCR segments JSON ({"video":..., "segments":[{...}]}) and annotate each segment
        with "emotions" (top labels list) and "probs" (dict).
        Writes annotated JSON to out_path or OUT_DIR / <video>_goemotions.json.
        Returns path to written file.
        """
        ocr_json_path = Path(ocr_json_path)
        if not ocr_json_path.exists():
            raise FileNotFoundError(f"OCR JSON not found: {ocr_json_path}")

        data = json.load(open(ocr_json_path, encoding="utf8"))
        segments = data.get("segments", [])
        texts = [seg.get("text", "") for seg in segments]

        preds = self.predict_batch(texts)

        for seg, p in zip(segments, preds):
            seg["emotions"] = p.get("labels", [])
            seg["probs"] = p.get("probs", {})

        video_name = data.get("video") or ocr_json_path.stem

        if out_path is None:
            out_path = OUT_DIR / f"{video_name}_goemotions.json"

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {out_path}")

        json.dump({"video": video_name, "segments": segments}, open(out_path, "w", encoding="utf8"), ensure_ascii=False, indent=2)
        return out_path

    def export_csv(self, annotated_json_path: Path, out_csv_path: Optional[Path] = None, include_text: bool = True) -> Path:
        """
        Export annotated JSON (with "probs") to CSV.
        CSV columns: start,end,(text),top1,top2,top3,top4,<label1>,<label2>,...
        """
        annotated_json_path = Path(annotated_json_path)
        if not annotated_json_path.exists():
            raise FileNotFoundError(f"Annotated JSON not found: {annotated_json_path}")

        data = json.load(open(annotated_json_path, encoding="utf8"))
        segments = data.get("segments", [])

        labels = list(self.labels)  # keep canonical order
        header = ["start", "end"]
        if include_text:
            header.append("text")
        header += ["top1", "top2", "top3", "top4"] + labels

        if out_csv_path is None:
            stem = annotated_json_path.stem
            out_csv_path = OUT_DIR / f"{stem}_linewise_emotions.csv"

        with open(out_csv_path, "w", newline="", encoding="utf8") as cf:
            writer = csv.writer(cf)
            writer.writerow(header)
            for seg in segments:
                start = seg.get("start", "")
                end = seg.get("end", "")
                text = seg.get("text", "")
                emotions = seg.get("emotions", [])[:4]
                top = emotions + [""] * (4 - len(emotions))
                probs = seg.get("probs", {})
                row = [start, end]
                if include_text:
                    row.append(text)
                row += top
                row += [probs.get(lbl, 0.0) for lbl in labels]
                writer.writerow(row)

        return out_csv_path

    def summarize(self, annotated_json_path: Path, out_summary_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Compute aggregate emotion distribution across segments.
        Returns a dict and writes to OUT_DIR if out_summary_path is None/unspecified.
        """
        annotated_json_path = Path(annotated_json_path)
        if not annotated_json_path.exists():
            raise FileNotFoundError(f"Annotated JSON not found: {annotated_json_path}")

        data = json.load(open(annotated_json_path, encoding="utf8"))
        segments = data.get("segments", [])
        if not segments:
            dist = {lbl: 0.0 for lbl in self.labels}
            summary = {
                "video": data.get("video", annotated_json_path.stem),
                "dominant_emotion": None,
                "emotion_distribution": dist,
                "segments_count": 0
            }
            if out_summary_path:
                json.dump(summary, open(out_summary_path, "w", encoding="utf8"), indent=2, ensure_ascii=False)
            else:
                json.dump(summary, open(OUT_DIR / f"{annotated_json_path.stem}_goemotions_summary.json", "w", encoding="utf8"), indent=2, ensure_ascii=False)
            return summary

        accum = {lbl: 0.0 for lbl in self.labels}
        count = 0
        for seg in segments:
            probs = seg.get("probs", {})
            if not probs:
                continue
            for lbl in self.labels:
                accum[lbl] += float(probs.get(lbl, 0.0))
            count += 1

        if count == 0:
            avg = {lbl: 0.0 for lbl in self.labels}
        else:
            avg = {lbl: accum[lbl] / count for lbl in self.labels}

        dominant = max(avg.items(), key=lambda x: x[1])[0] if avg else None

        summary = {
            "video": data.get("video", annotated_json_path.stem),
            "dominant_emotion": dominant,
            "emotion_distribution": avg,
            "segments_count": count
        }

        if out_summary_path:
            json.dump(summary, open(out_summary_path, "w", encoding="utf8"), indent=2, ensure_ascii=False)
        else:
            json.dump(summary, open(OUT_DIR / f"{annotated_json_path.stem}_goemotions_summary.json", "w", encoding="utf8"), indent=2, ensure_ascii=False)

        return summary

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _cli():
    p = argparse.ArgumentParser(prog="goemotions.infer", description="Run GoEmotions on OCR segments JSON and export outputs.")
    p.add_argument("--ocr-json", help="Path to OCR segments JSON (required for most ops).")
    p.add_argument("--model-dir", help="Optional path to goemotions model dir (overrides hardcoded).")
    p.add_argument("--out-json", help="Explicit annotated JSON path (written under output if provided as filename).")
    p.add_argument("--csv", action="store_true", help="Also write CSV linewise export (default name under output).")
    p.add_argument("--out-csv", help="Explicit CSV path.")
    p.add_argument("--summary", action="store_true", help="Write summary JSON (avg distribution).")
    p.add_argument("--out-summary", help="Path to summary JSON output.")
    p.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing annotated JSON file.")
    args = p.parse_args()

    if not args.ocr_json:
        raise SystemExit("Error: --ocr-json is required for this CLI tool.")

    pipeline = GoEmotionsPipeline(base_dir=Path(args.model_dir) if args.model_dir else None)

    ocr_json_path = Path(args.ocr_json)
    if not ocr_json_path.exists():
        # also allow repo-relative resolution
        alt = REPO_ROOT / args.ocr_json
        if alt.exists():
            ocr_json_path = alt
        else:
            raise FileNotFoundError(f"OCR JSON not found: {args.ocr_json}")

    out_json = None
    if args.out_json:
        out_json = Path(args.out_json)
        # if user provided a filename only (no dirs) write to OUT_DIR
        if not out_json.parent.exists():
            out_json = OUT_DIR / out_json.name

    annotated_path = pipeline.predict_json(ocr_json_path, out_path=out_json, overwrite=not args.no_overwrite)

    if args.csv:
        out_csv = Path(args.out_csv) if args.out_csv else None
        if out_csv and not out_csv.parent.exists():
            out_csv = OUT_DIR / out_csv.name
        csv_path = pipeline.export_csv(annotated_path, out_csv_path=out_csv)
        print(f"Wrote CSV: {csv_path}")

    if args.summary:
        out_summary = Path(args.out_summary) if args.out_summary else None
        if out_summary and not out_summary.parent.exists():
            out_summary = OUT_DIR / out_summary.name
        summary = pipeline.summarize(annotated_path, out_summary_path=out_summary)
        if out_summary:
            print(f"Wrote summary: {out_summary}")
        else:
            dom = summary.get("dominant_emotion")
            cnt = summary.get("segments_count", 0)
            print(f"Summary: dominant={dom} (segments={cnt})")

    print(f"Wrote annotated JSON: {annotated_path}")

if __name__ == "__main__":
    _cli()
