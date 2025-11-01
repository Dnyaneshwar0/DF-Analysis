#!/usr/bin/env python3
"""
GoEmotions inference — frontend-ready JSON outputs (silent version)

Generates:
 - <video>_goemotions.json
 - <video>_goemotions_summary.json
 - <video>_linegraph.json
 - <video>_tabledata.json
"""

import joblib, json
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")



# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "data" / "emotion" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# GoEmotions Pipeline
# ------------------------------------------------------------
class GoEmotionsPipeline:
    def __init__(self, base_dir: Optional[Path] = None):
        BASE_DIR = Path(base_dir) if base_dir else (
            REPO_ROOT / "models" / "emotion" / "goemotions_model"
        )
        self.model_path = BASE_DIR / "goemotions_clf.joblib"
        self.tfidf_path = BASE_DIR / "goemotions_tfidf.joblib"
        self.labels_path = BASE_DIR / "labels.txt"

        self.vectorizer = joblib.load(self.tfidf_path)
        self.model = joblib.load(self.model_path)
        self.labels = [x.strip() for x in open(self.labels_path, encoding="utf8") if x.strip()]

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict emotions for a single text segment."""
        if not text.strip():
            return {"text": text, "labels": [], "probs": {}, "conf": 0.0}

        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        probs_dict = {label: float(p) for label, p in zip(self.labels, probs)}
        top3 = sorted(probs_dict.items(), key=lambda x: -x[1])[:3]
        labels, vals = zip(*top3)
        conf = max(probs)

        return {
            "text": text,
            "labels": list(labels),
            "probs": {l: probs_dict[l] for l in labels},
            "conf": conf,
        }

    def predict_json(self, ocr_json_path: Path) -> Path:
        """Annotate OCR JSON with top-3 emotions and confidence."""
        data = json.load(open(ocr_json_path))
        segs = data.get("segments", [])
        preds = [self.predict(s.get("text", "")) for s in segs]

        for s, p in zip(segs, preds):
            s["emotions"] = p["labels"]
            s["probs"] = p["probs"]
            s["conf"] = p["conf"]

        out_path = OUT_DIR / f"{data['video']}_goemotions.json"
        json.dump({"video": data["video"], "segments": segs}, open(out_path, "w"), indent=2)
        return out_path

    def summarize(self, annotated_json: Path) -> Dict[str, Any]:
        """Compute average emotion distribution."""
        data = json.load(open(annotated_json))
        segs = data.get("segments", [])
        if not segs:
            summary = {
                "video": data["video"],
                "dominant_emotion": None,
                "emotion_distribution": {},
            }
            json.dump(summary, open(OUT_DIR / f"{data['video']}_goemotions_summary.json", "w"), indent=2)
            return summary

        accum, total = {}, 0
        for s in segs:
            for lbl, val in s["probs"].items():
                accum[lbl] = accum.get(lbl, 0) + val
            total += 1

        avg = {lbl: v / total for lbl, v in accum.items()}
        dom = max(avg.items(), key=lambda x: x[1])[0]
        summary = {"video": data["video"], "dominant_emotion": dom, "emotion_distribution": avg}
        json.dump(summary, open(OUT_DIR / f"{data['video']}_goemotions_summary.json", "w"), indent=2)
        return summary

    def make_linegraph_json(self, annotated_json: Path, summary: Dict[str, Any]) -> Path:
        """Generate JSON for frontend line graph rendering."""
        data = json.load(open(annotated_json))
        segs = data.get("segments", [])
        top3 = [lbl for lbl, _ in sorted(summary["emotion_distribution"].items(), key=lambda x: -x[1])[:3]]

        line_data = []
        for s in segs:
            entry = {"time": s.get("start", 0.0)}
            for lbl in top3:
                entry[lbl] = round(s.get("probs", {}).get(lbl, 0.0), 4)
            line_data.append(entry)

        out_path = OUT_DIR / f"{data['video']}_linegraph.json"
        json.dump({"video": data["video"], "top_emotions": top3, "data": line_data}, open(out_path, "w"), indent=2)
        return out_path

    def make_tabledata_json(self, annotated_json: Path) -> Path:
        """Generate JSON for frontend emotion table display."""
        data = json.load(open(annotated_json))
        segs = data.get("segments", [])

        table_rows = []
        for s in segs:
            table_rows.append({
                "start": s.get("start", 0.0),
                "end": s.get("end", 0.0),
                "text": s.get("text", ""),
                "emotions": s.get("emotions", []),
                "intensities": [round(v, 4) for v in s.get("probs", {}).values()],
                "confidence": round(s.get("conf", 0.0), 4),
            })

        out_path = OUT_DIR / f"{data['video']}_tabledata.json"
        json.dump({"video": data["video"], "segments": table_rows}, open(out_path, "w"), indent=2)
        return out_path


# ------------------------------------------------------------
# Main callable for backend
# ------------------------------------------------------------
def run_inference(ocr_json_path: Path, base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Run full inference pipeline silently, return output file paths."""
    pipeline = GoEmotionsPipeline(base_dir=base_dir)
    annotated = pipeline.predict_json(ocr_json_path)
    summary = pipeline.summarize(annotated)
    line_path = pipeline.make_linegraph_json(annotated, summary)
    table_path = pipeline.make_tabledata_json(annotated)

    return {
        "annotated_json": annotated,
        "summary_json": OUT_DIR / f"{summary['video']}_goemotions_summary.json",
        "linegraph_json": line_path,
        "tabledata_json": table_path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GoEmotions inference (silent)")
    parser.add_argument("--ocr-json", required=True, help="Path to OCR captions JSON")
    args = parser.parse_args()
    run_inference(Path(args.ocr_json))
