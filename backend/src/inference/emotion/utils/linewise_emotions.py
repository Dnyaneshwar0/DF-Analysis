#!/usr/bin/env python3
"""
scripts/linewise_emotions.py
Produce line-wise GoEmotions predictions (CSV + JSON) from a cleaned segments JSON.

Usage:
  # from project root, in the venv that has the joblib/scikit-learn stack (.venv_goe)
  python scripts/linewise_emotions.py \
    --segments outputs/eddiecaptions_segments_clean.json \
    --out-json outputs/eddiecaptions_linewise_emotions.json \
    --out-csv  outputs/eddiecaptions_linewise_emotions.csv
"""
import argparse, json, csv, math
from pathlib import Path
import joblib
import numpy as np

def load_models(base="models/goemotions_model"):
    vec = joblib.load(f"{base}/goemotions_tfidf.joblib")
    clf = joblib.load(f"{base}/goemotions_clf.joblib")
    try:
        labels = [str(x) for x in clf.classes_]
    except Exception:
        labels = None
    return vec, clf, labels

def predict_probabilities(vec, clf, texts):
    X = vec.transform(texts)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
    elif hasattr(clf, "decision_function"):
        from scipy.special import softmax
        df = clf.decision_function(X)
        probs = softmax(df, axis=1)
    else:
        preds = clf.predict(X)
        probs = []
        for p in preds:
            v = [1.0 if p==c else 0.0 for c in clf.classes_]
            probs.append(v)
        probs = np.array(probs)
    return np.array(probs)

def topk_from_probvec(probvec, labels, k=3):
    idx = np.argsort(probvec)[::-1][:k]
    out = []
    for i in idx:
        lab = labels[i] if labels is not None else str(i)
        out.append((lab, float(probvec[i])))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--segments", required=True, help="Cleaned segments JSON")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    segf = Path(args.segments)
    if not segf.exists():
        raise SystemExit(f"Segments file not found: {segf}")

    data = json.load(open(segf, encoding="utf8"))
    segments = data.get("segments", [])
    if not segments:
        raise SystemExit("No segments present in input JSON.")

    vec, clf, labels = load_models()
    texts = [s.get("text","").strip() for s in segments]
    probs = predict_probabilities(vec, clf, texts)  # shape (n, n_labels)

    # build outputs
    out_json = {"source": data.get("source","ocr_captions"), "segments": []}
    csv_rows = []
    header = ["start","end","duration","dominant","top1","top1_prob","top2","top2_prob","top3","top3_prob","text"]

    for i, s in enumerate(segments):
        probvec = probs[i]
        # map probabilities to label names (keys as str)
        if labels:
            prob_dict = {labels[j]: float(probvec[j]) for j in range(len(labels))}
        else:
            prob_dict = {str(j): float(probvec[j]) for j in range(len(probvec))}
        # dominant
        dom_idx = int(np.argmax(probvec))
        dominant = labels[dom_idx] if labels else str(dom_idx)
        # topk
        topk = topk_from_probvec(probvec, labels or [str(j) for j in range(len(probvec))], k=args.topk)
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start))
        duration = end - start if end >= start else 0.0
        text = s.get("text","").strip()
        # JSON entry
        out_seg = dict(s)
        out_seg["emotions"] = prob_dict
        out_seg["dominant"] = dominant
        out_json["segments"].append(out_seg)
        # CSV row
        row = {
            "start": start,
            "end": end,
            "duration": round(duration, 3),
            "dominant": dominant,
            "top1": topk[0][0] if len(topk)>0 else "",
            "top1_prob": round(topk[0][1], 4) if len(topk)>0 else 0.0,
            "top2": topk[1][0] if len(topk)>1 else "",
            "top2_prob": round(topk[1][1], 4) if len(topk)>1 else 0.0,
            "top3": topk[2][0] if len(topk)>2 else "",
            "top3_prob": round(topk[2][1], 4) if len(topk)>2 else 0.0,
            "text": text
        }
        csv_rows.append(row)

    # write JSON (numpy-safe)
    def _safe(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        return str(o)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf8") as fh:
        json.dump(out_json, fh, indent=2, ensure_ascii=False, default=_safe)

    # write CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", encoding="utf8", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    print("Wrote:", args.out_json, "and", args.out_csv)
    print("Rows:", len(csv_rows))

if __name__ == "__main__":
    main()
