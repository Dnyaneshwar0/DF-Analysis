#!/usr/bin/env python3
"""
captions_to_emotions.py
Load segments JSON, run GoEmotions pipeline (joblib TF-IDF + classifier),
and append emotion probabilities to each segment.
"""
import argparse, json
from pathlib import Path
import joblib, numpy as np


def load_goemotions_models(base="models/goemotions_model"):
    vec = joblib.load(f"{base}/goemotions_tfidf.joblib")
    clf = joblib.load(f"{base}/goemotions_clf.joblib")
    try:
        # Cast all labels to strings to avoid JSON key type issues
        labels = [str(x) for x in list(clf.classes_)]
    except Exception:
        labels = None
    return vec, clf, labels


def predict_probs(clf, vec, texts):
    X = vec.transform(texts)
    # try predict_proba, if not available use decision_function or predict
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
        # shape (n_samples, n_classes)
        return probs
    elif hasattr(clf, "decision_function"):
        df = clf.decision_function(X)
        # attempt to convert to probabilities via softmax
        from scipy.special import softmax
        return softmax(df, axis=1)
    else:
        preds = clf.predict(X)
        # map to sparse one-hot
        out = []
        for p in preds:
            v = [1.0 if p == c else 0.0 for c in clf.classes_]
            out.append(v)
        return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("segments_json")
    p.add_argument("out_json")
    args = p.parse_args()

    segs = json.load(open(args.segments_json, encoding="utf8")).get("segments", [])
    if not segs:
        print("No segments found.")
        return

    vec, clf, labels = load_goemotions_models()
    texts = [s["text"] for s in segs]
    probs = predict_probs(clf, vec, texts)
    all_segments = []

    for i, s in enumerate(segs):
        prob_vec = probs[i].tolist()
        # Map probabilities to label names
        if labels:
            em = {str(labels[j]): float(prob_vec[j]) for j in range(len(labels))}
            dominant = str(labels[int(np.argmax(prob_vec))])
        else:
            em = {f"class_{j}": float(prob_vec[j]) for j in range(len(prob_vec))}
            dominant = max(em, key=em.get)

        seg_out = dict(s)
        seg_out["emotions"] = em
        seg_out["dominant"] = dominant
        all_segments.append(seg_out)

    out = {"source": "ocr_captions", "segments": all_segments}

    # --- JSON dump with numpy-safe serialization ---
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    def safe_json(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        return str(o)

    with open(args.out_json, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=safe_json)

    print("Wrote", args.out_json)


if __name__ == "__main__":
    main()
