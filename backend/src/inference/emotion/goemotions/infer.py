# backend/src/inference/emotion/goemotions/infer.py
"""
GoEmotions inference module — supports both CLI and importable class usage.

- CLI: python3 -m src.inference.emotion.goemotions.infer
- Class: from src.inference.emotion.goemotions.infer import GoEmotionsPipeline
This version uses a hardcoded model directory path. If you move the repo, update BASE_DIR.
"""

import joblib
import numpy as np
from pathlib import Path
import json

class GoEmotionsPipeline:
    def __init__(self):
        # HARD-CODED path to the goemotions model directory (update if you move the repo)
        BASE_DIR = Path("/home/ampm/projects/DF-Analysis/backend/models/emotion/goemotions_model")

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

    def predict(self, text: str):
        """Predict emotions for a single text."""
        if not text or not text.strip():
            return {"text": text, "labels": [], "probs": {}}

        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]

        probs_dict = {label: float(p) for label, p in zip(self.labels, probs)}
        top_labels = [label for label, p in sorted(probs_dict.items(), key=lambda x: -x[1])[:4]]
        return {"text": text, "labels": top_labels, "probs": probs_dict}

    def predict_batch(self, texts):
        """Predict emotions for multiple texts."""
        return [self.predict(t) for t in texts]

# CLI mode (for quick tests)
if __name__ == "__main__":
    example_text = "I have faith in you"
    pipe = GoEmotionsPipeline()
    out = pipe.predict(example_text)
    print(json.dumps(out, indent=2, ensure_ascii=False))
