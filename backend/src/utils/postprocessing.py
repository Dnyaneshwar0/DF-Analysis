# backend/src/utils/postprocessing.py
import numpy as np
from typing import List, Dict, Tuple
import math

def bbox_iou(boxA: List[int], boxB: List[int]) -> float:
    """
    Compute IoU (intersection over union) for two bboxes in [x,y,w,h] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    return float(interArea) / float(unionArea)

def cluster_detections_into_tracks(dets: List[Dict], iou_threshold: float = 0.4, max_frame_gap: int = 4) -> List[List[Dict]]:
    """
    Greedy online clustering of detections into tracks using IoU and frame continuity.
    - dets: list of dicts each with keys: 'frame_index', 'bbox' (x,y,w,h), 'score'
    - returns list of tracks, each track is a list of detection dicts
    """
    # sort detections by frame index (stable order)
    dets_sorted = sorted(dets, key=lambda d: d["frame_index"])
    tracks = []  # list of lists

    for d in dets_sorted:
        placed = False
        for track in tracks:
            last = track[-1]
            # enforce temporal continuity and IoU overlap
            if (d["frame_index"] - last["frame_index"]) <= max_frame_gap:
                if bbox_iou(d["bbox"], last["bbox"]) >= iou_threshold:
                    track.append(d)
                    placed = True
                    break
        if not placed:
            tracks.append([d])
    return tracks

def summarize_track(track: List[Dict]) -> Dict:
    """Return summary statistics for a track (list of detections)."""
    scores = np.asarray([float(t["score"]) for t in track], dtype=float)
    frames = [int(t["frame_index"]) for t in track]
    bboxes = [t["bbox"] for t in track]
    return {
        "n": int(len(track)),
        "frame_start": int(min(frames)),
        "frame_end": int(max(frames)),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "scores": [float(s) for s in scores.tolist()],
        "bboxes": bboxes,
    }

def combine_detections(
    detections: List[Dict],
    iou_threshold: float = 0.4,
    max_frame_gap: int = 4,
    decision_threshold: float = 0.5,
    top_k_for_decision: int = 5
) -> Dict:
    """
    Combine a list of per-crop detections into a single video-level result.

    Input: detections: list of dicts with at least {"frame_index":int, "bbox":[x,y,w,h], "score":float}
    Returns: dictionary with:
       - label: "fake" or "real"
       - confidence: float in [0,1] (higher => more certain)
       - metrics: aggregated numeric stats
       - tracks: per-track summaries
       - per_detection: original list (optional)
    Strategy (default):
      1. Cluster detections into tracks (IoU + temporal continuity).
      2. Compute per-track statistics (mean/median/max).
      3. Use a decision rule that prefers strong evidence (high max) but also considers per-track means and proportion of detections above threshold.
      4. Confidence is composed from max_score and fraction_above_threshold for robustness.
    """
    out = {
        "label": None,
        "confidence": None,
        "metrics": {},
        "tracks": [],
        "per_detection": detections
    }

    if not detections:
        out["label"] = "real"
        out["confidence"] = 0.0
        out["metrics"] = {
            "n_detections": 0,
            "n_tracks": 0,
            "mean_score": None,
            "median_score": None,
            "max_score": None,
            "min_score": None,
            "prop_above_threshold": None
        }
        return out

    tracks = cluster_detections_into_tracks(detections, iou_threshold=iou_threshold, max_frame_gap=max_frame_gap)
    track_summaries = [summarize_track(t) for t in tracks]

    # Flatten scores
    all_scores = np.asarray([float(d["score"]) for d in detections], dtype=float)
    n_dets = int(all_scores.size)
    n_tracks = len(tracks)
    mean_score = float(all_scores.mean())
    median_score = float(np.median(all_scores))
    max_score = float(all_scores.max())
    min_score = float(all_scores.min())
    prop_above = float((all_scores >= decision_threshold).sum()) / float(n_dets)

    # Track-level: find top tracks by max score and by mean score
    track_maxes = [t["max_score"] for t in track_summaries]
    track_means = [t["mean_score"] for t in track_summaries]

    # Compute decision: heuristic ensemble (tunable)
    # Rule precedence:
    #  - If any detection has very high confidence (>= 0.95) -> FAKE
    #  - Else if a track has max >= 0.9 -> FAKE
    #  - Else if proportion of detections above threshold is sufficiently large -> FAKE
    #  - Else if global mean >= threshold -> FAKE
    #  - Otherwise REAL
    label = "real"
    if max_score >= 0.95:
        label = "fake"
    elif len(track_maxes) and max(track_maxes) >= 0.90:
        label = "fake"
    elif prop_above >= 0.4 and (max_score >= 0.6 or mean_score >= 0.55):
        # moderate evidence across many crops
        label = "fake"
    elif mean_score >= decision_threshold and prop_above >= 0.2:
        label = "fake"
    else:
        label = "real"

    # Compute a confidence score in [0,1]
    # Combine: strong contribution from max_score, scaled contribution from prop_above and mean_score
    # Normalize: max_score in [0,1], mean_score in [0,1], prop_above in [0,1].
    conf = 0.0
    conf += 0.6 * max_score
    conf += 0.25 * mean_score
    conf += 0.15 * prop_above
    # If label == real, invert confidence to represent 'certainty of real' (?) we want confidence always for chosen label
    # So clamp to [0,1]
    conf = float(max(0.0, min(1.0, conf)))

    # Provide also top-k detections/tracks for inspection
    # Top-k detections by score
    sorted_by_score = sorted(detections, key=lambda d: float(d["score"]), reverse=True)
    top_k_dets = sorted_by_score[:top_k_for_decision]

    out["label"] = label
    out["confidence"] = conf
    out["metrics"] = {
        "n_detections": n_dets,
        "n_tracks": n_tracks,
        "mean_score": mean_score,
        "median_score": median_score,
        "max_score": max_score,
        "min_score": min_score,
        "prop_above_threshold": prop_above
    }
    out["tracks"] = track_summaries
    out["top_detections"] = top_k_dets
    return out
