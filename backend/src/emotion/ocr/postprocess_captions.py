#!/usr/bin/env python3
"""
postprocess_captions.py
Takes raw frame-level OCR JSON and groups consecutive identical (or near-identical)
texts into caption segments with start/end timestamps.
"""
import json, argparse, difflib
from pathlib import Path

def normalize(s):
    return " ".join(s.strip().split()).lower()

def similar(a,b):
    if not a or not b: return False
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.85

def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_json")
    p.add_argument("out_json")
    p.add_argument("--min_len", type=int, default=3)
    p.add_argument("--merge_gap", type=float, default=0.25)
    args = p.parse_args()
    data = json.load(open(args.raw_json, encoding="utf8"))
    frames = data.get("frames", [])
    segments=[]
    cur=None
    for fr in frames:
        txt = normalize(fr.get("text",""))
        ts = fr.get("timestamp",0.0)
        if len(txt) < args.min_len:
            # treat as empty
            txt=""
        if cur is None:
            if txt:
                cur={"text":txt,"start":ts,"end":ts,"conf":fr.get("conf") or 0.0}
        else:
            if txt==cur["text"] or similar(txt, cur["text"]):
                cur["end"]=ts
                # update conf as average
                cur["conf"] = (cur.get("conf",0.0)+ (fr.get("conf") or 0.0))/2.0
            else:
                # finalize
                segments.append(cur)
                if txt:
                    cur={"text":txt,"start":ts,"end":ts,"conf":fr.get("conf") or 0.0}
                else:
                    cur=None
    if cur:
        segments.append(cur)
    # Merge very short segments into neighbors if needed
    merged=[]
    for seg in segments:
        if merged and seg["start"] - merged[-1]["end"] <= args.merge_gap and seg["text"]==merged[-1]["text"]:
            merged[-1]["end"]=seg["end"]
            merged[-1]["conf"] = (merged[-1]["conf"] + seg["conf"])/2.0
        else:
            merged.append(seg)
    out = {"video": Path(args.raw_json).stem, "segments": merged}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json,"w",encoding="utf8"), indent=2, ensure_ascii=False)
    print("Wrote", args.out_json, "segments:", len(merged))

if __name__ == "__main__":
    main()
