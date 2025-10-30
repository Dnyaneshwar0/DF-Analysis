#!/usr/bin/env python3
"""
ocr_batch.py
Walks a frames directory, runs ocr_frame.py on each image (in order),
and writes raw JSON detections to outputs/<video>_raw_captions.json

Usage:
  python ocr/ocr_batch.py ocr/frames/<video_dir> outputs/<video>_raw_captions.json --interval 0.5
"""
import os, sys, json, subprocess, argparse
from pathlib import Path
def run_ocr_on_frame(frame, crop_ratio):
    cmd = [sys.executable, "ocr/ocr_frame.py", frame, str(crop_ratio)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"image": frame, "text":"", "conf":0.0}
    try:
        return json.loads(proc.stdout.strip())
    except Exception:
        return {"image": frame, "text":proc.stdout.strip(), "conf":0.0}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames_dir")
    p.add_argument("out_json")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between frames (used to compute timestamps)")
    p.add_argument("--crop", type=float, default=0.25, help="bottom crop fraction")
    args = p.parse_args()
    frames = sorted([str(x) for x in Path(args.frames_dir).glob("*.jpg")])
    results=[]
    for idx,f in enumerate(frames):
        ts = round(idx * args.interval, 3)
        res = run_ocr_on_frame(f, args.crop)
        res["timestamp"] = ts
        results.append(res)
        print(f"[{idx}] {ts}s -> {res.get('text')[:80]}")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json,"w",encoding="utf8") as fh:
        json.dump({"frames": results, "interval": args.interval, "source_dir": args.frames_dir}, fh, indent=2, ensure_ascii=False)
    print("Wrote", args.out_json)

if __name__ == "__main__":
    main()
