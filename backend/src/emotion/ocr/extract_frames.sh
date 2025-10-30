#!/usr/bin/env bash
# Usage: ./extract_frames.sh path/to/video.mp4 0.5
VIDEO="$1"
INTERVAL="${2:-0.5}"   # seconds between samples
OUTDIR="${3:-ocr/frames/$(basename "${VIDEO%.*}")}"
mkdir -p "$OUTDIR"
# fps = 1/interval
FPS=$(python3 - <<PY
i = $INTERVAL
print(1.0 / i)
PY
)
ffmpeg -hide_banner -loglevel error -i "$VIDEO" -vf "scale=-2:720,fps=${FPS}" "$OUTDIR/frame_%05d.jpg"
echo "Frames written to $OUTDIR (interval=${INTERVAL}s)"
