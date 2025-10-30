#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Extract frames from a video at a fixed interval (for OCR).
# Works inside backend/ for DF-Analysis project.
#
# Usage:
#   bash src/inference/emotion/ocr/extract_frames.sh data/emotion/input/eddiecaptions.mp4 0.5
#
# Output:
#   backend/src/inference/emotion/ocr/frames/<video_basename>/frame_00001.jpg
# -----------------------------------------------------------------------------

set -e

VIDEO="$1"
INTERVAL="${2:-0.5}"   # seconds between frames
BACKEND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
FRAME_DIR="$BACKEND_ROOT/src/inference/emotion/ocr/frames"
OUTDIR="$FRAME_DIR/$(basename "${VIDEO%.*}")"

mkdir -p "$OUTDIR"

# Compute fps = 1/interval
FPS=$(python3 - <<PY
i = float("$INTERVAL")
print(1.0 / i)
PY
)

# The video path should be relative to backend/, not repo root
VIDEO_PATH="$BACKEND_ROOT/$VIDEO"

if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video not found at $VIDEO_PATH"
    exit 1
fi

# Extract frames using ffmpeg
ffmpeg -hide_banner -loglevel error -i "$VIDEO_PATH" -vf "scale=-2:720,fps=${FPS}" "$OUTDIR/frame_%05d.jpg"

echo "Frames written to: $OUTDIR (interval=${INTERVAL}s)"
