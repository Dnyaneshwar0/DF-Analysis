OCR Caption Extraction (burned-in)
-------------------------------
1) Install system deps:
   sudo apt update
   sudo apt install -y ffmpeg tesseract-ocr libtesseract-dev

2) Python deps (add to requirements.txt):
   pytesseract
   easyocr   # optional
   python-Levenshtein  # optional

3) Quick run:
   ./ocr/extract_frames.sh data/your_video.mp4 0.5
   python3 ocr/ocr_batch.py ocr/frames/your_video outputs/your_video_raw_captions.json --interval 0.5
   python3 ocr/postprocess_captions.py outputs/your_video_raw_captions.json outputs/your_video_segments.json
   python3 ocr/captions_to_emotions.py outputs/your_video_segments.json outputs/your_video_captions_emotions.json
