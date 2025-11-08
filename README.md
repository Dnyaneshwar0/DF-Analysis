# AffectForensics

AffectForensics is an multimodal video forensics system that integrates:
- Deepfake detection
- Manipulation reverse-engineering
- Multimodal emotional signal interpretation (face, voice, and text)

The system outputs a unified forensic report with supporting evidence rather than a single binary label.  
This enables diagnostic interpretation instead of opaque classification.

---

## Web Interface

| Component | Framework       | Purpose                              |
|-----------|-----------------|--------------------------------------|
| Backend   | Flask (Python)  | Model inference + HTTP API           |
| Frontend  | React + TailwindCSS | Visualization and report analysis UI |

Local runtime:
- Backend → `http://localhost:5000`
- Frontend → `http://localhost:3000`

---

## Installation

**WSL 2 is strongly recommended** for Windows users.

### WSL 2 Setup (Recommended)

```powershell
# Run in PowerShell (Admin)
wsl --install -d Ubuntu
```

Then in **Ubuntu terminal**:

```bash
sudo apt update && sudo apt install -y \
    ffmpeg tesseract-ocr tesseract-ocr-eng \
    libsndfile1 libgl1 libglib2.0-0 \
    libsm6 libxrender1 libxext6
```

---

### Windows Native (Not Recommended)

You'll need to manually install:

- [FFmpeg](https://ffmpeg.org/download.html#build-windows)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

> Add `ffmpeg` and `tesseract` to your `PATH`.

---

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/app.py
```

### Frontend

```bash
cd frontend
npm install
npm start
```

---

## Repository Structure (Overview)

```text
.
├── backend/
│   ├── src/
│   │   ├── app.py                 # Flask backend entry point
│   │   ├── routes/                # REST API endpoints
│   │   └── inference/
│   │       ├── detection/         # Deepfake classifier
│   │       ├── reverseEng/        # Manipulation pattern analysis
│   │       └── emotion/           # Multimodal emotion (video/audio/text)
│   ├── models/                    # Pre-trained model weights
│   └── data/                      # Local cache & annotated output storage
│
└── frontend/
    └── src/
        ├── pages/                 # Workflow: Upload → Analysis → Results
        ├── components/            # Reusable UI + visualization blocks
        └── App.js                 # React app root
```
---

## Methodology Summary

| Subsystem           | Dataset / Basis                        | Operational Output                        |
|---------------------|----------------------------------------|-------------------------------------------|
| Deepfake Detection  | Convolutional face-sequence classifier | Authenticity + confidence score           |
| Reverse Engineering | Manipulation signature comparison      | Predicted manipulation origin/type        |
| Emotion (Video)     | RAF-DB                                 | Temporal facial affect timeline           |
| Emotion (Audio)     | RAVDESS / CREMA-D                      | Acoustic emotion probability distribution |
| Emotion (Text)      | GoEmotions                             | Semantic affect inference from language   |



## Contributors

<a href="https://github.com/Dnyaneshwar0">
  <img src="https://avatars.githubusercontent.com/u/97013118?s=48&v=4" width="60" style="border-radius:50%;" alt="Parth Gujarkar" />
</a>
<a href="https://github.com/blast678">
  <img src="https://avatars.githubusercontent.com/u/98542059?v=4" width="60" style="border-radius:50%;" alt="Mithilesh Deshmukh" />
</a>
<a href="https://github.com/ampm14">
  <img src="https://avatars.githubusercontent.com/u/162469695?v=4" width="60" style="border-radius:50%;" alt="Aishwarya Mhatre" />
</a>

**Parth Gujarkar** – [@Dnyaneshwar0](https://github.com/Dnyaneshwar0)  
**Mithilesh Deshmukh** – [@blast678](https://github.com/blast678)  
**Aishwarya Mhatre** – [@ampm14](https://github.com/ampm14)


```