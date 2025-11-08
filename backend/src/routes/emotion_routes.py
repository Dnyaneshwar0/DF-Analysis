import os, tempfile, uuid, logging, hashlib, time, shutil, json, sys
from pathlib import Path
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

# import orchestrator entry (unchanged)
try:
    from src.inference.emotion.df_emo import main as run_emotion
except Exception:
    run_emotion = None

emotion_bp = Blueprint("emotion", __name__)
logger = logging.getLogger(__name__)

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}

# repo paths
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "emotion" / "output"
RAW_DIR = REPO_ROOT / "data" / "emotion" / "input" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return Path(filename.lower()).suffix in ALLOWED_EXT

# ---- idempotency helpers ----
def _stable_stem(file_path: Path) -> str:
    h = hashlib.sha1()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    base = Path(file_path).stem
    return f"emo_{h.hexdigest()[:32]}_{base}"

def _existing_result(stem: str):
    merged = OUT_DIR / f"{stem}_multimodel_summary.json"
    webmp4 = OUT_DIR / f"out_{stem}_web.mp4"
    if merged.exists() and webmp4.exists():
        try:
            return json.loads(merged.read_text())
        except Exception:
            return None
    return None

def _acquire_lock(stem: str, timeout=60):
    lock = OUT_DIR / f"{stem}.lock"
    start = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return fd, lock
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Another run holds lock for {stem}")
            time.sleep(0.25)

def _release_lock(fd, lock_path: Path):
    try:
        os.close(fd)
    finally:
        try:
            lock_path.unlink()
        except Exception:
            pass

@emotion_bp.route("/analyze", methods=["POST"])
def analyze_route():
    if run_emotion is None:
        return jsonify({"error": "Emotion inference module not available (import failed)"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # save to tmp first
    filename = secure_filename(file.filename)
    tmp_path = Path(tempfile.gettempdir()) / f"emo_{uuid.uuid4().hex}_{filename}"
    file.save(str(tmp_path))

    try:
        # compute deterministic stem
        stem = _stable_stem(tmp_path)

        # reuse if already computed
        cached = _existing_result(stem)
        if cached:
            return jsonify({"status": "success", "result": cached}), 200

        # move/rename to RAW_DIR with stable name so orchestrator uses it as stem
        raw_path = RAW_DIR / f"{stem}{Path(filename).suffix.lower()}"
        if not raw_path.exists():
            # atomic-ish move into repo raw dir
            shutil.move(str(tmp_path), str(raw_path))
        else:
            # raw already exists; discard tmp
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        # single-writer lock
        fd, lockp = _acquire_lock(stem)
        try:
            # re-check cache after lock (another worker may have finished)
            cached = _existing_result(stem)
            if cached:
                return jsonify({"status": "success", "result": cached}), 200

            # run orchestrator with the stable-named file
            result = run_emotion([str(raw_path)])

            # orchestrator returns merged dict; if it returned an error, map it
            if isinstance(result, dict) and result.get("error"):
                return jsonify({"status": "error", "result": result}), 400

            # prefer reading the merged json we expect (authoritative)
            merged_path = OUT_DIR / f"{stem}_multimodel_summary.json"
            if merged_path.exists():
                result = json.loads(merged_path.read_text())

            return jsonify({"status": "success", "result": result}), 200
        finally:
            _release_lock(fd, lockp)

    except TimeoutError as e:
        logger.warning(str(e))
        return jsonify({"error": str(e)}), 429
    except Exception as e:
        logger.exception("Emotion analysis failed")
        return jsonify({"error": str(e)}), 500
    finally:
        # best-effort cleanup of tmp file if it still exists
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove temp file")
