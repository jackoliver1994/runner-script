"""
Coqui TTS runner — subprocess-based chunk workers with INFINITE retries until
each chunk produces a .status.json state="ok" and a non-empty WAV.
Improved: at startup check `ffmpeg` availability (runs `which ffmpeg`), and if missing
attempts to install via apt-get in Colab. All original features preserved.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import glob
import re
import time
import sys
import subprocess
import traceback
from datetime import datetime
import numpy as np
import tempfile
import wave
import shutil
from typing import List, Tuple, Dict, Any
import random
import json

current_directory = os.getcwd()

# ---------- FFmpeg paths (user-provided) ----------
ffmpeg_root_path = os.path.join(current_directory, "ffmpeg")
if sys.platform.startswith("win"):
    ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg.exe")
    ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe.exe")
else:
    ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg")
    ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe")

# ---------- USER CONFIG ----------
folder_path = os.path.join(current_directory, "narration_response")
# max_retries kept for compatibility but not used to stop retries
max_retries = 100
retry_delay = 1  # seconds between retries / loops

# Desired total narration minutes (None to disable scaling)
TARGET_MINUTES = 10  # set to None to use base speed without scaling

# Concurrency (child-process workers for chunk-level management). Keep conservative on low-RAM systems.
MAX_WORKERS = 2

# Preferred TTS models (jenny prioritized)
preferred_models = [
    "tts_models/en/jenny/jenny",
    # add other models as fallbacks if desired
]

# Storytelling style adjustments
storytelling_settings = {
    "speed": float(np.round(np.random.uniform(0.40, 0.50), 2)),
    "pause_short": ",",
    "pause_medium": "...",
    "pause_long": " -- ",
    "intonation": "warm",
    "speaker": None,
    "description": (
        "Slow, emotional pacing with soft breathing pauses and warm tone. "
        "Ideal for cinematic storytelling, narration, and emotional monologues."
    ),
    "pause_short_ms": 220,
    "pause_medium_ms": 700,
    "pause_long_ms": 1200,
}

MIN_SPEED = 0.35
MAX_SPEED = 1.20

# keep temporary chunk files for debugging if True (you requested False)
PRESERVE_TMPDIR = False

# per-chunk time limit (parent will kill the subprocess if exceeding this)
CHUNK_WORKER_TIMEOUT = 240  # seconds (increase if models are slow)


# ---------- HELP: ensure ffmpeg available (Colab-friendly) ----------
def ensure_ffmpeg_available_and_install_if_missing():
    """
    Check for ffmpeg on PATH. If missing, try:
      - provided ffmpeg binary (ffmpeg_path)
      - apt-get install (if available)
      - pip install imageio-ffmpeg (preferred pip that includes binary)
      - pip install ffmpeg (per user request; may or may not provide binary)
    Updates global ffmpeg_path if found and returns the path (or None).
    """
    global ffmpeg_path
    # 1) PATH
    found = shutil.which("ffmpeg")
    if found:
        print(f"{_ts()} ✅ ffmpeg found on PATH: {found}")
        ffmpeg_path = found
        return found

    # 2) provided binary (configured path)
    if ffmpeg_path and os.path.exists(ffmpeg_path):
        print(
            f"{_ts()} ℹ️ Found provided ffmpeg binary at configured ffmpeg_path: {ffmpeg_path}"
        )
        return ffmpeg_path

    print(
        f"{_ts()} ⚠️ ffmpeg not found on PATH or at configured ffmpeg_path ({ffmpeg_path})."
    )

    # 3) apt-get (Colab / Debian-like systems)
    apt_get = (
        shutil.which("apt-get")
        or os.path.exists("/usr/bin/apt-get")
        or os.path.exists("/bin/apt-get")
    )
    if apt_get:
        try:
            print(
                f"{_ts()} ℹ️ Attempting to install ffmpeg using apt-get. This may take a moment..."
            )
            subprocess.check_call(
                ["apt-get", "update"], stdout=sys.stdout, stderr=sys.stderr
            )
            subprocess.check_call(
                ["apt-get", "-y", "install", "ffmpeg"],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            found = shutil.which("ffmpeg")
            if found:
                print(f"{_ts()} ✅ ffmpeg installed via apt-get: {found}")
                ffmpeg_path = found
                return found
            else:
                print(
                    f"{_ts()} ❌ apt-get finished but ffmpeg still not found on PATH."
                )
        except Exception:
            tb = traceback.format_exc()
            print(f"{_ts()} ❌ Failed to install ffmpeg via apt-get: {tb}")

    # 4) Try pip installing imageio-ffmpeg (this package ships a binary usable via imageio_ffmpeg.get_ffmpeg_exe())
    try:
        print(
            f"{_ts()} ℹ️ Attempting to install 'imageio-ffmpeg' via pip and use its bundled ffmpeg..."
        )
        _pip_install("imageio-ffmpeg")
        try:
            import imageio_ffmpeg as _iioff  # type: ignore

            exe = _iioff.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                print(f"{_ts()} ✅ Found ffmpeg via imageio-ffmpeg: {exe}")
                ffmpeg_path = exe
                return exe
            # some distributions expose 'ffmpeg' name; check PATH again
            found = shutil.which("ffmpeg")
            if found:
                print(
                    f"{_ts()} ✅ ffmpeg now on PATH after imageio-ffmpeg install: {found}"
                )
                ffmpeg_path = found
                return found
        except Exception as _e:
            print(f"{_ts()} ⚠️ Could not query imageio-ffmpeg binary: {_e}")
    except Exception:
        print(f"{_ts()} ⚠️ pip install imageio-ffmpeg failed or unavailable.")

    # 5) Per user request: try pip install 'ffmpeg' (less reliable; may just be wrapper)
    try:
        print(
            f"{_ts()} ℹ️ Attempting to 'pip install ffmpeg' as a last-resort attempt..."
        )
        _pip_install("ffmpeg")
        found = shutil.which("ffmpeg")
        if found:
            print(f"{_ts()} ✅ ffmpeg found on PATH after pip install: {found}")
            ffmpeg_path = found
            return found
        else:
            print(
                f"{_ts()} ⚠️ pip-installed 'ffmpeg' did not place an ffmpeg binary on PATH."
            )
    except Exception:
        tb = traceback.format_exc()
        print(f"{_ts()} ⚠️ pip install ffmpeg failed: {tb}")

    # Nothing found
    print(
        f"{_ts()} ⚠️ Unable to locate or install ffmpeg automatically. Please install ffmpeg manually."
    )
    return None


# ---------- UTILS ----------
def _pip_install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def ensure_coqui_models_installed(
    models: List[str], auto_install_tts: bool = True
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    model_manager = None
    try:
        from TTS.utils.manage import ModelManager  # type: ignore

        model_manager = ModelManager()
    except Exception:
        if auto_install_tts:
            try:
                print("TTS not found — installing package 'TTS'...")
                _pip_install("TTS")
                from TTS.utils.manage import ModelManager  # type: ignore

                model_manager = ModelManager()
            except Exception as e:
                err = traceback.format_exc()
                for m in models:
                    results[m] = {
                        "status": "error",
                        "path": None,
                        "detail": f"Failed to install/import TTS: {e}\n{err}",
                    }
                return results
        else:
            for m in models:
                results[m] = {
                    "status": "error",
                    "path": None,
                    "detail": "TTS not installed and auto_install_tts is False.",
                }
            return results

    for model_id in models:
        mid = model_id.strip()
        if not mid:
            continue
        try:
            print(f"Checking/downloading model: {mid}")
            model_path = model_manager.download_model(mid)
            results[mid] = {
                "status": "ok",
                "path": model_path,
                "detail": "Downloaded or already present.",
            }
        except Exception as e:
            tb = traceback.format_exc()
            results[mid] = {"status": "error", "path": None, "detail": f"{e}\n{tb}"}
    return results


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _nice_join(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def log_execution_time(
    start_time: float, end_time: float, show_ms: bool = False
) -> None:
    elapsed = max(0.0, end_time - start_time)
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    milliseconds = int((elapsed - int(elapsed)) * 1000)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    if show_ms and milliseconds:
        parts.append(f"{milliseconds} ms")
    human = _nice_join(parts)
    current_time = datetime.now().strftime("%I:%M:%S %p")
    print(
        f"[{start_time} to {end_time} now {current_time}] Execution completed in {human}."
    )


# ---------- COMPAT CHECK ----------
def ensure_numpy_pandas_compat(autofix=True):
    try:
        import numpy as _np
        import pandas as _pd

        print(f"{_ts()} Detected numpy {_np.__version__}, pandas {_pd.__version__}")
        return True
    except Exception as e:
        tb = traceback.format_exc()
        print(f"{_ts()} ⚠️  Import error while checking numpy/pandas: {tb}")
        if "numpy.dtype size changed" in str(e) or "binary incompatibility" in tb:
            recommended_numpy = "1.26.4"
            recommended_pandas = "2.2.2"
            pip_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                f"numpy=={recommended_numpy}",
                f"pandas=={recommended_pandas}",
            ]
            print(
                f"{_ts()} Attempting automatic repair by reinstalling: {' '.join(pip_cmd)}"
            )
            if not autofix:
                print("Run the pip command manually:", " ".join(pip_cmd))
                return False
            try:
                subprocess.check_call(pip_cmd)
                print(
                    f"{_ts()} ✅ Reinstallation completed. Restarting the script to apply changes..."
                )
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception:
                print(
                    f"{_ts()} ❌ Automatic reinstall failed. Please reinstall manually."
                )
                return False
        else:
            return False


ensure_numpy_pandas_compat(autofix=True)


# ---------- TEXT HELPERS ----------
def find_txt_file(folder: str) -> str:
    txt_files = glob.glob(os.path.join(folder, "*.txt"))
    return txt_files[0] if txt_files else None


def read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_bracket_texts(full_text: str) -> List[str]:
    if not full_text:
        return []
    matches = re.findall(r"\[(.*?)\]", full_text, re.DOTALL)
    cleaned = [m.strip() for m in matches if m and m.strip()]
    return cleaned


def normalize_punctuation_for_tts(text: str) -> str:
    if text is None:
        return text
    text = text.replace("—", "--").replace("–", "--").replace("―", "--")
    text = text.replace("…", "...")
    text = re.sub(r"\.{2,}", "...", text)
    text = re.sub(r"[^\S\n]{2,}", " ", text)
    return text.strip()


def sanitize_text_for_tts(text: str) -> str:
    if text is None:
        return text
    replacements = {"“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "―": "--"}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"[^\x00-\x7F]", "", text)
    return text.strip()


def smart_sentence_split(text: str, max_chunk_words: int = 110) -> List[str]:
    if not text:
        return []
    text = normalize_punctuation_for_tts(text)
    sentences = re.findall(r"[^.!?]+[.!?]*", text, re.DOTALL)
    chunks = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if len(words) <= max_chunk_words:
            chunks.append(s)
            continue
        subparts = re.split(r"[,;]|--", s)
        cur = ""
        for sp in subparts:
            sp = sp.strip()
            if not sp:
                continue
            candidate = sp if not cur else cur + " " + sp
            if len(candidate.split()) > max_chunk_words:
                if cur:
                    chunks.append(cur.strip())
                sp_words = sp.split()
                i = 0
                while i < len(sp_words):
                    piece = " ".join(sp_words[i : i + max_chunk_words])
                    chunks.append(piece.strip())
                    i += max_chunk_words
                cur = ""
            else:
                cur = candidate
        if cur:
            chunks.append(cur.strip())
    final = []
    for c in chunks:
        if not re.search(r"[.!?]$", c):
            if len(c.split()) < 6:
                c = c + "..."
            else:
                c = c + "."
        final.append(re.sub(r"\.{2,}", "...", c.strip()))
    return final


# ----------------- SUBTITLE (SRT) HELPERS -----------------
def _wav_duration_seconds(path: str) -> float:
    """
    Return duration in seconds for WAV or similar audio file.
    Tries Python wave first (works for PCM WAVs), and falls back to ffprobe
    (if ffprobe available) for other formats.
    """
    try:
        # try wave (works for standard PCM WAVs)
        import wave as _wave

        with _wave.open(path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate and frames >= 0:
                return float(frames) / float(rate)
    except Exception:
        pass

    # fallback to ffprobe if present
    try:
        ffprobe = (
            ffprobe_path
            if (ffprobe_path and os.path.exists(ffprobe_path))
            else shutil.which("ffprobe")
        )
        if not ffprobe:
            # try to detect from ffmpeg package helper (imageio-ffmpeg) if installed
            try:
                import imageio_ffmpeg as _iioff

                ffprobe = shutil.which("ffprobe") or None
            except Exception:
                ffprobe = None

        if ffprobe:
            cmd = [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            dur = out.decode().strip()
            try:
                return float(dur)
            except Exception:
                pass
    except Exception:
        pass

    # last resort: return 0.0 so caller can decide
    return 0.0


def _format_srt_timestamp(seconds: float) -> str:
    """
    Format seconds -> SRT timestamp 'HH:MM:SS,mmm'
    """
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    s = total % 60
    total //= 60
    m = total % 60
    total //= 60
    h = total
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _sanitize_for_srt(text: str) -> str:
    """
    Lightweight sanitize for subtitle text:
      - collapse multiple spaces, strip
      - remove silent markers if present
      - replace newlines with space
    """
    if not text:
        return ""
    t = text.replace("\r", " ").replace("\n", " ").strip()
    # remove the internal silence markers if any (defensive)
    t = t.replace("__SILENCE__:", "")
    t = re.sub(r"\s{2,}", " ", t)
    return t


def write_srt_from_chunks(
    chunks: List[str],
    tasks: List[Tuple[int, str, str, int]],
    results: Dict[int, Tuple[int, bool, str, Any]],
    base_output_path: str,
) -> str:
    """
    Generate an SRT file aligned with the concatenation order built earlier.

    - chunks: original chunks list (may contain "__SILENCE__:" markers)
    - tasks: the tasks list used earlier (tuple: (i, chunk, chunk_fname, trailing_pause_ms))
    - results: mapping of chunk index -> (idx, success_bool, path, err)
    - base_output_path: intended output path (we create <base> .srt next to it)
    Returns the path to the written .srt
    """
    srt_path = os.path.splitext(base_output_path)[0] + ".srt"
    entries = []
    current_time = 0.0
    for i in range(len(chunks)):
        # Determine path for this chunk (could be silence WAV or real TTS wav)
        wavp = None
        if i in results and results[i][1] and results[i][2]:
            wavp = results[i][2]
        else:
            # fallback to expected tmp chunk path if available in tasks
            try:
                expected = tasks[i][2]
                if os.path.exists(expected):
                    wavp = expected
            except Exception:
                wavp = None

        # If chunk text is silence marker, treat as silence audio (no subtitle)
        chunk_text = chunks[i] if isinstance(chunks[i], str) else str(chunks[i])
        is_silence_chunk = isinstance(chunk_text, str) and chunk_text.startswith(
            "__SILENCE__:"
        )

        # duration of the produced chunk audio
        dur = 0.0
        if wavp and os.path.exists(wavp):
            dur = _wav_duration_seconds(wavp)
            # defensively clamp small durations
            if dur <= 0:
                # if path mentions ms, try to extract
                m = re.search(r"_(\d+)ms\.wav$", wavp)
                if m:
                    try:
                        dur = int(m.group(1)) / 1000.0
                    except Exception:
                        dur = 0.0

        # If this chunk is speech (not silence), add a subtitle entry covering its audio duration
        if not is_silence_chunk:
            clean_text = _sanitize_for_srt(chunk_text)
            if clean_text:
                start = current_time
                end = current_time + dur
                # If duration missing (0), set a tiny minimal display time (0.3s)
                if end <= start + 0.02:
                    end = start + max(0.3, dur or 0.3)
                entries.append((len(entries) + 1, clean_text, start, end))

        # Advance timeline by this chunk duration
        current_time += dur

        # Advance timeline by trailing pause (if any) stored in tasks
        try:
            pause_ms = (
                int(tasks[i][3]) if tasks and tasks[i] and len(tasks[i]) > 3 else 0
            )
            if pause_ms and pause_ms > 0:
                current_time += pause_ms / 1000.0
        except Exception:
            pass

    # Write SRT file
    try:
        with open(srt_path, "w", encoding="utf-8") as s:
            for idx, text, start, end in entries:
                s.write(f"{idx}\n")
                s.write(
                    f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n"
                )
                # wrap text lines at ~42 chars for readability (soft)
                # but avoid inserting markup, keep plain lines
                maxlen = 42
                words = text.split()
                line = ""
                for w in words:
                    if len(line) + 1 + len(w) <= maxlen or line == "":
                        line = (line + " " + w).strip()
                    else:
                        s.write(line + "\n")
                        line = w
                if line:
                    s.write(line + "\n")
                s.write("\n")
        print(f"{_ts()} ✅ Wrote subtitle SRT: {srt_path} ({len(entries)} entries)")
    except Exception as e:
        print(f"{_ts()} ⚠️ Failed to write SRT file: {e}")

    return srt_path


# ---------- GPU CHECK helpers ----------
def is_torch_cuda_available() -> bool:
    try:
        import torch

        has = getattr(torch.cuda, "is_available", lambda: False)()
        if has:
            cnt = getattr(torch.cuda, "device_count", lambda: 0)()
            return bool(cnt)
        return False
    except Exception:
        return False


def child_local_cuda_hint() -> bool:
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            return True
        if shutil.which("nvidia-smi"):
            return True
        if is_torch_cuda_available():
            return True
    except Exception:
        pass
    return False


# ---------- FILE STATUS HELPERS ----------
def _read_json_safe(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_filesize_kb(path: str) -> int:
    try:
        return int(os.path.getsize(path) / 1024)
    except Exception:
        return 0


# ---------- SILENCE WAV GENERATOR ----------
def generate_silence_wav(
    duration_ms: int,
    out_path: str,
    sample_rate: int = 22050,
    nchannels: int = 1,
    sampwidth: int = 2,
) -> str:
    if duration_ms <= 0:
        raise ValueError("duration_ms must be > 0")
    nframes = int(sample_rate * (duration_ms / 1000.0))
    block_frames = 4096
    silent_block = (b"\x00" * sampwidth) * nchannels * block_frames
    with wave.open(out_path, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        frames_written = 0
        while frames_written < nframes:
            to_write = min(block_frames, nframes - frames_written)
            if to_write == block_frames:
                w.writeframes(silent_block)
            else:
                w.writeframes((b"\x00" * sampwidth) * nchannels * to_write)
            frames_written += to_write
    return out_path


# ---------- RANDOMIZED BREATH / PAUSE INJECTION ----------
def _estimate_wpm_from_speed(speed: float, base_wpm: int = 110) -> float:
    return base_wpm * float(speed)


def _select_breath_style(wpm: float):
    if wpm <= 110:
        return (8, 12, "slow")
    elif wpm <= 160:
        return (12, 20, "normal")
    elif wpm <= 200:
        return (20, 30, "energetic")
    else:
        return (30, 40, "fast")


def inject_random_pauses_into_chunks(
    chunks: List[str], speed: float, base_wpm: int = 110, rng: random.Random = None
) -> List[str]:
    if rng is None:
        rng = random.Random()
    new_chunks: List[str] = []
    wpm = _estimate_wpm_from_speed(speed, base_wpm=base_wpm)
    min_w, max_w, style = _select_breath_style(wpm)
    short_base = storytelling_settings.get("pause_short_ms", 220)
    med_base = storytelling_settings.get("pause_medium_ms", 700)
    long_base = storytelling_settings.get("pause_long_ms", 1200)
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith("__SILENCE__:"):
            new_chunks.append(chunk)
            continue
        words = chunk.split()
        if not words:
            continue
        avg_words_between_breaths = rng.randint(min_w, max_w)
        i = 0
        cur_words = []
        words_since_breath = 0
        if style == "slow":
            pause_weights = {"short": 0.55, "medium": 0.35, "long": 0.10}
        elif style == "normal":
            pause_weights = {"short": 0.70, "medium": 0.25, "long": 0.05}
        elif style == "energetic":
            pause_weights = {"short": 0.85, "medium": 0.13, "long": 0.02}
        else:
            pause_weights = {"short": 0.90, "medium": 0.095, "long": 0.005}
        max_pauses = max(1, int(len(words) / max(1, avg_words_between_breaths)))
        pauses_inserted = 0
        if rng.random() < 0.05:
            new_chunks.append(chunk)
            continue
        while i < len(words):
            cur_words.append(words[i])
            words_since_breath += 1
            i += 1
            threshold = max(
                1,
                int(rng.gauss((min_w + max_w) / 2.0, max(1.0, (max_w - min_w) / 4.0))),
            )
            threshold = max(min_w, min(max_w, threshold))
            punct_here = re.search(r"[,\.\?!;:]$", cur_words[-1]) is not None
            insert_prob = 0.9 if punct_here else 0.6
            if (
                words_since_breath >= threshold
                and pauses_inserted < max_pauses
                and rng.random() < insert_prob
            ):
                r = rng.random()
                cum = 0.0
                chosen = "short"
                for k, wv in pause_weights.items():
                    cum += wv
                    if r <= cum:
                        chosen = k
                        break
                if chosen == "short":
                    base = short_base
                    jitter = rng.randint(-50, 80)
                    ms = max(120, int(base + jitter))
                elif chosen == "medium":
                    base = med_base
                    jitter = rng.randint(-120, 300)
                    ms = max(300, int(base + jitter))
                else:
                    if style == "slow":
                        jitter = rng.randint(400, 1800)
                        ms = max(900, int(long_base + jitter))
                    else:
                        jitter = rng.randint(200, 1200)
                        ms = max(700, int(long_base + jitter))
                seg = " ".join(cur_words).strip()
                if seg:
                    if not re.search(r"[.!?]$", seg) and len(seg.split()) < 6:
                        seg = seg + "..."
                    new_chunks.append(seg)
                new_chunks.append(f"__SILENCE__:{ms}")
                pauses_inserted += 1
                cur_words = []
                words_since_breath = 0
        if cur_words:
            seg = " ".join(cur_words).strip()
            if seg:
                if not re.search(r"[.!?]$", seg) and len(seg.split()) < 6:
                    seg = seg + "..."
                new_chunks.append(seg)
    return new_chunks


# ---------- CHILD WORKER SCRIPT (written to tmp dir) ----------
child_script_source = r"""
# child script to synthesize a single chunk. invoked by parent via subprocess.
import sys, os, json, traceback
def _atomic_write_json(path, data):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def _safe_filesize_kb(p):
    try:
        return int(os.path.getsize(p)/1024)
    except:
        return 0

def main():
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--infile", required=True)
        parser.add_argument("--out", required=True)
        parser.add_argument("--speed", required=True)
        parser.add_argument("--speaker", default=None)
        parser.add_argument("--models", required=True)  # json list
        parser.add_argument("--status", required=True)
        args = parser.parse_args()
        status_file = args.status
        _atomic_write_json(status_file, {"state":"started", "pid":os.getpid(), "ts": __import__("datetime").datetime.now().isoformat()})
        with open(args.infile, "r", encoding="utf-8") as f:
            text = f.read()
        models = json.load(open(args.models, "r", encoding="utf-8"))
        speed = float(args.speed)
        speaker = args.speaker if args.speaker and args.speaker != "None" else None
        last_exc = None
        from TTS.api import TTS as _TTS
        for model in models:
            try:
                try:
                    tts = _TTS(model, gpu=False)
                except TypeError:
                    tts = _TTS(model, False)
                local_speaker = speaker
                if local_speaker is None and getattr(tts, "speakers", None):
                    local_speaker = tts.speakers[0]
                tts.tts_to_file(text=text, file_path=args.out, speed=speed, speaker=local_speaker)
                if os.path.exists(args.out) and _safe_filesize_kb(args.out) > 0:
                    _atomic_write_json(status_file, {"state":"ok", "path": args.out, "size_kb": _safe_filesize_kb(args.out), "model": model, "pid": os.getpid(), "ts": __import__("datetime").datetime.now().isoformat()})
                    return 0
                else:
                    raise RuntimeError("Output missing/empty after tts_to_file for model "+model)
            except Exception:
                last_exc = traceback.format_exc()
                _atomic_write_json(status_file, {"state":"error", "model": model, "error": last_exc, "pid": os.getpid(), "ts": __import__("datetime").datetime.now().isoformat()})
                continue
        _atomic_write_json(status_file, {"state":"error", "error": last_exc or "unknown", "pid": os.getpid(), "ts": __import__("datetime").datetime.now().isoformat()})
        return 2
    except Exception:
        tb = traceback.format_exc()
        try:
            status_arg = None
            if "--status" in sys.argv:
                try:
                    status_arg = sys.argv[sys.argv.index("--status")+1]
                except Exception:
                    status_arg = None
            if status_arg:
                _atomic_write_json(status_arg, {"state":"error","error":tb})
        except Exception:
            pass
        sys.stderr.write(tb)
        return 3

if __name__ == "__main__":
    sys.exit(main())
"""


# ---------- WAV PARAMS / FFmpeg HELPERS ----------
def _read_wav_params(path: str) -> Tuple[int, int, int]:
    """Return (nchannels, sampwidth, framerate) for a wav file using wave module."""
    try:
        with wave.open(path, "rb") as w:
            return (w.getnchannels(), w.getsampwidth(), w.getframerate())
    except Exception:
        raise


def _all_wavs_have_same_params(paths: List[str]) -> bool:
    if not paths:
        return True
    try:
        base = _read_wav_params(paths[0])
    except Exception:
        return False
    for p in paths[1:]:
        try:
            if _read_wav_params(p) != base:
                return False
        except Exception:
            return False
    return True


def _ffmpeg_resample_to_uniform(
    paths: List[str],
    tmpdir: str,
    ffmpeg_exe: str,
    target_rate: int = 22050,
    target_channels: int = 1,
) -> List[str]:
    """
    Use ffmpeg to re-encode each path to PCM S16LE, target_rate Hz, target_channels, write to tmp converted files.
    Returns list of re-encoded file paths in the same order.
    This function verifies ffmpeg_exe and attempts automatic recovery via ensure_ffmpeg_available_and_install_if_missing().
    """
    # Ensure ffmpeg_exe is valid; attempt to recover if not
    if not ffmpeg_exe or not os.path.exists(ffmpeg_exe):
        # Try resolving using global helper
        ff = ensure_ffmpeg_available_and_install_if_missing()
        if ff:
            ffmpeg_exe = ff
        else:
            raise RuntimeError(
                "ffmpeg executable not found. Cannot re-encode WAVs without ffmpeg."
            )

    converted = []
    for i, p in enumerate(paths):
        base = os.path.basename(p)
        outp = os.path.join(tmpdir, f"converted_{i:03d}_" + base)
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            p,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(target_rate),
            "-ac",
            str(target_channels),
            outp,
        ]
        try:
            subprocess.check_call(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed to re-encode {p}: {e}")
        converted.append(outp)
    return converted


# ---------- PYTHON WAV CONCAT (keeps original behavior but more tolerant) ----------
def _python_wave_concat(
    chunk_files: List[str], base_output_path: str, try_fix_with_ffmpeg: bool = True
) -> str:
    """
    Concatenate WAV files using the Python wave module. If parameters mismatch and ffmpeg available,
    try to re-encode chunks to uniform params and concat again.
    Returns path to the concatenated WAV on success.
    """
    concatenated_wav = os.path.splitext(base_output_path)[0] + "_coqui_storytelling.wav"
    if not chunk_files:
        raise RuntimeError("No chunk files provided to concat.")

    # If all params same -> proceed
    if _all_wavs_have_same_params(chunk_files):
        with wave.open(chunk_files[0], "rb") as w0:
            params = w0.getparams()
            frames = [w0.readframes(w0.getnframes())]
        for cf in chunk_files[1:]:
            with wave.open(cf, "rb") as w:
                frames.append(w.readframes(w.getnframes()))
        with wave.open(concatenated_wav, "wb") as outw:
            outw.setparams(params)
            for fr in frames:
                outw.writeframes(fr)
        print(f"{_ts()} ✅ Python WAV concatenation produced: {concatenated_wav}")
        return concatenated_wav

    # Params mismatch -> try ffmpeg if allowed
    ffmpeg_exe = (
        ffmpeg_path
        if (ffmpeg_path and os.path.exists(ffmpeg_path))
        else shutil.which("ffmpeg")
    )
    if try_fix_with_ffmpeg and ffmpeg_exe:
        print(
            f"{_ts()} ⚠️ WAV parameter mismatch detected. Attempting to re-encode chunks to uniform params using ffmpeg: {ffmpeg_exe}"
        )
        tmp_reencode_dir = tempfile.mkdtemp(prefix="coqui_reencode_")
        try:
            converted = _ffmpeg_resample_to_uniform(
                chunk_files,
                tmp_reencode_dir,
                ffmpeg_exe,
                target_rate=22050,
                target_channels=1,
            )
            # Now try python concat on converted files
            with wave.open(converted[0], "rb") as w0:
                params = w0.getparams()
                frames = [w0.readframes(w0.getnframes())]
            for cf in converted[1:]:
                with wave.open(cf, "rb") as w:
                    frames.append(w.readframes(w.getnframes()))
            with wave.open(concatenated_wav, "wb") as outw:
                outw.setparams(params)
                for fr in frames:
                    outw.writeframes(fr)
            print(
                f"{_ts()} ✅ Python WAV concatenation produced after re-encoding: {concatenated_wav}"
            )
            return concatenated_wav
        finally:
            try:
                shutil.rmtree(tmp_reencode_dir)
            except Exception:
                pass

    # If no ffmpeg available or re-encode failed -> raise helpful error
    raise RuntimeError(
        "Incompatible WAV parameters between chunks; cannot concatenate using Python wave. "
        "ffmpeg not available to re-encode chunks. Install ffmpeg or provide a working ffmpeg binary."
    )


def _ffmpeg_concat_encode(
    chunk_files: List[str], base_output_path: str, ffmpeg_exe: str
) -> str:
    """
    Use ffmpeg concat demuxer and re-encode to consistent PCM WAV, returning concatenated WAV path.
    Ensures ffmpeg_exe exists (and will attempt install via ensure_ffmpeg_available_and_install_if_missing()).
    """
    # resolve ffmpeg_exe if missing
    if not ffmpeg_exe or not os.path.exists(ffmpeg_exe):
        ff = ensure_ffmpeg_available_and_install_if_missing()
        if ff:
            ffmpeg_exe = ff
        else:
            raise RuntimeError(
                "ffmpeg executable not found. Cannot perform ffmpeg concat/encode."
            )

    tmpdir = (
        os.path.dirname(chunk_files[0])
        if chunk_files
        else tempfile.mkdtemp(prefix="coqui_ffconcat_")
    )
    concat_list = os.path.join(tmpdir, "ffconcat_list.txt")
    # write a safe concat file (ffmpeg expects: file 'path')
    with open(concat_list, "w", encoding="utf-8") as L:
        for cf in chunk_files:
            # choose quoting intelligently
            if "'" in cf and '"' not in cf:
                L.write(f'file "{cf}"\n')
            else:
                # default to single quotes; if both quotes present, escape single quotes
                if "'" in cf and '"' in cf:
                    safe = cf.replace("'", r"'\''")
                    L.write(f"file '{safe}'\n")
                else:
                    L.write(f"file '{cf}'\n")

    concatenated_wav = os.path.splitext(base_output_path)[0] + "_coqui_storytelling.wav"
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "22050",
        "-ac",
        "1",
        concatenated_wav,
    ]
    subprocess.check_call(cmd)
    return concatenated_wav


# ---------- MAIN ORCHESTRATION: spawn subprocess workers ----------
def synthesize_chunks_and_concatenate(
    chunks: List[str], base_output_path: str, speed: float, speaker
) -> str:
    """
    Spawn subprocess workers (one-per-chunk, controlled by MAX_WORKERS), synthesize each chunk
    (or generate silence), and concatenate results into a single WAV/MP3.

    Preserves original behaviour:
      - infinite retry of failing chunks
      - writes child_script_source to tmpdir and invokes it
      - respects CHUNK_WORKER_TIMEOUT and retry_delay
      - attempts ffmpeg concat/encode, falls back to python concat + re-encode
      - returns WAV or MP3 depending on base_output_path ext
    Improvements:
      - stronger, cross-platform subprocess process-group handling so timeouts kill children reliably
      - clearer timeout/kill logic (terminate -> wait -> kill)
      - safer reading of child stderr tails and status files
      - prevents zombie processes by waiting where appropriate
      - improved ffmpeg discovery (tries pip install imageio-ffmpeg / ffmpeg if missing)
    """
    import signal

    tmpdir = tempfile.mkdtemp(prefix="coqui_tts_chunks_")
    print(f"{_ts()} 🧾 Temporary chunk dir: {tmpdir} (preserve={PRESERVE_TMPDIR})")

    # write child script once
    child_script_path = os.path.join(tmpdir, "coqui_tts_child.py")
    with open(child_script_path, "w", encoding="utf-8") as f:
        f.write(child_script_source)

    try:
        use_cuda_hint = child_local_cuda_hint()

        # Build tasks and trailing pause selection (same as before)
        tasks: List[Tuple[int, str, str, int]] = []
        for i, chunk in enumerate(chunks):
            chunk_fname = os.path.join(tmpdir, f"chunk_{i:03d}.wav")
            trailing_pause_ms = 0
            if isinstance(chunk, str) and chunk.startswith("__SILENCE__:"):
                trailing_pause_ms = 0
            else:
                if re.search(r"\.\.\.$", chunk):
                    trailing_pause_ms = storytelling_settings.get("pause_long_ms", 1200)
                elif re.search(r"[.!?]$", chunk):
                    trailing_pause_ms = storytelling_settings.get(
                        "pause_medium_ms", 700
                    )
                elif "," in chunk:
                    trailing_pause_ms = storytelling_settings.get("pause_short_ms", 220)
                else:
                    trailing_pause_ms = 0
            tasks.append((i, chunk, chunk_fname, trailing_pause_ms))

        active = []
        results = {}
        total_tasks = len(tasks)
        attempts: Dict[int, int] = {i: 0 for i in range(total_tasks)}
        pending_indices = [i for i in range(total_tasks)]

        # create models json in tmpdir for child consumption
        models_json_path = os.path.join(tmpdir, "models.json")
        with open(models_json_path, "w", encoding="utf-8") as f:
            json.dump(preferred_models, f)

        # Helper to safely start subprocess in its own process group/session
        def _popen_in_group(cmd, stdout_f, stderr_f):
            # On POSIX use start_new_session, on Windows use CREATE_NEW_PROCESS_GROUP
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                return subprocess.Popen(
                    cmd,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    creationflags=creationflags,
                )
            else:
                # start_new_session available -> creates new session/process group
                return subprocess.Popen(
                    cmd, stdout=stdout_f, stderr=stderr_f, start_new_session=True
                )

        # Main worker loop (spawn/monitor/requeue)
        while pending_indices or active:
            # spawn as many as allowed
            while len(active) < MAX_WORKERS and pending_indices:
                i = pending_indices.pop(0)
                idx, chunk, outp, pause_ms = tasks[i]
                attempts[i] += 1

                # silence markers handled immediately
                if isinstance(chunk, str) and chunk.startswith("__SILENCE__:"):
                    try:
                        try:
                            ms = int(chunk.split(":", 1)[1])
                        except Exception:
                            ms = storytelling_settings.get("pause_medium_ms", 700)
                        silence_path = os.path.join(
                            tmpdir, f"silence_{i:03d}_{ms}ms.wav"
                        )
                        generate_silence_wav(
                            ms,
                            silence_path,
                            sample_rate=22050,
                            nchannels=1,
                            sampwidth=2,
                        )
                        results[i] = (i, True, silence_path, None)
                        print(
                            f"{_ts()} 🔇 Inserted silence for task {i} -> {silence_path} ({ms}ms)"
                        )
                    except Exception:
                        tb = traceback.format_exc()
                        results[i] = (i, False, outp, tb)
                        print(
                            f"{_ts()} ❌ Failed to generate silence for task {i}: {tb}"
                        )
                    continue

                # write chunk text to file for child to read
                text_path = os.path.join(tmpdir, f"chunk_{i:03d}.txt")
                with open(text_path, "w", encoding="utf-8") as tf:
                    tf.write(chunk)

                status_path = outp + ".status.json"
                stderr_path = os.path.join(tmpdir, f"chunk_{i:03d}.err.log")
                stdout_path = os.path.join(tmpdir, f"chunk_{i:03d}.out.log")

                cmd = [
                    sys.executable,
                    child_script_path,
                    "--infile",
                    text_path,
                    "--out",
                    outp,
                    "--speed",
                    str(speed),
                    "--speaker",
                    str(speaker) if speaker is not None else "None",
                    "--models",
                    models_json_path,
                    "--status",
                    status_path,
                ]

                try:
                    so = open(stdout_path, "wb")
                    se = open(stderr_path, "wb")
                    p = _popen_in_group(cmd, so, se)
                except Exception:
                    tb = traceback.format_exc()
                    print(f"{_ts()} ❌ Failed to start subprocess for chunk {i}: {tb}")
                    # ensure files closed
                    try:
                        so.close()
                    except Exception:
                        pass
                    try:
                        se.close()
                    except Exception:
                        pass
                    # requeue immediately (infinite retry)
                    pending_indices.append(i)
                    time.sleep(retry_delay)
                    continue

                active.append(
                    (
                        p,
                        i,
                        outp,
                        pause_ms,
                        time.time(),
                        text_path,
                        status_path,
                        stderr_path,
                        stdout_path,
                    )
                )
                print(
                    f"{_ts()} 🛠️ Started subprocess for chunk {i+1}/{total_tasks} -> {outp} (pid={p.pid})"
                )

            # monitor active items (iterate over a copy to allow removals)
            for proc_tuple in active[:]:
                (
                    proc,
                    idx,
                    outp,
                    pause_ms,
                    started,
                    text_path,
                    status_path,
                    stderr_path,
                    stdout_path,
                ) = proc_tuple
                elapsed = time.time() - started

                # Timeout handling: terminate then escalate to kill process group
                if elapsed > CHUNK_WORKER_TIMEOUT and proc.poll() is None:
                    try:
                        print(
                            f"\n{_ts()} ⚠️ Subprocess for chunk {idx} timed out (> {CHUNK_WORKER_TIMEOUT}s). Terminating (pid={proc.pid})."
                        )
                        proc.terminate()
                    except Exception:
                        print(
                            f"\n{_ts()} ⚠️ Failed to terminate subprocess for chunk {idx} (pid={proc.pid})."
                        )
                    # give short grace then escalate
                    time.sleep(1.5)
                    if proc.poll() is None:
                        try:
                            # kill process group where possible (POSIX)
                            if os.name != "nt":
                                try:
                                    pgid = os.getpgid(proc.pid)
                                    os.killpg(pgid, signal.SIGKILL)
                                except Exception:
                                    proc.kill()
                            else:
                                # Windows: attempt CTRL_BREAK_EVENT for group, else kill
                                try:
                                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                                except Exception:
                                    proc.kill()
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                        print(
                            f"{_ts()} ⚠️ Subprocess for chunk {idx} force-killed (pid={proc.pid})."
                        )

                ret = proc.poll()
                if ret is None:
                    continue

                # process ended; remove from active and close any opened fds
                try:
                    active.remove(proc_tuple)
                except ValueError:
                    pass

                # ensure stdout/stderr files are flushed/closed before reading tails
                try:
                    if os.path.exists(stdout_path):
                        # no-op; child wrote directly to file
                        pass
                except Exception:
                    pass

                status_info = {}
                if os.path.exists(status_path):
                    try:
                        status_info = _read_json_safe(status_path)
                    except Exception:
                        status_info = {}

                # SUCCESS condition: status "ok" AND reported path exists AND non-empty
                if status_info.get("state") == "ok":
                    path = status_info.get("path", outp)
                    if os.path.exists(path) and _safe_filesize_kb(path) > 0:
                        results[idx] = (idx, True, path, None)
                        print(
                            f"{_ts()} 🟢 Worker finished for chunk {idx+1} -> {path} (pause_after={pause_ms}ms)"
                        )
                        # close file handles if any (best-effort)
                        try:
                            proc.stdout and proc.stdout.close()
                        except Exception:
                            pass
                        continue
                    else:
                        print(
                            f"{_ts()} ⚠️ Status-file OK but output file missing/empty: {path}"
                        )

                if status_info and status_info.get("state") == "error":
                    print(
                        f"{_ts()} ❌ Subprocess reported error for chunk {idx} (pid={proc.pid})"
                    )
                    err = status_info.get("error", "")
                    if err:
                        print(err[:2000])
                    # show child stderr tail
                    try:
                        if os.path.exists(stderr_path):
                            with open(stderr_path, "rb") as se:
                                se.seek(0, os.SEEK_END)
                                sz = se.tell()
                                se.seek(max(0, sz - 2000))
                                tail = se.read().decode(errors="replace")
                                print(
                                    f"  --- child stderr tail ---\n{tail}\n  --- end stderr tail ---"
                                )
                    except Exception:
                        pass
                else:
                    # attempt to show stderr tail even when no status file
                    print(
                        f"{_ts()} ❌ No completion status for chunk {idx}. Treating as transient error."
                    )
                    try:
                        if os.path.exists(stderr_path):
                            with open(stderr_path, "rb") as se:
                                se.seek(0, os.SEEK_END)
                                sz = se.tell()
                                se.seek(max(0, sz - 2000))
                                tail = se.read().decode(errors="replace")
                                print(
                                    f"  --- child stderr tail ---\n{tail}\n  --- end stderr tail ---"
                                )
                    except Exception:
                        pass

                # infinite retry behavior: always requeue failing chunk (preserve original)
                print(
                    f"{_ts()} 🔁 Re-queueing chunk {idx} for retry (attempt {attempts[idx]})."
                )
                pending_indices.append(idx)
                time.sleep(retry_delay)

            # small sleep to avoid busy-wait
            time.sleep(0.05)

            # Compose ordered list of files for concat including silence files
        chunk_files_seq: List[str] = []
        for i in range(len(chunks)):
            if i in results and results[i][1] and os.path.exists(results[i][2]):
                wavp = results[i][2]
            else:
                wavp = os.path.join(tmpdir, f"chunk_{i:03d}.wav")
                if not os.path.exists(wavp):
                    sf = wavp + ".status.json"
                    st = _read_json_safe(sf)
                    if (
                        st
                        and st.get("state") == "ok"
                        and os.path.exists(st.get("path", ""))
                    ):
                        wavp = st.get("path")
                    else:
                        raise RuntimeError(f"Expected chunk file missing: {wavp}")
            chunk_files_seq.append(wavp)
            # append trailing silence file if needed
            pause_ms = tasks[i][3]
            if pause_ms and pause_ms > 0:
                silence_path = os.path.join(tmpdir, f"silence_{i:03d}_{pause_ms}ms.wav")
                generate_silence_wav(
                    pause_ms, silence_path, sample_rate=22050, nchannels=1, sampwidth=2
                )
                chunk_files_seq.append(silence_path)

        # --- NEW: generate subtitle (.srt) aligned with chunks ---
        try:
            srt_path = write_srt_from_chunks(chunks, tasks, results, base_output_path)
            print(f"{_ts()} ℹ️ Subtitle file generated: {srt_path}")
        except Exception:
            print(f"{_ts()} ⚠️ Subtitle generation failed:\n{traceback.format_exc()}")
        # --------------------------------------------------------

        # FFmpeg selection: prefer provided binary else fallback to PATH ffmpeg.
        # If none found, call ensure helper (which may attempt pip installs).
        ffmpeg_exe = (
            ffmpeg_path
            if (ffmpeg_path and os.path.exists(ffmpeg_path))
            else shutil.which("ffmpeg")
        )
        if not ffmpeg_exe:
            ff = ensure_ffmpeg_available_and_install_if_missing()
            if ff:
                ffmpeg_exe = ff

        if ffmpeg_exe:
            try:
                print(
                    f"{_ts()} 🔁 Using ffmpeg ({ffmpeg_exe}) for concatenation & encoding."
                )
                concatenated_wav = _ffmpeg_concat_encode(
                    chunk_files_seq, base_output_path, ffmpeg_exe
                )
            except subprocess.CalledProcessError as cpe:
                print(
                    f"{_ts()} ❌ ffmpeg concat failed, attempting python concat with re-encode fallback. Error: {cpe}"
                )
                concatenated_wav = _python_wave_concat(
                    chunk_files_seq, base_output_path, try_fix_with_ffmpeg=True
                )
            except Exception as ex:
                print(
                    f"{_ts()} ⚠️ ffmpeg concat raised exception: {ex}. Trying python concat with re-encode fallback."
                )
                concatenated_wav = _python_wave_concat(
                    chunk_files_seq, base_output_path, try_fix_with_ffmpeg=True
                )
        else:
            # No ffmpeg provided; try Python concat which may re-encode via PATH ffmpeg if available
            try:
                concatenated_wav = _python_wave_concat(
                    chunk_files_seq, base_output_path, try_fix_with_ffmpeg=True
                )
            except Exception as e:
                # raise a clearer error
                raise RuntimeError(
                    f"Incompatible WAV parameters and ffmpeg not available to re-encode. Install ffmpeg or provide a binary. Original: {e}"
                )

        # If final desired is mp3, try to convert using ffmpeg (provided path first, then PATH fallback)
        desired_ext = os.path.splitext(base_output_path)[1].lower()
        if desired_ext == ".mp3":
            # Ensure ffmpeg_exe is resolved now (may have been None earlier)
            if not ffmpeg_exe or not os.path.exists(ffmpeg_exe):
                ff = ensure_ffmpeg_available_and_install_if_missing()
                ffmpeg_exe = ff if ff else shutil.which("ffmpeg")

            if ffmpeg_exe:
                try:
                    final_mp3 = base_output_path
                    conv_cmd = [
                        ffmpeg_exe,
                        "-y",
                        "-i",
                        concatenated_wav,
                        "-codec:a",
                        "libmp3lame",
                        "-q:a",
                        "2",
                        final_mp3,
                    ]
                    subprocess.check_call(conv_cmd)
                    print(f"{_ts()} ✅ MP3 conversion completed: {final_mp3}")
                    return final_mp3
                except subprocess.CalledProcessError:
                    print(
                        f"{_ts()} ❌ ffmpeg (provided) mp3 conversion failed. Trying PATH ffmpeg fallback."
                    )
            # fallback: PATH ffmpeg
            ff_on_path = shutil.which("ffmpeg")
            if ff_on_path:
                try:
                    final_mp3 = base_output_path
                    conv_cmd = [
                        ff_on_path,
                        "-y",
                        "-i",
                        concatenated_wav,
                        "-codec:a",
                        "libmp3lame",
                        "-q:a",
                        "2",
                        final_mp3,
                    ]
                    subprocess.check_call(conv_cmd)
                    print(
                        f"{_ts()} ✅ MP3 conversion completed via PATH ffmpeg: {final_mp3}"
                    )
                    return final_mp3
                except subprocess.CalledProcessError:
                    print(
                        f"{_ts()} ❌ PATH ffmpeg mp3 conversion failed. Returning WAV."
                    )
                    return concatenated_wav
            else:
                print(
                    f"{_ts()} ⚠️ No ffmpeg available to convert to MP3. Returning WAV."
                )
                return concatenated_wav
        else:
            return concatenated_wav

    finally:
        if PRESERVE_TMPDIR:
            print(f"{_ts()} ℹ️ Preserving temporary dir for debugging: {tmpdir}")
        else:
            try:
                shutil.rmtree(tmpdir)
                print(f"{_ts()} 🧹 Removed temporary dir: {tmpdir}")
            except Exception:
                print(
                    f"{_ts()} ⚠️ Failed to remove temporary dir: {tmpdir}. You can remove it manually."
                )


# ---------- ESTIMATE & SCALE SPEED ----------
def estimate_duration_seconds(
    text: str, base_speed: float, base_wpm: int = 110
) -> float:
    words = len(text.split())
    if words == 0:
        return 0.0
    seconds = (words / base_wpm) * 60.0 / max(1e-6, base_speed)
    return seconds


def compute_scaled_speed_for_target(
    text: str, base_speed: float, target_seconds: float, base_wpm: int = 110
) -> float:
    est = estimate_duration_seconds(text, base_speed, base_wpm=base_wpm)
    if est <= 0 or target_seconds <= 0:
        return base_speed
    scale = est / target_seconds
    new_speed = base_speed * scale
    new_speed = max(MIN_SPEED, min(MAX_SPEED, new_speed))
    return float(round(new_speed, 3))


# ---------- MAIN ----------
def main_loop():
    while True:
        try:
            txt_file = find_txt_file(folder_path)
            if not txt_file:
                print(
                    f"{_ts()} ℹ️ No .txt file found in folder: {folder_path}. Waiting {retry_delay}s and retrying..."
                )
                time.sleep(retry_delay)
                continue

            print(f"{_ts()} 📄 Found text file: {txt_file}")
            full_text = read_text(txt_file)
            segments = extract_bracket_texts(full_text)
            if not segments:
                print(
                    f"{_ts()} ℹ️ No bracketed '[ ... ]' segments found in {txt_file}. Waiting {retry_delay}s and retrying..."
                )
                time.sleep(retry_delay)
                continue

            narration_text = (
                (" " + storytelling_settings["pause_medium"] + " ")
                .join(segments)
                .strip()
            )
            narration_text = normalize_punctuation_for_tts(narration_text)
            narration_text = sanitize_text_for_tts(narration_text)

            base_speed = float(storytelling_settings["speed"])
            if TARGET_MINUTES is not None and TARGET_MINUTES > 0:
                target_seconds = TARGET_MINUTES * 60
                adjusted_speed = compute_scaled_speed_for_target(
                    narration_text, base_speed, target_seconds
                )
                print(
                    f"{_ts()} 🔢 Estimated duration at base_speed {base_speed}: {estimate_duration_seconds(narration_text, base_speed):.1f}s"
                )
                print(
                    f"{_ts()} 🎯 Target duration: {target_seconds:.1f}s -> adjusted speed: {adjusted_speed}"
                )
                speed_to_use = adjusted_speed
            else:
                speed_to_use = base_speed

            chunks = smart_sentence_split(narration_text, max_chunk_words=90)
            print(
                f"{_ts()} 🧩 Split narration into {len(chunks)} chunks for safer, paced synthesis."
            )
            for i, c in enumerate(chunks[:5]):
                print(f"  chunk {i+1}: {c[:120]}{'...' if len(c)>120 else ''}")

            try:
                chunks = inject_random_pauses_into_chunks(chunks, speed_to_use)
                print(
                    f"{_ts()} 🔀 Injected randomized breath pauses. New total chunks (including silence markers): {len(chunks)}"
                )
            except Exception:
                traceback.print_exc()
                print(
                    f"{_ts()} ⚠️ Pause injection failed — continuing with original chunks."
                )

            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            output_file_path = os.path.join(
                folder_path, f"{base_name}_coqui_storytelling.mp3"
            )
            print(f"{_ts()} 🎧 Will synthesize to: {output_file_path}")
            print(f"{_ts()} 🗣️ Settings: {storytelling_settings['description']}")
            print(
                f"{_ts()} 🛠️ speed (used)={speed_to_use} speaker={storytelling_settings['speaker']}"
            )

            final_path = synthesize_chunks_and_concatenate(
                chunks, output_file_path, speed_to_use, storytelling_settings["speaker"]
            )
            print(
                f"{_ts()} 🏁 Coqui expressive storytelling narration completed: {final_path}"
            )

            break

        except Exception as e:
            print(f"{_ts()} ❌ Fatal error in main loop: {e}")
            traceback.print_exc()
            print(
                f"{_ts()} 🔄 Waiting {retry_delay}s before retrying the whole process..."
            )
            time.sleep(retry_delay)


if __name__ == "__main__":
    try:
        from multiprocessing import set_start_method as _ssm

        _ssm("spawn", force=True)
    except Exception:
        pass

    # Ensure ffmpeg present (Colab-friendly auto-install if missing)
    ff = ensure_ffmpeg_available_and_install_if_missing()
    if not ff:
        print(
            f"{_ts()} ⚠️ ffmpeg not available. The script will continue, but concatenation/resampling may require ffmpeg. Install it to avoid errors."
        )

    start = time.time()
    res = ensure_coqui_models_installed(preferred_models, auto_install_tts=True)
    for model, info in res.items():
        print(model, "->", info)
    main_loop()
    end = time.time()
    log_execution_time(start, end)
