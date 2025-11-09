#!/usr/bin/env python3
"""
Ultimate Image-to-Video composer using FFmpeg + FFprobe.

Improvements:
 - No side effects at import time (wraps runtime logic in main()) — friendly for GitHub Actions/tests.
 - Robust ensure_ffmpeg_binaries(): checks user-provided path, PATH, imageio-ffmpeg, and pip fallback.
 - Detects GitHub Actions / CI and avoids attempting interactive installs there; gives actionable guidance.
 - Uses logging instead of bare prints for clearer CI logs.
 - Preserves all original features (serial-first ordering including 'promt' typo, per-image segments,
   concat demuxer, final safe re-encode merge, temp cleanup).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import shutil
import glob
import sys
import logging
from typing import Optional, List

# -------------------- Configuration / Defaults --------------------
current_directory = os.getcwd()
ffmpeg_root_path = os.path.join(current_directory, "ffmpeg")

# cross-platform default paths (user may override by placing ffmpeg/ffprobe under ffmpeg/bin)
if sys.platform.startswith("win"):
    default_ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg.exe")
    default_ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe.exe")
else:
    default_ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg")
    default_ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe")

# These globals will be updated by ensure_ffmpeg_binaries()
ffmpeg_path = default_ffmpeg_path
ffprobe_path = default_ffprobe_path

image_folder = os.path.join(current_directory, "image_response")
narration_folder = os.path.join(current_directory, "narration_response")
output_video = os.path.join(current_directory, "image_response", "final_output.mp4")

frame_rate = 24
target_resolution = (1920, 1080)
video_crf = 18
video_preset = "slow"
pix_fmt = "yuv420p"
audio_bitrate = "192k"
audio_sample_rate = 48000

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("img2vid")


# -------------------- Helpers --------------------
def _is_exec(path: Optional[str]) -> bool:
    return bool(path and os.path.exists(path) and os.access(path, os.X_OK))


def ensure_ffmpeg_binaries() -> None:
    """
    Ensure ffmpeg and ffprobe executables are available and set global ffmpeg_path / ffprobe_path.

    Resolution order:
      1) Already-configured default ffmpeg_path / ffprobe_path (local ffmpeg/ directory)
      2) Look in PATH via shutil.which
      3) Try to use imageio-ffmpeg if installed
      4) If not in CI (GitHub Actions), attempt pip install imageio-ffmpeg and re-check
      5) Last-resort: pip install ffmpeg package (may not yield usable binary)
      6) Check common system locations

    NOTE: If running in CI (GITHUB_ACTIONS=true) we avoid auto pip-installs and instead raise
    an actionable error so CI workflow can install ffmpeg via package manager or setup actions.
    """
    global ffmpeg_path, ffprobe_path

    # 0) quick success if already valid
    if _is_exec(ffmpeg_path) and _is_exec(ffprobe_path):
        logger.debug(
            "Using preconfigured ffmpeg/ffprobe at %s / %s", ffmpeg_path, ffprobe_path
        )
        return

    # 1) PATH lookup
    path_ffmpeg = shutil.which("ffmpeg")
    path_ffprobe = shutil.which("ffprobe")
    if path_ffmpeg:
        ffmpeg_path = path_ffmpeg
    if path_ffprobe:
        ffprobe_path = path_ffprobe
    if _is_exec(ffmpeg_path) and _is_exec(ffprobe_path):
        logger.info("Found ffmpeg/ffprobe on PATH: %s / %s", ffmpeg_path, ffprobe_path)
        return

    # 2) Try to import imageio_ffmpeg if already available
    imageio_ffmpeg_mod = None
    try:
        import importlib

        imageio_ffmpeg_mod = importlib.import_module("imageio_ffmpeg")
    except Exception:
        imageio_ffmpeg_mod = None

    if imageio_ffmpeg_mod:
        try:
            exe = imageio_ffmpeg_mod.get_ffmpeg_exe()
            if exe and _is_exec(exe):
                ffmpeg_path = exe
                # try to find ffprobe next to ffmpeg
                probe_candidate = os.path.join(os.path.dirname(exe), "ffprobe")
                if sys.platform.startswith("win"):
                    probe_candidate += ".exe"
                if _is_exec(probe_candidate):
                    ffprobe_path = probe_candidate
                elif shutil.which("ffprobe"):
                    ffprobe_path = shutil.which("ffprobe")
                if _is_exec(ffmpeg_path) and _is_exec(ffprobe_path):
                    logger.info("Using ffmpeg from imageio-ffmpeg: %s", ffmpeg_path)
                    return
        except Exception:
            # fall through to installation attempts
            imageio_ffmpeg_mod = None

    # If running inside CI (GitHub Actions) avoid attempting interactive installs:
    in_github_actions = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    if in_github_actions:
        # give actionable guidance rather than attempting to pip-install inside CI
        msg = (
            "ffmpeg/ffprobe not found in CI environment (GITHUB_ACTIONS=true). "
            "Do NOT attempt to pip-install in CI — instead add a step to your workflow to "
            "install ffmpeg. Example (Ubuntu runner):\n\n"
            "  - name: Install ffmpeg\n"
            "    run: sudo apt-get update && sudo apt-get install -y ffmpeg\n\n"
            "Or use an action that provides ffmpeg. Ensure ffmpeg and ffprobe are available on PATH "
            "or provide a local ./ffmpeg/bin/ directory in the repo.\n"
            f"Tried default location: {ffmpeg_root_path} and PATH."
        )
        raise FileNotFoundError(msg)

    # Not in CI: attempt to install imageio-ffmpeg via pip (safe fallback)
    installed_imageio = False
    try:
        logger.warning(
            "ffmpeg/ffprobe not found — attempting to install imageio-ffmpeg via pip..."
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "imageio-ffmpeg"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import importlib

        imageio_ffmpeg_mod = importlib.import_module("imageio_ffmpeg")
        installed_imageio = True
    except Exception as e:
        logger.warning(
            "pip install imageio-ffmpeg failed (will try other fallbacks): %s", e
        )
        imageio_ffmpeg_mod = None

    if imageio_ffmpeg_mod:
        try:
            exe = imageio_ffmpeg_mod.get_ffmpeg_exe()
            if exe and _is_exec(exe):
                ffmpeg_path = exe
                probe_candidate = os.path.join(os.path.dirname(exe), "ffprobe")
                if sys.platform.startswith("win"):
                    probe_candidate += ".exe"
                if _is_exec(probe_candidate):
                    ffprobe_path = probe_candidate
                elif shutil.which("ffprobe"):
                    ffprobe_path = shutil.which("ffprobe")
                if _is_exec(ffmpeg_path) and _is_exec(ffprobe_path):
                    logger.info(
                        "Using ffmpeg provided by imageio-ffmpeg: %s", ffmpeg_path
                    )
                    return
        except Exception:
            pass

    # Try pip install ffmpeg (may not provide a usable platform binary, but it's a fallback)
    try:
        logger.warning("Attempting 'pip install ffmpeg' as a last-ditch fallback...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "ffmpeg"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.debug(
            "pip install ffmpeg attempt failed or produced no usable binaries: %s", e
        )

    # Re-check PATH and common locations
    path_ffmpeg = shutil.which("ffmpeg")
    path_ffprobe = shutil.which("ffprobe")
    if path_ffmpeg:
        ffmpeg_path = path_ffmpeg
    if path_ffprobe:
        ffprobe_path = path_ffprobe

    for maybe in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/bin/ffmpeg"]:
        if _is_exec(maybe):
            ffmpeg_path = maybe
            break
    for maybe in ["/usr/bin/ffprobe", "/usr/local/bin/ffprobe", "/bin/ffprobe"]:
        if _is_exec(maybe):
            ffprobe_path = maybe
            break

    if _is_exec(ffmpeg_path) and _is_exec(ffprobe_path):
        logger.info(
            "Found ffmpeg/ffprobe after fallback attempts: %s / %s",
            ffmpeg_path,
            ffprobe_path,
        )
        return

    # Final error with clear instructions
    raise FileNotFoundError(
        "ffmpeg/ffprobe executables could not be found.\n"
        f"Tried: default local ({ffmpeg_root_path}), PATH, imageio-ffmpeg, pip 'ffmpeg'.\n"
        "Please install ffmpeg on your system (e.g. apt/yum/brew/choco) or provide a local 'ffmpeg' directory "
        "with bin/ffmpeg and bin/ffprobe, or ensure ffmpeg/ffprobe are on PATH."
    )


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run subprocess command and return CompletedProcess. Raises SystemExit for failures like before,
    but logs more context for CI-friendly debugging.
    """
    logger.debug("Running command: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=check)
        if res.stderr:
            # truncate to avoid massive logs but keep useful info
            logger.debug("STDERR (truncated 800 chars): %s", (res.stderr or "")[:800])
        return res
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s", " ".join(e.cmd))
        logger.error("Return code: %s", e.returncode)
        logger.error("STDOUT: %s", e.stdout)
        logger.error("STDERR: %s", e.stderr)
        raise SystemExit("FFmpeg execution failed. See logs above for details.")


def probe_duration(path: str) -> Optional[float]:
    """
    Return duration in seconds for media file. Prefer ffprobe, fallback to ffmpeg stderr parsing.
    Returns None if duration could not be determined.
    """
    if _is_exec(ffprobe_path):
        cmd = [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        try:
            res = run(cmd)
            out = (res.stdout or "").strip()
            if out:
                return float(out)
        except Exception:
            pass

    # fallback: ffmpeg -i
    if _is_exec(ffmpeg_path):
        try:
            res = subprocess.run(
                [ffmpeg_path, "-i", path], capture_output=True, text=True
            )
            stderr = res.stderr or ""
            m = re.search(r"Duration:\s+(\d{2}:\d{2}:\d{2}\.\d+)", stderr)
            if m:
                h, m_, s = m.group(1).split(":")
                secs = int(h) * 3600 + int(m_) * 60 + float(s)
                return float(secs)
        except Exception:
            pass

    return None


def find_best_audio(folder: str) -> Optional[str]:
    """Find best audio (longest duration, prefer .wav)."""
    candidates = []
    for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac"):
        candidates.extend(glob.glob(os.path.join(folder, ext)))
    if not candidates:
        return None
    best, best_dur = None, -1.0
    for path in candidates:
        dur = probe_duration(path)
        if dur is None:
            logger.warning("Skipping unreadable audio: %s", os.path.basename(path))
            continue
        score = dur + (0.1 if path.lower().endswith(".wav") else 0.0)
        logger.info(
            "Audio candidate: %s — %.2fs (score %.2f)",
            os.path.basename(path),
            dur,
            score,
        )
        if score > best_dur:
            best, best_dur = path, score
    return best


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split("(\d+)", s)]


def collect_images_serial_first(folder: str) -> List[str]:
    """Collect images with serial order first (supports 'promt' typo)."""
    files = [
        f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not files:
        return []
    serial_pattern = re.compile(
        r"^(.*?(?:image[_\-\s]*promp?t)[_\-\s]*)(\d+)(\.[^.]+)$", re.IGNORECASE
    )
    matches = []
    for f in files:
        m = serial_pattern.match(f)
        if m:
            matches.append((int(m.group(2)), f))
    if matches:
        matches.sort(key=lambda x: x[0])
        ordered = [os.path.join(folder, f) for _, f in matches]
        remainder = [f for f in files if f not in {n for _, n in matches}]
        remainder.sort(key=natural_sort_key)
        ordered += [os.path.join(folder, f) for f in remainder]
        return ordered
    files.sort(key=natural_sort_key)
    return [os.path.join(folder, f) for f in files]


def ensure_even_dimensions_filter(width: int, height: int) -> str:
    return (
        "scale='trunc(iw*min({W}/iw\\,{H}/ih)/2)*2':"
        "'trunc(ih*min({W}/iw\\,{H}/ih)/2)*2',"
        "pad={W}:{H}:(ow-iw)/2:(oh-ih)/2"
    ).format(W=width, H=height)


# -------------------- Main Flow --------------------
def main() -> None:
    # Validate folders (fail early)
    if not os.path.isdir(image_folder):
        logger.error("Image folder not found: %s", image_folder)
        raise SystemExit(2)
    if not os.path.isdir(narration_folder):
        logger.error("Narration folder not found: %s", narration_folder)
        raise SystemExit(2)

    # Ensure ffmpeg binaries available; will raise FileNotFoundError with guidance if not
    try:
        ensure_ffmpeg_binaries()
    except FileNotFoundError as e:
        logger.error("ERROR: %s", e)
        raise SystemExit(3)

    # Collect images
    image_files = collect_images_serial_first(image_folder)
    if not image_files:
        logger.error("No images found in %s", image_folder)
        raise SystemExit(4)

    num_images = len(image_files)
    logger.info("Found %d images (serial-first). Showing up to 10:", num_images)
    for i, p in enumerate(image_files[:10], 1):
        logger.info("  %2d. %s", i, os.path.basename(p))

    # Choose audio
    chosen_audio = find_best_audio(narration_folder)
    if not chosen_audio:
        logger.error("No audio found in narration_folder (.wav/.mp3/.m4a/.flac).")
        raise SystemExit(5)

    logger.info("Using audio: %s", chosen_audio)
    audio_duration = probe_duration(chosen_audio)
    if audio_duration is None:
        logger.error("Could not parse audio duration for %s", chosen_audio)
        raise SystemExit(6)
    logger.info("Audio duration: %.3fs", audio_duration)

    per_image_dur = audio_duration / num_images
    logger.info("Each image duration: %.3fs", per_image_dur)

    tmpdir = tempfile.mkdtemp(prefix="img_vid_segments_")
    segment_files: List[str] = []

    try:
        vf_filter = ensure_even_dimensions_filter(*target_resolution)
        logger.info(
            "Creating %d video segments in temporary dir: %s", num_images, tmpdir
        )

        for idx, img in enumerate(image_files):
            seg_path = os.path.join(tmpdir, f"seg_{idx:03d}.mp4")
            cmd = [
                ffmpeg_path,
                "-y",
                "-loop",
                "1",
                "-i",
                img,
                "-t",
                f"{per_image_dur:.6f}",
                "-r",
                str(frame_rate),
                "-vf",
                vf_filter + ",format=" + pix_fmt,
                "-c:v",
                "libx264",
                "-preset",
                video_preset,
                "-crf",
                str(video_crf),
                "-pix_fmt",
                pix_fmt,
                "-movflags",
                "+faststart",
                seg_path,
            ]
            logger.info(
                "Segment %d/%d: %s", idx + 1, num_images, os.path.basename(seg_path)
            )
            run(cmd)
            segment_files.append(seg_path)

        # Write concat list
        concat_list = os.path.join(tmpdir, "segments.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        temp_concat = os.path.join(tmpdir, "temp_video.mp4")
        logger.info("Concatenating segments into %s", temp_concat)
        concat_cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
            "-c",
            "copy",
            temp_concat,
        ]
        run(concat_cmd)

        concat_duration = probe_duration(temp_concat)
        if concat_duration:
            logger.info("Concat video duration: %.3fs", concat_duration)

        # Safe merge (re-encode) — ensures compatible audio/video streams
        logger.info("Merging audio + video (safe re-encode)...")
        merge_cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            temp_concat,
            "-i",
            chosen_audio,
            "-c:v",
            "libx264",  # safe re-encode for guaranteed container compatibility
            "-preset",
            "veryfast",
            "-crf",
            str(18),
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-ar",
            str(audio_sample_rate),
            "-pix_fmt",
            pix_fmt,
            "-shortest",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            output_video,
        ]
        run(merge_cmd)

        logger.info("Success! Output video created: %s", output_video)
        logger.info("Expected duration (≈ audio): %.2fs", audio_duration)
        try:
            size_mb = os.path.getsize(output_video) / (1024**2)
            logger.info("Output size: %.2f MB", size_mb)
        except Exception:
            logger.debug("Could not determine output file size.")

    finally:
        try:
            shutil.rmtree(tmpdir)
            logger.info("Cleaned temporary directory: %s", tmpdir)
        except Exception:
            logger.warning("Could not remove temporary directory: %s", tmpdir)


if __name__ == "__main__":
    main()
