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
import platform
import urllib.request
import tarfile
import zipfile
import stat
import io
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

    Resolution order (preserves earlier behavior):
      1) Already-configured default ffmpeg_path / ffprobe_path (local ffmpeg/ directory)
      2) Look in PATH via shutil.which
      3) Try to use imageio-ffmpeg if installed
      4) If running in CI (GITHUB_ACTIONS=true) and not found, try to download a static ffmpeg build
         appropriate for the runner OS, extract into ./ffmpeg and use that.
      5) If not in CI, attempt pip install imageio-ffmpeg as before (unchanged).
      6) Fallback checks and final helpful error message.
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

    # 2) Try imageio-ffmpeg if available
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
            imageio_ffmpeg_mod = None

    # If running inside CI (GitHub Actions) — try to download static ffmpeg builds into ./ffmpeg
    in_github_actions = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    if in_github_actions:
        logger.info(
            "Running in CI (GITHUB_ACTIONS=true). Attempting to fetch a static ffmpeg build for the runner."
        )
        # Decide OS-specific candidate URLs (common, public static builds)
        sys_plat = platform.system().lower()
        candidate_urls = []
        if "linux" in sys_plat:
            # johnvansickle static builds for amd64 Linux (commonly used on ubuntu-latest)
            candidate_urls = [
                "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
            ]
        elif "windows" in sys_plat or sys_plat.startswith("mingw"):
            # gyan.dev provides windows builds (essentials zip)
            candidate_urls = [
                "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            ]
        elif "darwin" in sys_plat or "mac" in sys_plat:
            # evermeet builds for macOS
            candidate_urls = [
                "https://evermeet.cx/ffmpeg/ffmpeg-6.0.zip",  # best-effort; version may change
            ]
        else:
            candidate_urls = []

        if candidate_urls:
            try:
                _download_and_extract_ffmpeg(candidate_urls, ffmpeg_root_path)
                # After extraction, search for binaries under ffmpeg_root_path
                possible_ff = []
                for root, dirs, files in os.walk(ffmpeg_root_path):
                    for fname in files:
                        if fname.lower() in ("ffmpeg", "ffmpeg.exe"):
                            possible_ff.append(os.path.join(root, fname))
                # pick first executable candidate
                for exe in possible_ff:
                    if _is_exec(exe):
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
                                "Successfully installed and found ffmpeg/ffprobe in %s",
                                ffmpeg_root_path,
                            )
                            return
                logger.warning(
                    "Downloaded ffmpeg but could not locate usable ffmpeg/ffprobe executables under %s",
                    ffmpeg_root_path,
                )
            except Exception as exc:
                logger.warning("Automatic download/extract attempt failed: %s", exc)

        # If we reach here, try PATH and common locations once more before error
        path_ffmpeg = shutil.which("ffmpeg")
        path_ffprobe = shutil.which("ffprobe")
        if path_ffmpeg:
            ffmpeg_path = path_ffmpeg
        if path_ffprobe:
            ffprobe_path = path_ffprobe
        if _is_exec(ffmpeg_path) and _is_exec(ffprobe_path):
            logger.info(
                "Found ffmpeg/ffprobe on PATH after attempted download: %s / %s",
                ffmpeg_path,
                ffprobe_path,
            )
            return

        # Still not found — give CI-specific actionable guidance (but mention we tried to auto-download)
        msg = (
            "ffmpeg/ffprobe not found in CI environment (GITHUB_ACTIONS=true). "
            "Attempted to download static ffmpeg builds for the runner but could not locate usable binaries.\n\n"
            "Recommended fixes (pick one):\n"
            "  1) Add a step to your workflow to install ffmpeg (Ubuntu example):\n\n"
            "     - name: Install ffmpeg\n"
            "       run: sudo apt-get update && sudo apt-get install -y ffmpeg\n\n"
            "  2) Ensure your repo includes a ./ffmpeg/bin/ directory with ffmpeg and ffprobe checked in,\n"
            "     or provide prebuilt binaries at that path.\n\n"
            "  3) Use an action that provides ffmpeg (e.g. a community action) before running this script.\n\n"
            f"Tried default location: {ffmpeg_root_path} and PATH and attempted automatic download."
        )
        raise FileNotFoundError(msg)

    # Not in CI: attempt to pip install imageio-ffmpeg via pip (safe fallback)
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

    # Try pip install ffmpeg (may not provide platform binary)
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


def _download_and_extract_ffmpeg(
    urls: List[str], dest_root: str, timeout: int = 300
) -> None:
    """
    Try a list of candidate URLs; download first that succeeds and extract into dest_root.
    Supports .tar.xz, .tar.gz, .zip. Ensures extracted ffmpeg/ffprobe have executable bit set.
    Raises on total failure.
    """
    os.makedirs(dest_root, exist_ok=True)
    last_exc = None
    for url in urls:
        logger.info("Attempting to download ffmpeg from: %s", url)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                if resp.status not in (200,):
                    raise RuntimeError(f"HTTP status {resp.status} for {url}")
                data = resp.read()
            # Determine archive type
            buf = io.BytesIO(data)
            # tar.xz or tar.gz
            try:
                if url.endswith(".tar.xz") or url.endswith(".tar"):
                    buf.seek(0)
                    with tarfile.open(fileobj=buf, mode="r:*") as t:
                        t.extractall(path=dest_root)
                elif url.endswith(".tar.gz") or url.endswith(".tgz"):
                    buf.seek(0)
                    with tarfile.open(fileobj=buf, mode="r:gz") as t:
                        t.extractall(path=dest_root)
                elif url.endswith(".zip") or url.endswith(".ZIP"):
                    buf.seek(0)
                    with zipfile.ZipFile(buf) as z:
                        z.extractall(path=dest_root)
                else:
                    # try both tar and zip attempts if extension is unknown
                    buf.seek(0)
                    try:
                        with tarfile.open(fileobj=buf, mode="r:*") as t:
                            t.extractall(path=dest_root)
                    except tarfile.ReadError:
                        buf.seek(0)
                        with zipfile.ZipFile(buf) as z:
                            z.extractall(path=dest_root)
            except Exception as e:
                raise RuntimeError(f"Failed to extract archive from {url}: {e}")

            # After extraction, make any ffmpeg/ffprobe under dest_root executable
            for root, dirs, files in os.walk(dest_root):
                for fname in files:
                    if fname.lower() in (
                        "ffmpeg",
                        "ffmpeg.exe",
                        "ffprobe",
                        "ffprobe.exe",
                    ):
                        p = os.path.join(root, fname)
                        try:
                            st = os.stat(p)
                            os.chmod(p, st.st_mode | stat.S_IEXEC)
                        except Exception:
                            logger.debug("Could not set executable bit on %s", p)
            logger.info(
                "Downloaded and extracted ffmpeg from %s into %s", url, dest_root
            )
            return
        except Exception as e:
            last_exc = e
            logger.warning("Failed to download/extract from %s: %s", url, e)
            # try next URL
    raise RuntimeError(f"All download attempts failed. Last error: {last_exc}")


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


# -------------------- Subtitle burning helper --------------------
def _guess_color_hex(color: str) -> str:
    """
    Convert color names or hex '#RRGGBB' or 'RRGGBB' into ASS color bytes order AABBGGRR
    (we use AA=00 for fully opaque). Returns string like '00BBGGRR' suitable for &H... in force_style.
    Supports simple names 'white' and 'black' otherwise accepts 6-hex RGB.
    """
    if not color:
        color = "FFFFFF"
    color = color.strip().lower()
    name_map = {
        "white": "ffffff",
        "black": "000000",
        "red": "ff0000",
        "green": "00ff00",
        "blue": "0000ff",
    }
    if color in name_map:
        color = name_map[color]
    if color.startswith("#"):
        color = color[1:]
    if len(color) != 6 or not all(c in "0123456789abcdef" for c in color):
        # fallback white
        color = "ffffff"
    # ASS uses AABBGGRR; use AA=00 (opaque)
    rr = color[0:2]
    gg = color[2:4]
    bb = color[4:6]
    return f"00{bb}{gg}{rr}"  # string of 8 hex chars


def burn_in_subtitles_with_ffmpeg(
    ffmpeg_exe: str,
    input_video: str,
    srt_path: str,
    output_video: str,
    font: str = "Arial",
    fontsize: int = 48,
    fontcolor: str = "white",
    outline_color: str = "black",
    outline: int = 2,
    y_offset_px: int = 80,
    preserve_audio: bool = True,
) -> str:
    """
    Burn an SRT into a video using ffmpeg subtitles filter with a readable style.
    - ffmpeg_exe: path to ffmpeg binary
    - input_video: path to source video
    - srt_path: path to SRT file
    - output_video: final path to write (will be written to a temp file then moved)
    - font / fontsize / fontcolor / outline_color / outline: visual style
    - y_offset_px: vertical offset from bottom (in pixels)
    Returns the path to the written output_video (same as output_video param) on success.
    Raises RuntimeError on ffmpeg failure.
    """
    assert os.path.exists(input_video), f"Input video not found: {input_video}"
    assert os.path.exists(srt_path), f"SRT file not found: {srt_path}"

    # Build force_style argument (ASS style string). BorderStyle=1 uses outline+shadow per ASS.
    primary = _guess_color_hex(fontcolor)
    outline_c = _guess_color_hex(outline_color)
    # Escape single quotes in font name for ffmpeg filter argument
    font_escaped = font.replace("'", "\\'")
    force_style = (
        f"FontName={font_escaped},Fontsize={fontsize},"
        f"PrimaryColour=&H{primary}&,OutlineColour=&H{outline_c}&,"
        f"BorderStyle=1,Outline={outline},Shadow=0"
    )

    # Build subtitles filter carefully — for Windows paths we must escape backslashes and colons inside the filter.
    # ffmpeg expects the path as file path (it will internally hand over to libavfilter). We will quote the srt path.
    # To be safe, use the subtitles filter with the full path quoted.
    # On some platforms, ffmpeg requires utf-8 path handling — provide as-is.
    # Construct filter expression:
    # e.g. subtitles='/full/path/file.srt':force_style='...'
    srt_escaped = srt_path.replace("'", r"\'")
    vf_expr = f"subtitles='{srt_escaped}':force_style='{force_style}'"

    # Prepare temp output and run ffmpeg
    out_tmp = os.path.splitext(output_video)[0] + ".withsub.tmp.mp4"
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_video,
        "-vf",
        vf_expr,
    ]

    # Preserve audio stream (copy) if requested, otherwise re-encode in default aac
    if preserve_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate]

    # Re-encode video to ensure subtitles are burned-in and compat (safe settings)
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(video_crf),
        "-pix_fmt",
        pix_fmt,
        out_tmp,
    ]

    logger.info("Burning subtitles into video (temporary): %s", out_tmp)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("ffmpeg subtitles command failed. STDERR:\n%s", res.stderr)
        # Clean partial file if created
        try:
            if os.path.exists(out_tmp):
                os.remove(out_tmp)
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg subtitles embedding failed (rc={res.returncode})")

    # Move temp over target (atomic-ish)
    try:
        shutil.move(out_tmp, output_video)
        logger.info("Subtitle-burned video written: %s", output_video)
    except Exception as e:
        raise RuntimeError(f"Could not move subtitle output into place: {e}")

    return output_video


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

        # --- NEW: detect matching .srt for chosen audio and burn into the final video if present ---
        try:
            # Look for a .srt with the same basename as the chosen audio file
            srt_candidate = os.path.splitext(chosen_audio)[0] + ".srt"
            if os.path.exists(srt_candidate):
                logger.info(
                    "Found subtitle file for audio: %s", os.path.basename(srt_candidate)
                )
                # create a temp backup path (so we can replace atomically)
                output_with_subs = (
                    output_video  # we'll overwrite this path with subtitle-burned video
                )
                burn_in_subtitles_with_ffmpeg(
                    ffmpeg_exe=ffmpeg_path,
                    input_video=output_video,
                    srt_path=srt_candidate,
                    output_video=output_with_subs,
                    font="Arial",
                    fontsize=48,  # medium readable size for 1080p target
                    fontcolor="white",
                    outline_color="black",
                    outline=2,
                    y_offset_px=80,
                    preserve_audio=True,
                )
            else:
                logger.debug(
                    "No matching .srt found at: %s (skipping subtitle burn-in).",
                    srt_candidate,
                )
        except Exception as exc:
            logger.warning(
                "Subtitle burn-in step failed (continuing without burned subtitles): %s",
                exc,
            )
        # --- END subtitle step ---

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
