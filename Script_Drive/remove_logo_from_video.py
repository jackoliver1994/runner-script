#!/usr/bin/env python3
"""
Improved remove_logo_from_video.py
- Keeps _all original features_ (inpainting with OpenCV, scaling, audio merge with ffmpeg).
- Adds: ffmpeg presence check + best-effort static download, retries/exponential backoff for external calls,
        upload to transfer.sh (no API key; commonly used for large files).
- Usage: python remove_logo_from_video.py
"""

import sys
import cv2
import numpy as np
import subprocess
import os
import shutil
import time
import platform
import stat
import tarfile
import zipfile
import urllib.request
import requests  # pip install requests

# -------------------------------------------------------------
# Configuration / globals (kept names similar to original)
# -------------------------------------------------------------
current_directory = os.getcwd()

# ---------- FFmpeg paths (user-provided) ----------
ffmpeg_root_path = os.path.join(current_directory, "ffmpeg")
# cross-platform: default expected paths (may be updated by ensure_ffmpeg)
if sys.platform.startswith("win"):
    ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg.exe")
    ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe.exe")
else:
    ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg")
    ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe")

# Retry / backoff configuration
DEFAULT_RETRIES = 4
BACKOFF_FACTOR = 1.5  # exponential backoff multiplier


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def run_cmd(cmd, retries=DEFAULT_RETRIES, check_returncode=True):
    """Run a subprocess command with retries and exponential backoff. Returns CompletedProcess."""
    attempt = 0
    while True:
        attempt += 1
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if not check_returncode or result.returncode == 0:
                return result
            else:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, output=result.stdout, stderr=result.stderr
                )
        except Exception as e:
            if attempt >= retries:
                print(f"❌ Command failed after {attempt} attempts: {' '.join(cmd)}")
                # raise last error or return result if present
                if isinstance(e, subprocess.CalledProcessError):
                    return e
                raise
            sleep_time = BACKOFF_FACTOR ** (attempt - 1)
            print(
                f"⚠️ Command failed (attempt {attempt}/{retries}). Retrying in {sleep_time:.1f}s... Error: {e}"
            )
            time.sleep(sleep_time)


def is_executable(path):
    return os.path.isfile(path) and os.access(path, os.X_OK)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_executable(path):
    try:
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)
    except Exception:
        pass


# -------------------------------------------------------------
# FFmpeg presence / best-effort install
# -------------------------------------------------------------
def ensure_ffmpeg(verbose=True):
    """
    Ensure ffmpeg and ffprobe exist.
    1) Prefer system ffmpeg if on PATH.
    2) Otherwise, try to download static builds into ./ffmpeg/bin (best-effort).
    3) If automated download fails, print clear manual instructions.
    NOTE: Automatic download is best-effort because build URLs change over time.
    """
    global ffmpeg_path, ffprobe_path

    # 1) Prefer system binaries
    system_ffmpeg = shutil.which("ffmpeg")
    system_ffprobe = shutil.which("ffprobe")
    if system_ffmpeg and system_ffprobe:
        if verbose:
            print(f"✅ Found ffmpeg in PATH: {system_ffmpeg}")
        ffmpeg_path = system_ffmpeg
        ffprobe_path = system_ffprobe
        return True

    # 2) Check if bundled/previously downloaded binaries exist already
    if is_executable(ffmpeg_path) and is_executable(ffprobe_path):
        if verbose:
            print(f"✅ Found bundled ffmpeg: {ffmpeg_path}")
        return True

    # 3) Try to download a static build depending on platform (best-effort)
    system = platform.system().lower()
    ensure_dir(os.path.join(ffmpeg_root_path, "bin"))

    try:
        if system == "linux":
            # John Van Sickle static builds are a commonly used choice for Linux.
            # We attempt to fetch his amd64 static tar.xz and extract ffmpeg/ffprobe.
            # Source: https://johnvansickle.com/ffmpeg/
            url = (
                "https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz"
            )
            if verbose:
                print(
                    "ℹ️ Attempting to download static ffmpeg (linux) — this may take a while..."
                )
            archive_path = os.path.join(
                ffmpeg_root_path, "ffmpeg-git-amd64-static.tar.xz"
            )
            urllib.request.urlretrieve(url, archive_path)
            # extract ffmpeg and ffprobe from tar.xz
            with tarfile.open(archive_path, "r:xz") as tar:
                # look for the ffmpeg* bins and extract them
                members = tar.getmembers()
                for m in members:
                    if m.name.endswith("/ffmpeg") or m.name.endswith("/ffprobe"):
                        # extract and save into ./ffmpeg/bin/
                        target = os.path.join(
                            ffmpeg_root_path, "bin", os.path.basename(m.name)
                        )
                        with open(target, "wb") as out_f:
                            out_f.write(tar.extractfile(m).read())
                        set_executable(target)
            # cleanup archive
            try:
                os.remove(archive_path)
            except Exception:
                pass
            ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg")
            ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe")
            if verbose:
                print(f"✅ Downloaded ffmpeg to {ffmpeg_path}")
            return True

        elif system == "windows":
            # Try an easy-to-download build from gyan.dev (windows builds).
            # Note: gyan.dev often hosts builds at predictable names (ffmpeg-release-essentials.zip)
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            if verbose:
                print("ℹ️ Attempting to download static ffmpeg (windows)...")
            archive_path = os.path.join(ffmpeg_root_path, "ffmpeg-windows.zip")
            urllib.request.urlretrieve(url, archive_path)
            with zipfile.ZipFile(archive_path, "r") as zf:
                # extract ffmpeg.exe and ffprobe.exe to ./ffmpeg/bin/
                for name in zf.namelist():
                    base = os.path.basename(name)
                    if base.lower() in ("ffmpeg.exe", "ffprobe.exe"):
                        target = os.path.join(ffmpeg_root_path, "bin", base)
                        ensure_dir(os.path.dirname(target))
                        with open(target, "wb") as out_f:
                            out_f.write(zf.read(name))
                        set_executable(target)
            try:
                os.remove(archive_path)
            except Exception:
                pass
            ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg.exe")
            ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe.exe")
            if verbose:
                print(f"✅ Downloaded ffmpeg to {ffmpeg_path}")
            return True

        elif system == "darwin":
            # macOS: static builds exist but URLs/packaging vary; try ffmpeg.org downloads page as a hint.
            # Fallback to instructing user to install via brew or download manually.
            if verbose:
                print(
                    "⚠️ Automatic macOS download not implemented. Please install ffmpeg via Homebrew:"
                )
                print("    brew install ffmpeg")
            return False

        else:
            if verbose:
                print(f"⚠️ Unsupported platform for automated ffmpeg download: {system}")
            return False

    except Exception as e:
        print("⚠️ Automatic ffmpeg download/install failed:", e)
        print("Please install ffmpeg manually:")
        print(
            "  - Linux: follow https://johnvansickle.com/ffmpeg/ or your distro packages."
        )
        print(
            "  - Windows: get a build from https://www.gyan.dev/ffmpeg/ or https://github.com/BtbN/FFmpeg-Builds"
        )
        print("  - macOS: brew install ffmpeg")
        return False


# -------------------------------------------------------------
# Uploading: transfer.sh (no API key; supports large files)
# -------------------------------------------------------------
def upload_to_transfersh(file_path, retries=DEFAULT_RETRIES):
    """
    Upload a file to transfer.sh (no API key). Returns the file URL on success.
    Uses an HTTP PUT to https://transfer.sh/{filename}
    Common public instance: https://transfer.sh
    Note: Transfer.sh endpoints are community-run; availability/limits may change.
    See: the project repo and docs. :contentReference[oaicite:1]{index=1}
    """
    filename = os.path.basename(file_path)
    endpoint = f"https://transfer.sh/{filename}"
    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            with open(file_path, "rb") as fh:
                resp = requests.put(endpoint, data=fh, timeout=120)
            if resp.status_code in (200, 201):
                return resp.text.strip()
            else:
                print(
                    f"⚠️ transfer.sh upload failed (status {resp.status_code}): {resp.text}"
                )
        except Exception as e:
            print(f"⚠️ transfer.sh upload attempt {attempt} failed: {e}")
        # backoff
        time.sleep(BACKOFF_FACTOR ** (attempt - 1))
    # fallback: try temp.sh (another transfer service inspired by transfer.sh)
    try:
        print("ℹ️ Falling back to temp.sh for upload...")
        endpoint2 = f"https://temp.sh/{filename}"
        with open(file_path, "rb") as fh:
            resp2 = requests.put(endpoint2, data=fh, timeout=120)
        if resp2.status_code in (200, 201):
            return resp2.text.strip()
        else:
            print("⚠️ temp.sh fallback failed:", resp2.status_code, resp2.text)
    except Exception as e:
        print("⚠️ temp.sh fallback exception:", e)
    return None


# -------------------------------------------------------------
# Main processing function (original logic preserved & improved)
# -------------------------------------------------------------
def remove_logo_with_audio(
    video_path,
    logo_coords,
    output_path="video_no_logo.mp4",
    scale_factor=0.85,
    temp_path="temp_no_audio.mp4",
    inpaint_radius=3,
    ffmpeg_check_first=True,
):
    """
    Removes a logo from the video while keeping audio (original script behavior).
    Improvements:
     - calls ensure_ffmpeg at start (if ffmpeg_check_first)
     - uses run_cmd for ffmpeg merge with retries
     - attempts to upload final file to transfer.sh (no API key) and returns the URL (if requested separately)
    """
    # Optionally ensure ffmpeg is installed / available
    if ffmpeg_check_first:
        ensure_ffmpeg()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎥 Input resolution: {width}x{height}, FPS: {fps}, Frames: {total}")

    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)

    # Use mp4v (less CPU heavy than x264) for initial output (preserves your original choice)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (scaled_width, scaled_height))

    print(f"⚙️ Processing frames at {scaled_width}x{scaled_height}...")

    # Precompute mask once (original behavior preserved)
    mask = np.zeros((height, width), dtype=np.uint8)
    for r in logo_coords:
        mask[r["y"] : r["y"] + r["h"], r["x"] : r["x"] + r["w"]] = 255

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed
        if scale_factor != 1.0:
            frame = cv2.resize(
                frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
            )
            scaled_mask = cv2.resize(
                mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST
            )
        else:
            scaled_mask = mask

        # Inpaint to remove logo (keeps the same INPAINT_TELEA method)
        inpainted = cv2.inpaint(frame, scaled_mask, inpaint_radius, cv2.INPAINT_TELEA)
        out.write(inpainted)

        if frame_idx % 50 == 0:
            print(f"Frame {frame_idx}/{total}")
        frame_idx += 1

    cap.release()
    out.release()
    print("✅ Video frames processed successfully.")

    # Merge back original audio with medium-quality video compression (preserved from original)
    print("🎧 Merging audio (medium quality)...")

    merge_cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        temp_path,  # inpainted video
        "-i",
        video_path,  # original for audio
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "26",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-movflags",
        "+faststart",
        output_path,
    ]

    result = run_cmd(merge_cmd, retries=DEFAULT_RETRIES)
    if (
        isinstance(result, subprocess.CalledProcessError)
        or getattr(result, "returncode", 1) != 0
    ):
        # show ffmpeg stderr for debugging
        if hasattr(result, "stderr"):
            print("⚠️ FFmpeg merge failed:", result.stderr)
        else:
            print("⚠️ FFmpeg merge likely failed. See above.")
        # keep temp files for debugging
        return None
    else:
        print(f"🎬 Done! Output saved as: {output_path}")

    # Remove temp video safely
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            print("🧹 Temporary files cleaned up.")
        except Exception as e:
            print("⚠️ Could not remove temp file:", e)

    return output_path


# -------------------------------------------------------------
# Example usage / CLI-style main
# -------------------------------------------------------------
if __name__ == "__main__":
    # Paths (preserve your earlier example usage)
    video_path = os.path.join(current_directory, "image_response", "final_output.mp4")
    logo_regions = [{"x": 1139, "y": 1010, "w": 359, "h": 60}]
    final_output = os.path.join(
        current_directory, "image_response", "final_video_medium.mp4"
    )

    print(
        "ℹ️ Starting processing (will ensure ffmpeg and upload result to transfer.sh)..."
    )
    out = remove_logo_with_audio(
        video_path, logo_regions, final_output, scale_factor=0.85
    )

    if out:
        print(
            "ℹ️ Attempting to upload the result to transfer.sh (no API key required)..."
        )
        url = upload_to_transfersh(out)
        if url:
            print("🔗 Upload successful. Link:")
            print(url)
        else:
            print("⚠️ Upload failed. You can manually upload the file located at:", out)
