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

DEFAULT_RETRIES = 4
BACKOFF_FACTOR = 1.5
current_directory = os.getcwd()
# ---------- FFmpeg paths (user-provided) ----------
ffmpeg_root_path = os.path.join(current_directory, "ffmpeg")
# cross-platform: use .exe on Windows if present
if sys.platform.startswith("win"):
    ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg.exe")
    ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe.exe")
else:
    ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg")
    ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe")


# ----------------------------
# Utilities
# ----------------------------
def run_cmd(cmd, retries=DEFAULT_RETRIES, check_returncode=True):
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


# ----------------------------
# ffmpeg downloader (best-effort)
# ----------------------------
def ensure_ffmpeg(verbose=True):
    global ffmpeg_path, ffprobe_path
    system_ffmpeg = shutil.which("ffmpeg")
    system_ffprobe = shutil.which("ffprobe")
    if system_ffmpeg and system_ffprobe:
        if verbose:
            print(f"✅ Found ffmpeg in PATH: {system_ffmpeg}")
        ffmpeg_path = system_ffmpeg
        ffprobe_path = system_ffprobe
        return True

    if is_executable(ffmpeg_path) and is_executable(ffprobe_path):
        if verbose:
            print(f"✅ Found bundled ffmpeg: {ffmpeg_path}")
        return True

    system = platform.system().lower()
    ensure_dir(os.path.join(ffmpeg_root_path, "bin"))
    try:
        if system == "linux":
            url = (
                "https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz"
            )
            print("ℹ️ Downloading static ffmpeg (linux)...")
            archive_path = os.path.join(ffmpeg_root_path, "ffmpeg-static.tar.xz")
            urllib.request.urlretrieve(url, archive_path)
            with tarfile.open(archive_path, "r:xz") as tar:
                for m in tar.getmembers():
                    if m.name.endswith("/ffmpeg") or m.name.endswith("/ffprobe"):
                        target = os.path.join(
                            ffmpeg_root_path, "bin", os.path.basename(m.name)
                        )
                        with open(target, "wb") as out_f:
                            out_f.write(tar.extractfile(m).read())
                        set_executable(target)
            try:
                os.remove(archive_path)
            except Exception:
                pass
            ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg")
            ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe")
            print(f"✅ Downloaded ffmpeg -> {ffmpeg_path}")
            return True
        elif system == "windows":
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            print("ℹ️ Downloading static ffmpeg (windows)...")
            archive_path = os.path.join(ffmpeg_root_path, "ffmpeg-windows.zip")
            urllib.request.urlretrieve(url, archive_path)
            with zipfile.ZipFile(archive_path, "r") as zf:
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
            print(f"✅ Downloaded ffmpeg -> {ffmpeg_path}")
            return True
        elif system == "darwin":
            print("⚠️ On macOS please install ffmpeg via Homebrew: brew install ffmpeg")
            return False
        else:
            print(f"⚠️ Unsupported platform: {system}")
            return False
    except Exception as e:
        print("⚠️ ffmpeg download failed:", e)
        return False


# ----------------------------
# Video processing (unchanged)
# ----------------------------
def remove_logo_with_audio(
    video_path,
    logo_coords,
    output_path="video_no_logo.mp4",
    scale_factor=0.85,
    temp_path="temp_no_audio.mp4",
    inpaint_radius=3,
    ffmpeg_check_first=True,
):
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (scaled_width, scaled_height))
    print(f"⚙️ Processing frames at {scaled_width}x{scaled_height}...")
    mask = np.zeros((height, width), dtype=np.uint8)
    for r in logo_coords:
        mask[r["y"] : r["y"] + r["h"], r["x"] : r["x"] + r["w"]] = 255
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale_factor != 1.0:
            frame = cv2.resize(
                frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
            )
            scaled_mask = cv2.resize(
                mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST
            )
        else:
            scaled_mask = mask
        inpainted = cv2.inpaint(frame, scaled_mask, inpaint_radius, cv2.INPAINT_TELEA)
        out.write(inpainted)
        if frame_idx % 50 == 0:
            print(f"Frame {frame_idx}/{total}")
        frame_idx += 1
    cap.release()
    out.release()
    print("✅ Video frames processed successfully.")
    print("🎧 Merging audio (medium quality)...")
    merge_cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        temp_path,
        "-i",
        video_path,
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
        if hasattr(result, "stderr"):
            print("⚠️ FFmpeg merge failed:", result.stderr)
        else:
            print("⚠️ FFmpeg merge likely failed. See above.")
        return None
    else:
        print(f"🎬 Done! Output saved as: {output_path}")
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            print("🧹 Temporary files cleaned up.")
        except Exception as e:
            print("⚠️ Could not remove temp file:", e)
    return output_path


# Example usage
if __name__ == "__main__":
    video_path = os.path.join(current_directory, "image_response", "final_output.mp4")
    logo_regions = [{"x": 1139, "y": 1010, "w": 359, "h": 60}]
    remove_logo_with_audio(
        video_path,
        logo_regions,
        os.path.join(current_directory, "image_response", "final_video_medium.mp4"),
    )
