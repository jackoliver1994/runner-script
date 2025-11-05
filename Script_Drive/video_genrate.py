#!/usr/bin/env python3
"""
Ultimate Image-to-Video composer using FFmpeg + FFprobe.

- Always selects the best audio file from narration_folder (no explicit audio_file option).
- Keeps previous features: serial-first image ordering (supports 'promt' typo), natural sorting fallback,
  high-quality H.264 (CRF + preset), letterbox/pad to target resolution, even dimensions, per-image segment creation,
  concat demuxer, final audio encoding (AAC, resampled), robust ffprobe/ffmpeg error handling, temp cleanup.
- FIXED: safer final merge (no copy conflict), ensures compatible audio/video streams.
"""

import os
import re
import subprocess
import tempfile
import shutil
import glob
import sys

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
# -------------------------------------------------------------

# ---------- Basic validation ----------
for p in (ffmpeg_path, ffprobe_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required executable not found: {p}")

if not os.path.isdir(image_folder):
    raise FileNotFoundError(f"Image folder not found: {image_folder}")

if not os.path.isdir(narration_folder):
    raise FileNotFoundError(f"Narration folder not found: {narration_folder}")


# ---------- Helpers ----------
def run(cmd, check=True):
    """Run subprocess command and return CompletedProcess; prints stderr on error."""
    try:
        print("\n▶ Running:", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return res
    except subprocess.CalledProcessError as e:
        print("\n❌ FFmpeg command failed:")
        print("COMMAND:", " ".join(e.cmd))
        print("RETURN CODE:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise SystemExit("FFmpeg execution failed. See above for details.")


def probe_duration(path):
    """Return duration in seconds (float) for media file using ffprobe."""
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
        out = res.stdout.strip()
        if not out:
            return None
        return float(out)
    except Exception:
        return None


def find_best_audio(folder):
    """Find best audio (longest duration, prefer .wav)."""
    candidates = []
    for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac"):
        candidates.extend(glob.glob(os.path.join(folder, ext)))
    if not candidates:
        return None
    best, best_dur = None, -1
    for path in candidates:
        dur = probe_duration(path)
        if dur is None:
            print(f"⚠ Skipping (unreadable): {os.path.basename(path)}")
            continue
        score = dur + (0.1 if path.lower().endswith(".wav") else 0)
        print(
            f"Audio candidate: {os.path.basename(path)} — {dur:.2f}s (score {score:.2f})"
        )
        if score > best_dur:
            best, best_dur = path, score
    return best


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split("(\d+)", s)]


def collect_images_serial_first(folder):
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


def ensure_even_dimensions_filter(width, height):
    return (
        "scale='trunc(iw*min({W}/iw\\,{H}/ih)/2)*2':"
        "'trunc(ih*min({W}/iw\\,{H}/ih)/2)*2',"
        "pad={W}:{H}:(ow-iw)/2:(oh-ih)/2"
    ).format(W=width, H=height)


# ---------- Main flow ----------
image_files = collect_images_serial_first(image_folder)
if not image_files:
    raise SystemExit("No images found.")

num_images = len(image_files)
print(f"🖼 Found {num_images} images (serial-first).")
for i, p in enumerate(image_files[:10], 1):
    print(f"  {i:2d}. {os.path.basename(p)}")

chosen_audio = find_best_audio(narration_folder)
if not chosen_audio:
    raise SystemExit("No audio found in narration_folder (.wav/.mp3).")

print(f"\n🎵 Using audio: {chosen_audio}")
audio_duration = probe_duration(chosen_audio)
if audio_duration is None:
    raise SystemExit("Could not parse audio duration.")
print(f"Audio duration: {audio_duration:.3f}s")

per_image_dur = audio_duration / num_images
print(f"Each image duration: {per_image_dur:.3f}s")

tmpdir = tempfile.mkdtemp(prefix="img_vid_segments_")
segment_files = []

try:
    vf_filter = ensure_even_dimensions_filter(*target_resolution)
    print("\n🎞 Creating segments...")

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
        print(f"Segment {idx+1}/{num_images}: {os.path.basename(seg_path)}")
        run(cmd)
        segment_files.append(seg_path)

    # concat list
    concat_list = os.path.join(tmpdir, "segments.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for seg in segment_files:
            f.write(f"file '{seg}'\n")

    temp_concat = os.path.join(tmpdir, "temp_video.mp4")
    print("\n🔗 Concatenating segments...")
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
        print(f"Concat video duration: {concat_duration:.3f}s")

    # --- FIXED MERGE SECTION ---
    print("\n🎧 Merging audio + video (re-encode safe mode)...")
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
        "18",
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

    print(f"\n✅ Success! Output video created:\n{output_video}")
    print(f"Expected duration (≈ audio): {audio_duration:.2f}s")
    try:
        print(
            "Output size: {:.2f} MB".format(os.path.getsize(output_video) / (1024**2))
        )
    except Exception:
        pass

finally:
    try:
        shutil.rmtree(tmpdir)
        print(f"\n🧹 Cleaned temporary directory: {tmpdir}")
    except Exception:
        print(f"\n⚠ Could not remove temporary directory: {tmpdir}")

# End of script
