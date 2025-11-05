import sys
import cv2
import numpy as np
import subprocess
import os


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


def remove_logo_with_audio(video_path, logo_coords, output_path="video_no_logo.mp4"):
    """
    Removes a logo from the video while keeping (medium-quality) video + audio.
    Optimized for low-end systems (balanced performance and quality).
    """
    temp_path = "temp_no_audio.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎥 Input resolution: {width}x{height}, FPS: {fps}, Frames: {total}")

    # Scale down resolution slightly for faster low-system processing (optional)
    scale_factor = 0.85  # 85% of original size
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)

    # Use mp4v (less CPU heavy than x264) for initial output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (scaled_width, scaled_height))

    print(f"⚙️ Processing frames at {scaled_width}x{scaled_height}...")

    # Precompute mask once
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
            # Resize mask to same scale
            scaled_mask = cv2.resize(
                mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST
            )
        else:
            scaled_mask = mask

        # Inpaint to remove logo
        inpainted = cv2.inpaint(frame, scaled_mask, 3, cv2.INPAINT_TELEA)
        out.write(inpainted)

        # Light progress updates
        if frame_idx % 50 == 0:
            print(f"Frame {frame_idx}/{total}")
        frame_idx += 1

    cap.release()
    out.release()
    print("✅ Video frames processed successfully.")

    # Merge back original audio with medium-quality video compression
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
        "medium",  # slower = better compression, but not heavy
        "-crf",
        "26",  # 23 = high, 26 = medium quality (smaller file)
        "-c:a",
        "aac",  # re-encode to reduce size
        "-b:a",
        "128k",  # medium audio bitrate
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-movflags",
        "+faststart",
        output_path,
    ]

    # Run quietly, show success/failure
    result = subprocess.run(
        merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode == 0:
        print(f"🎬 Done! Output saved as: {output_path}")
    else:
        print("⚠️ FFmpeg merge failed:", result.stderr)

    # Remove temp video safely
    if os.path.exists(temp_path):
        os.remove(temp_path)
        print("🧹 Temporary files cleaned up.")


# Example usage
if __name__ == "__main__":
    video_path = os.path.join(current_directory, "image_response", "final_output.mp4")
    logo_regions = [{"x": 1139, "y": 1010, "w": 359, "h": 60}]
    remove_logo_with_audio(
        video_path,
        logo_regions,
        os.path.join(current_directory, "image_response", "final_video_medium.mp4"),
    )
