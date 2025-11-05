#!/usr/bin/env python3
"""
Single-text-file Pollinations generator (cleaned-only output)

- Set PROMPT_FOLDER to the folder where your .txt prompt file is located (default).
- Or pass a folder path as the first command-line argument to override.
- Extracts prompts inside [ ... ], generates images sequentially (retries with increasing timeouts),
  and saves the generated image as-is (no watermark cleaning).
"""
import os
import re
import time
import io
import sys
import requests
from datetime import datetime
from PIL import Image

current_directory = os.getcwd()
PROMPT_FOLDER = os.path.join(
    current_directory, "image_response"
)  # <--- set the folder containing the .txt file here
RETRY_ATTEMPTS = 10
START_TIMEOUT = 100  # seconds; will be 100, 200, 300 ... each retry
RETRY_DELAY = 1  # seconds between attempts
OUTPUT_QUALITY = 100  # saved JPEG quality
# ----------------------------


def find_first_txt(root):
    """Return the first .txt file path found in root, or None."""
    try:
        for entry in os.listdir(root):
            if entry.lower().endswith(".txt"):
                return os.path.join(root, entry)
    except FileNotFoundError:
        return None
    return None


def _nice_join(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def log_execution_time(
    start_time: float, end_time: float, show_ms: bool = False
) -> None:
    """
    Print current time (12-hour) and elapsed time as human-readable hours/minutes/seconds.
    :param start_time: float, start timestamp (time.time())
    :param end_time: float, end timestamp (time.time())
    :param show_ms: whether to include leftover milliseconds (default False)
    """
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
    # show seconds if nonzero or if no other part exists (so "0 seconds" is avoided unless elapsed < 1s)
    if seconds or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    if show_ms and milliseconds:
        parts.append(f"{milliseconds} ms")

    human = _nice_join(parts)
    current_time = datetime.now().strftime("%I:%M:%S %p")  # 12-hour with leading zero
    print(
        f"[{start_time} to {end_time} now {current_time}] Execution completed in {human}."
    )


def extract_prompts(text):
    """Extract prompts inside square brackets [ ... ]."""
    return [p.strip() for p in re.findall(r"\[(.*?)\]", text, re.DOTALL) if p.strip()]


def generate_image_bytes(prompt, attempts=RETRY_ATTEMPTS, start_timeout=START_TIMEOUT):
    """Return bytes of generated image or None on failure. Applies increasing timeouts per attempt."""
    timeout_val = start_timeout
    for attempt in range(1, attempts + 1):
        try:
            print(
                f"    → Attempt {attempt}/{attempts} (timeout={timeout_val}s)...",
                end=" ",
                flush=True,
            )
            url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
            resp = requests.get(url, timeout=timeout_val)
            if resp.status_code == 200 and resp.content:
                print("OK")
                return resp.content
            else:
                print(f"HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            print(f"Timeout after {timeout_val}s")
        except Exception as e:
            print(f"Error: {e}")
        timeout_val += start_timeout
        time.sleep(RETRY_DELAY)
    return None


def main(root_folder):
    txt_path = find_first_txt(root_folder)
    if not txt_path:
        print("No .txt file found in folder:", root_folder)
        return

    print("Using prompts file:", txt_path)
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    prompts = extract_prompts(content)
    if not prompts:
        print("No bracketed prompts found in the file.")
        return

    base = os.path.splitext(os.path.basename(txt_path))[0]
    print(f"Found {len(prompts)} prompts — starting sequential generation.\n")

    for i, prompt in enumerate(prompts, start=1):
        print("=" * 80)
        print(
            f"Prompt #{i}/{len(prompts)} preview: {prompt[:160]}{'...' if len(prompt) > 160 else ''}"
        )
        output_name = f"{base}_prompt_{i}.jpg"
        output_path = os.path.join(root_folder, output_name)

        if os.path.exists(output_path):
            print(f"  ⏭️ Output exists, skipping: {output_name}\n")
            continue

        img_bytes = generate_image_bytes(prompt)
        if not img_bytes:
            print(
                f"  ❌ Failed to generate image for prompt #{i} after retries. Moving to next.\n"
            )
            continue

        # Open image from bytes (no raw file written)
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"  ⚠️ Failed to open generated image bytes: {e}\n")
            continue

        # Save the original generated image (no cleaning)
        try:
            img.save(output_path, quality=OUTPUT_QUALITY)
            print(f"  ✅ Saved generated image: {output_name}\n")
        except Exception as e:
            print(f"  ⚠️ Failed to save image: {e}\n")

    print("All prompts processed.")


if __name__ == "__main__":
    # Use command line arg if provided, otherwise use PROMPT_FOLDER from config
    start = time.time()
    if len(sys.argv) > 1:
        root = os.path.abspath(sys.argv[1])
    else:
        root = os.path.abspath(PROMPT_FOLDER)

    print("Root folder:", root)
    main(root)
    end = time.time()
    log_execution_time(start, end)
