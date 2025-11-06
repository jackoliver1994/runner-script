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
    """Return the first .txt file path found in root, or None.

    Behavior:
    - If `root` is a file and ends with .txt, return it.
    - Otherwise check top-level entries in `root` (same as original).
    - If none found at top-level, do a shallow recursive search (os.walk) and return the first .txt found.
    """
    try:
        # If user passed a direct file path to a .txt, accept it
        if os.path.isfile(root) and root.lower().endswith(".txt"):
            return os.path.abspath(root)

        # If root is a directory, check top-level entries first (preserve original behavior)
        if os.path.isdir(root):
            for entry in os.listdir(root):
                if entry.lower().endswith(".txt"):
                    return os.path.join(root, entry)

            # fallback: a shallow recursive search to find the first text file in subfolders
            for dirpath, dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith(".txt"):
                        return os.path.join(dirpath, fn)
    except FileNotFoundError:
        return None
    except Exception:
        # Be conservative: don't raise in this helper, return None like original
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
    if seconds or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    if show_ms and milliseconds:
        parts.append(f"{milliseconds} ms")

    human = _nice_join(parts)

    # Format start and end times as readable stamps rather than raw floats
    try:
        start_stamp = datetime.fromtimestamp(start_time).strftime(
            "%Y-%m-%d %I:%M:%S %p"
        )
        end_stamp = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %I:%M:%S %p")
    except Exception:
        # Fallback if timestamps are malformed
        start_stamp = str(start_time)
        end_stamp = str(end_time)

    current_time = datetime.now().strftime("%I:%M:%S %p")  # 12-hour with leading zero
    print(
        f"[{start_stamp} -> {end_stamp} | now {current_time}] Execution completed in {human}."
    )


def extract_prompts(text):
    """Extract prompts inside square brackets [ ... ]."""
    return [p.strip() for p in re.findall(r"\[(.*?)\]", text, re.DOTALL) if p.strip()]


def generate_image_bytes(prompt, attempts=RETRY_ATTEMPTS, start_timeout=START_TIMEOUT):
    """
    Return bytes of generated image or None on failure. Applies increasing timeouts per attempt.
    - Uses urllib.parse.quote_plus to safely encode prompts into the URL.
    - Uses a requests.Session for connection reuse.
    - Retries with increasing timeout (start_timeout, 2*start_timeout, ...).
    - Logs HTTP status and common errors; returns None after attempts exhausted.
    """
    from urllib.parse import quote_plus
    from requests.exceptions import Timeout, RequestException

    timeout_val = start_timeout
    # Encode prompt safely for inclusion in a URL path segment
    encoded = quote_plus(prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{encoded}"

    # Use a Session for connection reuse
    session = requests.Session()
    try:
        for attempt in range(1, attempts + 1):
            try:
                print(
                    f"    → Attempt {attempt}/{attempts} (timeout={timeout_val}s)...",
                    end=" ",
                    flush=True,
                )
                # Explicitly pass timeout (seconds)
                resp = session.get(url, timeout=timeout_val)
                status = resp.status_code
                if status == 200 and resp.content:
                    print("OK")
                    return resp.content
                else:
                    # For non-200, print status and continue (some status codes like 429 may benefit from retry)
                    print(f"HTTP {status}")
            except Timeout:
                print(f"Timeout after {timeout_val}s")
            except RequestException as e:
                # Requests base exception for network/connection related errors
                print(f"Request error: {e}")
            except Exception as e:
                # Catch-all to avoid crashing the loop; keep original behavior of continuing
                print(f"Error: {e}")

            # increment timeout for next attempt and wait a small delay
            timeout_val += start_timeout
            time.sleep(RETRY_DELAY)
    finally:
        try:
            session.close()
        except Exception:
            pass

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
