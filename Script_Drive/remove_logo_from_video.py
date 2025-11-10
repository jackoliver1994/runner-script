#!/usr/bin/env python3
"""
remove_logo_from_video.py

- Preserves all existing functionality (frame inpainting, ffmpeg checks/download, audio merge).
- Adds a method to decrypt `user.json.enc` using `user_key.txt`. Supports:
    * Fernet (base64 key -> cryptography.Fernet)
    * OpenSSL salted format (header "Salted__") using password in user_key.txt (EVP_BytesToKey MD5 derivation)
    * Raw hex key with IV prepended in the encrypted file (16-byte IV + ciphertext)
- After decrypting, writes `user.json` to current directory (overwrites if exists).
- Uses client_secrets.json for YouTube OAuth as before.

Requirements (pip):
pip install opencv-python numpy requests google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client cryptography pycryptodome
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
import re
import ast
import json
import base64
import hashlib

# Optional dependency — many systems have requests; used earlier for ffmpeg download attempts
try:
    import requests
except Exception:
    requests = None

# Crypto libs for decryption fallback
try:
    from cryptography.fernet import Fernet
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad

    CRYPTO_LIBS_AVAILABLE = True
except Exception:
    CRYPTO_LIBS_AVAILABLE = False

# Google API imports (for YouTube upload)
try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import Request
    import google.oauth2.credentials
    import pickle

    GOOGLE_LIBS_AVAILABLE = True
except Exception:
    GOOGLE_LIBS_AVAILABLE = False

# -------------------------------------------------------------
# Config / defaults
# -------------------------------------------------------------
current_directory = os.getcwd()
METADATA_FILE = os.path.join(
    current_directory, "youtube_response", "youtube_metadata.txt"
)  # AI output saved here
client_secrets_file = os.path.join(current_directory, "client_secrets.json")
token_file = os.path.join(current_directory, "token.pickle")

# encrypted user json and key
encrypted_user_json = os.path.join(current_directory, "user.json.enc")
user_key_file = os.path.join(current_directory, "user_key.txt")
decrypted_user_json = os.path.join(current_directory, "client_secrets.json")

ffmpeg_root_path = os.path.join(current_directory, "ffmpeg")
if sys.platform.startswith("win"):
    ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg.exe")
    ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe.exe")
else:
    ffmpeg_path = os.path.join(ffmpeg_root_path, "bin", "ffmpeg")
    ffprobe_path = os.path.join(ffmpeg_root_path, "bin", "ffprobe")

DEFAULT_RETRIES = 4
BACKOFF_FACTOR = 1.5

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# Decrypt user.json.enc (new)
# -------------------------------------------------------------
def _evp_bytes_to_key(password: bytes, salt: bytes, key_len: int, iv_len: int):
    """
    EVP_BytesToKey using MD5 — compatible with OpenSSL `enc` salted format.
    Returns (key, iv)
    """
    dtot = b""
    prev = b""
    while len(dtot) < (key_len + iv_len):
        m = hashlib.md5(prev + password + salt).digest()
        dtot += m
        prev = m
    key = dtot[:key_len]
    iv = dtot[key_len : key_len + iv_len]
    return key, iv


def decrypt_user_json_enc(
    enc_path=encrypted_user_json, key_path=user_key_file, out_path=decrypted_user_json
):
    """
    Attempts to decrypt enc_path using key in key_path.
    Supports:
      - Fernet (key is base64 urlsafe and file is Fernet ciphertext)
      - OpenSSL salted format (header "Salted__" + 8 byte salt) with password in key_path
      - Raw hex key where enc file is: IV (16 bytes) + ciphertext => AES-256-CBC
    Writes plaintext to out_path on success and returns True.
    Raises RuntimeError on failure or if required crypto libs are missing.
    """
    if not os.path.exists(enc_path):
        print(f"ℹ️ Encrypted file not found at {enc_path}; skipping decryption.")
        return False

    if not os.path.exists(key_path):
        raise FileNotFoundError(f"user key file not found at {key_path}")

    if not CRYPTO_LIBS_AVAILABLE:
        raise RuntimeError(
            "Required crypto libraries are not available. Install with:\n"
            "  pip install cryptography pycryptodome"
        )

    with open(key_path, "r", encoding="utf-8") as f:
        key_text = f.read().strip()

    with open(enc_path, "rb") as f:
        enc_bytes = f.read()

    # 1) Try Fernet
    try:
        # Fernet keys are 32 urlsafe-base64 bytes -> length 44 characters when encoded
        # attempt to base64-decode and use as fernet key
        maybe_b = key_text.encode("utf-8")
        # Accept raw base64 or URL-safe
        try:
            # If key contains whitespace/newlines, strip them
            bkey = base64.urlsafe_b64decode(maybe_b + b"===")
        except Exception:
            bkey = None

        if bkey and len(bkey) == 32:
            # reconstruct URL-safe base64 for Fernet constructor
            fernet_key = base64.urlsafe_b64encode(bkey)
            try:
                f = Fernet(fernet_key)
                plaintext = f.decrypt(enc_bytes)
                with open(out_path, "wb") as outf:
                    outf.write(plaintext)
                print(
                    f"✅ Decrypted {enc_path} using Fernet key -> saved to {out_path}"
                )
                return True
            except Exception:
                # fallthrough to other attempts
                pass
    except Exception:
        pass

    # 2) Try OpenSSL salted format (header "Salted__")
    try:
        if enc_bytes.startswith(b"Salted__"):
            salt = enc_bytes[8:16]
            ciphertext = enc_bytes[16:]
            password = key_text.encode("utf-8")
            key, iv = _evp_bytes_to_key(password, salt, 32, 16)  # AES-256-CBC
            cipher = AES.new(key, AES.MODE_CBC, iv)
            plaintext = cipher.decrypt(ciphertext)
            try:
                plaintext = unpad(plaintext, AES.block_size)
            except ValueError:
                # padding error -> try to still write raw
                raise
            with open(out_path, "wb") as outf:
                outf.write(plaintext)
            print(
                f"✅ Decrypted {enc_path} using OpenSSL-salted format password -> saved to {out_path}"
            )
            return True
    except Exception as e:
        # continue to other attempts
        pass

    # 3) Try raw hex key with IV prepended in file (IV=first 16 bytes)
    try:
        # if key_text looks like hex
        hex_clean = key_text.strip().lower()
        if all(c in "0123456789abcdef" for c in hex_clean) and len(hex_clean) in (
            32,
            64,
        ):
            key_bytes = bytes.fromhex(hex_clean)
            # ensure key length of 16 or 32
            if len(key_bytes) not in (16, 32):
                # attempt to zero-pad/truncate
                if len(key_bytes) < 32:
                    key_bytes = key_bytes.ljust(32, b"\0")
                else:
                    key_bytes = key_bytes[:32]
            if len(enc_bytes) > 16:
                iv = enc_bytes[:16]
                ciphertext = enc_bytes[16:]
                cipher = AES.new(key_bytes, AES.MODE_CBC, iv)
                plaintext = cipher.decrypt(ciphertext)
                try:
                    plaintext = unpad(plaintext, AES.block_size)
                except ValueError:
                    # padding error -> still write raw plaintext attempt
                    raise
                with open(out_path, "wb") as outf:
                    outf.write(plaintext)
                print(
                    f"✅ Decrypted {enc_path} using raw hex key + IV (IV taken from file) -> saved to {out_path}"
                )
                return True
    except Exception:
        pass

    # If we reach here, decryption attempts failed
    raise RuntimeError(
        "Failed to decrypt the file. Supported key formats:\n"
        " - Fernet base64 key (cryptography.Fernet)\n"
        " - OpenSSL salted format (file starts with 'Salted__') using the key file as password\n"
        " - Raw hex key (16 or 32 byte hex) where file contains IV(16 bytes) + ciphertext\n\n"
        "Ensure the correct format and key are present in user_key.txt."
    )


# -------------------------------------------------------------
# FFmpeg presence / best-effort install (unchanged)
# -------------------------------------------------------------
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
            if verbose:
                print(
                    "ℹ️ Attempting to download static ffmpeg (linux) — this may take a while..."
                )
            archive_path = os.path.join(
                ffmpeg_root_path, "ffmpeg-git-amd64-static.tar.xz"
            )
            urllib.request.urlretrieve(url, archive_path)
            with tarfile.open(archive_path, "r:xz") as tar:
                members = tar.getmembers()
                for m in members:
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
            if verbose:
                print(f"✅ Downloaded ffmpeg to {ffmpeg_path}")
            return True

        elif system == "windows":
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            if verbose:
                print("ℹ️ Attempting to download static ffmpeg (windows)...")
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
            if verbose:
                print(f"✅ Downloaded ffmpeg to {ffmpeg_path}")
            return True

        elif system == "darwin":
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
# Metadata parsing (reads bracketed fields from file)
# -------------------------------------------------------------
def parse_bracket_field(text, label):
    """
    Finds the first occurrence of: label [ ... ]  (case-insensitive)
    Returns inner text or None. DOTALL allowed.
    """
    # allow label with optional whitespace and a bracket
    pattern = rf"(?i){re.escape(label)}\s*\[\s*(.*?)\s*\]"
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def read_metadata_from_file(path=METADATA_FILE):
    """
    Parses youtube metadata from the file. Expected fields (case-insensitive):
      title [ ... ]
      description [ ... ]    <- can be multi-line
      tags [ ... ]           <- JSON-like list OR comma-separated
      hashtags [ ... ]       <- comma-separated or JSON list
      categoryId [ ... ]     <- number
      CTA [ ... ]            <- multi-line or '||' separated
      thumbnail_texts [ ... ]<- JSON-like list
    Returns a dict with fields (strings / lists) and defaults if not found.
    """
    defaults = {
        "title": None,
        "description": None,
        "tags": [],
        "hashtags": [],
        "categoryId": None,
        "CTA": [],
        "thumbnail_texts": [],
    }

    if not os.path.exists(path):
        print(f"⚠️ Metadata file not found at: {path}. Using defaults / manual values.")
        return defaults

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse each field
    title = parse_bracket_field(content, "title")
    description = parse_bracket_field(content, "description")
    tags_raw = parse_bracket_field(content, "tags")
    hashtags_raw = parse_bracket_field(content, "hashtags")
    category_raw = parse_bracket_field(content, "categoryId")
    cta_raw = parse_bracket_field(content, "CTA")
    thumb_raw = parse_bracket_field(content, "thumbnail_texts")

    meta = {}

    meta["title"] = title if title else None
    meta["description"] = description if description else None

    # parse tags: try ast.literal_eval -> list, else comma-split
    meta["tags"] = []
    if tags_raw:
        try:
            parsed = ast.literal_eval(tags_raw)
            if isinstance(parsed, (list, tuple)):
                meta["tags"] = [str(x).strip() for x in parsed]
            else:
                # fallback to comma split
                meta["tags"] = [
                    t.strip() for t in str(tags_raw).split(",") if t.strip()
                ]
        except Exception:
            meta["tags"] = [t.strip() for t in str(tags_raw).split(",") if t.strip()]

    # hashtags
    meta["hashtags"] = []
    if hashtags_raw:
        try:
            parsed = ast.literal_eval(hashtags_raw)
            if isinstance(parsed, (list, tuple)):
                meta["hashtags"] = [h.strip() for h in parsed]
            else:
                meta["hashtags"] = [
                    h.strip() for h in re.split(r"[,\n]+", hashtags_raw) if h.strip()
                ]
        except Exception:
            meta["hashtags"] = [
                h.strip() for h in re.split(r"[,\n]+", hashtags_raw) if h.strip()
            ]

    # categoryId
    meta["categoryId"] = None
    if category_raw:
        category_raw_clean = category_raw.strip()
        # try to extract integer
        m = re.search(r"(\d+)", category_raw_clean)
        if m:
            meta["categoryId"] = m.group(1)
        else:
            meta["categoryId"] = None

    # CTA
    meta["CTA"] = []
    if cta_raw:
        # split by '||' or newlines
        if "||" in cta_raw:
            meta["CTA"] = [c.strip() for c in cta_raw.split("||") if c.strip()]
        else:
            meta["CTA"] = [c.strip() for c in cta_raw.splitlines() if c.strip()]

    # thumbnails
    meta["thumbnail_texts"] = []
    if thumb_raw:
        try:
            parsed = ast.literal_eval(thumb_raw)
            if isinstance(parsed, (list, tuple)):
                meta["thumbnail_texts"] = [str(x).strip() for x in parsed]
            else:
                meta["thumbnail_texts"] = [
                    t.strip() for t in str(thumb_raw).split(",") if t.strip()
                ]
        except Exception:
            meta["thumbnail_texts"] = [
                t.strip() for t in str(thumb_raw).split(",") if t.strip()
            ]

    # apply defaults if None
    for k, v in defaults.items():
        if meta.get(k) is None:
            meta[k] = v

    return meta


# -------------------------------------------------------------
# YouTube upload (unchanged logic from previous script)
# -------------------------------------------------------------
def get_authenticated_service(
    client_secrets_file=client_secrets_file, token_file=token_file
):
    if not GOOGLE_LIBS_AVAILABLE:
        raise RuntimeError(
            "Google client libraries are not available. Install with:\n"
            "  pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        )

    creds = None
    if os.path.exists(token_file):
        try:
            with open(token_file, "rb") as f:
                creds = pickle.load(f)
        except Exception:
            creds = None

    if not creds or not getattr(creds, "valid", False):
        if (
            creds
            and getattr(creds, "expired", False)
            and getattr(creds, "refresh_token", None)
        ):
            try:
                creds.refresh(Request())
            except Exception as e:
                print("⚠️ Failed to refresh credentials:", e)
                creds = None
        else:
            if not os.path.exists(client_secrets_file):
                raise FileNotFoundError(
                    f"OAuth client secrets file '{client_secrets_file}' not found. "
                    "Create OAuth credentials in Google Cloud Console and download as JSON."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, YOUTUBE_SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(token_file, "wb") as f:
            pickle.dump(creds, f)

    service = build("youtube", "v3", credentials=creds, cache_discovery=False)
    return service


def upload_to_youtube_resumable(
    service,
    file_path,
    title="Uploaded video",
    description="Uploaded via script",
    tags=None,
    categoryId="22",
    privacyStatus="unlisted",
    resumable_chunk_size=256 * 1024,
):
    if tags is None:
        tags = []

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": categoryId,
        },
        "status": {
            "privacyStatus": privacyStatus,
        },
    }

    media = MediaFileUpload(file_path, chunksize=resumable_chunk_size, resumable=True)
    request = service.videos().insert(
        part="snippet,status", body=body, media_body=media
    )

    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                print(f"⏫ Upload progress: {progress}%")
        except Exception as e:
            raise

    return response.get("id"), response


def upload_to_youtube_with_infinite_retries(
    file_path,
    metadata,
    client_secrets_file=client_secrets_file,
    token_file=token_file,
    sleep_between_retries=10,
):
    attempt = 0
    while True:
        attempt += 1
        try:
            print(
                f"🔁 YouTube upload attempt #{attempt} — obtaining authenticated service..."
            )
            service = get_authenticated_service(
                client_secrets_file=client_secrets_file, token_file=token_file
            )
            print("🔐 Authenticated with YouTube API. Starting resumable upload...")
            vid_id, response = upload_to_youtube_resumable(
                service,
                file_path,
                title=metadata.get("title") or "Uploaded video",
                description=metadata.get("description") or "",
                tags=metadata.get("tags") or [],
                categoryId=metadata.get("categoryId") or "22",
                privacyStatus=metadata.get("privacyStatus") or "unlisted",
            )
            print(f"✅ Upload successful! Video ID: {vid_id}")
            watch_url = f"https://youtu.be/{vid_id}"
            return vid_id, watch_url, response
        except Exception as e:
            print(f"⚠️ YouTube upload attempt #{attempt} failed with error: {e}")
            sleep_time = sleep_between_retries * min(60, 1 + attempt * 0.5)
            print(
                f"⏳ Sleeping for {sleep_time:.1f}s before retrying (will retry forever until success)..."
            )
            time.sleep(sleep_time)


# -------------------------------------------------------------
# Main video processing (unchanged)
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


# -------------------------------------------------------------
# CLI / __main__
# -------------------------------------------------------------
if __name__ == "__main__":
    # INPUTS you can edit if needed:
    video_path = os.path.join(current_directory, "image_response", "final_output.mp4")
    logo_regions = [{"x": 1139, "y": 1010, "w": 359, "h": 60}]
    final_output = os.path.join(
        current_directory, "image_response", "final_video_medium_mp4.mp4"
    )

    # 0) Attempt to decrypt encrypted user.json.enc if present
    try:
        if os.path.exists(encrypted_user_json) and os.path.exists(user_key_file):
            print("🔐 Found encrypted user JSON and key — attempting decryption...")
            success = decrypt_user_json_enc(
                encrypted_user_json, user_key_file, decrypted_user_json
            )
            if success:
                print(f"ℹ️ Decrypted user JSON available at: {decrypted_user_json}")
                # Optionally load it if you need to use any keys from it
                try:
                    with open(decrypted_user_json, "r", encoding="utf-8") as uj:
                        user_json_content = json.load(uj)
                        # You can use user_json_content if needed (e.g., alternative client secrets)
                except Exception:
                    pass
        else:
            print("ℹ️ No encrypted user JSON/key pair found — skipping decryption.")
    except Exception as e:
        print("❌ Decryption step failed:", e)
        # Do not exit; continue — user may still have client_secrets.json present

    # Read metadata from file (AI output saved here)
    metadata = read_metadata_from_file(METADATA_FILE)

    # If key fields missing, allow manual fallback
    youtube_title = metadata.get("title") or "My Processed Video"
    youtube_description = (
        metadata.get("description")
        or "Video processed by remove_logo_from_video.py — uploaded automatically."
    )
    youtube_tags = metadata.get("tags") or ["processed", "inpainting", "auto-upload"]
    youtube_categoryId = metadata.get("categoryId") or "22"
    youtube_hashtags = metadata.get("hashtags") or []
    youtube_ctas = metadata.get("CTA") or []
    youtube_thumbnail_texts = metadata.get("thumbnail_texts") or []
    youtube_privacy = (
        "private"  # keep default unless you add privacy inside metadata and parse it
    )

    print(
        "ℹ️ Starting processing (will ensure ffmpeg and then attempt YouTube upload with infinite retries)..."
    )
    out = remove_logo_with_audio(
        video_path, logo_regions, final_output, scale_factor=0.85
    )

    if out:
        print(
            "ℹ️ Processed video ready. Beginning YouTube upload (infinite retries until success)..."
        )
        # Compose metadata dict for uploader
        uploader_metadata = {
            "title": youtube_title,
            "description": youtube_description,
            "tags": youtube_tags,
            "categoryId": youtube_categoryId,
            "privacyStatus": youtube_privacy,
            "hashtags": youtube_hashtags,
            "CTA": youtube_ctas,
            "thumbnail_texts": youtube_thumbnail_texts,
        }

        # Attempt upload (will retry indefinitely until success)
        try:
            vid_id, watch_url, full_response = upload_to_youtube_with_infinite_retries(
                out,
                uploader_metadata,
                client_secrets_file=client_secrets_file,
                token_file=token_file,
                sleep_between_retries=10,
            )
            print("✅ Upload complete. Watch URL:", watch_url)
            # Optional: print CTAs and thumbnails recommendation
            if youtube_ctas:
                print("📌 Suggested CTAs (pin or comment):")
                for c in youtube_ctas:
                    print("-", c)
            if youtube_thumbnail_texts:
                print("🖼️ Thumbnail text ideas:")
                for t in youtube_thumbnail_texts:
                    print("-", t)
        except Exception as e:
            # Shouldn't be reached because upload wrapper never stops; defensive log
            print("❌ Unexpected error in YouTube upload wrapper:", e)
            print("The script will now exit.")
    else:
        print("❌ Processing failed — skipping YouTube upload.")
