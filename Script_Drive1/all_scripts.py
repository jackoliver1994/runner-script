#!/usr/bin/env python3
"""
Ultimate sequential Python runner (infinite retries per-script) — Unicode-safe.

Changes/fixes:
 - Ensures child processes use UTF-8 stdio (PYTHONIOENCODING / PYTHONUTF8).
 - Uses subprocess.run(..., encoding='utf-8', errors='replace') so the parent decodes child output safely.
 - Coerces TimeoutExpired stdout/stderr to strings safely (handles bytes).
 - Keeps all features: SCRIPTS array, infinite retries until success, console-only logging,
   --timeout, --env-file, --tail-output, --no-backoff, --dry-run, tqdm progress, and friendly timing.

Patch summary (fix for your reported error):
 - Removed the top-level `from mega import Mega` that caused immediate import-time errors
   when `mega` (or its dependency `tenacity`) was incompatible with the runtime.
 - Added a small compatibility shim that restores a minimal `asyncio.coroutine` symbol
   when missing (some older libraries call `asyncio.coroutine` and in Python 3.11+ it's absent).
 - Deferred / lazy-import of `mega.Mega` inside the upload helper so the rest of the script
   runs even if the MEGA client package isn't installed or is incompatible; when upload is attempted
   we try to import with the compatibility shim and give a clear ImportError message if it still fails.

This preserves every feature and leaves the upload logic (including infinite retries) unchanged.
"""

from __future__ import annotations
import os
import sys
import time
import subprocess
import logging
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime
import glob
import argparse
import random
import uuid
import os
import json
from base64 import b64decode
from cryptography.fernet import Fernet, InvalidToken

# ---------- USER SCRIPTS: set by you ----------
SCRIPTS: List[str] = []
# ----------------------------------------------

# Try to set parent streams to UTF-8 (best-effort; may not be supported in some older Python builds)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    # ignore; we'll handle safe writes elsewhere
    pass

# Ensure the environment has UTF-8 defaults for any child processes by default
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

# Try tqdm for nicer progress; fallback to simple textual progress
try:
    from tqdm import tqdm  # type: ignore

    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False


# -------------------------
# Small compatibility helper for environments (fixes: AttributeError: module 'asyncio' has no attribute 'coroutine')
# -------------------------
def _ensure_asyncio_coroutine_compat():
    """Some older libraries expect asyncio.coroutine to exist. On Python 3.11+ it may be absent.
    Provide a no-op compatibility symbol so import-time code that checks for or decorates with
    asyncio.coroutine won't raise AttributeError.

    This is intentionally minimal (a no-op decorator) — it only restores the symbol so legacy
    import-time code can proceed; it doesn't attempt to reimplement generator-based coroutines.
    """
    try:
        import asyncio

        if not hasattr(asyncio, "coroutine"):
            # no-op decorator — acceptable for import-time compatibility in many libraries
            def _coroutine_noop(fn):
                return fn

            asyncio.coroutine = _coroutine_noop
    except Exception:
        # don't fail hard here; importing mega will fail later with a clearer error if needed
        pass


def _import_mega_with_compat():
    """Attempt to import Mega while applying compatibility shim for asyncio.coroutine.

    Returns the Mega class if successful, otherwise raises ImportError with a helpful message.
    """
    try:
        # Apply shim BEFORE importing packages that may expect asyncio.coroutine
        _ensure_asyncio_coroutine_compat()
        from mega import Mega

        return Mega
    except Exception as e:
        # Re-raise as ImportError with clearer guidance
        raise ImportError(
            "Failed to import 'mega.Mega'. This can happen when the installed 'mega.py' or its "
            "dependency 'tenacity' is incompatible with your Python version (e.g. Python 3.11). "
            "Please try upgrading the packages: `pip install -U mega.py tenacity`, or run without MEGA upload. "
            f"Original error: {e}"
        )


# -------------------------
# Nice join + timing helper (your function)
# -------------------------
def _nice_join(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def log_execution_time(
    start_time: float, end_time: float, show_ms: bool = False
) -> None:
    """
    Print start/end and elapsed time as human-readable hours/minutes/seconds.
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
    start_h = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %I:%M:%S %p")
    end_h = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %I:%M:%S %p")
    current_time = datetime.now().strftime("%I:%M:%S %p")
    safe_print(
        f"[{start_h} -> {end_h} | now {current_time}] Execution completed in {human}."
    )


# -------------------------
# Helpers
# -------------------------
def discover_scripts_in_path(
    path: str, pattern: str = "*.py", exclude_self: bool = True
) -> List[str]:
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return []
    found = sorted(glob.glob(os.path.join(path, pattern)))
    if exclude_self:
        try:
            runner = os.path.abspath(__file__)
            found = [f for f in found if os.path.abspath(f) != runner]
        except Exception:
            pass
    return found


def read_env_file(env_file: str) -> Dict[str, str]:
    env = os.environ.copy()
    try:
        with open(env_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    except Exception as e:
        logging.warning(f"Could not load env file {env_file}: {e}")
    return env


def _prepare_child_env(parent_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Build environment for child process:
     - start from os.environ
     - overlay parent_env (if provided)
     - ensure PYTHONIOENCODING and PYTHONUTF8 are set to use UTF-8 stdio in the child
    """
    child_env = os.environ.copy()
    if parent_env:
        child_env.update(parent_env)
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    return child_env


def _coerce_to_str(maybe_bytes_or_str) -> str:
    """Coerce bytes or str to a safe string using utf-8 with replacement."""
    if maybe_bytes_or_str is None:
        return ""
    if isinstance(maybe_bytes_or_str, str):
        return maybe_bytes_or_str
    try:
        return maybe_bytes_or_str.decode("utf-8", errors="replace")
    except Exception:
        try:
            return maybe_bytes_or_str.decode(errors="replace")
        except Exception:
            return repr(maybe_bytes_or_str)


def run_subprocess_capture(
    cmd: List[str], timeout: Optional[float], env: Optional[Dict[str, str]] = None
) -> Tuple[int, str, str, float]:
    """
    Run subprocess and return (returncode, stdout, stderr, elapsed_seconds).
    IMPORTANT: use encoding='utf-8' and errors='replace' so the parent safely decodes child bytes.
    """
    child_env = _prepare_child_env(env)
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=child_env,
        )
        elapsed = time.time() - start
        # proc.stdout/proc.stderr are strings (decoded using utf-8 with replacements)
        return proc.returncode, proc.stdout or "", proc.stderr or "", elapsed
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start
        # e.stdout / e.stderr might be bytes or str depending on Python version: coerce to str safely
        stdout = _coerce_to_str(e.stdout)
        stderr = _coerce_to_str(e.stderr)
        stdout = (stdout or "") + "\n\n--- TIMEOUT ---\n"
        stderr = (stderr or "") + "\n\n--- TIMEOUT ---\n"
        return 124, stdout, stderr, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return 1, "", f"Exception when running subprocess: {repr(e)}", elapsed


# -------------------------
# Fallback simple progress
# -------------------------
class SimpleProgress:
    def __init__(self, total: int):
        self.total = total
        self.current = 0

    def set_description(self, desc: str):
        sys.stdout.write(desc + "\n")
        sys.stdout.flush()

    def update(self, n: int = 1):
        self.current += n
        pct = (self.current / self.total) * 100 if self.total else 100
        sys.stdout.write(f"[{self.current}/{self.total}] {pct:5.1f}%\n")
        sys.stdout.flush()

    def close(self):
        pass


# -------------------------
# Helper to safely write Unicode output to console (parent)
# -------------------------
def safe_print(text: str, err: bool = False) -> None:
    """
    Print text safely to stdout/stderr using UTF-8 write with replacement.
    Uses the buffer when available to avoid parent console encoding errors.
    """
    try:
        data = text.encode("utf-8", errors="replace")
        if not data.endswith(b"\n"):
            data += b"\n"
        if err:
            if hasattr(sys.stderr, "buffer"):
                sys.stderr.buffer.write(data)
                sys.stderr.buffer.flush()
            else:
                # fallback
                print(text, file=sys.stderr)
        else:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            else:
                # fallback
                print(text)
    except Exception:
        # Last-resort fallback to print (will use default encoding)
        try:
            if err:
                print(text, file=sys.stderr)
            else:
                print(text)
        except Exception:
            # nothing else to do
            pass


def write_stdout_bytes(text: str) -> None:
    safe_print(text, err=False)


def write_stderr_bytes(text: str) -> None:
    safe_print(text, err=True)


# -------------------------
# MEGA upload helpers
# -------------------------
def decrypt_user_credentials(
    user_enc_path: str = "user.json.enc", key_path: str = "user_key.txt"
) -> Dict[str, str]:
    """
    Decrypt user.json.enc using a Fernet key found in user_key.txt.
    Returns a dict: {"email": "...", "password": "..."}

    Assumptions:
    - user_key.txt contains the base64 Fernet key (as produced by Fernet.generate_key()) on one line.
    - user.json.enc was produced by Fernet.encrypt(b'...json bytes...').

    Raises:
      - FileNotFoundError if files missing
      - ValueError for parsing errors or wrong format
      - InvalidToken if decryption fails (bad key / corrupted)
    """
    if not os.path.exists(user_enc_path):
        raise FileNotFoundError(
            f"Encrypted credentials file not found: {user_enc_path}"
        )
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Key file not found: {key_path}")

    # Load key (strip whitespace/newline)
    with open(key_path, "rb") as kf:
        key_data = kf.read().strip()

    # If key file contains a readable base64 string, use it directly.
    # Some users store the key as text; ensure it's bytes.
    try:
        # Validate by attempting to create a Fernet instance
        f = Fernet(key_data)
    except Exception as e:
        # Helpful error explaining expectation
        raise ValueError(
            "Failed to initialise Fernet. Ensure user_key.txt contains the base64 Fernet key "
            "(what you get from Fernet.generate_key()). Original error: " + str(e)
        )

    # Read encrypted file bytes
    with open(user_enc_path, "rb") as ef:
        encrypted = ef.read()

    try:
        decrypted_bytes = f.decrypt(encrypted)
    except InvalidToken as e:
        raise InvalidToken(
            "Decryption failed. The key may be wrong or the encrypted file corrupted. "
            + str(e)
        )

    try:
        creds = json.loads(decrypted_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError("Decrypted data is not valid JSON: " + str(e))

    if not isinstance(creds, dict) or "email" not in creds or "password" not in creds:
        raise ValueError(
            "Decrypted JSON must be an object containing 'email' and 'password' keys."
        )

    return {"email": creds["email"], "password": creds["password"]}


def get_workflow_number() -> str:
    """
    Get GitHub workflow number dynamically when running inside GitHub Actions.
    Checks environment variables:
      - GITHUB_RUN_NUMBER (preferred)
      - GITHUB_RUN_ID
    If none found (e.g., running locally), returns a deterministic fallback string:
      '{YYYYMMDD}-{shortuuid8}'.
    """
    # GitHub Actions exposes GITHUB_RUN_NUMBER and GITHUB_RUN_ID
    wf = os.environ.get("GITHUB_RUN_NUMBER") or os.environ.get("GITHUB_RUN_ID")
    if wf:
        return str(wf)

    # Not in GitHub Actions — fallback to date + short uuid
    date_part = datetime.utcnow().strftime("%Y%m%d")
    uniq8 = uuid.uuid4().hex[:8]
    return f"{date_part}-{uniq8}"


def upload_youtube_response_to_mega(
    workflow_number: Optional[str] = None,
    user_enc_path: str = "user.json.enc",
    user_key_path: str = "user_key.txt",
    source_dir: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Upload the local folder "<cwd>/youtube_response" to MEGA into a folder named:
      YYYYMMDD_workflow{workflow_number}_{uniq8}

    - workflow_number: supplied or retrieved from get_workflow_number()
    - credentials: read by decrypt_user_credentials(user_enc_path, user_key_path)
      (this uses Fernet decryption as described in decrypt_user_credentials)
    - source_dir: optional path to the 'youtube_response' folder. Defaults to os.path.join(os.getcwd(), "youtube_response")
    - verbose: if True, pass verbose option to Mega client construction (where supported)

    Returns:
      { "mega_folder_name": str, "uploaded_files": int, "errors": [...], "details": [...] }
    """
    # Resolve workflow number
    if workflow_number is None:
        workflow_number = get_workflow_number()

    # Resolve source directory
    if source_dir is None:
        source_dir = os.path.join(os.getcwd(), "youtube_response")
    source_dir = os.path.abspath(source_dir)

    if not os.path.isdir(source_dir):
        raise ValueError(f"Source directory not found or not a directory: {source_dir}")

    # Decrypt credentials
    creds = decrypt_user_credentials(user_enc_path, user_key_path)
    email = creds.get("email")
    password = creds.get("password")
    if not email or not password:
        raise ValueError(
            "Decrypted credentials must contain non-empty 'email' and 'password'"
        )

    # Prepare MEGA folder name
    date_part = datetime.now().strftime("%Y%m%d")
    uniq_part = uuid.uuid4().hex[:8]
    mega_folder_name = f"{date_part}_workflow{workflow_number}_{uniq_part}"

    # Import Mega lazily with compatibility shim
    Mega = _import_mega_with_compat()

    # Init MEGA client
    mega_opts = {"verbose": True} if verbose else {}
    # mega.py accepts dict or nothing; handle both ways for compatibility
    try:
        mega = Mega(mega_opts) if mega_opts else Mega()
    except TypeError:
        # In case Mega() does not accept options, call without args
        mega = Mega()

    # Login
    try:
        m = mega.login(email, password)
    except Exception as e:
        raise RuntimeError(f"Failed to login to MEGA with provided credentials: {e}")

    uploaded_count = 0
    errors = []
    details = []

    # Create top-level folder
    top_folder_id = None
    try:
        created = m.create_folder(mega_folder_name)
        if isinstance(created, dict) and created:
            top_folder_id = created.get(mega_folder_name) or list(created.values())[-1]
        else:
            # attempt to find the folder
            found = m.find(mega_folder_name)
            if found:
                top_folder_id = found[0]
    except Exception as e:
        # try to find if creation failed but folder exists
        try:
            found = m.find(mega_folder_name)
            if found:
                top_folder_id = found[0]
            else:
                raise RuntimeError(
                    f"Failed creating MEGA folder '{mega_folder_name}': {e}"
                )
        except Exception as e2:
            raise RuntimeError(
                f"Failed creating/finding MEGA folder '{mega_folder_name}': {e2}"
            )

    # Walk and upload
    for dirpath, dirnames, filenames in os.walk(source_dir):
        rel_dir = os.path.relpath(dirpath, source_dir)
        try:
            if rel_dir == ".":
                dest_node_id = top_folder_id
            else:
                # create nested path under the top folder
                remote_subpath = "/".join(
                    [mega_folder_name, rel_dir.replace(os.sep, "/")]
                )
                created = m.create_folder(remote_subpath)
                if isinstance(created, dict) and created:
                    # last value is the folder id
                    dest_node_id = list(created.values())[-1]
                else:
                    # fallback
                    last_segment = os.path.basename(rel_dir)
                    found = m.find(last_segment)
                    dest_node_id = found[0] if found else top_folder_id
        except Exception as e:
            errors.append(f"Error creating/finding remote folder for '{rel_dir}': {e}")
            dest_node_id = top_folder_id

        for fname in filenames:
            local_path = os.path.join(dirpath, fname)
            try:
                uploaded_node = m.upload(local_path, dest_node_id)
                uploaded_count += 1
                public_link = None
                try:
                    public_link = m.get_upload_link(uploaded_node)
                except Exception:
                    public_link = None

                details.append(
                    {
                        "local_path": local_path,
                        "remote_parent_node": dest_node_id,
                        "uploaded_node": uploaded_node,
                        "public_link": public_link,
                    }
                )
            except Exception as e:
                errors.append(f"Failed to upload '{local_path}': {e}")

    result = {
        "mega_folder_name": mega_folder_name,
        "uploaded_files": uploaded_count,
        "errors": errors,
        "details": details,
    }
    return result


# -------------------------
# Main orchestration
# -------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sequential Python runner with infinite retries per script (Unicode-safe)."
    )
    parser.add_argument(
        "--path",
        "-p",
        default=".",
        help="Directory to discover scripts when SCRIPTS is empty (default current).",
    )
    parser.add_argument(
        "--pattern",
        default="*.py",
        help="Glob pattern to discover scripts (default '*.py').",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-script timeout in seconds (default: none).",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional env file KEY=VALUE lines to set environment for scripts.",
    )
    parser.add_argument(
        "--show-ms", action="store_true", help="Show milliseconds in timing output."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run, but don't execute scripts.",
    )
    parser.add_argument(
        "--no-backoff",
        action="store_true",
        help="Disable backoff pause between retries (not recommended).",
    )
    parser.add_argument(
        "--tail-output",
        type=int,
        default=0,
        help="If >0, print only the last N lines of stdout/stderr per attempt (default 0 = print all).",
    )
    args = parser.parse_args(argv)

    # Logging to console only
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Decide scripts: SCRIPTS array has priority; if empty, discover in path
    if SCRIPTS:
        logging.info("Using scripts from top-level SCRIPTS array.")
        final_scripts = [os.path.abspath(s) for s in SCRIPTS]
    else:
        logging.info(
            f"SCRIPTS is empty; discovering scripts in {args.path} matching {args.pattern}"
        )
        discovered = discover_scripts_in_path(
            args.path, args.pattern, exclude_self=True
        )
        final_scripts = [os.path.abspath(s) for s in discovered]

    final_scripts = [s for s in final_scripts if os.path.isfile(s)]
    if not final_scripts:
        logging.error("No scripts found to run. Populate SCRIPTS or discover scripts.")
        return 2

    # Load env file if provided (these values will be layered onto the child's env)
    env = None
    if args.env_file:
        env = read_env_file(args.env_file)

    python_exec = sys.executable or "python"
    total_start = time.time()

    total = len(final_scripts)
    if TQDM_AVAILABLE:
        pbar = tqdm(total=total, desc="Overall Progress", unit="script")
    else:
        pbar = SimpleProgress(total=total)
        logging.info(
            "tqdm not installed: using simple textual progress. Install tqdm for nicer bars: pip install tqdm"
        )

    summary = []  # list of dicts for final summary

    try:
        for idx, script_path in enumerate(final_scripts, start=1):
            script_name = os.path.basename(script_path)
            desc = f"[{idx}/{total}] Running: {script_name} (will retry until success)"
            try:
                pbar.set_description(desc)
            except Exception:
                pass

            logging.info(f"=== START SCRIPT: {script_name} ===")
            if args.dry_run:
                logging.info(f"DRY-RUN: would run {python_exec} {script_path}")
                summary.append(
                    {
                        "name": script_name,
                        "path": script_path,
                        "attempts": 0,
                        "elapsed": 0.0,
                        "status": "DRY-RUN",
                    }
                )
                pbar.update(1)
                continue

            script_total_start = time.time()
            attempt = 0
            succeeded = False
            total_attempt_time = 0.0

            # Infinite retry loop until success (rc == 0)
            try:
                while True:
                    attempt += 1
                    attempt_tag = f"Attempt #{attempt}"
                    logging.info(
                        f"[{script_name}] {attempt_tag} - launching: {python_exec} {script_path}"
                    )
                    rc, stdout, stderr, attempt_elapsed = run_subprocess_capture(
                        [python_exec, script_path], timeout=args.timeout, env=env
                    )
                    total_attempt_time += attempt_elapsed

                    # Print attempt summary + outputs safely (write bytes)
                    line_sep = "=" * 80
                    write_stdout_bytes(line_sep)
                    write_stdout_bytes(
                        f"[{script_name}] {attempt_tag} completed in {attempt_elapsed:.2f}s with return code: {rc}"
                    )
                    write_stdout_bytes("-" * 80)
                    if stdout:
                        if args.tail_output > 0:
                            out_lines = stdout.splitlines()
                            tail = "\n".join(out_lines[-args.tail_output :])
                            write_stdout_bytes(
                                f"--- STDOUT (last {args.tail_output} lines) ---\n{tail}"
                            )
                        else:
                            write_stdout_bytes(f"--- STDOUT ---\n{stdout}")
                    else:
                        write_stdout_bytes("--- STDOUT: <empty> ---")
                    write_stdout_bytes("-" * 80)
                    if stderr:
                        if args.tail_output > 0:
                            err_lines = stderr.splitlines()
                            tail_err = "\n".join(err_lines[-args.tail_output :])
                            write_stderr_bytes(
                                f"--- STDERR (last {args.tail_output} lines) ---\n{tail_err}"
                            )
                        else:
                            write_stderr_bytes(f"--- STDERR ---\n{stderr}")
                    else:
                        write_stderr_bytes("--- STDERR: <empty> ---")
                    write_stdout_bytes(line_sep)

                    if rc == 0:
                        logging.info(f"[{script_name}] succeeded on attempt {attempt}.")
                        succeeded = True
                        break
                    else:
                        logging.warning(
                            f"[{script_name}] attempt {attempt} failed (rc={rc}). Will retry indefinitely until success."
                        )
                        # Backoff to avoid busy looping (can disable with --no-backoff)
                        if args.no_backoff:
                            backoff = 0.0
                        else:
                            backoff = min(
                                60.0, (2 ** min(attempt - 1, 6))
                            )  # 1,2,4,8,16,32,64 -> capped 60
                            jitter = backoff * 0.1
                            backoff = max(
                                0.0, backoff + random.uniform(-jitter, jitter)
                            )
                        if backoff > 0:
                            logging.info(
                                f"[{script_name}] sleeping {backoff:.1f}s before next attempt."
                            )
                            try:
                                time.sleep(backoff)
                            except KeyboardInterrupt:
                                logging.warning(
                                    "KeyboardInterrupt during backoff sleep -> exiting."
                                )
                                raise
                        # loop continues
            except KeyboardInterrupt:
                logging.warning("KeyboardInterrupt detected: stopping execution.")
                script_total_end = time.time()
                log_execution_time(
                    script_total_start, script_total_end, show_ms=args.show_ms
                )
                summary.append(
                    {
                        "name": script_name,
                        "path": script_path,
                        "attempts": attempt,
                        "elapsed": time.time() - script_total_start,
                        "status": "INTERRUPTED",
                    }
                )
                try:
                    pbar.close()
                except Exception:
                    pass
                safe_print("Exiting due to user interruption.")
                return 130

            script_total_end = time.time()
            script_total_elapsed = script_total_end - script_total_start
            # print total script timing using your helper
            log_execution_time(
                script_total_start, script_total_end, show_ms=args.show_ms
            )

            # add to summary
            summary.append(
                {
                    "name": script_name,
                    "path": script_path,
                    "attempts": attempt,
                    "elapsed": script_total_elapsed,
                    "status": "OK" if succeeded else "FAILED",
                }
            )

            # update progress only after success
            pbar.update(1)
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt detected in outer loop: exiting.")
    finally:
        try:
            pbar.close()
        except Exception:
            pass

    total_end = time.time()
    total_elapsed = total_end - total_start

    # Print final summary
    write_stdout_bytes("\n" + "=" * 100)
    write_stdout_bytes("FINAL RUN SUMMARY")
    write_stdout_bytes("=" * 100)
    longest_name = max((len(s["name"]) for s in summary), default=10)
    hdr = f"{'SCRIPT'.ljust(longest_name)}  STATUS     ATTEMPTS   TIME(s)"
    write_stdout_bytes(hdr)
    write_stdout_bytes("-" * 100)
    for s in summary:
        name = s["name"].ljust(longest_name)
        status = s.get("status", "N/A").ljust(9)
        attempts = str(s.get("attempts", "N/A")).rjust(8)
        time_s = f"{s.get('elapsed', 0.0):.2f}".rjust(10)
        write_stdout_bytes(f"{name}  {status}   {attempts}   {time_s}")
    write_stdout_bytes("-" * 100)
    write_stdout_bytes(f"Total elapsed time for full run: {total_elapsed:.2f} seconds")
    log_execution_time(total_start, total_end, show_ms=args.show_ms)
    write_stdout_bytes("=" * 100)

    any_fail = any(
        (s.get("status") != "OK" and s.get("status") != "DRY-RUN") for s in summary
    )

    # -----------------------------
    # Optional: upload youtube_response to MEGA (safe, non-destructive add-on)
    # -----------------------------
    try:
        youtube_dir = os.path.abspath(os.path.join(os.getcwd(), "youtube_response"))
        user_enc_path = os.path.abspath(os.path.join(os.getcwd(), "user.json.enc"))
        user_key_path = os.path.abspath(os.path.join(os.getcwd(), "user_key.txt"))

        if os.path.isdir(youtube_dir):
            logging.info(
                "Detected ./youtube_response folder — attempting optional MEGA upload."
            )
            # Prefer calling helper functions if defined; otherwise skip upload gracefully.
            try:
                # get_workflow_number() should be implemented as discussed earlier
                workflow_number = None
                if "get_workflow_number" in globals():
                    try:
                        workflow_number = get_workflow_number()
                    except Exception as e:
                        logging.warning(
                            f"get_workflow_number() failed: {e}; falling back to env/ts."
                        )
                        workflow_number = (
                            os.environ.get("GITHUB_RUN_NUMBER")
                            or os.environ.get("GITHUB_RUN_ID")
                            or f"local-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
                        )
                else:
                    workflow_number = (
                        os.environ.get("GITHUB_RUN_NUMBER")
                        or os.environ.get("GITHUB_RUN_ID")
                        or f"local-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
                    )

                # Only attempt encrypted-credentials path if both files exist
                if os.path.exists(user_enc_path) and os.path.exists(user_key_path):
                    if (
                        "decrypt_user_credentials" in globals()
                        and "upload_youtube_response_to_mega" in globals()
                    ):
                        # Infinite retry loop for upload until success (or user interrupts)
                        upload_attempt = 0
                        while True:
                            upload_attempt += 1
                            try:
                                logging.info(
                                    f"MEGA upload attempt #{upload_attempt} (workflow: {workflow_number})"
                                )
                                # decrypt just to validate credentials (the upload function will re-decrypt if it expects to)
                                creds = decrypt_user_credentials(
                                    user_enc_path, user_key_path
                                )
                                logging.info(
                                    "Decrypted MEGA credentials successfully (will use to upload)."
                                )
                                upload_res = upload_youtube_response_to_mega(
                                    workflow_number=workflow_number,
                                    user_enc_path=user_enc_path,
                                    user_key_path=user_key_path,
                                    source_dir=youtube_dir,
                                    verbose=False,
                                )
                                # Consider upload successful when there are no reported errors.
                                errors = upload_res.get("errors") or []
                                uploaded_files = upload_res.get("uploaded_files", 0)
                                logging.info(
                                    f"MEGA upload result: folder={upload_res.get('mega_folder_name')} files={uploaded_files} errors={len(errors)}"
                                )
                                if not errors:
                                    logging.info("MEGA upload completed successfully.")
                                    break  # success -> exit retry loop
                                else:
                                    # Log errors and retry indefinitely
                                    logging.warning(
                                        f"MEGA upload reported {len(errors)} error(s); will retry until success. First error: {errors[0]}"
                                    )
                            except KeyboardInterrupt:
                                logging.warning(
                                    "KeyboardInterrupt detected during MEGA upload attempts — stopping retries."
                                )
                                raise
                            except Exception as e:
                                logging.warning(
                                    f"MEGA upload attempt #{upload_attempt} raised exception: {e}; will retry until success."
                                )

                            # Backoff before retrying, obey --no-backoff
                            if args.no_backoff:
                                backoff = 0.0
                            else:
                                # Exponential backoff with cap and jitter
                                backoff = min(60.0, (2 ** min(upload_attempt - 1, 6)))
                                jitter = backoff * 0.1
                                backoff = max(
                                    0.0, backoff + random.uniform(-jitter, jitter)
                                )
                            if backoff > 0:
                                logging.info(
                                    f"Sleeping {backoff:.1f}s before next MEGA upload attempt."
                                )
                                try:
                                    time.sleep(backoff)
                                except KeyboardInterrupt:
                                    logging.warning(
                                        "KeyboardInterrupt during upload backoff sleep -> exiting upload loop."
                                    )
                                    raise
                        # finished upload retry loop
                    else:
                        logging.warning(
                            "MEGA upload helpers not present in this module. To enable upload, add get_workflow_number(), decrypt_user_credentials() and upload_youtube_response_to_mega() to this file."
                        )
                else:
                    logging.info(
                        "Skipping MEGA upload: credentials files not found at expected paths: "
                        f"{user_enc_path}, {user_key_path} (or keep them intentionally absent)."
                    )
            except KeyboardInterrupt:
                logging.warning(
                    "User interrupted while attempting MEGA upload. Continuing to end."
                )
            except Exception as e:
                # Protect main flow — do not let upload problems change exit code or behavior.
                logging.warning(f"Unexpected error while attempting MEGA upload: {e}")
        else:
            logging.info(
                "No ./youtube_response folder detected — skipping optional MEGA upload."
            )
    except Exception as e:
        # super-defensive - ensure main still returns the original result even if our upload code breaks
        logging.warning(f"Upload-checking logic encountered an unexpected error: {e}")

    return 0 if not any_fail else 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
