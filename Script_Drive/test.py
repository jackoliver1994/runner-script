#!/usr/bin/env python3
"""
full_robust_runner.py

Self-contained runner for GitHub Actions:
 - Robust POST requests with retry/backoff and safe JSON handling
 - Rotating logs + console output
 - Heartbeat file written periodically
 - Optional repeated runs for X hours (default: single-shot)
 - Dumps last responses and full tracebacks on errors
 - Produces artifacts.zip containing logs, dumps, and outputs for upload by Actions

Usage (CI recommended env):
  python full_robust_runner.py
Or with overrides:
  python full_robust_runner.py --url "https://example.com/api" --timeout 100 --run-hours 6

Configuration via ENV (preferred in Actions) or CLI args:
  API_URL, MESSAGE, REQUEST_TIMEOUT, REQUEST_RETRIES, REQUEST_BACKOFF,
  HEARTBEAT_PATH, OUTPUTS_DIR, RUN_HOURS, INTERVAL_SECONDS, ARTIFACT_ZIP
"""

from __future__ import annotations
import argparse
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, Timeout
from urllib3.util.retry import Retry

# ---------- Defaults (override via env) ----------
DEFAULT_URL = os.getenv("API_URL", "https://apifreellm.com/api/chat")
DEFAULT_MESSAGE = os.getenv("MESSAGE", "Hello from GitHub Actions")
DEFAULT_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "100"))
DEFAULT_RETRIES = int(os.getenv("REQUEST_RETRIES", "3"))
DEFAULT_BACKOFF = float(os.getenv("REQUEST_BACKOFF", "1"))
DEFAULT_LOGFILE = os.getenv("REQUEST_LOGFILE", "robust_request.log")
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5
DEFAULT_HEARTBEAT = os.getenv("HEARTBEAT_PATH", "heartbeat/heartbeat.txt")
DEFAULT_OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs")
DEFAULT_RUN_HOURS = float(os.getenv("RUN_HOURS", "0"))  # 0 means single run
DEFAULT_INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "60"))
DEFAULT_ARTIFACT_ZIP = os.getenv("ARTIFACT_ZIP", "artifacts.zip")

# ---------- Helpers ----------
def ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_text(prefix: str, text: str) -> str:
    fname = f"{prefix}_{ts()}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text or "")
    return os.path.abspath(fname)

# ---------- Logging ----------
def setup_logging(logfile: str, console_level=logging.INFO, file_level=logging.DEBUG):
    ensure_dir(os.path.dirname(os.path.abspath(logfile)) or ".")
    logger = logging.getLogger("full_robust_runner")
    logger.setLevel(logging.DEBUG)
    # remove prior handlers to support re-run in same process
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logfile, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.debug("Logging to console and %s", os.path.abspath(logfile))
    return logger

# ---------- HTTP session builder ----------
def build_session(retries: int, backoff: float) -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def safe_parse_json(text: Optional[str]):
    if not text:
        return None
    text = text.strip()
    if text == "":
        return None
    try:
        return json.loads(text)
    except Exception:
        return None

# ---------- POST and result handling ----------
def send_post(session: requests.Session, url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float, logger: logging.Logger, attempt: int = 1) -> Dict[str, Any]:
    info = {
        "ok": False,
        "status": None,
        "elapsed": None,
        "json": None,
        "text": None,
        "headers": None,
        "dump_file": None,
        "error": None,
    }
    try:
        logger.info("Attempt %d: POST %s (timeout=%ss)", attempt, url, timeout)
        resp = session.post(url, json=payload, headers=headers or {}, timeout=timeout)
        info["status"] = resp.status_code
        info["headers"] = dict(resp.headers) if resp.headers else {}
        info["elapsed"] = float(resp.elapsed.total_seconds()) if getattr(resp, "elapsed", None) else None
        text = resp.text or ""
        info["text"] = text
        parsed = None
        try:
            parsed = resp.json()
        except Exception as e:
            logger.debug("resp.json() failed: %s", str(e))
            parsed = safe_parse_json(text)
        info["json"] = parsed
        if 200 <= resp.status_code < 300:
            info["ok"] = True
            logger.info("Received %s in %.3fs", resp.status_code, info["elapsed"] or 0.0)
        else:
            info["error"] = f"HTTP {resp.status_code}"
            logger.warning("Non-2xx status: %s", resp.status_code)
        if text and parsed is None:
            dump_path = save_text("last_response", text)
            info["dump_file"] = dump_path
            logger.warning("Non-JSON response body saved to %s", dump_path)
        return info
    except Timeout as e:
        tb = traceback.format_exc()
        info["error"] = f"Timeout: {e}"
        dump = save_text("last_response", f"Timeout\n\n{tb}")
        info["dump_file"] = dump
        logger.warning("Timeout: %s; dump -> %s", e, dump)
        return info
    except RequestException as e:
        tb = traceback.format_exc()
        info["error"] = f"RequestException: {e}"
        dump = save_text("last_response", f"RequestException\n\n{tb}")
        info["dump_file"] = dump
        logger.warning("RequestException: %s; dump -> %s", e, dump)
        return info
    except Exception as e:
        tb = traceback.format_exc()
        info["error"] = f"Unexpected: {e}"
        dump = save_text("last_response", f"Unexpected\n\n{tb}")
        info["dump_file"] = dump
        logger.error("Unexpected exception: %s; dump -> %s", e, dump)
        return info

# ---------- Heartbeat thread ----------
class HeartbeatThread(threading.Thread):
    def __init__(self, path: str, interval: int, logger: logging.Logger):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self.running = threading.Event()
        self.running.set()
        self.logger = logger

    def run(self):
        ensure_dir(os.path.dirname(self.path) or ".")
        self.logger.info("Heartbeat started -> %s (every %ds)", self.path, self.interval)
        while self.running.is_set():
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    f.write("heartbeat: " + datetime.utcnow().isoformat() + "Z\n")
                # also touch a timestamped file for easier artifact debugging
                with open(f"{self.path}.last", "w", encoding="utf-8") as f2:
                    f2.write(datetime.utcnow().isoformat() + "Z\n")
            except Exception as e:
                self.logger.warning("Heartbeat write failed: %s", e)
            for _ in range(self.interval):
                if not self.running.is_set():
                    break
                time.sleep(1)
        self.logger.info("Heartbeat stopped")

    def stop(self):
        self.running.clear()

# ---------- Artifact collection ----------
def make_artifact_zip(zip_name: str, logfile: str, outputs_dir: str, logger: logging.Logger) -> str:
    logger.info("Creating artifact zip: %s", zip_name)
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        # include logfile and rotating backups
        if os.path.exists(logfile):
            z.write(logfile, arcname=os.path.basename(logfile))
        # include rotated backups in same dir
        base = os.path.splitext(logfile)[0]
        for candidate in sorted(os.listdir(".")):
            if candidate.startswith(os.path.basename(base)):
                if os.path.isfile(candidate):
                    try:
                        z.write(candidate, arcname=candidate)
                    except Exception:
                        logger.debug("Could not add %s", candidate)
        # include last_response_*.txt and error_trace_*.txt
        for f in sorted([f for f in os.listdir(".") if f.startswith("last_response_") or f.startswith("error_trace_") or f.startswith("run_summary_")]):
            try:
                z.write(f, arcname=f)
            except Exception:
                logger.debug("Could not add %s", f)
        # include outputs (videos)
        if os.path.isdir(outputs_dir):
            for root, _, files in os.walk(outputs_dir):
                for fn in files:
                    path = os.path.join(root, fn)
                    arc = os.path.relpath(path, ".")
                    try:
                        z.write(path, arcname=arc)
                    except Exception:
                        logger.debug("Could not add %s", path)
    logger.info("Artifact zip created: %s", os.path.abspath(zip_name))
    return os.path.abspath(zip_name)

# ---------- CLI / Main ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", "-u", default=DEFAULT_URL)
    p.add_argument("--message", "-m", default=DEFAULT_MESSAGE)
    p.add_argument("--timeout", "-t", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("--retries", "-r", type=int, default=DEFAULT_RETRIES)
    p.add_argument("--backoff", "-b", type=float, default=DEFAULT_BACKOFF)
    p.add_argument("--logfile", "-l", default=DEFAULT_LOGFILE)
    p.add_argument("--heartbeat", default=DEFAULT_HEARTBEAT)
    p.add_argument("--outputs", default=DEFAULT_OUTPUTS_DIR)
    p.add_argument("--run-hours", type=float, default=DEFAULT_RUN_HOURS, help="0=single run, >0 run this many hours retrying every interval")
    p.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    p.add_argument("--artifact-zip", default=DEFAULT_ARTIFACT_ZIP)
    p.add_argument("--raw-json", help="raw JSON payload string (overrides --message)")
    return p.parse_args()

def main():
    args = parse_args()
    logger = setup_logging(args.logfile)
    logger.info("full_robust_runner starting; args=%s", vars(args))

    ensure_dir(args.outputs)
    # heartbeat
    hb = HeartbeatThread(path=args.heartbeat, interval=max(10, args.interval_seconds), logger=logger)
    hb.start()

    session = build_session(retries=args.retries, backoff=args.backoff)
    headers = {"Content-Type": "application/json"}

    if args.raw_json:
        try:
            payload = json.loads(args.raw_json)
        except Exception as e:
            logger.critical("Invalid --raw-json: %s", e)
            hb.stop()
            hb.join(timeout=2)
            sys.exit(2)
    else:
        payload = {"message": args.message}

    start_time = datetime.utcnow()
    end_time = start_time + timedelta(hours=args.run_hours) if args.run_hours > 0 else start_time
    iteration = 0
    any_failure = False
    last_info = None

    try:
        while True:
            iteration += 1
            attempt_info = None
            # make a single call with top-level retry loop (session also has low-level retries)
            for attempt in range(1, args.retries + 1):
                attempt_info = send_post(session=session, url=args.url, payload=payload, headers=headers, timeout=args.timeout, logger=logger, attempt=attempt)
                logger.debug("Attempt %d info: %s", attempt, {k: attempt_info.get(k) for k in ("ok","status","error","dump_file")})
                if attempt_info.get("ok"):
                    break
                wait = args.backoff * (2 ** (attempt - 1))
                logger.info("Waiting %.1fs before next attempt", wait)
                time.sleep(wait)
            last_info = attempt_info
            if not attempt_info.get("ok"):
                any_failure = True
                logger.error("Iteration %d: request failed after %d retries. Last error: %s", iteration, args.retries, attempt_info.get("error"))
            else:
                logger.info("Iteration %d: request succeeded status=%s", iteration, attempt_info.get("status"))

            # If run_hours == 0 => single-shot; break after first iteration
            if args.run_hours <= 0:
                break

            # if time is up, stop
            now = datetime.utcnow()
            if now >= end_time:
                logger.info("Run-hours elapsed; stopping main loop")
                break

            # wait until next iteration but keep heartbeat running (sleep in small chunks)
            secs = args.interval_seconds
            logger.info("Sleeping %ds until next iteration (heartbeat will continue)", secs)
            for _ in range(secs):
                time.sleep(1)
                # early exit if no more time
                if datetime.utcnow() >= end_time:
                    break

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        any_failure = True
    except Exception as exc:
        any_failure = True
        trace = traceback.format_exc()
        trace_file = save_text("error_trace", trace)
        logger.critical("Unhandled exception: %s; trace saved to %s", exc, trace_file)
    finally:
        # stop heartbeat thread gently
        hb.stop()
        hb.join(timeout=5)
        # write run summary
        summary = {
            "started_at": start_time.isoformat() + "Z",
            "ended_at": datetime.utcnow().isoformat() + "Z",
            "iterations": iteration,
            "last_status": last_info.get("status") if last_info else None,
            "last_error": last_info.get("error") if last_info else None,
            "any_failure": any_failure,
        }
        summary_path = save_text("run_summary", json.dumps(summary, indent=2))
        logger.info("Wrote run summary to %s", summary_path)

        # create artifact zip
        try:
            artifact = make_artifact_zip(args.artifact_zip, args.logfile, args.outputs, logger)
            print("ARTIFACT_ZIP=" + artifact)
        except Exception as e:
            logger.warning("Failed to create artifact zip: %s", e)

        # final exit
        if any_failure:
            logger.info("Completed with failures; exit code 1")
            sys.exit(1)
        else:
            logger.info("Completed successfully; exit code 0")
            sys.exit(0)

if __name__ == "__main__":
    main()
