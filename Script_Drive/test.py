#!/usr/bin/env python3
"""
robust_request.py

Robust POST request script with:
 - configurable timeout, retries, backoff
 - safe JSON parsing (handles empty / non-JSON bodies)
 - detailed logging to stdout and a log file (for CI)
 - proper exit codes for CI

Usage examples:
  python robust_request.py --url "https://apifreellm.com/api/chat" --message "Hello"
  API_URL=https://... python robust_request.py
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, HTTPError, Timeout
from urllib3.util.retry import Retry

# ---------- Defaults (override via CLI or env) ----------
DEFAULT_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "100"))     # seconds
DEFAULT_RETRIES = int(os.getenv("REQUEST_RETRIES", "3"))
DEFAULT_BACKOFF = float(os.getenv("REQUEST_BACKOFF", "1"))      # base backoff factor
DEFAULT_LOGFILE = os.getenv("REQUEST_LOGFILE", "robust_request.log")
DEFAULT_URL = os.getenv("API_URL", "https://apifreellm.com/api/chat")

# ---------- Logging ----------
logger = logging.getLogger("robust_request")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)

# File handler
fh = logging.FileHandler(DEFAULT_LOGFILE)
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
logger.addHandler(fh)


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


def safe_parse_json(text: str):
    """
    Try to parse JSON. Return parsed object or None if not JSON.
    """
    if text is None:
        return None
    text = text.strip()
    if text == "":
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def send_post(
    session: requests.Session,
    url: str,
    json_data,
    headers: dict,
    timeout: float,
    attempt_num: int = 1,
):
    """
    Send a POST and return a structured result dict with:
      - ok (bool)
      - status_code (int or None)
      - json (object or None)
      - text (str)
      - headers (dict)
      - elapsed (float seconds)
      - error (str or None)
    """
    result = {
        "ok": False,
        "status_code": None,
        "json": None,
        "text": None,
        "headers": None,
        "elapsed": None,
        "error": None,
    }
    try:
        logger.info(f"Attempt #{attempt_num} -> POST {url} (timeout={timeout}s)")
        resp = session.post(url, json=json_data, headers=headers or {}, timeout=timeout)
        result["status_code"] = resp.status_code
        result["headers"] = dict(resp.headers)
        result["elapsed"] = float(resp.elapsed.total_seconds()) if resp.elapsed else None
        # get text safely (but limit length in logs)
        text = resp.text or ""
        result["text"] = text

        # Try to parse JSON in a robust way
        parsed = None
        try:
            parsed = resp.json()
        except Exception:
            parsed = safe_parse_json(text)

        result["json"] = parsed

        # Treat 2xx as success (allow API-specific error field to handle logic later)
        if 200 <= resp.status_code < 300:
            result["ok"] = True
        else:
            result["error"] = f"HTTP {resp.status_code}"
            result["ok"] = False

        return result

    except Timeout as e:
        tb = traceback.format_exc()
        result["error"] = f"Timeout after {timeout}s: {e}"
        logger.debug(tb)
        return result
    except RequestException as e:
        tb = traceback.format_exc()
        result["error"] = f"RequestException: {e}"
        logger.debug(tb)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        result["error"] = f"Unexpected error: {e}"
        logger.debug(tb)
        return result


def main():
    parser = argparse.ArgumentParser(description="Robust POST to an API (with retries & logging).")
    parser.add_argument("--url", "-u", default=DEFAULT_URL, help="API URL")
    parser.add_argument("--message", "-m", default=None, help="Message to send (JSON payload will be {message: ...})")
    parser.add_argument("--timeout", "-t", type=float, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    parser.add_argument("--retries", "-r", type=int, default=DEFAULT_RETRIES, help="Number of retries for transient errors")
    parser.add_argument("--backoff", "-b", type=float, default=DEFAULT_BACKOFF, help="Backoff factor for retries")
    parser.add_argument("--logfile", "-l", default=DEFAULT_LOGFILE, help="Log file path")
    parser.add_argument("--raw-json", help="Send this raw JSON string as request body (instead of --message)")
    parser.add_argument("--header", action="append", help="Extra headers in Key:Value form (can be passed multiple times)")

    args = parser.parse_args()

    # Allow changing logfile after parser (reconfigure file handler)
    global fh
    if args.logfile and args.logfile != DEFAULT_LOGFILE:
        # remove old file handler and add new
        logger.removeHandler(fh)
        fh = logging.FileHandler(args.logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    url = args.url
    timeout = args.timeout
    retries = args.retries
    backoff = args.backoff

    # Build headers
    headers = {"Content-Type": "application/json"}
    if args.header:
        for h in args.header:
            if ":" in h:
                k, v = h.split(":", 1)
                headers[k.strip()] = v.strip()
            else:
                logger.warning("Ignored header entry (invalid format): %s", h)

    # Build payload
    if args.raw_json:
        try:
            payload = json.loads(args.raw_json)
        except Exception:
            logger.error("Invalid JSON passed to --raw-json")
            sys.exit(2)
    elif args.message is not None:
        payload = {"message": args.message}
    else:
        # If no message provided, try reading MESSAGE env var or default example payload
        env_msg = os.getenv("MESSAGE")
        if env_msg:
            payload = {"message": env_msg}
        else:
            payload = {"message": "Hello, how are you?"}

    logger.debug("Final payload: %s", json.dumps(payload))
    logger.debug("Headers: %s", headers)
    logger.info(f"Posting to {url} with retries={retries} timeout={timeout}s backoff={backoff}")

    session = build_session(retries=retries, backoff=backoff)

    # Attempt loop: session's Retry will do lower-level retries for some failures, but
    # we also perform higher-level attempts to capture structured logs between attempts.
    final_result = None
    for attempt in range(1, retries + 1):
        result = send_post(session=session, url=url, json_data=payload, headers=headers, timeout=timeout, attempt_num=attempt)

        # Log details (truncate long bodies to keep CI logs readable)
        truncated_text = (result["text"][:10000] + "...(truncated)") if result["text"] and len(result["text"]) > 10000 else (result["text"] or "")
        logger.info(f"Attempt {attempt} result: status={result['status_code']} elapsed={result['elapsed']}s ok={result['ok']}")
        logger.debug("Response headers: %s", json.dumps(result["headers"]) if result["headers"] else {})
        logger.debug("Response body (truncated 10k):\n%s", truncated_text)
        if result["error"]:
            logger.warning("Attempt %d error: %s", attempt, result["error"])

        # If ok and valid JSON or non-empty body -> accept
        if result["ok"]:
            final_result = result
            break

        # else, wait a bit before next attempt (exponential backoff)
        wait_seconds = backoff * (2 ** (attempt - 1))
        logger.info("Waiting %.1fs before next attempt (backoff)...", wait_seconds)
        try:
            # Be careful: in CI you may want to avoid very long sleeps; this is mild.
            import time
            time.sleep(wait_seconds)
        except KeyboardInterrupt:
            logger.error("Interrupted during backoff wait")
            break

    # Final evaluation
    if final_result is None:
        # All attempts failed
        logger.error("All %d attempts failed. See logs: %s", retries, args.logfile)
        # optionally dump last response text to separate file for debugging
        if result and result.get("text"):
            dump_file = f"last_response_{datetime.utcnow():%Y%m%dT%H%M%SZ}.txt"
            with open(dump_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
            logger.info("Wrote last response body to %s", dump_file)
        sys.exit(1)

    # We have a final_result (2xx). Try to interpret JSON payload
    parsed = final_result.get("json")
    if parsed:
        # Example API semantics from your original script:
        # if parsed.get("status") == "success": print response
        status_field = parsed.get("status") if isinstance(parsed, dict) else None
        if status_field == "success":
            logger.info("API reported success.")
            # print the "response" field if present
            api_response = parsed.get("response") if isinstance(parsed, dict) else parsed
            print("AI Response:", api_response)
            sys.exit(0)
        else:
            # Not a success according to API — still print parsed JSON for debugging
            logger.warning("API did not return success status. Full JSON: %s", json.dumps(parsed, indent=2) if isinstance(parsed, dict) else str(parsed))
            # If API included an error field, surface it
            api_error = parsed.get("error") if isinstance(parsed, dict) else None
            if api_error:
                logger.error("API error: %s", api_error)
            print("API result (non-success):", parsed)
            sys.exit(1)
    else:
        # No JSON returned — but request succeeded (2xx). Print raw text for debugging.
        logger.warning("Response had no JSON body. Raw text printed (may be empty).")
        print("Raw response text:")
        print(final_result.get("text") or "")
        # treat as failure for CI (change this behavior if your API sometimes returns non-JSON successes)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc)
        logger.critical(traceback.format_exc())
        sys.exit(2)
