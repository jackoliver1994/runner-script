#!/usr/bin/env python3
"""
story_pipeline.py

Refactored from user's original script into a modular, callable design.
- All major actions are methods on StoryPipeline (generate_script, generate_images, generate_narration, generate_youtube_metadata).
- ChatAPI.send_message accepts per-call timeout and controllable retry options.
- Image prompt generation supports batching (smaller requests) with backoff/retry to reduce timeouts.
- Helpers to extract bracketed blocks and ensure the "single bracketed block only" requirement.

Use:
    pipeline = StoryPipeline(api_url="https://apifreellm.com/api/chat")
    pipeline.generate_image_prompts(script_text, img_number=150, batch_size=50)
"""

import requests
import json
import time
import sys
import os
import random
import re
import ast
import math
import difflib
import concurrent.futures
import concurrent.futures as cf
from typing import Optional, List, Dict
from threading import Thread, Event
from datetime import datetime


# ---------------------- LOADING SPINNER ----------------------
class LoadingSpinner:
    def __init__(self, message: str = "Waiting for response..."):
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.stop_event = Event()
        self.thread = None
        self.message = message

    def spin(self):
        while not self.stop_event.is_set():
            for char in self.spinner_chars:
                sys.stdout.write("\r" + f"{self.message} {char}")
                sys.stdout.flush()
                time.sleep(0.1)
                if self.stop_event.is_set():
                    break
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def start(self):
        self.stop_event.clear()
        self.thread = Thread(target=self.spin, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()


# ---------------------- CHAT API HANDLER ----------------------
class ChatAPI:
    def __init__(
        self, url: str = "https://apifreellm.com/api/chat", default_timeout: int = 1000
    ):
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        self.base_timeout = default_timeout

    def send_message(
        self,
        message: str,
        timeout: Optional[int] = None,
        spinner_message: str = "Waiting for response...",
        initial_backoff: float = 2.0,
        max_backoff: float = 8.0,
        timeout_growth: int = 100,
        max_timeout_cap: int = 1000000,
        use_cloudscraper_on_403: bool = True,
        use_playwright_on_403: bool = False,
        proxy: Optional[str] = None,
    ) -> str:
        """
        Robust send_message — preserves original features (infinite retry, spinner, backoff, cloudscraper/playwright fallbacks),
        but adds thorough URL validation/normalization to avoid invalid-root_url issues (e.g. 'https:/') and safer solver usage.

        Notes:
        - If self.url is misconfigured (no host), this will raise ValueError early (helps avoid infinite loops on bad config).
        - All original features retained.
        """
        import requests
        import time
        import random
        import json
        import traceback
        import concurrent.futures as cf
        from urllib.parse import urlparse, urlunparse

        # lazy optional imports for challenge-solving
        try:
            import cloudscraper
        except Exception:
            cloudscraper = None

        PLAYWRIGHT_AVAILABLE = False
        if use_playwright_on_403:
            try:
                from playwright.sync_api import (
                    sync_playwright,
                    TimeoutError as PlaywrightTimeoutError,
                )

                PLAYWRIGHT_AVAILABLE = True
            except Exception:
                PLAYWRIGHT_AVAILABLE = False

        # ensure session reuse
        if not hasattr(self, "_session") or self._session is None:
            self._session = requests.Session()
        session = self._session

        # Validate and normalize self.url -> ensure scheme + netloc
        parsed = urlparse(self.url or "")
        if not parsed.scheme:
            # default to https if not provided
            parsed = parsed._replace(scheme="https")
        if not parsed.netloc:
            # If path contains host-like component (e.g. "https:/host.com"), try quick fix:
            # attempt to re-parse a common bad-case: "https:/" or "https:////host"
            guessed = (self.url or "").replace(":///", "://").replace(":/", "://")
            parsed2 = urlparse(guessed)
            if parsed2.netloc:
                parsed = parsed2
            else:
                raise ValueError(
                    f"ChatAPI.send_message: invalid url configured for ChatAPI.url -> '{self.url}'. "
                    "No host/netloc found. Please set a valid URL like 'https://example.com/api/chat'."
                )
        # root_url = scheme://netloc  (no trailing slash)
        root_url = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))

        # initial timeout and backoff
        configured_timeout = int(timeout or getattr(self, "base_timeout", 60))
        configured_timeout = min(configured_timeout, max_timeout_cap)
        backoff = float(initial_backoff)
        attempt = 0

        # track last solver times to avoid hammering them repeatedly
        last_solver_time = {"cloudscraper": 0.0, "playwright": 0.0}
        SOLVER_MIN_INTERVAL = 30.0  # seconds between solver attempts

        # spinner (if available in your codebase)
        spinner = None
        try:
            if "LoadingSpinner" in globals():
                spinner = LoadingSpinner(spinner_message)
        except Exception:
            spinner = None

        def looks_like_html_challenge(status_code, text, headers):
            """
            Broader HTML-challenge detector (same heuristics as before).
            """
            try:
                if status_code in (403, 522):
                    return True
                t = (text or "").lower()
                if (
                    "just a moment" in t
                    or "attention required" in t
                    or "please enable javascript" in t
                    or "cloudflare" in t
                    or "connection timed out" in t
                ):
                    return True
                ct = (headers or {}).get("Content-Type", "") or ""
                if status_code == 403 and "text/html" in ct.lower():
                    return True
            except Exception:
                pass
            return False

        def try_cloudscraper_once():
            """
            Attempt to use cloudscraper once to solve a detected HTML challenge.
            This function:
            - honours SOLVER_MIN_INTERVAL frequency
            - normalizes root_url and uses it for a GET seed
            - retries a POST with safer/fallback headers if necessary
            - extracts embedded JSON if present
            Returns parsed JSON dict on success, None otherwise.
            """
            if not cloudscraper:
                print("→ cloudscraper not installed; skipping cloudscraper solver.")
                return None

            now = time.time()
            if now - last_solver_time["cloudscraper"] < SOLVER_MIN_INTERVAL:
                print(
                    "→ cloudscraper attempted recently; skipping immediate re-attempt."
                )
                return None
            last_solver_time["cloudscraper"] = now

            try:
                print(
                    "→ Trying cloudscraper to solve challenge (GET seed + POST attempt)..."
                )
                s = (
                    cloudscraper.create_scraper()
                )  # may raise; let it surface and be caught below
                proxies = {"http": proxy, "https": proxy} if proxy else None

                # 1) seed cookies by GET to the root (helps many CF flows)
                try:
                    g = s.get(
                        root_url,
                        timeout=min(30, max(10, configured_timeout // 4)),
                        proxies=proxies,
                    )
                    print(
                        "→ cloudscraper GET seed status:",
                        getattr(g, "status_code", None),
                    )
                except Exception as ge:
                    print("→ cloudscraper GET seed failed (continuing to POST):", ge)

                # 2) POST with configured timeout
                r = None
                try:
                    r = s.post(
                        self.url,
                        json={"message": message},
                        timeout=configured_timeout,
                        proxies=proxies,
                    )
                except Exception as e:
                    print("→ cloudscraper POST attempt exception:", e)
                    # fallback: second POST with extended timeout
                    try:
                        r = s.post(
                            self.url,
                            json={"message": message},
                            timeout=max(60, configured_timeout),
                            proxies=proxies,
                        )
                    except Exception as e2:
                        print("→ cloudscraper fallback POST exception:", e2)
                        return None

                if r is None:
                    return None

                print("→ cloudscraper status:", getattr(r, "status_code", None))
                ct = (r.headers.get("Content-Type") or "").lower()
                body = r.text or ""

                # 3) If JSON content-type -> parse and return
                if "application/json" in ct:
                    try:
                        data = r.json()
                        return data
                    except Exception:
                        print("→ cloudscraper returned invalid JSON; snippet:")
                        print(body[:1000])

                # 4) If non-JSON HTML, attempt to extract embedded JSON (Next.js, __NEXT_DATA__, window.*)
                try:
                    m = re.search(
                        r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>([\s\S]+?)</script>',
                        body,
                        flags=re.I,
                    )
                    if not m:
                        m = re.search(
                            r"window\.__[A-Z0-9_]+__\s*=\s*({[\s\S]+?});", body
                        )
                    if m:
                        candidate = m.group(1)
                        try:
                            data = json.loads(candidate)
                            print(
                                "→ cloudscraper extracted JSON from HTML and parsed it."
                            )
                            return data
                        except Exception:
                            try:
                                data = json.loads(candidate.rstrip(" ;\n\r\t"))
                                return data
                            except Exception:
                                pass
                except Exception:
                    pass

                # 5) Try a second targeted POST with adjusted headers (UA + Accept JSON)
                try:
                    fallback_headers = dict(
                        getattr(self, "headers", {"Content-Type": "application/json"})
                    )
                    fallback_headers.update(
                        {
                            "Accept": "application/json, text/plain, */*",
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
                        }
                    )
                    print("→ cloudscraper retrying POST with fallback headers...")
                    r2 = s.post(
                        self.url,
                        json={"message": message},
                        headers=fallback_headers,
                        timeout=max(60, configured_timeout),
                        proxies=proxies,
                    )
                    print(
                        "→ cloudscraper fallback POST status:",
                        getattr(r2, "status_code", None),
                    )
                    ct2 = (r2.headers.get("Content-Type") or "").lower()
                    if "application/json" in ct2:
                        try:
                            return r2.json()
                        except Exception:
                            print(
                                "→ cloudscraper fallback returned invalid JSON; snippet:"
                            )
                            print((r2.text or "")[:1000])
                    # try extract JSON again from fallback HTML
                    body2 = r2.text or ""
                    m2 = re.search(
                        r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>([\s\S]+?)</script>',
                        body2,
                        flags=re.I,
                    )
                    if m2:
                        try:
                            return json.loads(m2.group(1))
                        except Exception:
                            pass
                except Exception as e:
                    print("→ cloudscraper fallback POST exception:", e)
                    traceback.print_exc()

                # 6) didn't produce parseable JSON — save an HTML snippet for offline debugging and return None
                try:
                    snippet_path = os.path.join(
                        "debug_cloudscraper",
                        f"cloudscraper_html_{int(time.time())}.html",
                    )
                    os.makedirs(os.path.dirname(snippet_path), exist_ok=True)
                    with open(snippet_path, "w", encoding="utf-8") as fh:
                        fh.write(body[:100000])
                    print(
                        f"→ cloudscraper returned non-JSON HTML; saved snippet -> {snippet_path}"
                    )
                except Exception as e_save:
                    print("→ Failed to save cloudscraper HTML snippet:", e_save)

                print(
                    "→ cloudscraper did not solve the challenge or returned non-JSON."
                )
                return None

            except Exception as e:
                print("→ cloudscraper attempt exception (top-level):", e)
                traceback.print_exc()
                return None

        def single_request(sess, req_timeout):
            """Perform a single POST using provided session; returns dict with ok/status/headers/text or ok=False and exc."""
            try:
                headers = getattr(
                    self, "headers", getattr(self, "default_headers", None)
                )
                if headers is None:
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    }
                proxies = {"http": proxy, "https": proxy} if proxy else None
                r = sess.post(
                    self.url,
                    headers=headers,
                    json={"message": message},
                    timeout=(10, req_timeout),
                    proxies=proxies,
                )
                return {
                    "ok": True,
                    "status": r.status_code,
                    "headers": dict(r.headers or {}),
                    "text": r.text,
                }
            except Exception as e:
                return {"ok": False, "exc": e, "trace": traceback.format_exc()}

        def try_playwright_once():
            if not PLAYWRIGHT_AVAILABLE:
                print("→ Playwright not available; skipping Playwright solver.")
                return None
            now = time.time()
            if now - last_solver_time["playwright"] < SOLVER_MIN_INTERVAL:
                print("→ Playwright attempted recently; skipping immediate re-attempt.")
                return None
            last_solver_time["playwright"] = now
            try:
                print("→ Trying Playwright (headless) to solve challenge...")
                from playwright.sync_api import sync_playwright

                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    ctx = browser.new_context()
                    page = ctx.new_page()
                    try:
                        page.goto(
                            root_url,
                            wait_until="load",
                            timeout=max(30000, int(configured_timeout * 1000)),
                        )
                    except Exception as e:
                        print(
                            "→ Playwright navigation exception (may still set cookies):",
                            e,
                        )
                    time.sleep(2)
                    cookies = ctx.cookies()
                    ua = page.evaluate("() => navigator.userAgent")
                    browser.close()
                # use requests session with cookies & ua
                s = requests.Session()
                for c in cookies:
                    try:
                        s.cookies.set(
                            c.get("name"),
                            c.get("value"),
                            domain=c.get("domain"),
                            path=c.get("path", "/"),
                        )
                    except Exception:
                        pass
                headers = dict(
                    getattr(
                        self,
                        "headers",
                        getattr(
                            self,
                            "default_headers",
                            {"Content-Type": "application/json"},
                        ),
                    )
                )
                headers["User-Agent"] = ua
                proxies = {"http": proxy, "https": proxy} if proxy else None
                r = s.post(
                    self.url,
                    json={"message": message},
                    headers=headers,
                    timeout=configured_timeout,
                    proxies=proxies,
                )
                print("→ Playwright-backed request status:", r.status_code)
                ct = (r.headers.get("Content-Type") or "").lower()
                if "application/json" in ct:
                    try:
                        return r.json()
                    except Exception:
                        print("→ Playwright returned invalid JSON; snippet:")
                        print((r.text or "")[:1000])
                        return None
                else:
                    print("→ Playwright returned non-JSON snippet:")
                    print((r.text or "")[:1000])
                    return None
            except Exception as e:
                print("→ Playwright attempt exception:", e)
                traceback.print_exc()
                return None

        print(
            "send_message starting infinite retry loop (preserving original log style)."
        )
        # infinite retry loop until success
        while True:
            attempt += 1

            # Print log like original
            print(
                f"\nAttempt #{attempt} — timeout={int(configured_timeout)}s — sending request..."
            )

            # start spinner if available
            if spinner:
                try:
                    spinner.start()
                except Exception:
                    pass

            try:
                # run request in thread to enforce attempt-level timeout and avoid blocking
                with cf.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(single_request, session, configured_timeout)
                    try:
                        wrapper = fut.result(timeout=configured_timeout + 5)
                    except cf.TimeoutError:
                        # attempt timed out - best-effort cleanup and retry after backoff
                        try:
                            fut.cancel()
                        except Exception:
                            pass
                        try:
                            session.close()
                        except Exception:
                            pass
                        self._session = requests.Session()
                        session = self._session
                        if spinner:
                            try:
                                spinner.stop()
                            except Exception:
                                pass
                        print(
                            "                                                                                 "
                        )
                        print(
                            f"⚠️ Attempt #{attempt} timed out after {configured_timeout}s. Backing off {backoff:.1f}s..."
                        )
                        time.sleep(backoff + random.random())
                        backoff = min(backoff * 2, max_backoff)
                        configured_timeout = min(
                            configured_timeout + timeout_growth, max_timeout_cap
                        )
                        continue
                    except Exception as e:
                        # unexpected issue waiting on future
                        if spinner:
                            try:
                                spinner.stop()
                            except Exception:
                                pass
                        print(
                            "                                                                                 "
                        )
                        print("⚠️ Exception while waiting for request future:", e)
                        traceback.print_exc()
                        time.sleep(backoff + random.random())
                        backoff = min(backoff * 2, max_backoff)
                        configured_timeout = min(
                            configured_timeout + timeout_growth, max_timeout_cap
                        )
                        continue

                if spinner:
                    try:
                        spinner.stop()
                    except Exception:
                        pass

                if not wrapper.get("ok"):
                    # network exception inside worker
                    exc = wrapper.get("exc")
                    print(
                        "                                                                                 "
                    )
                    print(f"⚠️ Network/worker exception on attempt #{attempt}: {exc!r}")
                    try:
                        print(wrapper.get("trace"))
                    except Exception:
                        pass
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    configured_timeout = min(
                        configured_timeout + timeout_growth, max_timeout_cap
                    )
                    continue

                status = int(wrapper.get("status", 0) or 0)
                headers = wrapper.get("headers") or {}
                body = wrapper.get("text") or ""

                print(
                    "                                                                                 "
                )
                print(f"✅ Response status: {status}")

                # Detect Cloudflare-like challenge and log identical message
                if looks_like_html_challenge(status, body, headers):
                    print("⚠️ Detected Cloudflare-like anti-bot page (HTTP 403).")
                    # Try cloudscraper once per-challenge (best-effort)
                    if use_cloudscraper_on_403 and cloudscraper:
                        data = try_cloudscraper_once()
                        if isinstance(data, dict):
                            if data.get("status") == "success":
                                resp_text = data.get("response", "")
                                print(
                                    "⚠️ cloudscraper returned status=success — returning response."
                                )
                                return resp_text
                            else:
                                print(
                                    "→ cloudscraper returned JSON but not status==success (snippet):"
                                )
                                try:
                                    print(json.dumps(data, indent=2)[:1000])
                                except Exception:
                                    print(data)
                        else:
                            print(
                                "→ cloudscraper did not solve the challenge or returned non-JSON."
                            )
                    # Try Playwright once if enabled
                    if use_playwright_on_403 and PLAYWRIGHT_AVAILABLE:
                        data = try_playwright_once()
                        if isinstance(data, dict):
                            if data.get("status") == "success":
                                resp_text = data.get("response", "")
                                print(
                                    "⚠️ Playwright-backed request returned status=success — returning response."
                                )
                                return resp_text
                            else:
                                print(
                                    "→ Playwright returned JSON but not status==success (snippet):"
                                )
                                try:
                                    print(json.dumps(data, indent=2)[:1000])
                                except Exception:
                                    print(data)
                        else:
                            print(
                                "→ Playwright did not solve the challenge or returned non-JSON."
                            )

                    # keep original behavior: backoff and continue infinite attempts
                    print(
                        "                                                                                 "
                    )
                    print(
                        "Attempt continuing after Cloudflare detection; backing off before next attempt."
                    )
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    configured_timeout = min(
                        configured_timeout + timeout_growth, max_timeout_cap
                    )
                    continue

                # Handle client errors (4xx) - log snippet and keep retrying indefinitely
                if 400 <= status < 500:
                    print(
                        "                                                                                 "
                    )
                    print(f"⚠️ Client error HTTP {status}. Response snippet:")
                    print((body or "")[:1000])
                    print("Attempt continuing (infinite retry mode). Backing off...")
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    configured_timeout = min(
                        configured_timeout + timeout_growth, max_timeout_cap
                    )
                    continue

                # Handle server errors (5xx) or rate limits
                if status == 429 or 500 <= status < 600:
                    print(
                        "                                                                                 "
                    )
                    print(
                        f"⚠️ Server error / rate-limit HTTP {status}. Response snippet:"
                    )
                    print((body or "")[:1000])
                    print("Attempt continuing after backoff...")
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    configured_timeout = min(
                        configured_timeout + timeout_growth, max_timeout_cap
                    )
                    continue

                # Try parse JSON
                ct = (headers.get("Content-Type") or "").lower()
                parsed = None
                if "application/json" in ct or (
                    isinstance(body, str)
                    and (body.strip().startswith("{") or body.strip().startswith("["))
                ):
                    try:
                        parsed = json.loads(body) if body else {}
                    except Exception:
                        print(
                            "                                                                                 "
                        )
                        print("⚠️ Failed to parse JSON response. Snippet:")
                        print((body or "")[:1000])
                        print("Will continue retrying...")
                        time.sleep(backoff + random.random())
                        backoff = min(backoff * 2, max_backoff)
                        configured_timeout = min(
                            configured_timeout + timeout_growth, max_timeout_cap
                        )
                        continue
                else:
                    # non-json non-html: log and continue
                    print(
                        "                                                                                 "
                    )
                    print(
                        "⚠️ Received non-JSON response (not an HTML challenge). Snippet:"
                    )
                    print((body or "")[:1000])
                    print("Attempt continuing (infinite retry mode).")
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    configured_timeout = min(
                        configured_timeout + timeout_growth, max_timeout_cap
                    )
                    continue

                # If parsed JSON exists, check for expected success
                if isinstance(parsed, dict) and parsed.get("status") == "success":
                    resp_text = parsed.get("response", "")
                    print(
                        "                                                                                 "
                    )
                    print("✅ API returned status=success. Returning response.")
                    return resp_text

                # JSON present but not success -> log and continue
                print(
                    "                                                                                 "
                )
                print("→ API returned JSON but not status==success. Payload snippet:")
                try:
                    print(json.dumps(parsed, indent=2)[:2000])
                except Exception:
                    print(parsed)
                print("Attempt continuing (infinite retry mode).")
                time.sleep(backoff + random.random())
                backoff = min(backoff * 2, max_backoff)
                configured_timeout = min(
                    configured_timeout + timeout_growth, max_timeout_cap
                )
                continue

            except KeyboardInterrupt:
                if spinner:
                    try:
                        spinner.stop()
                    except Exception:
                        pass
                print("Interrupted by user. Aborting send_message.")
                raise
            except Exception as e:
                if spinner:
                    try:
                        spinner.stop()
                    except Exception:
                        pass
                print(
                    "                                                                                 "
                )
                print("⚠️ Unexpected exception in send_message loop:", e)
                traceback.print_exc()
                print("Attempt continuing after backoff...")
                time.sleep(backoff + random.random())
                backoff = min(backoff * 2, max_backoff)
                configured_timeout = min(
                    configured_timeout + timeout_growth, max_timeout_cap
                )
                continue


# ---------------------- UTILITIES ----------------------
def _nice_join(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def ensure_ffmpeg_available() -> Optional[str]:
    """
    Try to find ffmpeg executable. Returns absolute path or None.
    Strategy:
     1) check system PATH
     2) check imageio-ffmpeg (preferred fallback)
     3) try to pip-install imageio-ffmpeg automatically (best-effort)
    Note: we avoid installing arbitrary 'ffmpeg' wheel that may not provide a binary.
    """
    import shutil, subprocess, sys

    # 1) system PATH
    exe = shutil.which("ffmpeg")
    if exe:
        print(f"[ffmpeg] found system ffmpeg at: {exe}")
        return exe

    # 2) try imageio-ffmpeg (it exposes ffmpeg binary via get_ffmpeg_exe)
    try:
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
        if ff:
            print(f"[ffmpeg] found imageio-ffmpeg binary at: {ff}")
            return ff
    except Exception:
        pass

    # 3) try to pip install imageio-ffmpeg (best-effort)
    try:
        print("[ffmpeg] attempting to pip install imageio-ffmpeg as a fallback...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "imageio-ffmpeg"]
        )
        import importlib

        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        if ff:
            print(f"[ffmpeg] installed and found imageio-ffmpeg binary at: {ff}")
            return ff
    except Exception as e:
        print(f"[ffmpeg] pip install fallback failed: {e}")

    print("[ffmpeg] ffmpeg not available on PATH and imageio-ffmpeg not available.")
    return None


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


def save_response(folder_name: str, file_name: str, content: str):
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Saved: {file_path}")


def extract_largest_bracketed(text: str) -> Optional[str]:
    matches = re.findall(r"\[([^\]]+)\]", text, flags=re.DOTALL)
    if not matches:
        return None
    largest = max(matches, key=lambda s: len(s))
    return largest.strip()


def extract_all_bracketed_blocks(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r"\[([^\]]+)\]", text, flags=re.DOTALL)]


def escape_for_coqui_tts(text: str) -> str:
    """
    Escapes and normalizes text for safe, natural-sounding Coqui TTS synthesis.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([.,!?;:])([^\s])", r"\1 \2", text)
    text = re.sub(r"--", "—", text)
    if not re.search(r"[.!?…]$", text):
        text += "."
    return text


def clean_script_text(script_text: str) -> str:
    s = script_text or ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Remove markdown formatting
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"__(.*?)__", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\*(.*?)\*", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"_(.*?)_", r"\1", s, flags=re.DOTALL)

    # Remove scene headings like INT., EXT., etc.
    s = re.sub(r"(?im)^\s*(INT|EXT|INT/EXT|INT\.|EXT\.).*$", "", s)

    # --- Remove any leading label followed by a colon (all caps, capitalized, lowercase, spaces) ---
    s = re.sub(r'(?m)^\s*[A-Za-z\s0-9\-\–\—\'"“”&.,]{1,}:\s*', "", s)

    # Remove lines that are just a colon
    s = re.sub(r"(?m)^\s*:\s*$", "", s)

    # Remove parenthetical directions
    s = re.sub(r"\([^)]*\)", "", s)

    # Remove the word NARRATOR
    s = re.sub(r"\bNARRATOR\b", "", s, flags=re.IGNORECASE)

    # Remove remaining asterisks
    s = s.replace("*", "")

    # Collapse multiple blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()

    # Fix spacing after punctuation
    s = re.sub(r'([.!?])\s*([A-Z0-9"\'])', r"\1 \2", s)

    return s


# ---------------------- STORY / ASSET PIPELINE ----------------------
class StoryPipeline:
    def __init__(
        self,
        api_url: str = "https://apifreellm.com/api/chat",
        default_timeout: int = 1000,
    ):
        self.chat = ChatAPI(url=api_url, default_timeout=default_timeout)

    def _build_script_prompt(
        self,
        niche: str,
        person: str,
        timing_minutes: int,
        words_per_minute: int = 250,
        topic: str = "",
    ) -> str:
        """
        Universal cinematic prompt with structure + writing rules (keeps all existing features).
        Improved for clarity and stronger enforcement of bracket-only output and word-count targets,
        without removing existing features.
        """
        words_target = timing_minutes * words_per_minute
        person_clean = str(person).strip()
        topic_clean = str(topic).strip()

        if person_clean:
            if topic_clean:
                person_section = (
                    f"Center the narrative on {person_clean} within the context of the topic '{topic_clean}'. "
                    "Tell their story with emotional depth: origins, defining first struggles, turning points, setbacks, breakthroughs, and legacy. "
                    "Weave 3-7 verifiable facts (dates, achievements, quirks) naturally into the storytelling to deepen stakes."
                )
            else:
                person_section = (
                    f"Center the narrative on {person_clean}. "
                    "Tell their story with emotional depth: origins, defining first struggles, turning points, setbacks, breakthroughs, and legacy. "
                    "Weave 3-7 verifiable facts (dates, achievements, quirks) naturally into the storytelling to deepen stakes."
                )
        else:
            if topic_clean:
                person_section = (
                    f"Do not center the story on a real person. Instead, write an original third-person story relevant to niche '{niche}' and topic '{topic_clean}'. "
                    "Create a protagonist who embodies the emotional essence of the topic and trace their origins, conflicts, transformations, and legacy."
                )
            else:
                person_section = (
                    f"Do not center the story on a real person. Instead, write an original third-person story relevant to niche '{niche}'. "
                    "Create a protagonist, trace origins, conflicts, transformation, and legacy."
                )

        # Strong, explicit bracket + word-count enforcement appended to the end of the prompt.
        prompt = (
            "You are an expert cinematic storyteller and YouTube scriptwriter.\n\n"
            f"{person_section}\n\n"
            f"Write a powerful, cinematic long-form storytelling script (minimum 10 minutes) about the subject above. "
            f"Length: produce approximately {words_target} words (this is ~{timing_minutes} minutes at {words_per_minute} wpm). "
            f"Aim for between 90% and 110% if exact isn't possible, but prefer to meet the target within ±1% if you can.\n\n"
            "Tone: immersive, emotional, deeply human — like a master storyteller holding the audience’s attention from start to finish.\n\n"
            "Structure (use naturally, do not label in output):\n"
            "  - Hook (0:00-0:30): gripping opening that instantly pulls viewers with emotion, curiosity, or conflict.\n"
            "  - Act 1 — Origins: humble beginnings, key influences, early dreams, defining first struggles.\n"
            "  - Act 2 — Turning Points & Conflicts: failures, risks, doubts, betrayals; build tension and pacing.\n"
            "  - Act 3 — Breakthrough & Mastery: vivid sensory storytelling, decisive action, transformation.\n"
            "  - Act 4 — Legacy, Reflection & Lessons: emotional depth and 3 memorable takeaways that feel earned.\n"
            "  - Closing Line: one powerful quotable sentence to linger in the mind.\n\n"
            "Writing style (stick to these rules):\n"
            "  - Show, don't tell: use concrete sensory details, vivid imagery, and emotional interiority, any headings, any extra content.\n"
            "  - Tension-release rhythm: mix punchy sentences with slower reflective lines.\n"
            "  - Include brief quotes, internal thoughts, or imagined monologues for intimacy.\n"
            "  - Avoid repetition: do NOT repeat paragraphs or large blocks of text. Use callbacks and echoes instead of restatement.\n"
            "  - Keep transitions smooth and momentum-building; each scene should deepen emotion or advance narrative.\n"
            "  - Maintain authenticity; avoid exaggeration — emotional truth over hype.\n\n"
            "Formatting and output rules (CRITICAL):\n"
            "  - OUTPUT EXACTLY ONE PAIR OF SQUARE BRACKETS AND NOTHING ELSE: a single pair of square brackets containing ONLY the full script text. "
            "The assistant must not output any additional text, headings, labels, JSON, commentary, or metadata outside that single bracketed block. "
            "Example valid output: [The full script goes here ...].\n"
            "  - Count words in the usual sense. Produce exactly the target words if possible; otherwise get as close as possible within ±1% tolerance. "
            "If you cannot precisely hit the target, prefer to be slightly under rather than exceeding the upper bound.\n\n"
            "When you continue or condense content (if asked), do NOT repeat the last paragraph; continue seamlessly and maintain voice and pacing. "
            "Produce exactly one bracketed script block and nothing else: output a single opening bracket [ then the entire script content followed by a single closing bracket ] — include no other characters, whitespace, newlines, headings, labels, metadata, counts, commentary, instructions, fragments of the prompt, code fences, or BOM before or after; the bracketed text must be the complete script with no explanatory notes, stage directions, or parenthetical remarks not part of the script; if the script cannot be produced, return exactly []; the response must contain absolutely nothing else.\n"
            "Preserve important facts and beats. Generate now."
        )
        return prompt

    def generate_script(
        self,
        niche: str,
        person: str,
        timing_minutes: int = 10,
        words_per_minute: int = 250,
        timeout: Optional[int] = None,
        strict: bool = True,
        max_attempts: int = 100,
        topic: str = "",
    ) -> str:
        """
        COMPLETE generate_script implementation — infinite-retry until success (no internal caps).

        Behavior summary:
        - Keeps all original features from your initial design: bracketed-only outputs in strict mode,
          strengthen/retry prompts, continuation seeded with last paragraph, model condense attempts,
          deterministic trimming fallback, fuzzy dedupe, cleaning hooks, and saving.
        - **No wall-clock or API-call caps**: the method will keep retrying until a satisfactory
          non-empty cleaned script that meets the target (within tolerance) is produced. This is
          intentional per your request. The only "cap" is success.
        - The saved artifact is EXACTLY one bracketed block and nothing else: `[<cleaned script>]`.
        - The function will not save empty content. If a generated candidate cleans to empty, it will
          keep retrying.

        WARNING: Running without caps can consume unbounded resources. Use in a controlled environment.
        """

        # Derived values
        words_target = timing_minutes * words_per_minute
        tolerance = max(1, int(words_target * 0.01))

        # Base prompt
        prompt = self._build_script_prompt(
            niche=niche,
            person=person,
            timing_minutes=timing_minutes,
            words_per_minute=words_per_minute,
            topic=topic,
        )

        timeout = timeout or getattr(self.chat, "base_timeout", None)

        # Helper regex and utilities
        single_block_re = re.compile(r"^\s*\[[\s\S]*\]\s*$", flags=re.DOTALL)

        def _word_count(text: str) -> int:
            return len(re.findall(r"\w+", text or ""))

        def _get_paragraphs(text: str) -> list:
            if not text:
                return []
            paras = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", text) if p.strip()]
            if not paras:
                paras = [p.strip() for p in re.split(r"\n|\r\n", text) if p.strip()]
            return paras

        def _normalize(s: str) -> str:
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s or "").lower()).strip()

        def _remove_exact_and_fuzzy_duplicates(
            text: str, fuzzy_threshold: float = 0.90
        ) -> str:
            paras = _get_paragraphs(text)
            if not paras:
                return text
            kept = []
            normals = []
            for p in paras:
                np = _normalize(p)
                if not np:
                    continue
                duplicate = False
                if np in normals:
                    duplicate = True
                else:
                    for k in normals:
                        if len(np) < 40 or len(k) < 40:
                            continue
                        if (
                            difflib.SequenceMatcher(None, np, k).ratio()
                            >= fuzzy_threshold
                        ):
                            duplicate = True
                            break
                if not duplicate:
                    kept.append(p)
                    normals.append(np)
            return "\n\n".join(kept).strip()

        def _log_clean_state(label: str, text: str):
            wc = _word_count(text)
            remaining = words_target - wc
            print(
                f"[{label}] After cleaning: {wc} words; Remaining to target: {remaining} ({words_target}±{tolerance})"
            )
            return wc, remaining

        def _strengthen_prompt(
            base_prompt: str, previous_words: Optional[int], attempt_no: int
        ) -> str:
            extra = (
                "\n\nIMPORTANT: You MUST output EXACTLY ONE bracketed block and NOTHING ELSE. "
                "The output must start with '[' and end with ']' and contain no characters outside those brackets. "
                "The bracketed block should contain ONLY the full script text (no labels, no JSON, no commentary, no metadata, no tags). "
                "There must be no additional bracketed blocks, and no leading or trailing whitespace or blank lines outside the brackets. "
                f"Now adjust the script so that it contains exactly {words_target} words (count words in the usual sense—whitespace-separated). "
                f"If you cannot hit exactly {words_target}, produce a script that is as close as possible within ±{tolerance} words. "
                "Prioritize an exact match; if multiple outputs tie for closeness, any may be used. "
                "Do not include any explanations, diagnostics, or extra output — only the single bracketed script block."
                "Produce exactly one bracketed script block and nothing else: output a single opening bracket [ then the entire script content followed by a single closing bracket ] — include no other characters, whitespace, newlines, headings, labels, metadata, counts, commentary, instructions, fragments of the prompt, code fences, or BOM before or after; the bracketed text must be the complete script with no explanatory notes, stage directions, or parenthetical remarks not part of the script; if the script cannot be produced, return exactly []; the response must contain absolutely nothing else.\n"
            )
            if previous_words is not None:
                diff = words_target - previous_words
                extra += f"Previous attempt had {previous_words} words ({'short' if diff>0 else 'long' if diff<0 else 'exact'} by {abs(diff)}). "
                if diff > 0:
                    extra += "Extend and enrich naturally to reach the target. "
                elif diff < 0:
                    extra += (
                        "Tightly condense and remove redundancies (preserve beats). "
                    )
            extra += f"Attempt #{attempt_no}."
            return base_prompt + extra

        def _extract_candidate(resp: str) -> str:
            # prefer the largest bracketed block if present; otherwise return whole response
            try:
                brs = re.findall(r"\[[\s\S]*?\]", resp)
                if brs:
                    return max(brs, key=len)[1:-1].strip()
            except Exception:
                pass
            return resp.strip()

        def _heuristic_trim_to_target(text: str, target_words: int) -> str:
            # deterministic conservative trimming preserving anchors
            paras = _get_paragraphs(text)
            if not paras:
                return text
            protect_first = min(1, len(paras))
            protect_last = min(1, len(paras) - protect_first) if len(paras) > 1 else 0
            para_sents = []
            for p in paras:
                sents = [
                    s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", p) if s.strip()
                ]
                if not sents:
                    sents = [p.strip()]
                para_sents.append(sents)
            flat = []
            loc = []
            for pi, sents in enumerate(para_sents):
                for si, s in enumerate(sents):
                    flat.append(_normalize(s))
                    loc.append((pi, si))
            n = len(flat)
            if n == 0:
                return text
            scores = [0.0] * n
            for i in range(n):
                si = flat[i]
                if not si or len(si) < 20:
                    scores[i] = 0.0
                    continue
                tot = 0.0
                cnt = 0
                for j in range(n):
                    if i == j:
                        continue
                    sj = flat[j]
                    if not sj:
                        continue
                    tot += difflib.SequenceMatcher(None, si, sj).ratio()
                    cnt += 1
                scores[i] = (tot / cnt) if cnt else 0.0
            removable = []
            for idx, (pi, si) in enumerate(loc):
                if pi < protect_first or pi >= len(paras) - protect_last:
                    continue
                sent_text = para_sents[pi][si]
                wc_sent = _word_count(sent_text)
                removable.append((scores[idx], wc_sent, pi, si, sent_text))
            removable.sort(key=lambda x: (x[0], x[1]), reverse=True)
            current_text = text
            current_wc = _word_count(current_text)
            removals_by_para = {}
            for score, wc_sent, pi, si, stext in removable:
                if current_wc <= target_words:
                    break
                if len(para_sents[pi]) <= 1:
                    continue
                already = removals_by_para.get(pi, 0)
                if (already + 1) / len(para_sents[pi]) > 0.6:
                    continue
                para_sents[pi][si] = ""
                removals_by_para[pi] = already + 1
                new_paras = []
                for sents in para_sents:
                    sents_clean = [s for s in sents if s and s.strip()]
                    if sents_clean:
                        new_paras.append(" ".join(sents_clean))
                current_text = "\n\n".join(new_paras).strip()
                current_wc = _word_count(current_text)
                if current_wc <= target_words:
                    break
            if _word_count(current_text) > target_words:
                paras_now = _get_paragraphs(current_text)
                cand_idxs = [
                    i
                    for i in range(len(paras_now))
                    if i >= protect_first and i < len(paras_now) - protect_last
                ]
                cand_idxs_sorted = sorted(
                    cand_idxs, key=lambda i: _word_count(paras_now[i])
                )
                for i in cand_idxs_sorted:
                    if _word_count(current_text) <= target_words:
                        break
                    paras_now[i] = ""
                    current_text = "\n\n".join(
                        [p for p in paras_now if p.strip()]
                    ).strip()
            if _word_count(current_text) > target_words:
                words = re.findall(r"\S+", current_text)
                paras_now = _get_paragraphs(current_text)
                cum = []
                total = 0
                for p in paras_now:
                    wc_p = _word_count(p)
                    cum.append((total, total + wc_p))
                    total += wc_p
                last_protected_idx = (
                    max(0, len(paras_now) - protect_last)
                    if protect_last > 0
                    else len(paras_now)
                )
                keep_last_start_word = (
                    cum[last_protected_idx][0] if last_protected_idx < len(cum) else 0
                )
                allowable = max(0, keep_last_start_word + 3)
                if target_words <= allowable:
                    truncated = " ".join(words[:target_words])
                else:
                    truncated = " ".join(words[: max(target_words, allowable)])
                return truncated.strip()
            return current_text.strip()

        # Accumulation state
        accumulated = ""
        attempt = 0

        # Finalize helper — saves ONLY if non-empty and always as a single bracketed block
        def _finalize_and_save(text: str) -> Optional[str]:
            final_text = _remove_exact_and_fuzzy_duplicates(text, fuzzy_threshold=0.92)
            if final_text.startswith("[") and final_text.endswith("]"):
                final_text = final_text[1:-1].strip()
            final_text = final_text.strip()
            if not final_text:
                # refuse to save empty content
                print(
                    "⚠️ Final text is empty after cleaning — will not save. Continuing retries."
                )
                return None
            # Save EXACTLY one bracketed block and nothing else
            save_response(
                "generated_complete_script",
                "generated_complete_script.txt",
                f"[{final_text}]",
            )
            return final_text

        # Main loop: keep trying until we produce a non-empty cleaned script close to target
        while True:
            attempt += 1

            # --- generation phase ---
            if not accumulated:
                # initial generation
                req_prompt = (
                    prompt
                    if attempt == 1
                    else _strengthen_prompt(prompt, None, attempt)
                )
                try:
                    resp = self.chat.send_message(
                        req_prompt,
                        timeout=timeout,
                        spinner_message=f"Generating initial script (attempt {attempt})...",
                    )
                except Exception as e:
                    print(f"⚠️ send_message failed on generation attempt {attempt}: {e}")
                    # exponential backoff (bounded) before retrying to avoid tight infinite loops
                    sleep_sec = min(8 + random.random() * 2, 30)
                    print(f"Sleeping {sleep_sec:.1f}s before retrying generation...")
                    time.sleep(sleep_sec)
                    continue

                # In strict mode we insist the model returns a bracketed block; otherwise we accept best candidate
                if strict:
                    if not single_block_re.match(resp):
                        print(
                            "⚠️ Strict mode: response did not contain a single bracketed block — retrying."
                        )
                        time.sleep(0.2)
                        continue
                candidate = _extract_candidate(resp)
                try:
                    cleaned_candidate = clean_script_text(candidate) or candidate
                except Exception:
                    cleaned_candidate = candidate
                cleaned_candidate = _remove_exact_and_fuzzy_duplicates(
                    cleaned_candidate, fuzzy_threshold=0.90
                )
                wc, remaining = _log_clean_state("Initial", cleaned_candidate)

                # If too long, attempt model condense then deterministic trim
                if wc > words_target + tolerance:
                    condense_tries = min(6, max(1, max_attempts - attempt))
                    prev_long = wc
                    condensed_candidate = cleaned_candidate
                    for ct in range(condense_tries):
                        condense_prompt = (
                            "You were given a previously generated script (below). The cleaned version currently has "
                            f"{prev_long} words, but it must be reduced to exactly {words_target} words (or as close as possible within ±{tolerance}). "
                            "Tighten and condense the text: remove redundancies, merge sentences, shorten descriptive passages, and preserve the original narrative structure, beats, and meaning. "
                            "DO NOT invent new sections, scenes, characters, or facts. Do NOT change the sequence of events, character names, perspective, or core details. "
                            "Keep tone, tense, and voice consistent with the original. Prefer preserving essential lines and emotional beats even when shortening. "
                            "Produce exactly one bracketed script block and nothing else: output a single opening bracket [ then the entire script content followed by a single closing bracket ] — include no other characters, whitespace, newlines, headings, labels, metadata, counts, commentary, instructions, fragments of the prompt, code fences, or BOM before or after; the bracketed text must be the complete script with no explanatory notes, stage directions, or parenthetical remarks not part of the script; if the script cannot be produced, return exactly []; the response must contain absolutely nothing else.\n"
                            "Output EXACTLY ONE bracketed block and NOTHING ELSE — the bracketed block must contain only the full revised script text (no extra whitespace, commentary, metadata, or explanation).\n\n"
                            "PREVIOUS_SCRIPT_BEGIN\n"
                            f"{condensed_candidate}\n"
                            "PREVIOUS_SCRIPT_END\n"
                        )
                        try:
                            cond_resp = self.chat.send_message(
                                condense_prompt,
                                timeout=timeout,
                                spinner_message=f"Condensing (try {ct+1}/{condense_tries})...",
                            )
                        except Exception as e:
                            print(f"⚠️ Condense send_message failed: {e}")
                            break
                        if strict and not single_block_re.match(cond_resp):
                            print(
                                "⚠️ Strict mode: condense response not bracketed — retrying condense."
                            )
                            time.sleep(0.2)
                            continue
                        cond_inner = _extract_candidate(cond_resp)
                        try:
                            cond_clean = clean_script_text(cond_inner) or cond_inner
                        except Exception:
                            cond_clean = cond_inner
                        cond_clean = _remove_exact_and_fuzzy_duplicates(
                            cond_clean, fuzzy_threshold=0.90
                        )
                        cond_wc, _ = _log_clean_state(f"Condense {ct+1}", cond_clean)
                        if abs(cond_wc - words_target) <= tolerance:
                            accumulated = cond_clean
                            break
                        if cond_wc < prev_long:
                            condensed_candidate = cond_clean
                            prev_long = cond_wc
                            time.sleep(0.2)
                            continue
                        break
                    # if condense loop failed to meet target exactly, run deterministic trim
                    if not accumulated:
                        trimmed = _heuristic_trim_to_target(
                            condensed_candidate, words_target
                        )
                        try:
                            trimmed_clean = clean_script_text(trimmed) or trimmed
                        except Exception:
                            trimmed_clean = trimmed
                        accumulated = _remove_exact_and_fuzzy_duplicates(
                            trimmed_clean, fuzzy_threshold=0.90
                        )
                else:
                    # candidate is short or near-target — accept as accumulated block
                    accumulated = cleaned_candidate

                # If already within tolerance, attempt to save (must be non-empty)
                acc_wc = _word_count(accumulated)
                if abs(acc_wc - words_target) <= tolerance:
                    res = _finalize_and_save(accumulated)
                    if res is not None:
                        return res
                    else:
                        # didn't save (empty); clear accumulated and continue
                        accumulated = ""
                        continue

                # otherwise loop continues to request continuations
                continue

            # --- continuation phase: we have an accumulated block that is short ---
            acc_wc = _word_count(accumulated)
            remaining = max(0, words_target - acc_wc)
            last_para = (
                _get_paragraphs(accumulated)[-1] if _get_paragraphs(accumulated) else ""
            )

            cont_prompt = (
                "You are an expert cinematic scriptwriter and continuity editor. "
                "Below is a script already generated (PREV_BEGIN / PREV_END). Continue it seamlessly from the last paragraph so the final combined script reaches "
                f"approximately {words_target} words (add about {remaining} words). Do NOT repeat the last paragraph; continue naturally. Maintain voice, pacing, and narrative logic. "
                "Output ONLY the continuation text (no brackets, no labels, no metadata). Avoid repeating entire paragraphs; use callbacks, echoes, and thematic callbacks instead.\n\n"
                f"PREV_BEGIN\n{accumulated}\nPREV_END\n\n"
                f"LAST_PARAGRAPH_BEGIN\n{last_para}\nLAST_PARAGRAPH_END\n\n"
                "GUIDELINES (follow these tightly):\n"
                "- Seamlessness: bridge directly from LAST_PARAGRAPH so the result reads as one continuous video_script — no jarring resets, no reintroductory exposition.\n"
                "- Middle-act mastery: prioritize rising action, conflict escalation, turning points, stakes increase, and micro-resolutions that propel the story forward.\n"
                "- Maintain characters, names, facts, tone, and tense exactly as present in PREV and LAST_PARAGRAPH. If a scene or character is implied, continue that thread unless explicitly contradicted.\n"
                "- Show, don't tell: prefer sensory details, short scenes, beats, and concrete actions over long explanation. Use short+long sentences rhythmically to control pacing.\n"
                "- Callbacks, not copy: reference earlier lines or imagery with subtle callback phrases (echo words, repeated motifs, similar imagery) rather than copying whole sentences.\n"
                "- Flow & transitions: use graceful transitions between beats or scenes (one-sentence visual transitions, cut-to, or a brief descriptive line) without headings, timestamps, or labels.\n"
                "- Scene economy: each paragraph should function as a micro-scene or beat — introduce a small change, reveal, decision, or escalation that advances momentum.\n"
                "- Dialogue & tags: if characters speak, keep speaker attribution consistent with prior format. Use realistic, concise dialogue that reveals character or motive.\n"
                "- Continuity safety: never contradict established facts (names, timeline, locations, relationships). If uncertain, favor neutral phrasing that preserves continuity.\n"
                "- Length control: aim for ~{words_target} total words. If you overshoot slightly that's fine; if you undershoot, continue until the target is sensibly reached. When within ~3-7% of the target, create a satisfying mini-cliff or logical segue for the next batch.\n"
                "- Formatting: plain paragraphs only (no lists, no headers, no code). Keep natural paragraph breaks for beats. No editorial comments, no analysis, no instructions to the reader.\n"
                "- Safety & quality: keep language appropriate for a wide audience; avoid gratuitous profanity unless already present and integral to character voice.\n\n"
                "Produce exactly one bracketed script block and nothing else: output a single opening bracket [ then the entire script content followed by a single closing bracket ] — include no other characters, whitespace, newlines, headings, labels, metadata, counts, commentary, instructions, fragments of the prompt, code fences, or BOM before or after; the bracketed text must be the complete script with no explanatory notes, stage directions, or parenthetical remarks not part of the script; if the script cannot be produced, return exactly []; the response must contain absolutely nothing else.\n"
                "Deliver a single continuous piece that a human editor could paste after PREV_BEGIN and have the full script read as one complete, cinematic video script.\n"
            )
            try:
                cont_resp = self.chat.send_message(
                    cont_prompt,
                    timeout=timeout,
                    spinner_message=f"Continuation attempt (overall attempt {attempt})...",
                )
            except Exception as e:
                print(f"⚠️ Continuation send_message failed: {e}")
                time.sleep(0.2)
                continue

            # In strict mode we accept only continuation text (model may return unbracketed continuation)
            cont_candidate = _extract_candidate(cont_resp)
            try:
                cont_clean = clean_script_text(cont_candidate) or cont_candidate
            except Exception:
                cont_clean = cont_candidate

            # Append and dedupe
            combined = (accumulated.rstrip() + "\n\n" + cont_clean.strip()).strip()
            combined = _remove_exact_and_fuzzy_duplicates(
                combined, fuzzy_threshold=0.90
            )
            new_wc, remaining_after = _log_clean_state("After append", combined)

            if new_wc <= acc_wc:
                # no progress — try stronger regeneration append
                print(
                    "⚠️ Continuation produced no net progress; will retry with stronger generation prompt."
                )
                prompt = _strengthen_prompt(prompt, acc_wc, attempt + 1)
                gen_prompt = (
                    "Produce a new script section that continues the narrative below and does not repeat existing paragraphs. "
                    f"Aim to add about {remaining} words. Output EXACTLY ONE bracketed block with only the new section text.\n\nPREV_BEGIN\n{accumulated}\nPREV_END\n"
                )
                try:
                    gen_resp = self.chat.send_message(
                        gen_prompt,
                        timeout=timeout,
                        spinner_message="Generating appended chunk...",
                    )
                except Exception as e:
                    print(f"⚠️ Append-generation failed: {e}")
                    time.sleep(0.2)
                    continue
                if strict and not single_block_re.match(gen_resp):
                    print("⚠️ Strict mode: append generation not bracketed — retrying.")
                    time.sleep(0.2)
                    continue
                gen_candidate = _extract_candidate(gen_resp)
                try:
                    gen_clean = clean_script_text(gen_candidate) or gen_candidate
                except Exception:
                    gen_clean = gen_candidate
                new_combined = (
                    accumulated.rstrip() + "\n\n" + gen_clean.strip()
                ).strip()
                new_combined = _remove_exact_and_fuzzy_duplicates(
                    new_combined, fuzzy_threshold=0.90
                )
                new_wc2, remaining2 = _log_clean_state(
                    "After regeneration append", new_combined
                )
                if new_wc2 > acc_wc:
                    accumulated = new_combined
                    continue
                else:
                    time.sleep(0.2)
                    continue

            # progress made
            accumulated = combined

            # if reached or exceeded target -> finalize
            if _word_count(accumulated) >= words_target:
                # if over target, trim conservatively
                if _word_count(accumulated) > words_target + tolerance:
                    trimmed = _heuristic_trim_to_target(accumulated, words_target)
                    try:
                        trimmed_clean = clean_script_text(trimmed) or trimmed
                    except Exception:
                        trimmed_clean = trimmed
                    accumulated = _remove_exact_and_fuzzy_duplicates(
                        trimmed_clean, fuzzy_threshold=0.92
                    )
                res = _finalize_and_save(accumulated)
                if res is not None:
                    return res
                else:
                    # didn't save (empty) — reset and continue
                    accumulated = ""
                    continue

            # slight pause before next continuation (keeps CPU friendly)
            time.sleep(0.05)

        # This loop is intended to run until succeed; reaching here is unexpected
        raise RuntimeError("generate_script failed to terminate normally.")

    def _split_text_into_n_parts(self, text: str, n: int) -> List[str]:
        """
        Split `text` into `n` parts of roughly equal word count while trying to preserve sentence boundaries.
        Uses word-based targets: words_per_part = ceil(total_words / n).
        Returns list length == n (may contain empty strings when text is very short).
        """
        from typing import List
        import re, math

        text = (text or "").strip()
        if n <= 1 or not text:
            return [text] + [""] * (n - 1) if n > 0 else []

        # Sentence split (best-effort)
        sentences = re.split(r"(?<=[\.\?!])\s+", text)
        if not sentences:
            return [""] * n

        # words per sentence
        words_per_sentence = [len(re.findall(r"\S+", s)) for s in sentences]
        total_words = sum(words_per_sentence)
        if total_words == 0:
            return [""] * n

        target = math.ceil(total_words / n)

        parts = []
        current_sentences = []
        current_count = 0
        sentences_idx = 0

        while sentences_idx < len(sentences) and len(parts) < n - 1:
            w = words_per_sentence[sentences_idx]
            s = sentences[sentences_idx]

            # If adding exceeds target and we already have some content, close part
            if current_count + w > target and current_count > 0:
                parts.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_count = 0
                # do not increment sentences_idx yet (re-evaluate same sentence into next part)
                continue

            # otherwise add sentence
            current_sentences.append(s)
            current_count += w
            sentences_idx += 1

        # append remaining sentences as the final part
        # gather the rest of the sentences (including any not consumed)
        tail = []
        while sentences_idx < len(sentences):
            tail.append(sentences[sentences_idx])
            sentences_idx += 1

        # push current accumulation and tail merged as final part(s)
        if current_sentences:
            parts.append(" ".join(current_sentences).strip())
        if tail:
            parts.append(" ".join(tail).strip())

        # If fewer than n parts, pad with empty strings
        while len(parts) < n:
            parts.append("")

        # If more than n parts (rare), merge extras into the last part
        if len(parts) > n:
            parts = parts[: n - 1] + [" ".join(parts[n - 1 :]).strip()]

        return parts

    def generate_image_prompts(
        self,
        script_text: str,
        theme: str,
        img_number: int = 50,
        batch_size: int = 5,
        timeout_per_call: Optional[int] = None,
        save_each_batch: bool = True,
        parallel_batches: bool = False,  # new optional: speeds up large jobs but off by default
    ) -> List[str]:
        """
        Robust image prompt generation. Preserves strict per-paragraph quota semantics by default.
        - If parallel_batches=True, paragraph batches may be generated in parallel (still tries to respect quota).
        - Writes final validated prompts into image_response/image_prompts.txt (same behavior).
        """
        import os, re, math, random, time

        timeout_per_call = timeout_per_call or min(
            120, getattr(self.chat, "base_timeout", 120)
        )

        # quick guards
        if img_number <= 0:
            return []
        if batch_size <= 0:
            batch_size = 1

        response_folder = "image_response"
        final_fname = "image_prompts.txt"
        final_path = os.path.join(response_folder, final_fname)
        os.makedirs(response_folder, exist_ok=True)

        # compute paragraph splits
        num_paragraphs = max(1, math.ceil(img_number / max(1, batch_size)))
        try:
            paragraphs = self._split_text_into_n_parts(script_text, num_paragraphs)
        except Exception:
            # fallback
            from math import ceil

            words = script_text.split()
            avg = max(1, ceil(len(words) / num_paragraphs)) if script_text else 0
            paragraphs = [
                " ".join(words[i : i + avg]) for i in range(0, len(words), avg)
            ]
            while len(paragraphs) < num_paragraphs:
                paragraphs.append("")

        prompts = []
        seen_prompts = set()
        per_paragraph_cap = getattr(self, "max_prompt_attempts", None)
        per_paragraph_max = (
            per_paragraph_cap
            if (isinstance(per_paragraph_cap, int) and per_paragraph_cap > 0)
            else None
        )

        def _write_atomic(path, content):
            try:
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(content)
                os.replace(tmp, path)
            except Exception:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

        def _save_progress():
            try:
                content = "".join(f"[{p}]\n" for p in prompts)
                _write_atomic(final_path, content)
                print(f"💾 Saved progress: {len(prompts)} prompts -> {final_path}")
            except Exception as e:
                print(f"⚠️ Failed to save progress: {e}")

        # attempt builder and validator reuse from instance
        def _build_request(script_paragraph, req_count, seed):
            try:
                return self._build_image_prompt_request(
                    script_paragraph, theme, req_count
                )
            except Exception:
                return f"{script_paragraph}\n\nGenerate {req_count} ultra-detailed image prompts. Seed:{seed}"

        def _extract_blocks(resp_text):
            if not resp_text:
                return []
            # try bracketed first
            blocks = extract_all_bracketed_blocks(resp_text)
            if blocks:
                return [b.strip() for b in blocks if isinstance(b, str)]
            # fallback: lines of sufficient length
            lines = [ln.strip() for ln in resp_text.splitlines() if ln.strip()]
            candidates = []
            for ln in lines:
                if len(ln) < 12:
                    continue
                ln2 = re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip()
                if len(ln2.split()) >= 6:
                    candidates.append(ln2)
            return candidates

        def _is_valid(candidate):
            # reuse your same heuristic but keep it simpler and fast here to avoid subtle bugs
            if not candidate or len(candidate.strip()) < 8:
                return False
            c = candidate.strip("[]()\"'` ").strip()
            if len(c.split()) < 8 and not any(
                k in c.lower()
                for k in (
                    "camera",
                    "cinematic",
                    "photoreal",
                    "illustration",
                    "render",
                    "watercolor",
                )
            ):
                return False
            # reject obvious meta lines
            lower = c.lower()
            for bad in (
                "do you want",
                "please confirm",
                "which one",
                "error",
                "unable to",
                "service may be down",
                "example",
            ):
                if bad in lower:
                    return False
            return True

        # MAIN: sequential per-paragraph filling (keeps original semantics)
        for p_idx in range(num_paragraphs):
            if len(prompts) >= img_number:
                break
            paragraph = paragraphs[p_idx] or script_text or ""
            quota_for_para = min(batch_size, img_number - len(prompts))
            collected = 0
            attempts = 0
            print(f"\n➡️ Paragraph {p_idx+1}/{num_paragraphs} quota={quota_for_para}")

            while collected < quota_for_para and (
                per_paragraph_max is None or attempts < per_paragraph_max
            ):
                attempts += 1
                seed = random.randint(1000, 9999)
                need = quota_for_para - collected
                request_count = need
                enriched = f"{paragraph}\n\n# Paragraph {p_idx+1} | Attempt {attempts} | Seed {seed}\nGenerate {request_count} unique, highly-detailed image prompts. Output one bracketed prompt per line."
                req_body = _build_request(enriched, request_count, seed)

                try:
                    resp = self.chat.send_message(
                        req_body,
                        timeout=timeout_per_call,
                        spinner_message=f"Generating para {p_idx+1} prompts (attempt {attempts})...",
                    )
                except Exception as e:
                    print(f"  ⚠️ API error: {e}")
                    # save diagnostic
                    try:
                        _write_atomic(
                            os.path.join(
                                response_folder, f"para_{p_idx+1}_err_{attempts}.txt"
                            ),
                            str(e),
                        )
                    except Exception:
                        pass
                    time.sleep(0.6 + random.random() * 0.4)
                    continue

                blocks = _extract_blocks(resp)
                added = 0
                for blk in blocks:
                    if added >= request_count:
                        break
                    cand = blk.strip("[]()\"'` ").strip()
                    cand = re.sub(r"\s+", " ", cand)
                    if not _is_valid(cand):
                        continue
                    if theme.lower() not in cand.lower():
                        cand = f"{cand} | Theme: {theme}"
                    if cand in seen_prompts:
                        continue
                    seen_prompts.add(cand)
                    prompts.append(cand)
                    added += 1
                    collected += 1

                print(
                    f"  ⚪ added {added} prompts this attempt (collected {collected}/{quota_for_para}, total {len(prompts)}/{img_number})"
                )
                if save_each_batch and added:
                    _save_progress()
                time.sleep(0.3 + random.random() * 0.5)

            if collected < quota_for_para:
                print(
                    f"⚠️ Could not fill paragraph {p_idx+1} quota (collected {collected}/{quota_for_para}). Moving on."
                )

        # final dedupe & trim
        final_prompts = []
        for p in prompts:
            if p not in final_prompts:
                final_prompts.append(p)
            if len(final_prompts) >= img_number:
                break

        # write final canonical file
        try:
            _write_atomic(final_path, "".join(f"[{p}]\n" for p in final_prompts))
            print(
                f"\n✅ Final saved {len(final_prompts)}/{img_number} prompts -> {final_path}"
            )
        except Exception as e:
            print(f"\n⚠️ Failed final save: {e}")

        return final_prompts

    def _build_image_prompt_request(
        self, script_text: str, theme: str, img_number: int
    ) -> str:
        """
        Builds a robust, uniqueness-focused prompt for generating cinematic image prompts.
        This variant explicitly requests exhaustive micro-details and photographic specs; no length constraint.
        Output must be one bracketed prompt per line.

        This method prefers the rich prompt body you provided; it's preserved verbatim.
        """
        # If user previously supplied a custom builder on the instance, prefer that
        if hasattr(self, "_custom_image_prompt_builder") and callable(
            self._custom_image_prompt_builder
        ):
            try:
                return self._custom_image_prompt_builder(script_text, theme, img_number)
            except Exception:
                pass

        body = f"""
        You are an expert visual concept designer, cinematographer, production photographer, studio art director, and elite prompt engineer.

        🎬 TASK:
        Generate {img_number} completely UNIQUE, imaginative, and visually distinct image prompts 
        based on the paragraph below. Do NOT restrict prompt length — include exhaustive micro-details and photographic specifications.

        EACH PROMPT MUST:
        - Describe a single, self-contained scene (composition, subject(s), environment, mood).
        - Include camera type and lens focal length (e.g., 24mm / 50mm / 85mm), aperture (e.g., f/1.4), shutter speed, ISO, and perspective (wide/close/low/eye-level/aerial).
        - Specify exact shot framing (headroom, lead room, rule of thirds), and distance to subject.
        - Provide detailed lighting: time of day, key/fill/rim configuration, light quality (soft/hard), direction, color temperature or hex codes, and modifiers (softbox, grid, snoot, reflector).
        - Describe micro-details and tactile cues: fabric weave and stitching, skin pores, hair strands, lens dust, water droplets, rust, paint flake, reflections, specular highlights, bokeh character.
        - Define foreground / midground / background elements and depth layering plus atmospheric effects (fog, haze, smoke, rain, mist, particulate).
        - Provide color palette with primary/secondary accents and example hex codes; include tonal contrast and color grading notes (e.g., Kodak Portra, teal-orange, high-contrast filmic).
        - Offer stylistic references (artists, films, photographers) and choose a rendering style (photoreal, hyperreal 3D, cinematic CG, oil paint, anime cel-shaded).
        - Include post-processing/rendering guidance: grain size, lens flare style, denoiser/upscaler hints, sharpness, highlight rolloff.
        - If people are present, include ages, ethnicities, wardrobe layers, poses, micro-expressions, eye catchlights, and number of people. If objects/vehicles, include make/model, era, condition, and materials.
        - Provide exact aspect ratio (e.g., 16:9, 2:3, 9:16) where relevant.
        - Optionally append negative guidance (what to avoid).

        Always append `| Theme: {theme}` at the end of each prompt.

        🎞 SCRIPT PARAGRAPH (inspiration — use this to generate the prompts, do not output it verbatim unless required to convey the scene):
        {script_text.strip()}

        ⚙️ OUTPUT FORMAT:
        - Output exactly one prompt per line, wrapped in square brackets [ ... ].
        - No extra commentary, numbering, or metadata.
        - Preserve maximal detail — longer prompts are acceptable.
        """.strip()
        return body

    def generate_narration(
        self,
        script_text: str,
        timing_minutes: int,
        words_per_minute: int = 250,
        bracket_label: str = "narrate_prompt",
        timeout: Optional[int] = None,
    ) -> str:
        """
        Generate TTS-ready narration. Returns narration text (escaped for Coqui).
        Expects the API to return a single bracketed block that begins with 'narrate_prompt' label.
        """
        words_target = timing_minutes * words_per_minute
        prompt = (
            "Using the script below, write a single, polished third-person narration derived directly from it, suitable for Coqui TTS (Jenny). "
            f"Length: produce approximately {words_target} words (aim for 90%-110%), matching the {timing_minutes}-minute length. "
            "Preserve punctuation ('--', ':', '?') and natural pauses for voice synthesis. "
            "Formatting STRICT REQUIREMENTS: Output exactly ONE bracketed block and nothing else. The block must begin with the literal label 'narrate_prompt' on the first line, followed by a newline and then the narration text. "
            "Example valid output: [narrate_prompt\nThis is the narration text ...]. Do not include other labels or metadata inside the brackets. "
            "Script follows: ===\n"
            f"{script_text}\n"
            "===\n"
            "Generate now."
        )
        resp = self.chat.send_message(
            prompt,
            timeout=timeout or self.chat.base_timeout,
            spinner_message="Generating narration...",
        )
        narr_block = extract_largest_bracketed(resp)
        if narr_block:
            # remove label if present
            if narr_block.lower().startswith(bracket_label.lower()):
                parts = narr_block.split("\n", 1)
                narr_content = parts[1].strip() if len(parts) > 1 else ""
            else:
                narr_content = narr_block.strip()
        else:
            # fallback
            blocks = extract_all_bracketed_blocks(resp)
            narr_content = max(blocks, key=len) if blocks else resp.strip()
            print(
                "⚠️ Narration bracket not found exactly as requested — using best available content."
            )

        # Escape for Coqui TTS
        narr_content = escape_for_coqui_tts(narr_content)
        save_response("narration_response", "narrate_prompt.txt", f"[{narr_content}]")
        return narr_content

    def generate_youtube_metadata(
        self, script_text: str, timing_minutes: int = 10, timeout: Optional[int] = None
    ) -> str:
        prompt = (
            f"Act as a professional YouTube growth strategist and SEO copywriter. "
            f"Your task is to generate a *complete, optimized YouTube metadata package* for a {timing_minutes}-minute video. "
            "Base everything strictly on the script provided below.\n\n"
            "The output must include the following sections clearly labeled:\n"
            "1. **TITLE (max 90 characters, including spaces)** — Craft a click-enticing, emotion-driven, curiosity-filled viral title. "
            "Ensure it’s relevant to the script and includes strong SEO keywords.\n"
            "2. **DESCRIPTION (max 4900 characters, including spaces)** — Write a fully optimized and engaging description that:\n"
            "   - Hooks the viewer in the first two lines.\n"
            "   - Summarizes the video naturally using SEO-rich language.\n"
            "   - Includes time-stamped highlights if applicable.\n"
            "   - Encourages watch time, comments, likes, and subscriptions.\n"
            "   - Includes relevant affiliate links or placeholders (e.g., '👇 Check this out: [link]').\n"
            "   - Adds CTAs to subscribe or follow.\n"
            "   - Ends with keyword-rich hashtags and key phrases.\n"
            "3. **TAGS (comma-separated)** — Generate 20–30 high-ranking SEO tags (mix of short-tail and long-tail keywords relevant to the video topic).\n"
            "4. **HASHTAGS** — Include 10–20 trending, niche-relevant hashtags formatted like #ExampleTag.\n"
            "5. **CTA SECTION** — Write 2–3 persuasive call-to-action lines viewers will see in pinned comments or end screens.\n"
            "6. **THUMBNAIL TEXT IDEAS (3 options)** — Create short, bold text phrases (max 5 words) that grab attention on a thumbnail.\n\n"
            "Important Instructions:\n"
            "- Keep tone natural, human, and engaging — avoid robotic phrasing.\n"
            "- Never exceed character limits.\n"
            "- Optimize for click-through rate (CTR), viewer retention, and YouTube search visibility.\n"
            "- Use powerful emotional triggers (e.g., curiosity, fear of missing out, inspiration, surprise, or value-driven phrases).\n"
            "- Return clean, properly formatted output with labeled sections.\n\n"
            f"Here is the full video script for context:\n===\n{script_text}\n===\n"
            "Now generate the complete optimized metadata package.\n\n"
            "ADDITIONAL MACHINE-READABLE BLOCK (required):\n"
            "After the labeled human-readable sections above, include a **MACHINE-READABLE BLOCK** containing the exact fields below, each on its own line, with the value placed inside square brackets `[...]`. "
            "The script that will consume this output will parse these bracketed fields. Be careful: keep the labels exactly as shown (case-insensitive allowed) and place only one matching pair of square brackets per field.\n\n"
            "Required bracketed fields (the content inside brackets can be multi-line for DESCRIPTION):\n"
            "title [the title of the video]\n"
            "description [the description of video with all des, SEO tags, time stamps, CTAs... (can be multi-line)]\n"
            'tags ["tag1", "tag2", "long tail tag 3", ...]  # Provide as a Python/JSON-style list OR comma separated list\n'
            "hashtags [#tag1, #tag2, #tag3, ...]\n"
            "categoryId [any id]  # Use a numeric ID (e.g., 22) from the list: \n"
            "    1 Film & Animation\n"
            "    2 Autos & Vehicles\n"
            "    10 Music\n"
            "    15 Pets & Animals\n"
            "    17 Sports\n"
            "    19 Travel & Events\n"
            "    20 Gaming\n"
            "    22 People & Blogs\n"
            "    23 Comedy\n"
            "    24 Entertainment\n"
            "    25 News & Politics\n"
            "    26 Howto & Style\n"
            "    27 Education\n"
            "    28 Science & Technology\n"
            "    29 Nonprofits & Activism\n"
            "CTA [A short set of 2-3 CTAs separated by '||' or newlines]\n"
            'thumbnail_texts ["Option 1", "Option 2", "Option 3"]\n\n'
            "EXAMPLE MACHINE-READABLE BLOCK (must follow the same formatting exactly):\n"
            "title [How I Removed a Watermark in 60 Seconds]\n"
            "description [Hook line...\nFull description here with timestamps and links...]\n"
            'tags ["remove watermark","video inpainting","ffmpeg tutorial"]\n'
            "hashtags [#removewatermark, #ffmpeg, #tutorial]\n"
            "categoryId [22]\n"
            "CTA [Subscribe for more||Watch the next video for advanced tips]\n"
            'thumbnail_texts ["Remove Watermark Fast","No Logo!","Pro Trick"]\n\n'
            "IMPORTANT: The consumer script will parse values between the first matching '[' and its corresponding ']' for each field. "
            "Do not include extra square brackets inside the field value. After generating this output, save the entire output to a file named 'youtube_metadata.txt' and the script will read it.\n\n"
            "Now generate the complete optimized metadata package and include the MACHINE-READABLE BLOCK at the end exactly as specified."
        )

        resp = self.chat.send_message(
            prompt,
            timeout=timeout or self.chat.base_timeout,
            spinner_message="Generating YouTube metadata...",
        )
        save_response("youtube_response", "youtube_metadata.txt", resp)
        return resp


# ---------------------- EXAMPLE USAGE ----------------------
if __name__ == "__main__":
    start = time.time()

    pipeline = StoryPipeline(
        api_url="https://apifreellm.com/api/chat", default_timeout=1000
    )

    # --- Example 1: Only generate the story/script (BRACKETED single block file saved) ---
    script = pipeline.generate_script(
        niche="Preschool-early-elementary children",
        person="",
        timing_minutes=10,
        timeout=100,
        topic="The Little Cloud Painter",
    )
    print("\n--- Script (first 400 chars) ---")
    print(script[:400])

    # --- Example 2: Generate image prompts in batches (helps with timeouts) ---
    # If your API frequently times out, reduce batch_size (e.g., 2 or 3)
    image_prompts = pipeline.generate_image_prompts(
        script_text=script,
        theme="water color illustrations, children's book, whimsical, vibrant colors Creativity + sharing + emotional regulation",
        img_number=150,  # set smaller for testing; set 50 in production
        batch_size=30,  # reduce if timeouts happen frequently
        timeout_per_call=100,
    )
    print(f"\nGenerated {len(image_prompts)} image prompts (sample):")
    for p in image_prompts[:5]:
        print(f"[{p}]")

    # --- Example 3: Only generate narration (suitable for Coqui) ---
    narration_text = pipeline.generate_narration(script, timing_minutes=10)
    print("\nNarration saved and ready for TTS")

    # # --- Example 4: Generate youtube metadata ---
    yt_meta = pipeline.generate_youtube_metadata(script, timing_minutes=10)
    print("\nYouTube metadata saved.")

    print("\n✅ Done. Use the pipeline methods to call only what you need.")
    end = time.time()
    log_execution_time(start, end)
