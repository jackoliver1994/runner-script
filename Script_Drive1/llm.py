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

from weakref import proxy
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
import traceback
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
    def __init__(self, url: str = "https://apifreellm.com/api/chat", default_timeout: int = 1000):
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        # keep the old attribute name used elsewhere:
        self.default_headers = self.headers
        self.base_timeout = default_timeout

    def fetch_proxies_from_proxyscrape(
        self, max_proxies: int = 50000, timeout: int = 10
    ) -> list:
        """
        Best-effort fetch of proxy list lines 'ip:port' from common ProxyScrape endpoints.
        Returns a list of unique 'ip:port' strings shuffled.
        Note: requesting 10_000_000 is accepted but provider/network will limit actual results.
        """
        import requests, random

        endpoints = [
            "https://api.proxyscrape.com/?request=displayproxies&proxytype=http&timeout=10000&country=all&ssl=all&anonymity=all",
            "https://api.proxyscrape.com/?request=displayproxies&proxytype=socks4&timeout=10000&country=all&ssl=all&anonymity=all",
            "https://api.proxyscrape.com/?request=displayproxies&proxytype=socks5&timeout=10000&country=all&ssl=all&anonymity=all",
            "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all",
        ]
        proxies = []
        try:
            s = requests.Session()
            for url in endpoints:
                if len(proxies) >= max_proxies:
                    break
                try:
                    r = s.get(url, timeout=min(timeout, 30))
                    if r.status_code != 200:
                        continue
                    text = (r.text or "").strip()
                    if not text:
                        continue
                    for ln in text.splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        # simple validation ip:port (very permissive)
                        if ":" in ln and len(ln.split(":")) >= 2:
                            proxies.append(ln)
                            if len(proxies) >= max_proxies:
                                break
                except Exception:
                    # ignore endpoint errors and continue
                    continue
        except Exception:
            pass
        # dedupe while preserving order-ish then shuffle
        proxies = list(dict.fromkeys(proxies))
        random.shuffle(proxies)
        return proxies

    def _pick_next_proxy_and_impersonation(self):
        """
        Rotate to next proxy from cached pool (fetch lazily if needed) and create headers impersonation.
        Returns (proxy_string_or_none, headers_dict).
        """
        import random, time

        # lazy init pool attribute
        if not hasattr(self, "_proxy_pool") or not self._proxy_pool:
            # fetch lazily (may be large)
            try:
                max_fetch = getattr(self, "max_proxy_fetch", 10000000)
                self._proxy_pool = self.fetch_proxies_from_proxyscrape(
                    max_proxies=max_fetch
                )
            except Exception:
                self._proxy_pool = []

        if not hasattr(self, "_proxy_index"):
            self._proxy_index = 0

        if not self._proxy_pool:
            proxy = None
        else:
            proxy = self._proxy_pool[self._proxy_index % len(self._proxy_pool)]
            self._proxy_index = (self._proxy_index + 1) % len(self._proxy_pool)

        ua_pool = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
        ]
        ua = random.choice(ua_pool)
        headers = dict(getattr(self, "headers", {"Content-Type": "application/json"}))
        headers.update(
            {
                "User-Agent": ua,
                "Accept": "application/json, text/plain, */*",
                # X-Forwarded-For optional mild impersonation
                "X-Forwarded-For": (
                    proxy.split(":")[0] if proxy else headers.get("X-Forwarded-For", "")
                ),
            }
        )
        return proxy, headers

    def send_message(
        self,
        message: str,
        timeout: int = None,
        spinner_message: str = "Waiting for response...",
        initial_backoff: float = 2.0,
        max_backoff: float = 8.0,
        timeout_growth: int = 100,
        max_timeout_cap: int = 1000000,
        use_cloudscraper_on_403: bool = True,
        use_playwright_on_403: bool = False,
        proxy: str = None,
        specific_error: list = [
            "response status: 403",
            "detected cloudflare-like anti-bot page",
            "Response status=403 length=9245",
            "Detected HTML challenge",
            "Cloudflare-like content",
            "Response status=403",
			"cloudflare", 
			"detected cloudflare", 
			"access denied", 
			"anti-bot"
        ],
    ):
        """
        Robust send_message preserving infinite retry semantics and all original features.
        Improvements:
        - start/stop LoadingSpinner when spinner_message is provided (non-blocking)
        - use log() wrapper (print with flush) for immediate logs
        - cap read timeouts to avoid C-level blocking of worker threads
        - improved exception logging (tracebacks)
        - keep infinite retry by default (while True:)
        """
        # local helper for consistent immediate logging
        def log(*args, **kwargs):
            # small timestamp for easier tracing
            ts = time.strftime("%H:%M:%S")
            kwargs.setdefault("flush", True)
            print(f"[{ts}]", *args, **kwargs)

        # Defensive defaults
        configured_timeout = int(timeout or getattr(self, "base_timeout", 60))
        configured_timeout = min(configured_timeout, max_timeout_cap)
        attempt = 0
        backoff = initial_backoff
        current_proxy = proxy or getattr(self, "_current_proxy", None)

        # Spinner: start only if spinner_message is truthy. Always stop in finally.
        spinner = None
        if spinner_message:
            try:
                spinner = LoadingSpinner(spinner_message)
                spinner.start()
            except Exception as e:
                log("⚠️ Failed to start spinner:", e)

        try:
            # optional libs
            try:
                import cloudscraper
            except Exception:
                cloudscraper = None

            PLAYWRIGHT_AVAILABLE = False
            if use_playwright_on_403:
                try:
                    from playwright.sync_api import sync_playwright
                    PLAYWRIGHT_AVAILABLE = True
                except Exception:
                    PLAYWRIGHT_AVAILABLE = False

            # Build canonical specific_error list
            specific_error = (
                specific_error
                if specific_error is not None
                else getattr(self, "specific_error", [])
            )
            specific_error = [s.lower() for s in specific_error or []]

            # Internal single_request used by the thread pool.
            def single_request(sess, req_timeout, use_headers, use_proxy):
                """
                Robust single_request — executed in a ThreadPoolExecutor to avoid blocking main thread.
                Returns dict with structure: {"ok": bool, "status_code": int, "body": str, "headers": dict} OR {"ok": False, "exc": Exception}
                """
                try:
                    if sess is None:
                        import requests
                        sess = requests.Session()
                    # avoid env proxies interfering (we manage proxies explicitly)
                    try:
                        sess.trust_env = False
                    except Exception:
                        pass

                    headers_local = dict(use_headers or {})
                    proxies_cfg = None
                    if use_proxy:
                        proxies_cfg = {"http": f"http://{use_proxy}", "https": f"http://{use_proxy}"}

                    # CAP read timeout: avoid C-level indefinite blocking (connect, read)
                    # req_timeout may be seconds; enforce min 10s, max 300s by default
                    read_timeout = int(req_timeout) if req_timeout else 60
                    read_timeout = min(max(10, read_timeout), 300)

                    # Prefer using a tuple (connect, read)
                    timeout_arg = (10, read_timeout)

                    # many endpoints in your code use json payload
                    import requests
                    r = sess.post(
                        self.url,
                        json={"message": message},
                        headers=headers_local,
                        timeout=timeout_arg,
                        proxies=proxies_cfg,
                    )

                    headers_resp = r.headers or {}
                    body = r.text if hasattr(r, "text") else (r.content.decode("utf-8", "ignore") if hasattr(r, "content") else None)
                    return {"ok": True, "status_code": r.status_code, "body": body, "headers": headers_resp}
                except Exception as e:
                    # return full exception object so outer code can inspect type/trace
                    return {"ok": False, "exc": e, "trace": traceback.format_exc()}

            log("send_message: starting infinite retry loop (preserving infinite behavior).")
            while True:
                attempt += 1
                log(f"Attempt #{attempt} — timeout={int(configured_timeout)}s — proxy={current_proxy or '<none>'}")

                # run request with thread pool to handle possible hang
                try:
                    with cf.ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(single_request, getattr(self, "_session", None), configured_timeout, getattr(self, "default_headers", {}), current_proxy)
                        # choose an upper bound to wait for the worker thread result so main thread can recover.
                        wrapper = None
                        wait_timeout = min(configured_timeout, 300) + 5
                        try:
                            wrapper = fut.result(timeout=wait_timeout)
                        except cf.TimeoutError:
                            log("⚠️ Worker timed out (fut.result). Attempting to cancel worker thread.")
                            try:
                                fut.cancel()
                            except Exception:
                                pass
                            # recreate session to avoid stuck/broken session state (non-destructive)
                            try:
                                self._session = None
                            except Exception:
                                pass
                            # backoff and continue the infinite retry loop
                            time.sleep(backoff + random.random())
                            backoff = min(backoff * 2, max_backoff)
                            configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)
                            continue
                except Exception as e:
                    log("⚠️ Unexpected executor error:", e)
                    log(traceback.format_exc())
                    # backoff and continue (preserve infinite behavior)
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)
                    continue

                # wrapper now contains the result of single_request
                if not wrapper:
                    log("⚠️ No wrapper result returned from worker — retrying.")
                    configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    continue

                if wrapper.get("ok") is False:
                    # network or other exception occurred inside worker
                    exc = wrapper.get("exc")
                    log(f"⚠️ Network exception on attempt #{attempt}: {type(exc).__name__} — {getattr(exc, 'args', '')}")
                    log(wrapper.get("trace"))
                    # increase timeouts/backoff and continue infinite retry
                    configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    continue

                # successful low-level HTTP request; inspect response
                status_code = wrapper.get("status_code", 0)
                body = wrapper.get("body", "")
                headers_resp = wrapper.get("headers", {}) or {}

                # debug small snippet
                ct = (headers_resp.get("Content-Type") or "").lower()
                if status_code == 403 and use_cloudscraper_on_403 and cloudscraper:
                    log("→ Detected 403; attempting cloudscraper.")
                    try:
                        scraper = cloudscraper.create_scraper()
                        r2 = scraper.post(self.url, json={"message": message}, headers=getattr(self, "default_headers", {}), timeout=(10, min(configured_timeout, 300)))
                        body = r2.text
                        headers_resp = r2.headers or {}
                        status_code = r2.status_code
                    except Exception:
                        log("⚠️ cloudscraper attempt failed:", traceback.format_exc())

                # Now handle common failure vs success heuristics (preserve original logic)
                # If response is JSON, parse and check for 'status==success' or other indicators
                parsed = None
                try:
                    if "application/json" in ct or (isinstance(body, str) and (body.strip().startswith("{") or body.strip().startswith("["))):
                        parsed = json.loads(body) if body else {}
                except Exception:
                    log("⚠️ Failed to parse JSON. Response snippet:")
                    log((body or "")[:2000])

                # Original success detection: if parsed and parsed.get("status") == "success" or similar
                if parsed and (
                    (isinstance(parsed, dict) and parsed.get("status") == "success")
                    or ("message" in parsed and parsed.get("message"))
                ):
                    log("✅ API returned success. Returning response.")
                    # return the parsed payload or raw body, matching previous design
                    return parsed if parsed else body

                # Otherwise nothing useful — print snippet and retry (preserve infinite retry)
                log("→ API returned JSON but no 'status==success' or useful payload. Payload snippet:")
                try:
                    log(json.dumps(parsed, indent=2)[:2000])
                except Exception:
                    log(parsed)
                log("Retrying (same proxy).")
                time.sleep(backoff + random.random())
                backoff = min(backoff * 2, max_backoff)
                configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)

        finally:
            if spinner:
                try:
                    spinner.stop()
                except Exception:
                    # do not raise; spinner stop shouldn't break flow
                    pass

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
        allow_infinite: bool = True,               # NEW: keep infinite by default
        stop_event: Optional[Event] = None,
    ) -> str:
        """
        COMPLETE generate_script implementation — infinite-retry until success (no internal caps).

        This implementation:
        - Calls ChatAPI.send_message repeatedly until a satisfactory bracketed script is produced
        - Uses the helper functions defined earlier (cleaning, trimming, strengthening, extraction)
        - Preserves the infinite-retry option (allow_infinite) and supports external stop_event
        """

        # Derived values
        words_target = timing_minutes * words_per_minute
        tolerance = max(1, int(words_target * 0.01))

        # Base prompt
        base_prompt = self._build_script_prompt(
            niche=niche,
            person=person,
            timing_minutes=timing_minutes,
            words_per_minute=words_per_minute,
            topic=topic,
        )

        timeout = timeout or getattr(self.chat, "base_timeout", None)

        # Helpers already defined above are used: _word_count, _get_paragraphs, _normalize, etc.
        single_block_re = re.compile(r"^\s*\[[\s\S]*\]\s*$", flags=re.DOTALL)

        def _word_count(text: str) -> int:
            return len(re.findall(r"\w+", text or ""))

        def _extract_resp_text(resp):
            # Accept dicts (parsed JSON) or raw text
            if isinstance(resp, dict):
                # common keys to consider
                for k in ("message", "body", "text", "result", "data"):
                    v = resp.get(k) if isinstance(resp, dict) else None
                    if v:
                        return v if isinstance(v, str) else str(v)
                # fallback: stringify dict
                try:
                    return json.dumps(resp)
                except Exception:
                    return str(resp)
            else:
                return str(resp or "")

        def _finalize_and_save_if_good(candidate_text: str):
            # Clean dedupe and save exactly one bracketed block
            bracketed = _finalize_and_save(candidate_text)
            if bracketed:
                return bracketed
            return None

        attempt = 0
        backoff = 1.0
        accumulated = ""  # accumulate best candidates so abort can save something
        last_wc = None

        while True:
            attempt += 1

            # Respect external stop
            if stop_event is not None and stop_event.is_set():
                print(f"Stop requested (attempt {attempt}). Finalizing best-effort.")
                if accumulated:
                    return _finalize_and_save_if_good(accumulated)
                return None

            if not allow_infinite and attempt > max_attempts:
                print(f"Max attempts ({max_attempts}) reached in non-infinite mode. Finalizing best-effort.")
                if accumulated:
                    return _finalize_and_save_if_good(accumulated)
                return None

            # Prepare prompt: on 1st attempt use base_prompt, otherwise strengthen based on last_wc
            if attempt == 1:
                prompt_to_send = base_prompt
            else:
                prompt_to_send = _strengthen_prompt(base_prompt, last_wc, attempt)

            try:
                resp = self.chat.send_message(
                    prompt_to_send,
                    timeout=timeout or getattr(self.chat, "base_timeout", None),
                    spinner_message=f"Generating script (attempt {attempt})...",
                )
            except Exception as e:
                print(f"⚠️ API call failed on attempt {attempt}: {e}")
                # exponential backoff but keep infinite behavior
                time.sleep(min(30, backoff) + random.random() * 0.5)
                backoff = min(backoff * 1.8, 60.0)
                continue

            resp_text = _extract_resp_text(resp)
            candidate = _extract_candidate(resp_text)

            # Clean the candidate text (remove brackets, normalize, etc)
            cleaned = clean_script_text(candidate or "")
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1].strip()

            # If the assistant strictly returned bracketed block with label/metadata, extract largest bracket only
            if strict:
                # prefer the largest bracketed block found in original response (already done by _extract_candidate),
                # but still ensure cleaned is not empty
                pass

            wc = _word_count(cleaned)
            last_wc = wc
            print(f"[Attempt {attempt}] Candidate words={wc}; target={words_target} ±{tolerance}")

            # If cleaned empty -> try again (maybe the model refused); preserve resp_text as diagnostic
            if not cleaned:
                print("⚠️ Candidate cleaned to empty. Retrying after backoff.")
                time.sleep(min(5 + attempt * 0.5, 20))
                backoff = min(backoff * 1.8, 60.0)
                continue

            # Accumulate if this candidate adds fresh content (simple dedupe by normalized start)
            if cleaned and cleaned not in accumulated:
                accumulated = (accumulated + "\n\n" + cleaned).strip()

            # If within tolerance -> finalize and save
            if abs(wc - words_target) <= tolerance:
                print(f"✅ Candidate within tolerance: {wc} words. Finalizing and saving.")
                final = _finalize_and_save_if_good(cleaned)
                if final:
                    return final
                # If couldn't save (somehow), keep retrying
                print("⚠️ Finalize/save did not produce an artifact; continuing retries.")
                time.sleep(1 + random.random())
                continue

            # If too long: try deterministic trimming heuristics and finalize if good
            if wc > words_target + tolerance:
                print(f"→ Candidate too long ({wc} > {words_target}). Attempting heuristic trim.")
                trimmed = _heuristic_trim_to_target(cleaned, words_target)
                t_wc = _word_count(trimmed)
                print(f"   Trimmed to {t_wc} words.")
                if abs(t_wc - words_target) <= tolerance:
                    print("✅ Trimmed candidate is within tolerance. Finalizing and saving.")
                    final = _finalize_and_save_if_good(trimmed)
                    if final:
                        return final
                # If trimming couldn't reach target, continue with stronger condense instructions next attempt
                print("→ Trim did not reach acceptable length; retrying with stronger condense instruction.")
                time.sleep(0.5 + random.random() * 0.7)
                backoff = min(backoff * 1.5, 60.0)
                continue

            # If too short: extend/continue
            if wc < words_target - tolerance:
                print(f"→ Candidate too short ({wc} < {words_target}). Asking model to extend.")
                # Build a short continuation instruction that seeds with the last paragraph to avoid repeats
                paras = _get_paragraphs(cleaned)
                last_para = paras[-1] if paras else ""
                continue_prompt = (
                    f"You produced the script below (inside brackets). The script is currently {wc} words but we need exactly {words_target} words "
                    f"(±{tolerance}). Continue the story seamlessly from the last paragraph and expand naturally to meet the word target, "
                    f"preserving voice, beats, and sensory detail. Do NOT repeat previous paragraphs; continue the narrative. "
                    f"Seed (last paragraph):\n\n{last_para}\n\nNow continue. Output exactly one bracketed block and nothing else."
                )
                try:
                    cont_resp = self.chat.send_message(
                        continue_prompt,
                        timeout=timeout or getattr(self.chat, "base_timeout", None),
                        spinner_message=f"Extending script (attempt {attempt})...",
                    )
                except Exception as e:
                    print(f"⚠️ Continue call failed: {e}")
                    time.sleep(min(5, backoff) + random.random() * 0.5)
                    backoff = min(backoff * 1.8, 60.0)
                    continue

                cont_text = _extract_resp_text(cont_resp)
                cont_candidate = _extract_candidate(cont_text)
                cont_cleaned = clean_script_text(cont_candidate or "")
                if cont_cleaned.startswith("[") and cont_cleaned.endswith("]"):
                    cont_cleaned = cont_cleaned[1:-1].strip()

                # merge (append) and re-evaluate
                merged = (cleaned + "\n\n" + cont_cleaned).strip()
                merged = _remove_exact_and_fuzzy_duplicates(merged, fuzzy_threshold=0.92)
                merged_wc = _word_count(merged)
                print(f"   After continuation, merged words={merged_wc}")

                if abs(merged_wc - words_target) <= tolerance:
                    print("✅ Continued script meets the target. Finalizing and saving.")
                    final = _finalize_and_save_if_good(merged)
                    if final:
                        return final
                else:
                    # adopt merged as accumulated and continue retrying with strengthened prompt next loop
                    accumulated = merged
                    last_wc = merged_wc
                    time.sleep(0.6 + random.random() * 0.6)
                    backoff = min(backoff * 1.5, 60.0)
                    continue

            # Fallback safety backoff before next attempt
            time.sleep(min(3 + backoff, 30))
            backoff = min(backoff * 1.4, 60.0)

        # unreachable, but explicit
        return None

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

    pipeline.chat.max_proxy_fetch = 50000
    pipeline.chat.specific_error = [
        "✅ response status: 403",
        "⚠️ detected cloudflare-like anti-bot page",
    ]

    # --- Example 1: Only generate the story/script (BRACKETED single block file saved) ---
    script = pipeline.generate_script(
        niche="Preschool-early-elementary children",
        person="",
        timing_minutes=2,
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
        img_number=30,  # set smaller for testing; set 50 in production
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
