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
        """Internal spinner loop (daemon thread)."""
        try:
            while not self.stop_event.is_set():
                for char in self.spinner_chars:
                    if self.stop_event.is_set():
                        break
                    try:
                        sys.stdout.write("\r" + f"{self.message} {char}")
                        sys.stdout.flush()
                    except Exception:
                        pass
                    time.sleep(0.1)
        except Exception:
            # Do not allow spinner exceptions to crash the program
            try:
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
            except Exception:
                pass

    def start(self):
        """Start the spinner thread (idempotent)."""
        try:
            self.stop_event.clear()
            if self.thread and self.thread.is_alive():
                return
            self.thread = Thread(target=self.spin, daemon=True)
            self.thread.start()
        except Exception:
            # best-effort: don't crash if threading fails
            pass

    def stop(self):
        """Stop spinner and join thread (if started)."""
        try:
            if self.thread and self.thread.is_alive():
                self.stop_event.set()
                self.thread.join(timeout=2.0)
        except Exception:
            # swallow exceptions from join to avoid interfering with main flow
            pass
        finally:
            try:
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
            except Exception:
                pass


# ---------------------- CHAT API HANDLER ----------------------
class ChatAPI:
    def __init__(self, url: str = "https://apifreellm.com/api/chat", default_timeout: int = 1000):
        """
        Initialize ChatAPI.
        - Creates consistent header attributes used by send_message and cloudscraper fallback.
        - Keeps a persistent _session placeholder (recreated as needed).
        """
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        # keep the older attribute name used in some call sites for backward compatibility
        self.default_headers = dict(self.headers)
        self.base_timeout = default_timeout
        # session can be created lazily inside send_message/single_request
        self._session = None
        # current proxy (string like "ip:port") when rotating proxies
        self._current_proxy = None
        # proxy pool attrs are created lazily in _pick_next_proxy_and_impersonation
        self._proxy_pool = getattr(self, "_proxy_pool", [])
        self._proxy_index = getattr(self, "_proxy_index", 0)

    def fetch_proxies_from_proxyscrape(self, max_proxies: int = 50000, timeout: int = 10) -> list:
        """
        Best-effort fetch of proxy list lines 'ip:port' from ProxyScrape-like endpoints.
        Returns shuffled deduplicated list. Safe to call repeatedly (idempotent-ish).
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
            s.trust_env = False
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
                        if ":" in ln and len(ln.split(":")) >= 2:
                            proxies.append(ln)
                            if len(proxies) >= max_proxies:
                                break
                except Exception:
                    # ignore endpoint errors; continue others
                    continue
        except Exception:
            pass
        # dedupe while preserving first-seen ordering then shuffle
        proxies = list(dict.fromkeys(proxies))
        random.shuffle(proxies)
        return proxies

    def _pick_next_proxy_and_impersonation(self):
        """
        Rotate to next proxy and build headers with mild impersonation.
        Sets self._current_proxy and returns (proxy, headers).

        Improvements:
        - Lazy-fetch proxy pool if empty.
        - Persist chosen proxy in self._current_proxy.
        - Ensure headers built include randomized UA and X-Forwarded-For.
        """
        import random, time

        # lazy fetch if pool empty
        try:
            if not getattr(self, "_proxy_pool", None):
                max_fetch = int(getattr(self, "max_proxy_fetch", 1000))
                try:
                    fetched = self.fetch_proxies_from_proxyscrape(max_proxies=max_fetch)
                    if isinstance(fetched, list) and fetched:
                        self._proxy_pool = fetched
                except Exception:
                    # keep whatever (maybe empty) and continue
                    self._proxy_pool = getattr(self, "_proxy_pool", []) or []
        except Exception:
            self._proxy_pool = getattr(self, "_proxy_pool", []) or []

        if not hasattr(self, "_proxy_index"):
            self._proxy_index = 0

        # If still empty -> no proxy available
        if not self._proxy_pool:
            self._current_proxy = None
            headers = dict(getattr(self, "headers", {"Content-Type": "application/json"}))
            ua = headers.get("User-Agent") or "Mozilla/5.0 (compatible; StoryPipeline/1.0)"
            headers.update({"User-Agent": ua})
            return None, headers

        # pick next proxy in round-robin
        idx = int(getattr(self, "_proxy_index", 0)) % len(self._proxy_pool)
        proxy = self._proxy_pool[idx]
        # advance index
        self._proxy_index = (idx + 1) % len(self._proxy_pool)
        # store current proxy on instance (so other parts can inspect)
        self._current_proxy = proxy

        # build mild impersonation headers
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
                # X-Forwarded-For is helpful for some endpoints; use proxy host IP if possible (no DNS lookup here)
                "X-Forwarded-For": (proxy.split(":")[0] if proxy else headers.get("X-Forwarded-For", "")),
            }
        )
        return proxy, headers

    def send_message(
        self,
        message: str,
        timeout: int = None,
        spinner_message: str = "Waiting for response.",
        initial_backoff: float = 2.0,
        max_backoff: float = 8.0,
        timeout_growth: int = 100,
        max_timeout_cap: int = 1000000,
        use_cloudscraper_on_403: bool = True,
        use_playwright_on_403: bool = False,
        proxy: str = None,
        specific_error: list = None,
    ):
        """
        Robust send_message preserving infinite retry semantics and all original features.

        Key fixes for proxy rotation:
        - Detects ProxyError / tunnel failures and removes the bad proxy from the pool.
        - Immediately rotates to a new proxy (and updates self._current_proxy) and retries.
        - When body matches specific_error markers, marks current proxy as bad and rotates.
        """
        import requests
        from requests.exceptions import ProxyError as ReqProxyError, RequestException
        import random

        if specific_error is None:
            specific_error = [
                "response status: 403",
                "detected cloudflare-like anti-bot page",
                "Response status=403 length=9245",
                "Detected HTML challenge",
                "Cloudflare-like content",
                "Response status=403",
                "cloudflare",
                "detected cloudflare",
                "access denied",
                "anti-bot",
            ]

        def log(*args, **kwargs):
            ts = time.strftime("%H:%M:%S")
            kwargs.setdefault("flush", True)
            print(f"[{ts}]", *args, **kwargs)

        configured_timeout = int(timeout or getattr(self, "base_timeout", 60))
        configured_timeout = min(configured_timeout, max_timeout_cap)
        attempt = 0
        backoff = initial_backoff

        # Start with explicit proxy argument if provided, otherwise use instance current proxy
        current_proxy = proxy or getattr(self, "_current_proxy", None)

        spinner = None
        if spinner_message:
            try:
                spinner = LoadingSpinner(spinner_message)
                spinner.start()
            except Exception:
                log("⚠️ Failed to start spinner (non-fatal).")

        # helper to remove a failing proxy from the pool safely
        def _mark_bad_proxy(p):
            try:
                if not p:
                    return
                pool = getattr(self, "_proxy_pool", None)
                if not pool:
                    return
                # remove all exact matches
                pool = [x for x in pool if x != p]
                self._proxy_pool = pool
                # if index out of range fix it
                if getattr(self, "_proxy_index", 0) >= max(1, len(self._proxy_pool)):
                    self._proxy_index = 0
                # clear instance current proxy
                if getattr(self, "_current_proxy", None) == p:
                    self._current_proxy = None
                log(f"🗑️ Marked bad proxy and removed from pool: {p} (remaining {len(self._proxy_pool)})")
            except Exception:
                try:
                    log("⚠️ Failed to mark bad proxy:", p)
                except Exception:
                    pass

        try:
            # lazy import cloudscraper/playwright if needed
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

            # normalize specific_error to lowercase for matching
            specific_error = [s.lower() for s in (specific_error or [])]

            def single_request(sess, req_timeout, use_headers, use_proxy):
                """
                Execute a single HTTP POST in a worker thread. Returns a dict describing outcome.
                """
                try:
                    if sess is None:
                        sess = requests.Session()
                    try:
                        sess.trust_env = False
                    except Exception:
                        pass

                    headers_local = dict(use_headers or {})
                    proxies_cfg = None
                    if use_proxy:
                        # requests expects scheme in proxy URL
                        p = str(use_proxy)
                        if not (p.startswith("http://") or p.startswith("https://") or p.startswith("socks5://") or p.startswith("socks4://")):
                            # assume http if no scheme
                            p = "http://" + p
                        proxies_cfg = {"http": p, "https": p}

                    read_timeout = int(req_timeout) if req_timeout else 60
                    read_timeout = min(max(10, read_timeout), 300)
                    timeout_arg = (10, read_timeout)

                    r = sess.post(
                        self.url,
                        json={"message": message},
                        headers=headers_local,
                        timeout=timeout_arg,
                        proxies=proxies_cfg,
                    )

                    headers_resp = r.headers or {}
                    body = r.text if hasattr(r, "text") else (r.content.decode("utf-8", "ignore") if hasattr(r, "content") else "")
                    return {"ok": True, "status_code": r.status_code, "body": body, "headers": headers_resp}
                except Exception as e:
                    return {"ok": False, "exc": e, "trace": traceback.format_exc()}

            log("send_message: starting retry loop (proxy-rotation enabled).")
            while True:
                attempt += 1

                # If no current_proxy and we have a pool, pick one now
                if not current_proxy and getattr(self, "_proxy_pool", None):
                    current_proxy, hdrs = self._pick_next_proxy_and_impersonation()
                    # _pick_next_proxy_and_impersonation sets self._current_proxy already
                    # but also update local headers for immediate call
                    use_headers = hdrs
                else:
                    # choose headers: prefer instance headers, fallback to default_headers
                    use_headers = dict(getattr(self, "headers", {})) or dict(getattr(self, "default_headers", {}))

                # run request inside a worker to avoid C-level read hangs
                try:
                    with cf.ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(single_request, getattr(self, "_session", None), configured_timeout, use_headers, current_proxy)
                        wrapper = None
                        wait_timeout = min(configured_timeout, 300) + 5
                        try:
                            wrapper = fut.result(timeout=wait_timeout)
                        except cf.TimeoutError:
                            log("⚠️ Worker timed out (fut.result). Cancelling worker and recreating session.")
                            try:
                                fut.cancel()
                            except Exception:
                                pass
                            try:
                                self._session = None
                            except Exception:
                                pass
                            time.sleep(backoff + random.random())
                            backoff = min(backoff * 2, max_backoff)
                            configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)
                            # rotate proxy on worker timeout to avoid stuck proxy
                            if current_proxy:
                                _mark_bad_proxy(current_proxy)
                                current_proxy = None
                            continue
                except Exception as e:
                    log("⚠️ Unexpected executor error:", e)
                    log(traceback.format_exc())
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)
                    continue

                if not wrapper:
                    log("⚠️ No wrapper result returned from worker — retrying.")
                    configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    continue

                if wrapper.get("ok") is False:
                    exc = wrapper.get("exc")
                    # Detect proxy-related exceptions and mark proxy bad + rotate immediately
                    try:
                        if isinstance(exc, ReqProxyError) or "ProxyError" in type(exc).__name__ or "Tunnel connection failed" in str(exc) or "Unable to connect to proxy" in str(exc):
                            log(f"⚠️ Proxy failure detected on attempt #{attempt}: {type(exc).__name__} — rotating proxy.")
                            log(wrapper.get("trace"))
                            # mark and remove the current proxy from pool
                            if current_proxy:
                                _mark_bad_proxy(current_proxy)
                                current_proxy = None
                            # pick the next proxy immediately (if available)
                            if getattr(self, "_proxy_pool", None):
                                current_proxy, hdrs = self._pick_next_proxy_and_impersonation()
                                use_headers = hdrs
                            time.sleep(min(backoff, 5) + random.random())
                            backoff = min(backoff * 2, max_backoff)
                            continue
                    except Exception:
                        # if detection failed, fall through to generic handling
                        pass

                    log(f"⚠️ Network exception on attempt #{attempt}: {type(exc).__name__} — {getattr(exc, 'args', '')}")
                    log(wrapper.get("trace"))
                    configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    continue

                status_code = wrapper.get("status_code", 0)
                body = wrapper.get("body", "") or ""
                headers_resp = wrapper.get("headers", {}) or {}

                ct = (headers_resp.get("Content-Type") or "").lower()
                # 403 handling: cloudscraper fallback (pass proxies to cloudscraper if we had one)
                if status_code == 403 and use_cloudscraper_on_403 and cloudscraper:
                    log("→ Detected 403; attempting cloudscraper fallback (using current proxy if present).")
                    try:
                        scraper = cloudscraper.create_scraper()
                        proxies_cfg = None
                        if current_proxy:
                            p = str(current_proxy)
                            if not (p.startswith("http://") or p.startswith("https://") or p.startswith("socks5://") or p.startswith("socks4://")):
                                p = "http://" + p
                            proxies_cfg = {"http": p, "https": p}
                        r2 = scraper.post(self.url, json={"message": message}, headers=getattr(self, "default_headers", {}), timeout=(10, min(configured_timeout, 300)), proxies=proxies_cfg)
                        body = r2.text
                        headers_resp = r2.headers or {}
                        status_code = r2.status_code
                    except Exception as e:
                        log("⚠️ cloudscraper attempt failed:", traceback.format_exc())
                        # cloudscraper attempt failed - likely proxy issue or blocking; mark bad proxy
                        if current_proxy:
                            _mark_bad_proxy(current_proxy)
                            current_proxy = None
                        time.sleep(min(backoff, 4) + random.random())
                        backoff = min(backoff * 2, max_backoff)
                        continue

                # attempt to parse JSON if it looks like JSON
                parsed = None
                try:
                    if "application/json" in ct or (isinstance(body, str) and (body.strip().startswith("{") or body.strip().startswith("["))):
                        parsed = json.loads(body) if body else {}
                except Exception:
                    log("⚠️ Failed to parse JSON. Response snippet:")
                    log((body or "")[:2000])

                # Accept responses similar to previous design:
                if parsed and ((isinstance(parsed, dict) and parsed.get("status") == "success") or ("message" in parsed and parsed.get("message"))):
                    log("✅ API returned success. Returning parsed payload or body.")
                    # persist the working proxy (keep it for next call), do not mark as bad
                    self._current_proxy = current_proxy
                    return parsed if parsed else body

                # If content-type not JSON, but body non-empty — check for anti-bot markers
                if not parsed and body:
                    low = (body or "").lower()
                    if any(err_marker in low for err_marker in specific_error):
                        log(f"→ Detected anti-bot / specific error marker in body (attempt {attempt}). Rotating proxy + retrying.")
                        # mark the current proxy as bad because it likely caused the challenge
                        if current_proxy:
                            _mark_bad_proxy(current_proxy)
                            current_proxy = None
                        # pick a fresh proxy for next attempt if available
                        if getattr(self, "_proxy_pool", None):
                            current_proxy, hdrs = self._pick_next_proxy_and_impersonation()
                            use_headers = hdrs
                        # small backoff and retry
                        time.sleep(backoff + random.random())
                        backoff = min(backoff * 2, max_backoff)
                        configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)
                        continue
                    # otherwise, treat it as a valid raw textual response (matching prior behaviour)
                    log("→ Returning raw body (non-JSON).")
                    self._current_proxy = current_proxy
                    return body

                # If parsed JSON but not recognized success shape, print diagnostics and retry after rotating proxy
                log("→ Received JSON but no recognized 'success' or message field. Payload (truncated):")
                try:
                    log(json.dumps(parsed, indent=2)[:2000])
                except Exception:
                    log(parsed)
                # rotate proxy on repeated unexpected JSON to attempt different route
                if current_proxy:
                    _mark_bad_proxy(current_proxy)
                    current_proxy = None
                log("Retrying after backoff; attempt:", attempt)
                time.sleep(backoff + random.random())
                backoff = min(backoff * 2, max_backoff)
                configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)

        finally:
            if spinner:
                try:
                    spinner.stop()
                except Exception:
                    pass

# ---------------------- UTILITIES ----------------------
def _nice_join(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]

def ensure_ffmpeg_available() -> Optional[str]:
    import subprocess, sys, importlib
    try:
        # 1) check PATH
        from shutil import which
        ff = which("ffmpeg")
        if ff:
            return ff
    except Exception:
        pass
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        if ff:
            return ff
    except Exception:
        pass
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        if ff:
            return ff
    except Exception:
        pass
    return None

def log_execution_time(start_time: float, end_time: float, show_ms: bool = False) -> None:
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
    current_time = datetime.now().strftime("%I:%M:%S %p")
    print(f"[{start_time} to {end_time} now {current_time}] Execution completed in {human}.")

def save_response(folder_name: str, file_name: str, content: str):
    try:
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Saved: {file_path}")
    except Exception as e:
        print(f"⚠️ Failed to save response {folder_name}/{file_name}: {e}")

def extract_largest_bracketed(text: str) -> Optional[str]:
    matches = re.findall(r"\[([^\]]+)\]", text, flags=re.DOTALL)
    if not matches:
        return None
    largest = max(matches, key=lambda s: len(s))
    return largest.strip()

def extract_all_bracketed_blocks(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r"\[([^\]]+)\]", text, flags=re.DOTALL)]

def escape_for_coqui_tts(text: str) -> str:
    # Simple character cleanup to avoid odd tokens in Coqui; keep punctuation that matters.
    text = (text or "").strip()
    # replace repeated whitespace
    text = re.sub(r"\s+", " ", text)
    # ensure sentences separated by two newlines for paragraph breaks
    paras = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", text) if p.strip()]
    out = "\n\n".join(paras)
    return out

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

    def _build_script_prompt(self, niche: str, person: str, timing_minutes: int = 10, words_per_minute: int = 250, topic: str = "") -> str:
        """
        Build the canonical story/script prompt, including strict bracket and word-count enforcement.
        This preserves your original structure and constraints but ensures consistent formatting.
        """
        words_target = timing_minutes * words_per_minute
        topic_clean = (topic or "").strip()
        if person and person.strip():
            person_section = (
                f"Write the story centered on a fictionalized protagonist inspired by '{person}'. "
                "Use the person only as inspiration — do NOT present real private facts or personal data about living persons. "
                "If the name is public and widely-known, use fictionalized depiction only."
            )
        else:
            person_section = (
                f"Do not center the story on a real person. Instead, write an original third-person story relevant to niche '{niche}'."
            )

        prompt = (
            "You are an expert cinematic storyteller and YouTube scriptwriter.\n\n"
            f"{person_section}\n\n"
            f"Write a powerful, cinematic long-form storytelling script relevant to '{topic_clean}' and niche '{niche}'. "
            f"Length: produce approximately {words_target} words (~{timing_minutes} minutes at {words_per_minute} wpm). "
            "Aim to hit the target within ±1% if possible. Prefer slightly under if exact isn't possible.\n\n"
            "Structure (use naturally, do not label in output):\n"
            "  - Hook (0:00-0:30)\n"
            "  - Act 1 — Origins\n"
            "  - Act 2 — Turning Points & Conflicts\n"
            "  - Act 3 — Breakthrough & Mastery\n"
            "  - Act 4 — Legacy, Reflection & Lessons\n"
            "  - Closing Line: one powerful quotable sentence\n\n"
            "Style: immersive, cinematic, show-not-tell, sensory detail, clean transitions, emotional hooks.\n\n"
            "CRITICAL FORMATTING: OUTPUT EXACTLY ONE PAIR OF SQUARE BRACKETS AND NOTHING ELSE. "
            "The content inside the brackets must be the complete script text with no labels, metadata, or commentary outside the brackets. "
            f"Target words: {words_target} (±1%). If you cannot meet it exactly, produce the closest script within that tolerance. "
            "If you cannot produce a script that meets these rules, output exactly [] (an empty bracket pair).\n\n"
            "Produce now."
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
        allow_infinite: bool = True,
        stop_event: Optional[Event] = None,
    ) -> str:
        """
        Generate a cinematic bracketed script. Keeps infinite retry semantics by default.
        Returns the saved bracketed script (string) or None on graceful stop/no-result.
        """

        words_target = timing_minutes * words_per_minute
        tolerance = max(1, int(words_target * 0.01))

        base_prompt = self._build_script_prompt(
            niche=niche,
            person=person,
            timing_minutes=timing_minutes,
            words_per_minute=words_per_minute,
            topic=topic,
        )

        timeout = timeout or getattr(self.chat, "base_timeout", None)

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

        def _remove_exact_and_fuzzy_duplicates(text: str, fuzzy_threshold: float = 0.90) -> str:
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
                        if difflib.SequenceMatcher(None, np, k).ratio() >= fuzzy_threshold:
                            duplicate = True
                            break
                if not duplicate:
                    kept.append(p)
                    normals.append(np)
            return "\n\n".join(kept).strip()

        def _strengthen_prompt(base_prompt: str, previous_words: Optional[int], attempt_no: int) -> str:
            extra = (
                "\n\nIMPORTANT: You MUST output EXACTLY ONE bracketed block and NOTHING ELSE. "
                "The output must start with '[' and end with ']' and contain no characters outside those brackets. "
                "The bracketed block should contain ONLY the full script text (no labels, no JSON, no commentary). "
                f"Now adjust the script so that it contains exactly {words_target} words. "
                f"If you cannot hit exactly {words_target}, produce a script as close as possible within ±{tolerance} words. "
                "Do not output anything outside the single bracketed block."
            )
            if previous_words is not None:
                diff = words_target - previous_words
                extra += f" Previous attempt had {previous_words} words (difference {diff}). "
                if diff > 0:
                    extra += "Extend and enrich to reach the target."
                elif diff < 0:
                    extra += "Condense tightly to reach the target."
            extra += f" Attempt #{attempt_no}."
            return base_prompt + extra

        def _extract_candidate(resp: str) -> str:
            try:
                brs = re.findall(r"\[[\s\S]*?\]", resp)
                if brs:
                    return max(brs, key=len)[1:-1].strip()
            except Exception:
                pass
            return resp.strip()

        def _heuristic_trim_to_target(text: str, target_words: int) -> str:
            paras = _get_paragraphs(text)
            if not paras:
                return text
            protect_first = min(1, len(paras))
            protect_last = min(1, max(0, len(paras) - protect_first))
            words = re.findall(r"\S+", text)
            if len(words) <= target_words:
                return text
            # try to remove middle content conserving first+last anchors
            if len(paras) >= 3:
                head = paras[0]
                tail = paras[-1]
                middle = " ".join(paras[1:-1])
                middle_words = re.findall(r"\S+", middle)
                allowed_middle = max(0, target_words - len(re.findall(r"\S+", head)) - len(re.findall(r"\S+", tail)))
                if allowed_middle <= 0:
                    out = " ".join((re.findall(r"\S+", head) + re.findall(r"\S+", tail))[:target_words])
                    return out
                trimmed_middle = " ".join(middle_words[:allowed_middle])
                merged = (head + "\n\n" + trimmed_middle + "\n\n" + tail).strip()
                return merged
            # fallback: simple truncate
            return " ".join(words[:target_words]).strip()

        def _finalize_and_save(text: str) -> Optional[str]:
            final_text = _remove_exact_and_fuzzy_duplicates(text, fuzzy_threshold=0.92)
            if final_text.startswith("[") and final_text.endswith("]"):
                final_text = final_text[1:-1].strip()
            final_text = final_text.strip()
            if not final_text:
                print("⚠️ Final text empty after cleaning — not saving.")
                return None
            bracketed = f"[{final_text}]"
            save_response("generated_complete_script", "generated_complete_script.txt", bracketed)
            return bracketed

        attempt = 0
        backoff = 1.0
        accumulated = ""
        last_wc = None

        while True:
            attempt += 1

            if stop_event is not None and stop_event.is_set():
                print(f"Stop requested (attempt {attempt}). Finalizing best-effort.")
                if accumulated:
                    return _finalize_and_save(accumulated)
                return None

            if not allow_infinite and attempt > max_attempts:
                print(f"Max attempts ({max_attempts}) reached in non-infinite mode. Finalizing best-effort.")
                if accumulated:
                    return _finalize_and_save(accumulated)
                return None

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
                time.sleep(min(30, backoff) + random.random() * 0.5)
                backoff = min(backoff * 1.8, 60.0)
                continue

            resp_text = resp if isinstance(resp, str) else (json.dumps(resp) if isinstance(resp, (dict, list)) else str(resp or ""))
            candidate = _extract_candidate(resp_text)
            cleaned = candidate or ""
            # clean up common bracket artifacts
            cleaned = re.sub(r"^\[+|\]+$", "", cleaned).strip()
            cleaned = _remove_exact_and_fuzzy_duplicates(cleaned, fuzzy_threshold=0.94)
            wc = _word_count(cleaned)
            last_wc = wc
            print(f"[Attempt {attempt}] Candidate words={wc}; target={words_target} ±{tolerance}")

            if not cleaned:
                print("⚠️ Candidate cleaned to empty. Retrying after backoff.")
                time.sleep(min(5 + attempt * 0.5, 20))
                backoff = min(backoff * 1.8, 60.0)
                continue

            if cleaned and cleaned not in accumulated:
                accumulated = (accumulated + "\n\n" + cleaned).strip()

            if abs(wc - words_target) <= tolerance:
                print(f"✅ Candidate within tolerance: {wc} words. Finalizing and saving.")
                final = _finalize_and_save(cleaned)
                if final:
                    return final
                print("⚠️ Finalize/save did not produce artifact; continuing.")
                time.sleep(1 + random.random())
                continue

            if wc > words_target + tolerance:
                print(f"→ Candidate too long ({wc} > {words_target}). Attempting heuristic trim.")
                trimmed = _heuristic_trim_to_target(cleaned, words_target)
                t_wc = _word_count(trimmed)
                print(f"   Trimmed to {t_wc} words.")
                if abs(t_wc - words_target) <= tolerance:
                    print("✅ Trimmed candidate is within tolerance. Finalizing and saving.")
                    final = _finalize_and_save(trimmed)
                    if final:
                        return final
                print("→ Trim did not reach acceptable length; will retry with condense instruction.")
                time.sleep(0.5 + random.random() * 0.7)
                backoff = min(backoff * 1.5, 60.0)
                continue

            if wc < words_target - tolerance:
                print(f"→ Candidate too short ({wc} < {words_target}). Asking model to extend.")
                paras = _get_paragraphs(cleaned)
                last_para = paras[-1] if paras else ""
                continue_prompt = (
                    f"You produced the script below (inside brackets). The script is currently {wc} words but we need {words_target} "
                    f"words (±{tolerance}). Continue seamlessly from the last paragraph and expand naturally to meet the target. "
                    f"Seed (last paragraph):\n\n{last_para}\n\nNow continue and output exactly one bracketed block and nothing else."
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

                cont_text = cont_resp if isinstance(cont_resp, str) else (json.dumps(cont_resp) if isinstance(cont_resp, (dict, list)) else str(cont_resp or ""))
                cont_candidate = _extract_candidate(cont_text)
                cont_cleaned = re.sub(r"^\[+|\]+$", "", (cont_candidate or "")).strip()
                merged = (cleaned + "\n\n" + cont_cleaned).strip()
                merged = _remove_exact_and_fuzzy_duplicates(merged, fuzzy_threshold=0.92)
                merged_wc = _word_count(merged)
                print(f"   After continuation, merged words={merged_wc}")

                if abs(merged_wc - words_target) <= tolerance:
                    print("✅ Continued script meets the target. Finalizing and saving.")
                    final = _finalize_and_save(merged)
                    if final:
                        return final
                else:
                    accumulated = merged
                    last_wc = merged_wc
                    time.sleep(0.6 + random.random() * 0.6)
                    backoff = min(backoff * 1.5, 60.0)
                    continue

            time.sleep(min(3 + backoff, 30))
            backoff = min(backoff * 1.4, 60.0)

        # unreachable
        return None

    def _split_text_into_n_parts(self, text: str, n: int) -> List[str]:
        """
        Split text into n parts of roughly equal word count while trying to preserve sentence boundaries.
        Returns length==n list (may contain empty strings).
        """
        from typing import List
        import re, math

        text = (text or "").strip()
        if n <= 1 or not text:
            return [text] + [""] * (n - 1) if n > 0 else []

        sentences = re.split(r"(?<=[\.\?!])\s+", text)
        if not sentences:
            return [""] * n

        words_per_sentence = [len(re.findall(r"\S+", s)) for s in sentences]
        total_words = sum(words_per_sentence)
        if total_words == 0:
            return [""] * n

        target = math.ceil(total_words / n)
        parts = []
        current_sentences = []
        current_count = 0
        idx = 0

        while idx < len(sentences) and len(parts) < n - 1:
            w = words_per_sentence[idx]
            s = sentences[idx]
            if current_count + w > target and current_count > 0:
                parts.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_count = 0
                continue
            current_sentences.append(s)
            current_count += w
            idx += 1

        tail = []
        while idx < len(sentences):
            tail.append(sentences[idx])
            idx += 1

        if current_sentences:
            parts.append(" ".join(current_sentences).strip())
        if tail:
            parts.append(" ".join(tail).strip())

        # ensure exactly n elements
        while len(parts) < n:
            parts.append("")
        if len(parts) > n:
            # merge tail items into last to fit
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
        parallel_batches: bool = False,
    ) -> List[str]:
        """
        Generate image prompts from script_text. Preserves per-paragraph quota semantics.
        If parallel_batches=True, will attempt to generate paragraph batches in parallel (bounded).
        """
        import os, re, math, random, time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        timeout_per_call = timeout_per_call or min(120, getattr(self.chat, "base_timeout", 120))
        if img_number <= 0:
            return []
        if batch_size <= 0:
            batch_size = 1

        response_folder = "image_response"
        final_fname = "image_prompts.txt"
        final_path = os.path.join(response_folder, final_fname)
        os.makedirs(response_folder, exist_ok=True)

        num_paragraphs = max(1, math.ceil(img_number / max(1, batch_size)))
        try:
            paragraphs = self._split_text_into_n_parts(script_text, num_paragraphs)
        except Exception:
            from math import ceil
            words = script_text.split()
            avg = max(1, ceil(len(words) / num_paragraphs)) if script_text else 0
            paragraphs = [" ".join(words[i : i + avg]) for i in range(0, len(words), avg)]
            while len(paragraphs) < num_paragraphs:
                paragraphs.append("")

        prompts = []
        seen_prompts = set()
        per_paragraph_cap = getattr(self, "max_prompt_attempts", None)
        per_paragraph_max = (per_paragraph_cap if (isinstance(per_paragraph_cap, int) and per_paragraph_cap > 0) else None)

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

        def _build_request(script_paragraph, req_count, seed):
            try:
                return self._build_image_prompt_request(script_paragraph, theme, req_count)
            except Exception:
                return f"{script_paragraph}\n\nGenerate {req_count} ultra-detailed image prompts. Seed:{seed}"

        def _extract_blocks(resp_text):
            if not resp_text:
                return []
            blocks = extract_all_bracketed_blocks(resp_text)
            if blocks:
                return [b.strip() for b in blocks if isinstance(b, str)]
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
            if not candidate or len(candidate.strip()) < 8:
                return False
            c = candidate.strip("[]()\"'` ").strip()
            if len(c.split()) < 8 and not any(k in c.lower() for k in ("camera", "cinematic", "photoreal", "illustration", "render", "watercolor")):
                return False
            lower = c.lower()
            for bad in ("do you want", "please confirm", "which one", "error", "unable to", "service may be down", "example"):
                if bad in lower:
                    return False
            return True

        # helper to process a single paragraph into prompts (retries inside)
        def _process_paragraph(p_idx):
            local_prompts = []
            paragraph = paragraphs[p_idx] or script_text or ""
            quota_for_para = min(batch_size, max(1, img_number - len(prompts))) if batch_size > 0 else img_number
            collected = 0
            attempts = 0
            seed = random.randint(1000, 9999)
            while collected < quota_for_para and (per_paragraph_max is None or attempts < (per_paragraph_max or 9999)):
                attempts += 1
                need = quota_for_para - collected
                request_count = need
                enriched = f"{paragraph}\n\n# Paragraph {p_idx+1} | Attempt {attempts} | Seed {seed}\nGenerate {request_count} unique, highly-detailed image prompts. Output one bracketed prompt per line."
                req_body = _build_request(enriched, request_count, seed)
                try:
                    resp = self.chat.send_message(req_body, timeout=timeout_per_call, spinner_message=f"Generating para {p_idx+1} prompts (attempt {attempts}).")
                except Exception as e:
                    print(f"  ⚠️ API error: {e}")
                    try:
                        _write_atomic(os.path.join(response_folder, f"para_{p_idx+1}_err_{attempts}.txt"), str(e))
                    except Exception:
                        pass
                    time.sleep(0.6 + random.random() * 0.4)
                    continue

                blocks = _extract_blocks(resp if isinstance(resp, str) else (json.dumps(resp) if isinstance(resp,(dict,list)) else str(resp or "")))
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
                    local_prompts.append(cand)
                    added += 1
                    collected += 1
                print(f"  ⚪ para {p_idx+1} added {added} prompts this attempt (collected {collected}/{quota_for_para}).")
                if save_each_batch and added:
                    # merge local into global and save
                    prompts.extend(local_prompts[-added:])
                    _save_progress()
                time.sleep(0.3 + random.random() * 0.5)
            if collected < quota_for_para:
                print(f"⚠️ Could not fill paragraph {p_idx+1} quota (collected {collected}/{quota_for_para}).")
            return local_prompts

        # Main: either sequential or parallel
        if parallel_batches and len(paragraphs) > 1:
            # limit parallelism to small pool to avoid overloading API
            max_workers = min(4, len(paragraphs))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_process_paragraph, i): i for i in range(len(paragraphs))}
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                        # extend prompts maintaining order best-effort
                        for p in res:
                            if p not in prompts:
                                prompts.append(p)
                                if len(prompts) >= img_number:
                                    break
                        if save_each_batch:
                            _save_progress()
                    except Exception as e:
                        print(f"⚠️ Paragraph worker error: {e}")
        else:
            for p_idx in range(len(paragraphs)):
                if len(prompts) >= img_number:
                    break
                res = _process_paragraph(p_idx)
                # extend canonical prompts
                for p in res:
                    if p not in prompts:
                        prompts.append(p)
                    if len(prompts) >= img_number:
                        break

        # final dedupe and trim
        final_prompts = []
        for p in prompts:
            if p not in final_prompts:
                final_prompts.append(p)
            if len(final_prompts) >= img_number:
                break

        try:
            _write_atomic(final_path, "".join(f"[{p}]\n" for p in final_prompts))
            print(f"\n✅ Final saved {len(final_prompts)}/{img_number} prompts -> {final_path}")
        except Exception as e:
            print(f"\n⚠️ Failed final save: {e}")

        return final_prompts

    def _build_image_prompt_request(self, script_text: str, theme: str, img_number: int) -> str:
        """
        Build a rich, exhaustive image prompt generator request. Preserves the verbose spec you used before.
        """
        if hasattr(self, "_custom_image_prompt_builder") and callable(self._custom_image_prompt_builder):
            try:
                return self._custom_image_prompt_builder(script_text, theme, img_number)
            except Exception:
                pass

        body = f"""
    You are an expert visual concept designer, cinematographer, production photographer, studio art director, and elite prompt engineer.

    TASK:
    Generate {img_number} UNIQUE, imaginative image prompts based on the paragraph below. Do NOT restrict prompt length — include exhaustive micro-details and photographic specifications.

    EACH PROMPT MUST:
    - Describe a single, self-contained scene (composition, subject(s), environment, mood).
    - Include camera type and lens focal length (e.g., 24mm / 50mm / 85mm), aperture (e.g., f/1.4), shutter speed, ISO, and perspective.
    - Specify shot framing, distance to subject, lighting, modifiers (softbox, grid), and micro-details (fabric weave, skin pores, droplets).
    - Define foreground/midground/background layering, atmospheric effects, and color palette with example hex codes.
    - Suggest cinematic color grading (e.g., Kodak Portra, teal-orange) and finishing styles (film grain, chromatic aberration).
    - Provide one-line mood caption and 2-3 short phrase tags for style.

    OUTPUT:
    Return EXACTLY ONE bracketed prompt per line. Example valid line: [Young girl painting clouds, watercolor illustration, 50mm, f/2.0, soft morning light, warm pastel palette, cinematic shallow DOF, --v 5] 

    Paragraph:
    {script_text}

    Theme / constraints: {theme}

    Generate now.
    """
        return body.strip()

    def generate_narration(self, script_text: str, timing_minutes: int = 10, words_per_minute: int = 250) -> str:
        """
        Produce narration text suitable for Coqui or other TTS systems.
        - Returns escaped narration string and writes file narration/narration.txt
        """
        words_target = timing_minutes * words_per_minute
        # simple cleaning & normalization preserving paragraphs
        paras = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", script_text) if p.strip()]
        # join and normalize spaces while keeping paragraph breaks
        cleaned = "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in paras)
        # naive length adapt: if too long, keep first words_target words; if too short, leave as-is
        wc = len(re.findall(r"\w+", cleaned))
        if wc > words_target:
            words = re.findall(r"\S+", cleaned)
            cleaned = " ".join(words[: words_target])
        escaped = escape_for_coqui_tts(cleaned)
        # save
        os.makedirs("narration", exist_ok=True)
        with open(os.path.join("narration", "narration.txt"), "w", encoding="utf-8") as f:
            f.write(escaped)
        print(f"💾 Saved narration (approx {min(wc, words_target)} words) -> narration/narration.txt")
        return escaped

    def generate_youtube_metadata(self, script_text: str, timing_minutes: int = 10, words_per_minute: int = 250, timeout: Optional[int] = None) -> dict:
        """
        Generate YouTube title, description and tags using ChatAPI; save raw response.
        Returns parsed response (or raw string if parsing not available).
        """
        timeout = timeout or min(120, getattr(self.chat, "base_timeout", 120))
        prompt = (
            "You are an expert YouTube metadata writer. Given the following script, produce a JSON object with keys: "
            "'title' (<=100 chars), 'description' (>=200 chars), 'tags' (list of 8-20 keywords), and 'chapter_timestamps' (optional list of time->label). "
            "Only return valid JSON (no surrounding text). Script:\n\n" + script_text
        )
        resp = self.chat.send_message(prompt, timeout=timeout, spinner_message="Generating YouTube metadata.")
        # save raw
        save_response("youtube_response", "youtube_metadata.txt", resp if isinstance(resp, str) else json.dumps(resp))
        # try to parse JSON if possible
        parsed = None
        try:
            if isinstance(resp, (dict, list)):
                parsed = resp
            else:
                parsed = json.loads(resp)
        except Exception:
            parsed = {"raw": resp}
        return parsed



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
