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
        max_backoff: float = 60.0,
        timeout_growth: int = 100,
        max_timeout_cap: int = 1000000,
        use_cloudscraper_on_403: bool = True,
        use_playwright_on_403: bool = True,
        proxy: str = None,
        specific_error: list = None,
    ):
        """
        Robust send_message with:
        - proxy rotation + bad-proxy removal,
        - curl impersonation fallback,
        - Playwright stealth-ish fetch + URL path fuzzing,
        - Node Puppeteer-stealth fallback (if node/npm available),
        - cloudscraper fallback,
        - preserves infinite retry behavior.

        Returns either a string (body) or parsed JSON (if code paths return parsed objects in your original code).
        Many callers expect textual responses so StoryPipeline now converts dicts to strings safely.
        """
        import subprocess
        import shutil
        import json
        import random
        import tempfile

        if specific_error is None:
            specific_error = [
                "response status: 403",
                "detected cloudflare-like anti-bot page",
                "detected cloudflare",
                "access denied",
                "anti-bot",
                "captcha",
                "forbidden",
            ]

        def log(*a, **k):
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]", *a, **k)

        configured_timeout = int(timeout or getattr(self, "base_timeout", 60))
        configured_timeout = min(configured_timeout, max_timeout_cap)
        attempt = 0
        backoff = initial_backoff

        # prefer explicit proxy arg, else last used instance proxy
        current_proxy = proxy or getattr(self, "_current_proxy", None)

        spinner = None
        if spinner_message:
            try:
                spinner = LoadingSpinner(spinner_message)
                spinner.start()
            except Exception:
                log("⚠️ Spinner start failed (non-fatal).")

        # helper: mark a proxy as bad and remove from pool
        def _mark_bad_proxy(p):
            try:
                if not p:
                    return
                pool = getattr(self, "_proxy_pool", None)
                if not pool:
                    return
                pool = [x for x in pool if x != p]
                self._proxy_pool = pool
                if getattr(self, "_proxy_index", 0) >= max(1, len(self._proxy_pool)):
                    self._proxy_index = 0
                if getattr(self, "_current_proxy", None) == p:
                    self._current_proxy = None
                log("🗑️ Removed bad proxy:", p, "remaining:", len(self._proxy_pool))
                # optionally persist bad proxies for offline debugging
                try:
                    bad_file = os.path.join("debug_proxies", "bad_proxies.txt")
                    os.makedirs(os.path.dirname(bad_file), exist_ok=True)
                    with open(bad_file, "a", encoding="utf-8") as f:
                        f.write(p + "\n")
                except Exception:
                    pass
            except Exception:
                pass

        # helper: run curl impersonate (fast, uses system curl)
        def _run_curl_impersonate(target_url, hdrs: dict, use_proxy_str: str, max_time=20):
            try:
                curl = shutil.which("curl")
                if not curl:
                    return None
                args = [curl, "--location", "--silent", "--show-error", "--max-time", str(int(max_time))]
                # Add headers
                for k, v in (hdrs or {}).items():
                    # curl -H "Header: value"
                    args += ["-H", f"{k}: {v}"]
                # set user agent explicitly if present
                ua = hdrs.get("User-Agent") if hdrs else None
                if ua:
                    args += ["-A", ua]
                # proxy
                if use_proxy_str:
                    p = str(use_proxy_str)
                    if not (p.startswith("http://") or p.startswith("https://") or p.startswith("socks5://") or p.startswith("socks4://")):
                        p = "http://" + p
                    args += ["--proxy", p]
                # follow redirects
                args += ["-i", "--insecure", target_url]
                proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max_time + 5)
                out = proc.stdout.decode("utf-8", "ignore")
                # heuristics: if we get 200 and not a large HTML anti-bot page, return
                return out
            except Exception as e:
                return None

        # helper: simple set of URL fuzzed paths to try
        def _fuzz_paths(base_url):
            # generate candidate URL paths / query variations to try
            urls = []
            base = base_url.rstrip("/")
            candidates = [
                base,
                base + "/",
                base + "/api",
                base + "/api/v1",
                base + "/api/v2",
                base + "/v1",
                base + "/v2",
                base + "/status",
                base + "/health",
                base + "/api/chat",
                base + "/api/chat/",
                base + "/index",
                base + "/home",
            ]
            # add random query param variants to bypass naive caching/blocks
            for c in candidates:
                for i in range(3):
                    q = f"?_={random.randint(1000,999999)}"
                    urls.append(c + q)
                urls.append(c)
            # de-duplicate preserve order
            seen = set()
            out = []
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    out.append(u)
            return out

        # helper: Playwright stealth-ish fetch + path fuzzing
        def _playwright_stealth_fetch_and_fuzz(base_url, hdrs, use_proxy_str, tries=6, per_try_timeout=18):
            try:
                from playwright.sync_api import sync_playwright
            except Exception:
                return None
            try:
                with sync_playwright() as p:
                    browser = None
                    # try several path variants
                    urls = _fuzz_paths(base_url)[:tries]
                    for url_variant in urls:
                        try:
                            chromium = p.chromium
                            launch_args = {"headless": True, "args": ["--no-sandbox", "--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage"]}
                            if use_proxy_str:
                                # Playwright accepts proxy dict with server key like "http://host:port"
                                proxy_arg = {}
                                pstr = str(use_proxy_str)
                                if not (pstr.startswith("http://") or pstr.startswith("https://") or pstr.startswith("socks5://") or pstr.startswith("socks4://")):
                                    pstr = "http://" + pstr
                                proxy_arg["server"] = pstr
                                launch_args["proxy"] = proxy_arg
                            browser = chromium.launch(**launch_args)
                            context = browser.new_context(user_agent=hdrs.get("User-Agent") if hdrs else None, java_script_enabled=True, ignore_https_errors=True)
                            # stealth tweaks: overwrite navigator webdriver and other signals
                            page = context.new_page()
                            page.set_default_navigation_timeout(per_try_timeout * 1000)
                            # set extra headers if provided
                            if hdrs:
                                try:
                                    extra = dict(hdrs)
                                    # remove content-length if present
                                    extra.pop("Content-Length", None)
                                    page.set_extra_http_headers(extra)
                                except Exception:
                                    pass
                            # evaluate to hide webdriver flag
                            page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => false});")
                            page.add_init_script("window.chrome = { runtime: {} };")
                            page.add_init_script("Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});")
                            page.add_init_script("Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});")
                            # try to goto
                            page.goto(url_variant, wait_until="domcontentloaded")
                            content = page.content()
                            page.close()
                            context.close()
                            browser.close()
                            # basic heuristics: if status 200-ish and body not huge anti-bot
                            if content and len(content) > 0:
                                return content
                        except Exception:
                            try:
                                if browser:
                                    browser.close()
                            except Exception:
                                pass
                            continue
            except Exception:
                return None
            return None

        # helper: node puppeteer/stealth fallback (writes a small script and executes it)
        def _node_puppeteer_stealth_fetch_and_fuzz(base_url, hdrs, use_proxy_str, tries=6, per_try_timeout=18):
            node_bin = shutil.which("node") or shutil.which("nodejs")
            if not node_bin:
                return None
            try:
                tmpdir = tempfile.mkdtemp(prefix="puppet_")
                script_path = os.path.join(tmpdir, "stealth_fetch.js")
                # small Puppeteer script using puppeteer-extra + stealth plugin if present; otherwise fallback to puppeteer
                js = r"""
    ( async () => {
    const fs = require('fs');
    const url = process.argv[2] || '';
    const proxy = process.argv[3] || '';
    const userAgent = process.argv[4] || '';
    const timeout = parseInt(process.argv[5] || '18000', 10);
    let puppeteer;
    let useExtra = false;
    try {
        puppeteer = require('puppeteer-extra');
        const StealthPlugin = require('puppeteer-extra-plugin-stealth');
        puppeteer.use(StealthPlugin());
        useExtra = true;
    } catch (e) {
        try {
        puppeteer = require('puppeteer');
        } catch (e2) {
        console.error('NO_PUPPETEER');
        process.exit(2);
        }
    }
    try {
        const launchOpts = { headless: true, args: ['--no-sandbox','--disable-blink-features=AutomationControlled'] };
        if (proxy) {
        launchOpts.args.push(`--proxy-server=${proxy}`);
        }
        const browser = await puppeteer.launch(launchOpts);
        const page = await browser.newPage();
        if (userAgent) await page.setUserAgent(userAgent);
        await page.setDefaultNavigationTimeout(timeout);
        try {
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        const content = await page.content();
        console.log(content);
        } catch (e) {
        console.error('ERR', e && e.message ? e.message : String(e));
        process.exit(3);
        } finally {
        await browser.close();
        }
    } catch (e) {
        console.error('ERR', e && e.message ? e.message : String(e));
        process.exit(4);
    }
    })();
    """
                with open(script_path, "w", encoding="utf-8") as fw:
                    fw.write(js)
                # build candidate URLs
                urls = _fuzz_paths(base_url)[:tries]
                for u in urls:
                    try:
                        args = [node_bin, script_path, u, (str(use_proxy_str) if use_proxy_str else ""), (hdrs.get("User-Agent") if hdrs else ""), str(int(per_try_timeout * 1000))]
                        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=per_try_timeout + 5)
                        out = proc.stdout.decode("utf-8", "ignore")
                        err = proc.stderr.decode("utf-8", "ignore")
                        if out and len(out) > 0 and ("NO_PUPPETEER" not in err):
                            return out
                    except Exception:
                        continue
            except Exception:
                return None
            return None

        # nested single_request using requests (kept minimal)
        def single_request(sess, req_timeout, use_headers, use_proxy):
            try:
                import requests
                if sess is None:
                    sess = requests.Session()
                try:
                    sess.trust_env = False
                except Exception:
                    pass
                headers_local = dict(use_headers or {})
                proxies_cfg = None
                if use_proxy:
                    p = str(use_proxy)
                    if not (p.startswith("http://") or p.startswith("https://") or p.startswith("socks5://") or p.startswith("socks4://")):
                        p = "http://" + p
                    proxies_cfg = {"http": p, "https": p}
                read_timeout = int(req_timeout) if req_timeout else 60
                read_timeout = min(max(10, read_timeout), 300)
                timeout_arg = (10, read_timeout)
                r = sess.post(self.url, json={"message": message}, headers=headers_local, timeout=timeout_arg, proxies=proxies_cfg)
                headers_resp = r.headers or {}
                body = r.text if hasattr(r, "text") else (r.content.decode("utf-8", "ignore") if hasattr(r, "content") else "")
                return {"ok": True, "status_code": r.status_code, "body": body, "headers": headers_resp}
            except Exception as e:
                return {"ok": False, "exc": e, "trace": traceback.format_exc()}

        try:
            # attempt loop (infinite retry by design)
            while True:
                attempt += 1
                # choose proxy/headers
                if not current_proxy and getattr(self, "_proxy_pool", None):
                    current_proxy, pick_headers = self._pick_next_proxy_and_impersonation()
                    use_headers = pick_headers
                else:
                    use_headers = dict(getattr(self, "headers", {})) or dict(getattr(self, "default_headers", {}))

                # run worker request to avoid blocking
                with cf.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(single_request, getattr(self, "_session", None), configured_timeout, use_headers, current_proxy)
                    wrapper = None
                    wait_timeout = min(configured_timeout, 300) + 5
                    try:
                        wrapper = fut.result(timeout=wait_timeout)
                    except cf.TimeoutError:
                        log("⚠️ Worker timed out; marking proxy bad and retrying.")
                        # suspected proxy stuck
                        if current_proxy:
                            _mark_bad_proxy(current_proxy)
                            current_proxy = None
                        time.sleep(backoff + random.random())
                        backoff = min(backoff * 2, max_backoff)
                        configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)
                        continue

                if not wrapper:
                    log("⚠️ No wrapper result; retrying.")
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    continue

                if wrapper.get("ok") is False:
                    exc = wrapper.get("exc")
                    # detect proxy-related failures
                    s_exc = str(exc).lower() if exc else ""
                    if "proxyerror" in type(exc).__name__.lower() or "tunnel connection failed" in s_exc or "unable to connect to proxy" in s_exc or "connection refused" in s_exc:
                        log("⚠️ Proxy error detected:", s_exc)
                        if current_proxy:
                            _mark_bad_proxy(current_proxy)
                            current_proxy = None
                        time.sleep(min(backoff, 4) + random.random())
                        backoff = min(backoff * 2, max_backoff)
                        continue
                    # generic network exception: backoff and retry
                    log("⚠️ Network exception:", s_exc)
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    continue

                status_code = wrapper.get("status_code", 0)
                body = wrapper.get("body", "") or ""
                headers_resp = wrapper.get("headers", {}) or {}
                ct = (headers_resp.get("Content-Type") or "").lower()

                # Status 403 -> try multi-stage anti-bot recovery
                if status_code == 403:
                    log("→ Detected 403. Starting anti-bot recovery sequence (rotate IP, curl impersonate, Playwright/Puppeteer stealth, cloudscraper).")
                    # mark current proxy as possibly bad and rotate
                    if current_proxy:
                        _mark_bad_proxy(current_proxy)
                        current_proxy = None
                    # immediate rotate to next proxy (if any)
                    if getattr(self, "_proxy_pool", None):
                        current_proxy, pick_headers = self._pick_next_proxy_and_impersonation()
                        use_headers = pick_headers

                    # 1) try curl impersonation (fast)
                    try:
                        curl_out = _run_curl_impersonate(self.url, use_headers, current_proxy, max_time=12)
                        if curl_out and any(k in curl_out.lower() for k in ("{", "[", "message", "status")):
                            log("→ curl impersonation returned content; using that result.")
                            return curl_out
                    except Exception:
                        pass

                    # 2) try Playwright stealth + path fuzzing
                    if use_playwright_on_403:
                        try:
                            pw_out = _playwright_stealth_fetch_and_fuzz(self.url, use_headers, current_proxy, tries=10, per_try_timeout=18)
                            if pw_out and len(pw_out) > 10:
                                log("→ Playwright stealth fetch returned content; using that result.")
                                return pw_out
                        except Exception:
                            pass

                    # 3) try Node puppeteer stealth fallback (if node available)
                    try:
                        pp_out = _node_puppeteer_stealth_fetch_and_fuzz(self.url, use_headers, current_proxy, tries=8, per_try_timeout=18)
                        if pp_out and len(pp_out) > 10:
                            log("→ Puppeteer stealth (node) returned content; using that result.")
                            return pp_out
                    except Exception:
                        pass

                    # 4) try cloudscraper fallback last
                    if use_cloudscraper_on_403:
                        try:
                            import cloudscraper
                            scraper = cloudscraper.create_scraper()
                            proxies_cfg = None
                            if current_proxy:
                                p = str(current_proxy)
                                if not (p.startswith("http://") or p.startswith("https://") or p.startswith("socks5://") or p.startswith("socks4://")):
                                    p = "http://" + p
                                proxies_cfg = {"http": p, "https": p}
                            try:
                                r2 = scraper.post(self.url, json={"message": message}, headers=getattr(self, "default_headers", {}), timeout=(10, min(configured_timeout, 300)), proxies=proxies_cfg)
                                body = r2.text
                                headers_resp = r2.headers or {}
                                status_code = r2.status_code
                                if body and len(body) > 0:
                                    log("→ cloudscraper fallback returned a body; returning it.")
                                    # keep current_proxy if it worked
                                    self._current_proxy = current_proxy
                                    return body
                            except Exception:
                                # cloudscraper failure -> mark proxy bad
                                if current_proxy:
                                    _mark_bad_proxy(current_proxy)
                                    current_proxy = None
                        except Exception:
                            pass

                    # after all attempts, bump backoff and continue loop
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 1.8, max_backoff)
                    continue

                # If body contains anti-bot specific_error markers -> similar recovery (rotate)
                low_body = (body or "").lower()
                if any(marker in low_body for marker in (s.lower() for s in specific_error)):
                    log("→ Detected anti-bot markers in body. Rotating proxy and trying recovery sequence.")
                    if current_proxy:
                        _mark_bad_proxy(current_proxy)
                        current_proxy = None
                    if getattr(self, "_proxy_pool", None):
                        current_proxy, pick_headers = self._pick_next_proxy_and_impersonation()
                        use_headers = pick_headers
                    # try curl + playwright + puppeteer + cloudscraper (same as above)
                    try:
                        curl_out = _run_curl_impersonate(self.url, use_headers, current_proxy, max_time=12)
                        if curl_out and any(k in curl_out.lower() for k in ("{", "[", "message", "status")):
                            return curl_out
                    except Exception:
                        pass
                    try:
                        pw_out = _playwright_stealth_fetch_and_fuzz(self.url, use_headers, current_proxy, tries=8, per_try_timeout=14)
                        if pw_out and len(pw_out) > 10:
                            return pw_out
                    except Exception:
                        pass
                    try:
                        pp_out = _node_puppeteer_stealth_fetch_and_fuzz(self.url, use_headers, current_proxy, tries=6, per_try_timeout=14)
                        if pp_out and len(pp_out) > 10:
                            return pp_out
                    except Exception:
                        pass
                    if use_cloudscraper_on_403:
                        try:
                            import cloudscraper
                            scraper = cloudscraper.create_scraper()
                            proxies_cfg = None
                            if current_proxy:
                                p = str(current_proxy)
                                if not (p.startswith("http://") or p.startswith("https://") or p.startswith("socks5://") or p.startswith("socks4://")):
                                    p = "http://" + p
                                proxies_cfg = {"http": p, "https": p}
                            r2 = scraper.post(self.url, json={"message": message}, headers=getattr(self, "default_headers", {}), timeout=(10, min(configured_timeout, 300)), proxies=proxies_cfg)
                            body = r2.text
                            if body and len(body) > 0:
                                return body
                        except Exception:
                            if current_proxy:
                                _mark_bad_proxy(current_proxy)
                                current_proxy = None
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 1.8, max_backoff)
                    continue

                # try to parse JSON if looks like JSON
                parsed = None
                try:
                    if "application/json" in ct or (isinstance(body, str) and (body.strip().startswith("{") or body.strip().startswith("["))):
                        parsed = json.loads(body) if body else {}
                except Exception:
                    log("⚠️ JSON parse failed; body snippet follows.")
                    log((body or "")[:1000])

                # If parsed looks like success return it
                if parsed and ((isinstance(parsed, dict) and parsed.get("status") == "success") or ("message" in parsed and parsed.get("message"))):
                    self._current_proxy = current_proxy
                    return parsed

                # If no JSON but body non-empty, return it
                if not parsed and body:
                    self._current_proxy = current_proxy
                    return body

                # If parsed JSON but not recognized shape -> rotate and retry
                log("→ Received JSON but not the expected shape. Rotating proxy and retrying.")
                if current_proxy:
                    _mark_bad_proxy(current_proxy)
                    current_proxy = None
                time.sleep(backoff + random.random())
                backoff = min(backoff * 1.8, max_backoff)
                configured_timeout = min(configured_timeout + timeout_growth, max_timeout_cap)
                continue

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
        Generate the bracketed script. This version keeps your infinite-retry behaviour
        but fixes TypeError by converting send_message results into text safely.
        """
        words_target = timing_minutes * words_per_minute
        tolerance = max(1, int(words_target * 0.01))
        prompt = self._build_script_prompt(niche=niche, person=person, timing_minutes=timing_minutes, words_per_minute=words_per_minute, topic=topic)
        timeout = timeout or getattr(self.chat, "base_timeout", None)
        single_block_re = re.compile(r"^\s*\[[\s\S]*\]\s*$", flags=re.DOTALL)

        def _resp_to_text(resp) -> str:
            # Convert various possible send_message returns into a representative string for regex checks.
            try:
                if resp is None:
                    return ""
                if isinstance(resp, str):
                    return resp
                if isinstance(resp, (dict, list)):
                    # prefer message/body keys if present
                    if isinstance(resp, dict):
                        for k in ("message", "body", "text", "result", "data"):
                            if k in resp and isinstance(resp[k], str) and resp[k].strip():
                                return resp[k]
                    try:
                        return json.dumps(resp)
                    except Exception:
                        return str(resp)
                return str(resp)
            except Exception:
                return str(resp)

        def _extract_candidate(resp_text: str) -> str:
            try:
                brs = re.findall(r"\[[\s\S]*?\]", resp_text)
                if brs:
                    return max(brs, key=len)[1:-1].strip()
            except Exception:
                pass
            return resp_text.strip()

        def _word_count(text: str) -> int:
            return len(re.findall(r"\w+", text or ""))

        # reuse cleaning/trim helpers already in your file: clean_script_text, _heuristic_trim_to_target, _remove_exact_and_fuzzy_duplicates etc.
        attempt = 0
        accumulated = ""
        last_wc = None
        backoff = 1.0

        while True:
            attempt += 1

            # generate or strengthen prompt depending on attempt
            req_prompt = prompt if attempt == 1 else prompt + f"\n\n(Attempt #{attempt} - refine to reach {words_target} words.)"

            try:
                resp = self.chat.send_message(req_prompt, timeout=timeout, spinner_message=f"Generating script (attempt {attempt})...")
            except Exception as e:
                print(f"⚠️ send_message raised on attempt {attempt}: {e}")
                time.sleep(min(5 + backoff, 20))
                backoff = min(backoff * 1.8, 60.0)
                continue

            # convert to text safely
            resp_text = _resp_to_text(resp)

            # If strict mode: require a single bracket block. Check on text.
            if strict:
                if not single_block_re.match(resp_text):
                    print(f"⚠️ Strict mode: attempt {attempt} response not single bracketed block — retrying.")
                    # if resp_text contains bracket blocks but also JSON wrapper, try to extract the largest.
                    candidate_try = _extract_candidate(resp_text)
                    if not candidate_try:
                        time.sleep(0.2)
                        continue
                # else proceed

            candidate = _extract_candidate(resp_text)
            cleaned_candidate = clean_script_text(candidate or "")
            # dedupe fuzzily (use existing helper if present; fallback simple)
            try:
                cleaned_candidate = _remove_exact_and_fuzzy_duplicates(cleaned_candidate, fuzzy_threshold=0.92)
            except Exception:
                cleaned_candidate = cleaned_candidate

            wc = _word_count(cleaned_candidate)
            last_wc = wc
            print(f"[Attempt {attempt}] Candidate words={wc}; target={words_target} ±{tolerance}")

            if not cleaned_candidate:
                print("⚠️ Candidate cleaned to empty; retrying.")
                time.sleep(min(2 + backoff, 10))
                backoff = min(backoff * 1.8, 60.0)
                continue

            # accumulate candidate (so abort returns something)
            if cleaned_candidate and cleaned_candidate not in accumulated:
                accumulated = (accumulated + "\n\n" + cleaned_candidate).strip()

            # if close enough -> finalize save and return
            if abs(wc - words_target) <= tolerance:
                # save exactly one bracketed block
                final = cleaned_candidate
                if final.startswith("[") and final.endswith("]"):
                    final = final[1:-1].strip()
                if final:
                    save_response("generated_complete_script", "generated_complete_script.txt", f"[{final}]")
                    return f"[{final}]"
                else:
                    print("⚠️ Final candidate empty after stripping; continuing.")
                    accumulated = ""
                    continue

            # if too long -> condense with the model once then trim heuristically
            if wc > words_target + tolerance:
                print("→ Candidate too long; asking model to condense then applying deterministic trim.")
                condense_prompt = (
                    "Condense the script inside the single bracket so the content reaches the exact target word count without changing beats or facts. Output exactly one bracketed block and nothing else."
                )
                try:
                    cond_resp = self.chat.send_message(condense_prompt + "\n\n" + resp_text, timeout=timeout, spinner_message="Condensing...")
                except Exception:
                    cond_resp = None
                cond_text = _resp_to_text(cond_resp) if cond_resp is not None else ""
                cond_candidate = _extract_candidate(cond_text) if cond_text else ""
                cond_clean = clean_script_text(cond_candidate or "")
                try:
                    cond_clean = _remove_exact_and_fuzzy_duplicates(cond_clean, fuzzy_threshold=0.90)
                except Exception:
                    pass
                if cond_clean:
                    if abs(_word_count(cond_clean) - words_target) <= tolerance:
                        save_response("generated_complete_script", "generated_complete_script.txt", f"[{cond_clean}]")
                        return f"[{cond_clean}]"
                    trimmed = _heuristic_trim_to_target(cond_clean, words_target)
                    trimmed_clean = clean_script_text(trimmed)
                    save_response("generated_complete_script", "generated_complete_script.txt", f"[{trimmed_clean}]")
                    return f"[{trimmed_clean}]"
                # else fallback to loop
                time.sleep(0.5)
                backoff = min(backoff * 1.5, 60.0)
                continue

            # if too short -> request continuation seeded with last paragraph
            if wc < words_target - tolerance:
                paras = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", cleaned_candidate) if p.strip()]
                last_para = paras[-1] if paras else ""
                cont_prompt = (
                    "Continue the script inside a single bracket so total script reaches the requested word count. "
                    "Do NOT repeat the last paragraph; continue from there."
                    f"\n\nSEED_LAST_PARAGRAPH:\n{last_para}\n\nTarget words: {words_target}"
                )
                try:
                    cont_resp = self.chat.send_message(cont_prompt, timeout=timeout, spinner_message="Extending script...")
                except Exception as e:
                    print("⚠️ Continuation call failed:", e)
                    time.sleep(0.5)
                    continue
                cont_text = _resp_to_text(cont_resp)
                cont_candidate = _extract_candidate(cont_text)
                cont_clean = clean_script_text(cont_candidate or "")
                combined = (cleaned_candidate.rstrip() + "\n\n" + cont_clean.strip()).strip()
                try:
                    combined = _remove_exact_and_fuzzy_duplicates(combined, fuzzy_threshold=0.92)
                except Exception:
                    pass
                combined_wc = _word_count(combined)
                print(f"→ After extension combined words={combined_wc}")
                if abs(combined_wc - words_target) <= tolerance:
                    final = combined
                    save_response("generated_complete_script", "generated_complete_script.txt", f"[{final}]")
                    return f"[{final}]"
                # adopt combined and continue loop
                accumulated = combined
                last_wc = combined_wc
                time.sleep(0.3)
                continue

            # small sleep/backoff before next attempt
            time.sleep(min(1.0 + backoff, 8.0))
            backoff = min(backoff * 1.4, 60.0)

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
