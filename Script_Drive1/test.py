#!/usr/bin/env python3
"""
Robust cloudscraper-first probe for apifreellm-style endpoints.
- Prints all diagnostics to stdout (no files).
- Uses cloudscraper if available (solves CF). Falls back to requests + Playwright.
- Retries server errors (5xx) and JSON responses with {"status":"error"} using exponential backoff.
- Does NOT retry on 4xx client errors (403, 400, ...).
Configure via env:
  API_URL (default https://apifreellm.com/api/chat)
  MESSAGE (default "Hello, how are you?")
  PROXY (optional proxy URL if you want to try proxy)
"""

import os, sys, time, json, traceback, random
import requests

# lazy imports
try:
    import cloudscraper
except Exception:
    cloudscraper = None

PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

API_URL = os.getenv("API_URL", "https://apifreellm.com/api/chat")
MESSAGE = os.getenv("MESSAGE", "Hello, how are you?")
PROXY = os.getenv("PROXY") or None
TIMEOUT = int(os.getenv("TIMEOUT", "100"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
INITIAL_BACKOFF = 2.0

def pretty(obj):
    try:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        print(obj)

def is_html(resp_text):
    if not resp_text:
        return False
    t = resp_text.strip().lower()
    return t.startswith("<!doctype") or t.startswith("<html") or "just a moment" in t or "please enable javascript" in t

def perform_post(sess_post, url, headers, payload, proxies=None):
    try:
        resp = sess_post(url, headers=headers, json=payload, timeout=TIMEOUT, proxies=proxies)
    except requests.Timeout:
        print("→ Request timed out")
        return None, "timeout"
    except requests.ConnectionError as e:
        print("→ Connection error:", e)
        return None, "connection_error"
    except Exception as e:
        print("→ Unexpected exception sending request:", e)
        traceback.print_exc()
        return None, "exception"

    print("→ status:", resp.status_code)
    print("→ response headers:", dict(resp.headers))
    text = resp.text or ""
    ct = resp.headers.get("Content-Type", "").lower()
    if "application/json" in ct:
        try:
            data = resp.json()
            return data, None
        except Exception as e:
            print("→ JSON decode failed:", e)
            print("→ text snippet:", text[:2000])
            return None, "json_decode"
    else:
        print("→ Non-JSON response (first 2000 chars):")
        print(text[:2000])
        if is_html(text):
            print("→ detected HTML / Cloudflare challenge page.")
            return None, "html_challenge"
        return None, "non_json"

def cloudscraper_attempt(url, payload, proxies=None):
    if cloudscraper is None:
        print("cloudscraper not installed — skipping this method.")
        return None, "no_cloudscraper"
    print("\n=== cloudscraper attempt ===")
    try:
        s = cloudscraper.create_scraper()
        # optional: respect PROXY
        if proxies:
            r = s.post(url, json=payload, timeout=TIMEOUT, proxies=proxies)
            ct = r.headers.get("Content-Type","")
            if "application/json" in ct:
                return r.json(), None
            else:
                return None, "non_json"
        else:
            r = s.post(url, json=payload, timeout=TIMEOUT)
            ct = r.headers.get("Content-Type","")
            if "application/json" in ct:
                return r.json(), None
            else:
                return None, "non_json"
    except Exception as e:
        print("cloudscraper exception:", repr(e))
        traceback.print_exc()
        return None, "cloudscraper_exception"

def requests_attempt(url, payload, proxies=None):
    print("\n=== requests attempt (browser-like headers) ===")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"),
        "Origin": "https://apifreellm.com",
        "Referer": "https://apifreellm.com/",
    }
    return perform_post(requests.post, url, headers, payload, proxies=proxies)

def playwright_attempt_then_requests(url, payload, proxies=None):
    print("\n=== Playwright attempt (solve challenge in headless browser, reuse cookies) ===")
    if not PLAYWRIGHT_AVAILABLE:
        print("Playwright not installed — skipping this method.")
        return None, "no_playwright"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            print("Opening site root to allow JS challenge to run...")
            try:
                page.goto("https://apifreellm.com/", wait_until="load", timeout=TIMEOUT*1000)
            except PlaywrightTimeoutError:
                print("Playwright load timed out (may still have cookies).")
            time.sleep(2)
            cookies = context.cookies()
            ua = page.evaluate("() => navigator.userAgent")
            browser.close()

        sess = requests.Session()
        for c in cookies:
            sess.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path", "/"))

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "User-Agent": ua,
            "Origin": "https://apifreellm.com",
            "Referer": "https://apifreellm.com/",
        }
        print("Using extracted UA:", ua)
        return perform_post(sess.post, url, headers, payload, proxies=proxies)
    except Exception as e:
        print("Playwright attempt exception:", e)
        traceback.print_exc()
        return None, "playwright_exception"

def robust_post_with_backoff(method_fn, url, payload, proxies=None, max_retries=MAX_RETRIES):
    attempt = 0
    backoff = INITIAL_BACKOFF
    while attempt < max_retries:
        attempt += 1
        print(f"\n-- attempt {attempt}/{max_retries} using {method_fn.__name__}")
        result, err = method_fn(url, payload, proxies=proxies)
        if result is not None:
            # if API returns JSON but with status:error — treat as server-side issue and retry
            if isinstance(result, dict) and result.get("status") == "error":
                print("API returned JSON with status=error:")
                pretty(result)
                # decide to retry for server/internal errors
                # if error explicitly says internal server error -> retry
                # but if it says authentication/client issue, break
                msg = result.get("error") or result.get("message") or ""
                if "internal" in msg.lower() or not msg:
                    if attempt < max_retries:
                        sleep = backoff + random.random()
                        print(f"Transient server error detected; retrying in {sleep:.1f}s...")
                        time.sleep(sleep)
                        backoff *= 2
                        continue
                    else:
                        return result, "server_error_final"
                else:
                    # non-transient error (client-like) — do not retry
                    return result, "api_client_error"
            else:
                # success-like JSON or other acceptable content
                print("Successful JSON response:")
                pretty(result)
                return result, None
        else:
            # no JSON or error code; err explains why
            print("Method returned no JSON; error:", err)
            # If it's an HTML challenge or 4xx, do not retry with the same method
            if err in ("html_challenge", "non_json", "cloudscraper_exception", "no_cloudscraper"):
                return None, err
            # For transient network errors, retry
            if err in ("timeout", "connection_error", "json_decode", "exception"):
                if attempt < max_retries:
                    sleep = backoff + random.random()
                    print(f"Transient transport error; retrying in {sleep:.1f}s...")
                    time.sleep(sleep)
                    backoff *= 2
                    continue
            return None, err
    return None, "max_retries_exceeded"

def main():
    payload = {"message": MESSAGE}
    proxies = {"http": PROXY, "https": PROXY} if PROXY else None

    print("Probe starting. API_URL:", API_URL)
    print("cloudscraper available:", bool(cloudscraper))
    print("playwright available:", PLAYWRIGHT_AVAILABLE)
    print("PROXY configured:", bool(PROXY))

    # Prefer cloudscraper (it solved CF in your run)
    if cloudscraper:
        res, err = robust_post_with_backoff(cloudscraper_attempt, API_URL, payload, proxies=proxies)
        if res is not None and err is None:
            print("\nDONE: cloudscraper succeeded and returned usable JSON.")
            return 0
        else:
            print("\ncloudscraper did not produce usable response. err:", err)

    # Fallback: Playwright (if available)
    if PLAYWRIGHT_AVAILABLE:
        res, err = robust_post_with_backoff(playwright_attempt_then_requests, API_URL, payload, proxies=proxies)
        if res is not None and err is None:
            print("\nDONE: Playwright path succeeded and returned usable JSON.")
            return 0
        else:
            print("\nPlaywright path failed. err:", err)

    # Last fallback: basic requests
    res, err = robust_post_with_backoff(requests_attempt, API_URL, payload, proxies=proxies)
    if res is not None and err is None:
        print("\nDONE: requests succeeded.")
        return 0

    print("\nALL METHODS exhausted. See logs above. Recommendations:")
    print(" - Check provider docs for exact required payload fields (maybe not 'message').")
    print(" - If server returns Internal Server Error repeatedly, contact provider or retry later.")
    print(" - For CI use, prefer self-hosted runner for stability.")
    return 2

if __name__ == '__main__':
    code = main()
    sys.exit(code)
