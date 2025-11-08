#!/usr/bin/env python3
"""
All-in-one diagnostic script to try multiple approaches against an API protected by Cloudflare.
Prints everything to console. No file writes.

Approaches tried (in order):
  1) requests + browser headers
  2) cloudscraper
  3) Playwright headless to solve JS challenge, extract cookies, then requests
  4) requests via user-provided proxy (if PROXY env var set)

Configure via environment:
  API_URL  - the target URL (default: https://apifreellm.com/api/chat)
  PROXY    - optional proxy URL (http://user:pass@host:port)
Note: Playwright browsers must be installed: `playwright install`.
"""

import os
import sys
import time
import json
import traceback
import requests

# Try to import cloudscraper and playwright lazily so script still runs if not installed
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
PROXY = os.getenv("PROXY")  # optional proxy URL
TIMEOUT = int(os.getenv("TIMEOUT", "100"))
MESSAGE = os.getenv("MESSAGE", "Hello, how are you?")
MAX_RETRIES = 1  # each method does a single attempt; we don't retry endlessly here

def pretty(obj):
    try:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        print(obj)

def is_html_body(resp_text):
    if not resp_text:
        return False
    t = resp_text.strip().lower()
    return t.startswith("<!doctype") or t.lstrip().startswith("<html") or "just a moment" in t or "please enable javascript" in t

def try_requests_headers(url):
    print("\n=== METHOD 1: requests with browser-like headers ===")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        ),
        "Origin": "https://apifreellm.com",
        "Referer": "https://apifreellm.com/",
    }
    payload = {"message": MESSAGE}
    proxies = {"http": PROXY, "https": PROXY} if PROXY else None

    try:
        print(f"Sending POST to {url} (timeout={TIMEOUT}s) with headers (masked):")
        masked = dict(headers)
        if "authorization" in masked:
            masked["authorization"] = masked["authorization"][:8] + "..."  # just in case
        print({k: (v if k.lower() not in ("authorization","x-api-key") else "****") for k,v in masked.items()})
        r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT, proxies=proxies)
        print("Status:", r.status_code)
        print("Response headers:", dict(r.headers))
        ct = r.headers.get("Content-Type","")
        if "application/json" in ct:
            try:
                pretty(r.json())
                return True
            except Exception as e:
                print("Failed to decode JSON:", e)
                print("First 2000 chars of body:")
                print(r.text[:2000])
                return False
        else:
            print("Non-JSON response (first 2000 chars):")
            print(r.text[:2000])
            if is_html_body(r.text):
                print("[detected HTML challenge / Cloudflare page]")
            return False
    except Exception as e:
        print("Exception while requests attempt:", repr(e))
        traceback.print_exc()
        return False

def try_cloudscraper(url):
    print("\n=== METHOD 2: cloudscraper (attempt to solve some CF challenges) ===")
    if cloudscraper is None:
        print("cloudscraper not installed. Skipping. Install: pip install cloudscraper")
        return False
    payload = {"message": MESSAGE}
    try:
        scraper = cloudscraper.create_scraper()
        r = scraper.post(url, json=payload, timeout=TIMEOUT)
        print("Status:", r.status_code)
        ct = r.headers.get("Content-Type","")
        if "application/json" in ct:
            try:
                pretty(r.json())
                return True
            except Exception as e:
                print("Failed to decode JSON:", e)
                print("Body (first 2000):")
                print(r.text[:2000])
                return False
        else:
            print("Non-JSON response (first 2000 chars):")
            print(r.text[:2000])
            if is_html_body(r.text):
                print("[detected HTML challenge / Cloudflare page]")
            return False
    except Exception as e:
        print("Exception while cloudscraper attempt:", repr(e))
        traceback.print_exc()
        return False

def try_playwright_then_requests(url):
    print("\n=== METHOD 3: Playwright headless -> extract cookies -> requests ===")
    if not PLAYWRIGHT_AVAILABLE:
        print("Playwright not installed or not available. Skipping. Install: pip install playwright and run `playwright install`")
        return False

    # Strategy:
    # 1) Navigate to the site root to let Cloudflare challenge run (may set cookies)
    # 2) Extract cookies & user-agent from the page context
    # 3) Use requests.Session with those cookies and same UA to POST to API_URL
    base_origin = os.getenv("PLAYWRIGHT_VISIT", "https://apifreellm.com/")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            print(f"Opening {base_origin} in headless browser (timeout {TIMEOUT*1000}ms)")
            try:
                page.goto(base_origin, wait_until="load", timeout=TIMEOUT*1000)
            except PlaywrightTimeoutError:
                print("Playwright navigation timed out (page may be slow). Continuing - cookies may still be present.")
            except Exception as e:
                print("Playwright navigation exception:", e)
            # Wait a little for JS challenges to run
            time.sleep(3)
            # get cookies
            cookies = context.cookies()
            ua = page.evaluate("() => navigator.userAgent")
            print("Collected cookies (truncated):", [{c['name']: c.get('value')[:30] + ("..." if len(c.get('value',''))>30 else "")} for c in cookies])
            browser.close()

        # Build a requests.Session with cookies from Playwright
        sess = requests.Session()
        for c in cookies:
            # requests cookiejar expects domain without leading dot sometimes; keep as-is
            sess.cookies.set(c['name'], c['value'], domain=c.get('domain', None), path=c.get('path','/'))

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": ua,
            "Origin": "https://apifreellm.com",
            "Referer": "https://apifreellm.com/",
        }
        print("Using extracted UA:", ua)
        print("Attempting POST to API_URL using session and cookies...")
        r = sess.post(url, headers=headers, json={"message": MESSAGE}, timeout=TIMEOUT)
        print("Status:", r.status_code)
        ct = r.headers.get("Content-Type","")
        if "application/json" in ct:
            try:
                pretty(r.json())
                return True
            except Exception as e:
                print("Failed to decode JSON after Playwright:", e)
                print(r.text[:2000])
                return False
        else:
            print("Non-JSON response (first 2000 chars):")
            print(r.text[:2000])
            if is_html_body(r.text):
                print("[detected HTML challenge / Cloudflare page even after Playwright]")
            return False
    except Exception as e:
        print("Playwright attempt raised exception:", repr(e))
        traceback.print_exc()
        return False

def try_proxy(url):
    print("\n=== METHOD 4: requests via PROXY (if PROXY env var set) ===")
    if not PROXY:
        print("No PROXY environment variable provided. Skipping.")
        return False
    print("Using proxy:", PROXY)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }
    try:
        r = requests.post(url, headers=headers, json={"message": MESSAGE}, timeout=TIMEOUT, proxies={"http": PROXY, "https": PROXY})
        print("Status:", r.status_code)
        ct = r.headers.get("Content-Type","")
        if "application/json" in ct:
            pretty(r.json())
            return True
        else:
            print("Non-JSON response (first 2000):")
            print(r.text[:2000])
            return False
    except Exception as e:
        print("Proxy attempt exception:", repr(e))
        traceback.print_exc()
        return False

def main():
    print("Ultimate probe starting for URL:", API_URL)
    print("Playwright available:", PLAYWRIGHT_AVAILABLE)
    print("cloudscraper available:", cloudscraper is not None)
    print("PROXY configured:", bool(PROXY))

    # Try 1: requests headers
    ok = try_requests_headers(API_URL)
    if ok:
        print("\nSUCCESS via Method 1 (requests + headers).")
        return 0

    # Try 2: cloudscraper
    ok = try_cloudscraper(API_URL)
    if ok:
        print("\nSUCCESS via Method 2 (cloudscraper).")
        return 0

    # Try 3: Playwright solve
    ok = try_playwright_then_requests(API_URL)
    if ok:
        print("\nSUCCESS via Method 3 (Playwright -> cookies -> requests).")
        return 0

    # Try 4: proxy
    ok = try_proxy(API_URL)
    if ok:
        print("\nSUCCESS via Method 4 (requests via PROXY).")
        return 0

    print("\nALL METHODS FAILED — see logs above.")
    print("Recommendations:")
    print(" - Confirm the provider has a proper programmatic API (official API domain/path).")
    print(" - Run the script locally (if it works locally but not on GitHub hosted runners, use a self-hosted runner).")
    print(" - Contact the provider to request API access or whitelist your runner IPs.")
    print(" - If you choose to experiment with cloudscraper/playwright/proxies, ensure it complies with provider TOS.")
    return 2

if __name__ == "__main__":
    code = main()
    sys.exit(code)
