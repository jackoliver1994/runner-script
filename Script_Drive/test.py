import requests
import json
import time
import traceback

url = "https://apifreellm.com/api/chat"
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

data = {"message": "Hello, how are you?"}
TIMEOUT = 100
MAX_RETRIES = 3
WAIT_BETWEEN = 5

def safe_request():
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=TIMEOUT)
        print(f"Status: {resp.status_code}")
        if "application/json" in resp.headers.get("Content-Type", ""):
            print("Response JSON:", json.dumps(resp.json(), indent=2))
        else:
            print("Non-JSON response (first 500 chars):")
            print(resp.text[:500])
    except Exception:
        traceback.print_exc()

for i in range(1, MAX_RETRIES + 1):
    print(f"\n=== Attempt {i}/{MAX_RETRIES} ===")
    safe_request()
    if i < MAX_RETRIES:
        time.sleep(WAIT_BETWEEN)
