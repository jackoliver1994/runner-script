import requests
import json
import time
import traceback

url = "https://apifreellm.com/api/chat"
headers = {"Content-Type": "application/json"}
data = {"message": "Hello, how are you?"}

MAX_RETRIES = 3
TIMEOUT = 100  # seconds
WAIT_BETWEEN_RETRIES = 5  # seconds


def safe_request(url, headers, data, timeout):
    """Send POST request safely and handle all exceptions."""
    try:
        print(f"⏳ Sending request to {url} with timeout={timeout}s ...")
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        print(f"✅ Response received with status code: {response.status_code}")

        if not response.text.strip():
            print("⚠️ Empty response body from server.")
            return None

        try:
            result = response.json()
            return result
        except json.JSONDecodeError:
            print("❌ Failed to parse JSON. Response text was:")
            print(response.text)
            return None

    except requests.Timeout:
        print(f"⏰ Request timed out after {timeout}s.")
    except requests.ConnectionError as e:
        print(f"🔌 Connection error: {e}")
    except requests.RequestException as e:
        print(f"❗ General request exception: {e}")
    except Exception as e:
        print("🔥 Unexpected error during request:")
        traceback.print_exc()

    return None


def main():
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n=== Attempt {attempt}/{MAX_RETRIES} ===")

        result = safe_request(url, headers, data, TIMEOUT)

        if result:
            if result.get("status") == "success":
                print(f"🤖 AI Response: {result.get('response', '[No content]')}")
                return
            else:
                print(f"❗ API Error: {result.get('error', 'Unknown error')}")
                return
        else:
            print(f"⚠️ Attempt {attempt} failed, retrying in {WAIT_BETWEEN_RETRIES}s...")
            time.sleep(WAIT_BETWEEN_RETRIES)

    print("❌ All attempts failed. Exiting script.")


if __name__ == "__main__":
    main()
