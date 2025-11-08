import requests
import json

url = "https://apifreellm.com/api/chat"
headers = {
    "Content-Type": "application/json"
}
data = {
    "message": "Hello, how are you?"
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

if result.get("status") == "success":
    print("AI Response:", result["response"])
else:
    print("Error:", result["error"])