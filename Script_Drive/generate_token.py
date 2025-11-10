#!/usr/bin/env python3
"""
encrypt_client_secrets_simple.py

Encrypts client_secrets.json in the current folder using Fernet (cryptography).
- Input: client_secrets.json  (must exist in current folder)
- Output: client_secrets.json.enc  (encrypted file; safe to commit)
- Key output: client_secrets_key.txt  (Fernet key; keep this secret and store in GitHub Secrets)

Usage:
    1. Place client_secrets.json in the same folder as this script.
    2. pip install cryptography
    3. Run: python encrypt_client_secrets_simple.py
"""

import os
from pathlib import Path
from cryptography.fernet import Fernet

# ---------- CONFIG: change only if needed ----------
INPUT_FILE = "user.json"
OUTPUT_FILE = "user.json.enc"
KEY_FILE = "user_key.txt"
# --------------------------------------------------


def main():
    cwd = os.getcwd()
    in_path = Path(cwd) / INPUT_FILE
    out_path = Path(cwd) / OUTPUT_FILE
    key_path = Path(cwd) / KEY_FILE

    if not in_path.exists():
        raise SystemExit(
            f"ERROR: {INPUT_FILE} not found in current folder ({cwd}). Put your client_secrets.json here."
        )

    # generate key
    key = Fernet.generate_key()
    f = Fernet(key)

    data = in_path.read_bytes()
    token = f.encrypt(data)

    out_path.write_bytes(token)
    key_path.write_text(key.decode(), encoding="utf-8")

    print(f"\n✅ Encrypted file written: {out_path}")
    print(f"✅ Key written to (local backup): {key_path}")
    print(
        "\nIMPORTANT: Copy the key below and store it securely (e.g., GitHub Actions secret named CLIENT_SECRETS_KEY)."
    )
    print("----- BEGIN FERNET KEY -----")
    print(key.decode())
    print("----- END FERNET KEY -----\n")
    print(
        "Do NOT commit the plaintext client_secrets.json or the key file to your repository."
    )
    print("You may safely commit client_secrets.json.enc to your repo.")


if __name__ == "__main__":
    main()
