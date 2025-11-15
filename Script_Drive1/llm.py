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
import subprocess
import importlib
from types import SimpleNamespace
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


# ---------------------- LOCAL LLM ADAPTER + CHAT API HANDLER (REPLACEMENT) ----------------------

try:
    LoadingSpinner  # type: ignore
except Exception:

    class LoadingSpinner:
        def __init__(self, msg=""):
            self.msg = msg

        def start(self):
            pass

        def stop(self):
            pass


class LocalLLM:
    def __init__(
        self,
        local_model: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 1024,
    ):
        self.local_model = local_model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.generator = None
        self._ready = False
        if self.local_model:
            self._attempt_init_with_repair()

    def _pip_install(self, package: str):
        """
        Robust pip installer used during runtime repair. Shows logs and tries a CPU torch wheel fallback.
        NOTE: Installing large packages (torch) can be slow in CI and may fail on restricted runners.
        """
        import shlex

        try:
            print(f"📦 pip installing {package} ...", flush=True)
            # split to handle things like "package==x.y" or "pkg[extra]"
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
            ] + shlex.split(package)
            subprocess.check_call(cmd)
            print(f"✅ Installed {package}", flush=True)
        except Exception as e:
            print(f"⚠️ pip install {package} failed: {e}", flush=True)
            # Special-case: torch - try CPU wheel if device='cpu' and default install failed
            if (
                "torch" in package
                and getattr(self, "device", None)
                and self.device.lower() == "cpu"
            ):
                try:
                    print(
                        "🔁 Attempting CPU-only torch wheel via PyTorch index (best-effort)...",
                        flush=True,
                    )
                    fallback_cmd = [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "--force-reinstall",
                        "torch",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                    ]
                    subprocess.check_call(fallback_cmd)
                    print(
                        "✅ Installed torch (cpu wheel) via PyTorch index.", flush=True
                    )
                except Exception as e2:
                    print("⚠️ CPU torch fallback failed:", e2, flush=True)

    def _try_imports(self):
        """
        Try to import needed libraries; return tuple(transformers_available, numpy_ok, error_message)
        """
        try:
            import transformers  # type: ignore

            transformers_ok = True
            transformers_err = None
        except Exception as e:
            transformers_ok = False
            transformers_err = str(e)
        # Check numpy import separately to detect ABI issues
        try:
            import numpy as np  # type: ignore

            numpy_ok = True
            numpy_err = None
        except Exception as e:
            numpy_ok = False
            numpy_err = str(e)
        # prefer reporting transformer import error if present, else numpy err
        return transformers_ok, numpy_ok, transformers_err or numpy_err

    def _repair_numpy_if_needed(self, err_msg: str) -> bool:
        """
        If the error message suggests a numpy ABI mismatch, attempt a safe reinstall/repair.
        Returns True if an attempt was made (even if it fails), False if not attempted.
        Controlled by ENV var LOCAL_LLM_AUTO_FIX (set to '0' to skip auto-fix).
        """
        import os

        auto_fix = os.environ.get("LOCAL_LLM_AUTO_FIX", "1") != "0"
        if not auto_fix:
            print("ℹ️ LOCAL_LLM_AUTO_FIX=0 — skipping auto-repair of numpy.")
            return False

        # Detect typical ABI message
        if (
            "numpy.dtype size changed" in (err_msg or "")
            or "binary incompatibility" in (err_msg or "").lower()
        ):
            print(
                "⚠️ Detected numpy binary-compatibility error. Attempting to reinstall numpy (best-effort)."
            )
            # Try a safe reinstall: upgrade and force-reinstall numpy to get matching wheel
            try:
                target = "numpy"
                self._pip_install(f"{target} --upgrade --force-reinstall")
                return True
            except Exception as e:
                print("⚠️ numpy reinstall attempt failed:", e, flush=True)
                return True
        return False

    def _ensure_transformers_and_deps(self):
        """
        Ensure transformers and a compatible backend are installed.
        Best-effort: pin huggingface-hub and transformers to compatible versions,
        install torch explicitly (attempt CPU wheel if device=='cpu'), and common extras.
        """
        try:
            transformers_ok, numpy_ok, err = self._try_imports()
        except Exception:
            transformers_ok, numpy_ok, err = (False, False, None)

        # If transformers not installed or import failed -> attempt to install compatible set
        if not transformers_ok:
            # First try a conservative, known-compatible set
            try:
                # pin huggingface-hub to eliminate split_torch_state_dict mismatch
                self._pip_install("huggingface-hub>=0.18.4")
                # install transformers with torch extras (this may pull a torch)
                self._pip_install("transformers[torch]>=4.35.2")
                # install accelerate/safetensors/sentencepiece
                self._pip_install("accelerate")
                self._pip_install("safetensors")
                self._pip_install("sentencepiece")
            except Exception as e:
                print(
                    "⚠️ Best-effort install of transformers extras failed:",
                    e,
                    flush=True,
                )

        # Ensure torch is present; try explicit cpu wheel if device == 'cpu'
        try:
            import torch  # type: ignore

            torch_ok = True
        except Exception:
            torch_ok = False

        if not torch_ok:
            try:
                if getattr(self, "device", None) and self.device.lower() == "cpu":
                    # Try CPU-specific wheel index (best-effort)
                    self._pip_install(
                        "torch --index-url https://download.pytorch.org/whl/cpu"
                    )
                else:
                    # generic torch install (let pip choose CUDA vs CPU if available)
                    self._pip_install("torch")
            except Exception as e:
                print("⚠️ Attempt to install torch failed:", e, flush=True)

        # re-check imports to update error info
        transformers_ok, numpy_ok, err = self._try_imports()

        # If transformers exists but we still see the 'split_torch_state_dict_into_shards' error,
        # force reinstall to a known combination that usually resolves the import mismatch.
        if err and "split_torch_state_dict_into_shards" in str(err):
            print(
                "⚠️ Detected huggingface_hub/transformers mismatch; attempting pinned reinstall.",
                flush=True,
            )
            try:
                self._pip_install("huggingface-hub==0.18.4")
                self._pip_install("transformers==4.35.2")
            except Exception as e:
                print("⚠️ Pin reinstall attempt failed:", e, flush=True)

        # final re-check
        transformers_ok, numpy_ok, err = self._try_imports()
        if not transformers_ok:
            print(
                "⚠️ transformers still not importable after attempts. Local LLM will remain unavailable.",
                flush=True,
            )

    def _init_pipeline(self):
        """
        Initialize the HF pipeline. Wrap in try/except to surface meaningful errors.
        Keeps 'trust_remote_code' semantics so remote repos with custom code can load.
        """
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # type: ignore

            kwargs = {"trust_remote_code": True}
            if self.device:
                if self.device.lower() == "cpu":
                    # force CPU map to avoid CUDA attempts
                    kwargs["device_map"] = "cpu"
                elif self.device.lower() == "cuda":
                    kwargs["device_map"] = "auto"

            print(
                f"🔁 Loading local model '{self.local_model}' (this may download weights)...",
                flush=True,
            )
            try:
                # preferred call (may download weights)
                self.generator = pipeline(
                    "text-generation", model=self.local_model, **kwargs
                )
            except Exception as e_pipeline:
                # fallback to explicit tokenizer + model loads to give more explicit errors
                print("ℹ️ pipeline() fallback triggered:", e_pipeline, flush=True)
                tokenizer = AutoTokenizer.from_pretrained(
                    self.local_model, use_fast=True, trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.local_model, trust_remote_code=True
                )

                def gen_fn(prompt, max_new_tokens=self.max_new_tokens):
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
                    return tokenizer.decode(outputs[0], skip_special_tokens=True)

                self.generator = SimpleNamespace(
                    __call__=lambda prompt, **kw: [
                        {
                            "generated_text": gen_fn(
                                prompt, kw.get("max_new_tokens", self.max_new_tokens)
                            )
                        }
                    ]
                )

            self._ready = bool(self.generator is not None)
            if self._ready:
                print("✅ Local generator ready.", flush=True)
            else:
                print("⚠️ Local generator creation returned None.", flush=True)
        except Exception as e:
            # raise a clear runtime error so higher-level code can decide whether to fallback or abort
            raise RuntimeError(f"Exception while initializing local pipeline: {e}")

    def _attempt_init_with_repair(self):
        """
        Attempt to initialize the pipeline, and if a numpy ABI error occurs, try to repair and retry once.
        """
        # Try once, if fails due to numpy ABI, attempt repair and try again (one retry)
        self._ensure_transformers_and_deps()
        try:
            self._init_pipeline()
        except RuntimeError as e:
            # Check message for numpy ABI and attempt repair once if found
            err_text = str(e)
            print(f"⚠️ {err_text}", flush=True)
            attempted = self._repair_numpy_if_needed(err_text)
            if attempted:
                # reload python modules in a best-effort way
                try:
                    print("🔁 Re-importing environment after repair...", flush=True)
                    importlib.invalidate_caches()
                    # attempt to re-import critical libs
                    importlib.reload(importlib.import_module("importlib"))
                except Exception:
                    pass
                # one retry
                try:
                    self._ensure_transformers_and_deps()
                    self._init_pipeline()
                except Exception as e2:
                    print(f"⚠️ Retry after repair also failed: {e2}", flush=True)
                    self._ready = False
                    self.generator = None
            else:
                # no repair attempted or not applicable
                self._ready = False
                self.generator = None

    def is_ready(self) -> bool:
        return bool(self._ready and self.generator is not None)

    def generate(
        self, prompt: str, timeout: int = 300, max_new_tokens: Optional[int] = None
    ) -> str:
        if not self.is_ready():
            raise RuntimeError("LocalLLM not ready.")
        max_tokens = max_new_tokens or self.max_new_tokens
        try:
            out = self.generator(prompt, max_new_tokens=max_tokens)
            if isinstance(out, list) and out:
                return out[0].get("generated_text", "")
            elif isinstance(out, str):
                return out
            else:
                return str(out)
        except Exception as e:
            raise RuntimeError(f"Local generation failed: {e}")


# Backwards-compatible ChatAPI wrapper (keeps remote retry/backoff/infinite loop)
class ChatAPI:
    def __init__(
        self,
        url: str = "https://apifreellm.com/api/chat",
        default_timeout: int = 100,
        local_model: Optional[str] = None,
        local_device: Optional[str] = None,
        require_local: bool = False,
        require_remote: bool = False,
        allow_fallback: bool = False,
    ):
        """
        ChatAPI constructor.
        - local_model: name or path for a local HF-compatible model (e.g. "mistral-small-3.1")
        - require_local: if True, insist on local-only; if local init fails, raise immediately (no remote fallback).
        - require_remote: if True, force remote-only usage even if local is available.
        - allow_fallback: if True, allow fallback between local <-> remote; otherwise prefer single-mode:
            * If local_model is provided and allow_fallback is False => treat as require_local by default.
            * If no local_model provided, remote usage is used (require_remote=False).
        """
        import requests  # expected to be available in original codebase

        self.requests = requests
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        self.base_timeout = default_timeout

        # Mode flags
        self.require_local = bool(require_local)
        self.require_remote = bool(require_remote)
        self.allow_fallback = bool(allow_fallback)

        # If caller provided a local_model and did not explicitly allow fallback,
        # interpret that as they want local-only behavior (no silent remote fallback).
        if local_model and not self.allow_fallback and not require_remote:
            # prefer local-only by default when user explicitly asked for local_model
            self.require_local = True
            self.require_remote = False

        # store wanted local model/device
        self.local_model = local_model
        self.local_device = local_device
        self.local_llm: Optional[LocalLLM] = None

        # Attempt local model initialization only if local_model is non-empty AND
        # we're not in strict remote-only mode.
        if self.local_model and not self.require_remote:
            try:
                print(
                    f"🔎 Attempting to initialize local model '{self.local_model}' on device='{self.local_device}'..."
                )
                self.local_llm = LocalLLM(
                    local_model=self.local_model, device=self.local_device
                )
                if not self.local_llm.is_ready():
                    if self.require_local:
                        raise RuntimeError(
                            "Local model specified but failed to initialize and 'require_local' is set."
                        )
                    else:
                        print(
                            "⚠️ Local model specified but failed to initialize. Will fall back to remote API."
                        )
                        self.local_llm = None
                        if not self.local_llm:
                            self._attempt_local_repair_and_retry()
            except Exception as e:
                print("⚠️ Error initializing LocalLLM:", e)
                if self.require_local:
                    raise
                self.local_llm = None

        # If user explicitly asked for remote only, and local existed, ensure we don't use it
        if self.require_remote:
            self.local_llm = None

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
        use_curl_impersonate: bool = True,
        proxy: str = None,
        specific_error: list = None,
    ):
        """
        Robust send_message preserving infinite retry semantics and adding:
        - proxy rotation when proxies fail,
        - cloudscraper fallback on Cloudflare/403,
        - Playwright stealth fallback (if available and enabled),
        - optional curl impersonation fallback.

        NOTE: This replaces the original block that used a f-string containing backslashes
        inside an expression (caused the SyntaxError). The fix builds JS-safe strings
        first (via json.dumps) and then constructs the JS snippet.
        """
        import requests
        import traceback
        import subprocess
        import json
        import random
        from requests.exceptions import ProxyError as ReqProxyError, RequestException

        if specific_error is None:
            specific_error = [
                "response status: 403",
                "detected cloudflare-like anti-bot page",
                "cloudflare",
                "access denied",
                "anti-bot",
                "enable javascript and cookies",
            ]
        specific_error = [s.lower() for s in specific_error]

        def log(*args, **kwargs):
            ts = time.strftime("%H:%M:%S")
            kwargs.setdefault("flush", True)
            print(f"[{ts}]", *args, **kwargs)

        configured_timeout = int(timeout or getattr(self, "base_timeout", 60))
        configured_timeout = min(configured_timeout, max_timeout_cap)
        attempt = 0
        backoff = initial_backoff

        # pick initial proxy (argument overrides instance)
        current_proxy = proxy or getattr(self, "_current_proxy", None)

        # local helper to mark bad proxy
        def _mark_bad_proxy_local(p):
            try:
                if not p:
                    return
                pool = getattr(self, "_proxy_pool", None)
                if pool and p in pool:
                    pool = [x for x in pool if x != p]
                    self._proxy_pool = pool
                    log("🗑️ Removed bad proxy from pool:", p, "remaining:", len(pool))
                if getattr(self, "_current_proxy", None) == p:
                    self._current_proxy = None
            except Exception:
                pass

        # spinner
        spinner = None
        if spinner_message:
            try:
                spinner = LoadingSpinner(spinner_message)
                spinner.start()
            except Exception:
                pass

        # optional imports
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

        session = None

        while True:
            attempt += 1
            try:
                log(
                    f"Attempt #{attempt} — timeout={configured_timeout}s — proxy={current_proxy}"
                )
                # build headers + proxy tuple from instance helper (if present)
                try:
                    if hasattr(self, "_pick_proxy_and_headers"):
                        prx, headers = self._pick_proxy_and_headers()
                        current_proxy = prx or current_proxy
                    else:
                        headers = dict(
                            getattr(
                                self, "headers", {"Content-Type": "application/json"}
                            )
                        )
                        # mild UA impersonation
                        headers["User-Agent"] = headers.get(
                            "User-Agent",
                            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
                        )
                except Exception:
                    headers = dict(
                        getattr(self, "headers", {"Content-Type": "application/json"})
                    )

                # create session if needed and avoid env proxies
                if session is None:
                    session = requests.Session()
                    try:
                        session.trust_env = False
                    except Exception:
                        pass
                    if current_proxy:
                        p = str(current_proxy)
                        if not (
                            p.startswith("http://")
                            or p.startswith("https://")
                            or p.startswith("socks5://")
                            or p.startswith("socks4://")
                        ):
                            p = "http://" + p
                        session.proxies.update({"http": p, "https": p})

                # perform HTTP request (wrapped so we can safely catch network-level errors)
                try:
                    spinner and spinner.start()
                    r = session.post(
                        self.url,
                        headers=headers,
                        json={"message": message},
                        timeout=(10, min(configured_timeout, 300)),
                    )
                    spinner and spinner.stop()
                except Exception as e:
                    spinner and spinner.stop()
                    # network-level failure -> mark proxy bad when appropriate and retry
                    log(
                        f"⚠️ Network exception on attempt #{attempt}: {type(e).__name__} — {getattr(e, 'args', '')}"
                    )
                    log(traceback.format_exc())
                    if current_proxy:
                        _mark_bad_proxy_local(current_proxy)
                        current_proxy = None
                        try:
                            # pick next proxy (best-effort)
                            if getattr(self, "_proxy_pool", None):
                                current_proxy = self._proxy_pool[
                                    getattr(self, "_proxy_index", 0)
                                    % max(1, len(self._proxy_pool))
                                ]
                                self._proxy_index = (
                                    getattr(self, "_proxy_index", 0) + 1
                                ) % max(1, len(self._proxy_pool))
                                self._current_proxy = current_proxy
                        except Exception:
                            pass
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, max_backoff)
                    continue

                # Normal response handling
                status = getattr(r, "status_code", None)
                body = getattr(r, "text", "") or ""
                headers_resp = r.headers or {}

                # detect cloudflare-like or specific error in body
                lower_body = (body or "").lower()
                matched_specific = any(tok in lower_body for tok in specific_error) or (
                    status == 403
                )

                # If proxy-level failure detected -> mark proxy bad and rotate
                if (
                    isinstance(r, requests.Response) and status in (502, 503, 504)
                ) or matched_specific:
                    log(
                        f"⚠️ Detected problematic response (status={status}). matched_specific={matched_specific}"
                    )
                    if current_proxy:
                        _mark_bad_proxy_local(current_proxy)
                        current_proxy = None
                        # try to pick new proxy
                        try:
                            if getattr(self, "_proxy_pool", None):
                                current_proxy = self._proxy_pool[
                                    getattr(self, "_proxy_index", 0)
                                    % max(1, len(self._proxy_pool))
                                ]
                                self._proxy_index = (
                                    getattr(self, "_proxy_index", 0) + 1
                                ) % max(1, len(self._proxy_pool))
                                self._current_proxy = current_proxy
                        except Exception:
                            pass

                    # cloudscraper fallback
                    if (
                        matched_specific
                        and use_cloudscraper_on_403
                        and cloudscraper is not None
                    ):
                        try:
                            log(
                                "→ Detected 403/challenge; attempting cloudscraper fallback (using current proxy if present)."
                            )
                            cs = cloudscraper.create_scraper()
                            if current_proxy:
                                p = str(current_proxy)
                                if not (
                                    p.startswith("http://")
                                    or p.startswith("https://")
                                    or p.startswith("socks5://")
                                    or p.startswith("socks4://")
                                ):
                                    p = "http://" + p
                                cs.proxies.update({"http": p, "https": p})
                            cs.headers.update(headers)
                            cs_resp = cs.post(
                                self.url,
                                json={"message": message},
                                timeout=(10, min(configured_timeout, 300)),
                            )
                            if cs_resp.status_code == 200:
                                log(
                                    "✅ cloudscraper succeeded; returning cloudscraper response."
                                )
                                try:
                                    return cs_resp.json()
                                except Exception:
                                    return cs_resp.text
                            else:
                                log(
                                    "⚠️ cloudscraper fallback returned status",
                                    cs_resp.status_code,
                                )
                        except Exception as e:
                            log(
                                "⚠️ cloudscraper attempt failed:", traceback.format_exc()
                            )
                            if current_proxy:
                                _mark_bad_proxy_local(current_proxy)
                                current_proxy = None
                            time.sleep(min(backoff, 4) + random.random())
                            backoff = min(backoff * 2, max_backoff)
                            continue

                    # Playwright fallback (if enabled)
                    if (
                        matched_specific
                        and use_playwright_on_403
                        and PLAYWRIGHT_AVAILABLE
                    ):
                        try:
                            log(
                                "→ Attempting Playwright stealth fetch to acquire cookies/session."
                            )
                            from playwright.sync_api import sync_playwright as _sync_pw

                            with _sync_pw() as pw:
                                browser = pw.chromium.launch(
                                    headless=True, args=["--no-sandbox"]
                                )
                                context = browser.new_context(
                                    user_agent=headers.get("User-Agent")
                                )
                                page = context.new_page()
                                # navigate to base url to solve challenge
                                page.goto(self.url, timeout=30000)

                                # Build JS-escaped / JSON-safe pieces outside of the f-string expression.
                                # json.dumps gives a JS-safe quoted string (includes surrounding quotes).
                                js_url = json.dumps(self.url)
                                js_body = json.dumps({"message": message})

                                fetch_script = (
                                    f"() => fetch({js_url}, {{"
                                    f"method: 'POST', headers: {{ 'Content-Type': 'application/json' }},"
                                    f"body: JSON.stringify({js_body})"
                                    f"}}).then(r => r.text()).then(t => t).catch(e => String(e));"
                                )

                                result = page.evaluate(fetch_script)
                                browser.close()
                                if result:
                                    log(
                                        "✅ Playwright fetch returned content; returning it."
                                    )
                                    # try to parse JSON before returning raw string
                                    try:
                                        return json.loads(result)
                                    except Exception:
                                        return result
                        except Exception as e:
                            log("⚠️ Playwright fallback failed:", traceback.format_exc())
                            # treat this as non-fatal and continue to other fallbacks / retry

                    # optional curl impersonation fallback (lightweight)
                    if matched_specific and use_curl_impersonate:
                        try:
                            log(
                                "→ Attempting curl impersonation fallback (best-effort)."
                            )
                            ua = headers.get(
                                "User-Agent", "Mozilla/5.0 (X11; Linux x86_64)"
                            )
                            curl_cmd = [
                                "curl",
                                "-sS",
                                "-X",
                                "POST",
                                "-H",
                                "Content-Type: application/json",
                                "-H",
                                f"User-Agent: {ua}",
                                "-d",
                                json.dumps({"message": message}),
                                self.url,
                            ]
                            if current_proxy:
                                curl_cmd.insert(-1, "--proxy")
                                curl_cmd.insert(-1, f"http://{current_proxy}")
                            out = subprocess.check_output(
                                curl_cmd,
                                stderr=subprocess.STDOUT,
                                timeout=max(60, configured_timeout),
                            )
                            out = out.decode("utf-8", errors="ignore")
                            if out:
                                log(
                                    "✅ curl fallback produced a response; returning it."
                                )
                                try:
                                    return json.loads(out)
                                except Exception:
                                    return out
                        except Exception:
                            log("⚠️ curl fallback failed:", traceback.format_exc())

                    # else backoff and loop to try next proxy / retry
                    log(f"⚠️ Backing off {backoff:.1f}s and retrying.")
                    time.sleep(backoff + random.uniform(0, 1.0))
                    backoff = min(backoff * 2, max_backoff)
                    configured_timeout = min(
                        configured_timeout + timeout_growth, max_timeout_cap
                    )
                    continue

                # If we get here, status code is not a 5xx/403 problematic case handled above.
                # Try parsing JSON if possible
                parsed = None
                try:
                    ct = (headers_resp.get("Content-Type") or "").lower()
                    if "application/json" in ct or (
                        isinstance(body, str)
                        and (
                            body.strip().startswith("{") or body.strip().startswith("[")
                        )
                    ):
                        parsed = json.loads(body) if body else {}
                except Exception:
                    log("⚠️ Failed to parse JSON. Response snippet:")
                    log((body or "")[:2000])

                if parsed and (
                    (isinstance(parsed, dict) and parsed.get("status") == "success")
                    or ("message" in parsed and parsed.get("message"))
                ):
                    log("✅ API returned success. Returning parsed payload or body.")
                    self._current_proxy = current_proxy
                    return parsed

                # If raw body but contains anti-bot markers, rotate/ retry
                if not parsed and body:
                    low = (body or "").lower()
                    if any(err_marker in low for err_marker in specific_error):
                        log(
                            f"→ Detected anti-bot / specific error marker in body (attempt {attempt}). Rotating proxy + retrying."
                        )
                        if current_proxy:
                            _mark_bad_proxy_local(current_proxy)
                            current_proxy = None
                        time.sleep(backoff + random.random())
                        backoff = min(backoff * 2, max_backoff)
                        configured_timeout = min(
                            configured_timeout + timeout_growth, max_timeout_cap
                        )
                        continue
                    log("→ Returning raw body (non-JSON).")
                    return body

                # Last resort: return parsed or raw body
                if parsed:
                    return parsed
                return body

            except Exception as e:
                log("⚠️ Unexpected error in send_message loop:", traceback.format_exc())
                time.sleep(backoff + random.random())
                backoff = min(backoff * 2, max_backoff)
                configured_timeout = min(
                    configured_timeout + timeout_growth, max_timeout_cap
                )
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
        default_timeout: int = 100,
        local_model: str | None = None,
        local_device: str | None = None,
        require_local: bool = False,
        require_remote: bool = False,
        allow_fallback: bool = False,
    ):
        """
        StoryPipeline initializer (robust, no-feature-removal).

        - local_model: name/path of local LLM to use (e.g. "mistral-small-3.1").
        - local_device: device for local model ("cpu" or "cuda").
        - require_local: if True -> local-only (raise if local unavailable).
        - require_remote: if True -> remote-only (never use local).
        - allow_fallback: if True -> allow fallback between local and remote on failures.

        Behavior:
        - If local_model is provided and allow_fallback is False and require_remote is False,
        default to local-only behavior (require_local=True).
        - If both require_local and require_remote are True -> raise ValueError.
        - Attempts to pass local flags into ChatAPI constructor. If ChatAPI doesn't accept them,
        the constructor falls back to post-construction attribute setting and a one-time
        local re-init attempt via _attempt_local_repair_and_retry().
        """
        # store pipeline-level attributes first
        self.api_url = api_url
        self.default_timeout = default_timeout
        self.local_model = local_model
        self.local_device = local_device

        # flags
        self.require_local = bool(require_local)
        self.require_remote = bool(require_remote)
        self.allow_fallback = bool(allow_fallback)

        # If a local_model was provided but the caller did not explicitly allow fallback or
        # require remote-only, prefer to allow fallback by default rather than enforce local-only.
        # This avoids hard failures when a given model id is not available locally or on HF
        # (e.g. shorthand names like "mistral-small-3.1" that may be private or require token).
        # If callers want strict local-only behavior, they can pass require_local=True explicitly.
        if self.local_model and not self.allow_fallback and not self.require_remote:
            # prefer safe fallback behavior by default
            self.allow_fallback = True
            self.require_local = False
            self.require_remote = False
            print(
                "ℹ️ local_model was provided but allow_fallback was not set — defaulting to allow remote fallback. "
                "Pass require_local=True to insist on local-only behavior."
            )

        if self.require_local and self.require_remote:
            raise ValueError(
                "Conflicting options: require_local and require_remote cannot both be True."
            )

        # Instantiate ChatAPI. Prefer to pass the local/flag args into ChatAPI constructor
        # so ChatAPI can run its own local init logic in its __init__ if implemented.
        try:
            # Prefer constructor signature supporting these extra args (if ChatAPI accepts them)
            self.chat = ChatAPI(
                url=self.api_url,
                default_timeout=self.default_timeout,
                local_model=self.local_model,
                local_device=self.local_device,
                require_local=self.require_local,
                require_remote=self.require_remote,
                allow_fallback=self.allow_fallback,
            )
        except TypeError:
            # older ChatAPI may not accept those args — create basic instance then set attributes
            self.chat = ChatAPI(url=self.api_url, default_timeout=self.default_timeout)
            # set attributes on chat so send_message / other methods can read them
            try:
                self.chat.local_model = self.local_model
                self.chat.local_device = self.local_device
                self.chat.require_local = self.require_local
                self.chat.require_remote = self.require_remote
                self.chat.allow_fallback = self.allow_fallback
            except Exception:
                # fallback to setattr for robustness
                setattr(self.chat, "local_model", self.local_model)
                setattr(self.chat, "local_device", self.local_device)
                setattr(self.chat, "require_local", self.require_local)
                setattr(self.chat, "require_remote", self.require_remote)
                setattr(self.chat, "allow_fallback", self.allow_fallback)

            # If local_model was requested, the ChatAPI may not have tried to init local LLM
            # because it was constructed earlier. Attempt one best-effort repair + re-init here.
            if self.local_model:
                try:
                    # if ChatAPI provides a helper, call it; else call pipeline helper that uses LocalLLM
                    if hasattr(self.chat, "_attempt_local_repair_and_retry"):
                        # prefer chat's own helper if present
                        self.chat._attempt_local_repair_and_retry()
                    else:
                        # Use the pipeline-level helper to attempt one repair and init
                        self._attempt_local_repair_and_retry()
                except Exception as e:
                    # If require_local is True, propagate the error; else continue and let fallback behavior decide
                    if self.require_local:
                        raise RuntimeError(
                            f"Local model requested but automatic re-init failed: {e}"
                        )

        # Now, post-construction sanity: if require_local is set, ensure local_llm is present and ready
        local_ready = False
        try:
            local_llm_obj = getattr(self.chat, "local_llm", None)
            if (
                local_llm_obj
                and hasattr(local_llm_obj, "is_ready")
                and local_llm_obj.is_ready()
            ):
                local_ready = True
            # some implementations use a _ready flag
            elif local_llm_obj and getattr(local_llm_obj, "_ready", False):
                local_ready = True
        except Exception:
            local_ready = False

        if self.require_local and not local_ready:
            # If we reached here and require_local is True, local didn't initialize earlier.
            # Attempt one additional best-effort repair/retry before failing hard.
            try:
                # try chat-level helper first
                if hasattr(self.chat, "_attempt_local_repair_and_retry"):
                    ok = self.chat._attempt_local_repair_and_retry()
                else:
                    ok = self._attempt_local_repair_and_retry()
                if not ok:
                    raise RuntimeError(
                        "Local-only requested but local LLM not ready after repair attempt."
                    )
            except Exception as e:
                # fail loudly — no silent fallback when require_local is True
                raise RuntimeError(
                    f"Local-only requested but local LLM could not be initialized: {e}"
                )

        # Final log summary (keeps original logging behavior)
        mode = (
            "remote-only"
            if self.require_remote
            else (
                "local-only"
                if self.require_local
                else (
                    "local-preferred with fallback"
                    if self.allow_fallback and self.local_model
                    else "remote-preferred"
                )
            )
        )
        print(
            f"StoryPipeline initialized ({mode}). api_url={self.api_url!r}, local_model={self.local_model!r}, allow_fallback={self.allow_fallback}"
        )

    def _resp_to_text(self, resp) -> str:
        """
        Normalize various possible response shapes into a string safe for regex/parsing.
        Handles: str, bytes, dict (OpenAI-like, nested 'message' structures, 'choices'),
        lists of parts, and falls back to json.dumps or str().
        """
        try:
            if resp is None:
                return ""
            # bytes -> decode
            if isinstance(resp, (bytes, bytearray)):
                try:
                    return resp.decode("utf-8")
                except Exception:
                    return resp.decode("latin-1", errors="ignore")
            # already a string
            if isinstance(resp, str):
                return resp

            # dict-like: common keys first
            if isinstance(resp, dict):
                for k in ("response", "message", "text", "content", "result"):
                    if k in resp and isinstance(resp[k], str):
                        return resp[k]
                # nested message.content style
                if "message" in resp and isinstance(resp["message"], dict):
                    inner = resp["message"]
                    for k in ("content", "text"):
                        if k in inner and isinstance(inner[k], str):
                            return inner[k]
                # OpenAI-like: choices -> first -> text / message/content
                if (
                    "choices" in resp
                    and isinstance(resp["choices"], list)
                    and resp["choices"]
                ):
                    first = resp["choices"][0]
                    if isinstance(first, dict):
                        for k in ("text", "message", "content"):
                            if k in first and isinstance(first[k], str):
                                return first[k]
                        if "message" in first and isinstance(first["message"], dict):
                            m = first["message"]
                            if "content" in m:
                                c = m["content"]
                                if isinstance(c, str):
                                    return c
                                if isinstance(c, list):
                                    parts = []
                                    for item in c:
                                        if isinstance(item, dict):
                                            for kk in ("text", "content"):
                                                if kk in item and isinstance(
                                                    item[kk], str
                                                ):
                                                    parts.append(item[kk])
                                        elif isinstance(item, str):
                                            parts.append(item)
                                    if parts:
                                        return " ".join(parts)

            # list-like: join string parts
            if isinstance(resp, (list, tuple)):
                parts = []
                for item in resp:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        # try same heuristics recursively for small nested dict
                        s = self._resp_to_text(item)
                        if s:
                            parts.append(s)
                    elif isinstance(item, (bytes, bytearray)):
                        try:
                            parts.append(item.decode("utf-8"))
                        except Exception:
                            parts.append(str(item))
                if parts:
                    return " ".join(parts)

            # fallback: json.dumps if possible, else str()
            try:
                return json.dumps(resp, ensure_ascii=False)
            except Exception:
                return str(resp)
        except Exception:
            try:
                return str(resp)
            except Exception:
                return ""

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

    def _attempt_local_repair_and_retry(self) -> bool:
        """
        Best-effort: attempt one auto-repair + re-init of the local model.
        Reuses LocalLLM constructor (which may perform pip installs / fixes).
        Returns True if re-init succeeded (self.chat.local_llm is ready); False otherwise.
        This helper runs at most once when invoked by the constructor.
        """
        try:
            print(
                "ℹ️ Local init failed earlier — attempting one automatic repair + re-initialize...",
                flush=True,
            )
            # If ChatAPI exposes a way to init local directly, prefer that.
            if hasattr(self.chat, "init_local_llm"):
                try:
                    ok = self.chat.init_local_llm(
                        local_model=self.local_model, device=self.local_device
                    )
                    if ok:
                        return True
                except Exception as e:
                    print("⚠️ chat.init_local_llm raised:", e, flush=True)

            # Try to construct a fresh LocalLLM and replace chat.local_llm if ready
            tmp = None
            try:
                tmp = LocalLLM(local_model=self.local_model, device=self.local_device)
            except Exception as e:
                # Some LocalLLM constructors may do auto-repair and then raise; log and continue
                print(
                    "⚠️ LocalLLM constructor raised during repair attempt:",
                    e,
                    flush=True,
                )

            if tmp:
                # check readiness
                try:
                    if hasattr(tmp, "is_ready") and tmp.is_ready():
                        self.chat.local_llm = tmp
                        print(
                            "✅ LocalLLM initialized successfully on retry.", flush=True
                        )
                        return True
                    elif getattr(tmp, "_ready", False):
                        self.chat.local_llm = tmp
                        print("✅ LocalLLM appears ready after retry.", flush=True)
                        return True
                except Exception as e:
                    print("⚠️ error checking tmp.is_ready():", e, flush=True)

            # No successful init
            print(
                "⚠️ Local repair + retry did not produce a ready local LLM.", flush=True
            )
            return False

        except Exception as e:
            print("⚠️ Unexpected error during local repair + retry:", e, flush=True)
            return False

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
        (Preserves all original features; added robust response normalization to handle dict responses.)
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
            # deterministic conservative trimming preserving anchors (unchanged logic)
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
                    # immediate retry (infinite loop until success)
                    time.sleep(0.2)
                    continue

                resp_text = self._resp_to_text(resp)

                # In strict mode we insist the model returns a bracketed block; otherwise we accept best candidate
                if strict:
                    if not single_block_re.match(resp_text):
                        print(
                            "⚠️ Strict mode: response did not contain a single bracketed block — retrying."
                        )
                        time.sleep(0.2)
                        continue
                candidate = _extract_candidate(resp_text)
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

                        cond_resp_text = self._resp_to_text(cond_resp)
                        if strict and not single_block_re.match(cond_resp_text):
                            print(
                                "⚠️ Strict mode: condense response not bracketed — retrying condense."
                            )
                            time.sleep(0.2)
                            continue
                        cond_inner = _extract_candidate(cond_resp_text)
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
                # (rest of the guidelines are unchanged)
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

            cont_resp_text = self._resp_to_text(cont_resp)
            # In strict mode we accept only continuation text (model may return unbracketed continuation)
            cont_candidate = _extract_candidate(cont_resp_text)
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

                gen_resp_text = self._resp_to_text(gen_resp)
                if strict and not single_block_re.match(gen_resp_text):
                    print("⚠️ Strict mode: append generation not bracketed — retrying.")
                    time.sleep(0.2)
                    continue
                gen_candidate = _extract_candidate(gen_resp_text)
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
    ) -> List[str]:
        """
        Strict per-paragraph batching; paragraph 1 must reach its full batch_size quota
        (or min(batch_size, remaining)) before moving to next paragraph. Robust parsing,
        dedupe, retries, delays, saving via save_response(folder, file, content) into
        folder "image_response" with final file "image_prompts.txt". Cleans intermediate
        files so only final file remains. Filters out non-image/meta lines so final file
        contains only true image prompts.

        Preserves all existing features.
        """
        import re
        import random
        import time
        import math
        import os
        from typing import List, Optional

        if img_number <= 0:
            return []

        timeout_per_call = timeout_per_call or min(
            120, getattr(self.chat, "base_timeout", 120)
        )
        prompts: List[str] = []
        seen_prompts: set = set()
        response_folder = "image_response"
        final_fname = "image_prompts.txt"
        final_path = os.path.join(response_folder, final_fname)

        # compute number of sequential paragraph batches
        num_paragraphs = max(1, math.ceil(img_number / max(1, batch_size)))

        # fallback paragraph splitter (keeps sentences intact where possible)
        def _split_text_into_n_parts_fallback(text: str, n: int):
            text = (text or "").strip()
            if not text:
                return [""] * n
            sents = re.split(r"(?<=[.!?])\s+", text)
            if len(sents) == 1:
                words = text.split()
                avg = max(1, math.ceil(len(words) / n))
                parts = []
                for i in range(0, len(words), avg):
                    parts.append(" ".join(words[i : i + avg]))
                while len(parts) < n:
                    parts.append("")
                return parts[:n]
            words = text.split()
            total_words = len(words)
            target = max(1, math.ceil(total_words / n))
            parts = []
            cur = []
            cw = 0
            for sent in sents:
                sw = len(sent.split())
                if cw + sw > target and cur:
                    parts.append(" ".join(cur).strip())
                    cur = [sent]
                    cw = sw
                else:
                    cur.append(sent)
                    cw += sw
            if cur:
                parts.append(" ".join(cur).strip())
            while len(parts) < n:
                parts.append("")
            return parts[:n]

        # build paragraphs (use instance method if available)
        try:
            paragraphs = self._split_text_into_n_parts(script_text, num_paragraphs)
            if not paragraphs or len(paragraphs) < num_paragraphs:
                paragraphs = _split_text_into_n_parts_fallback(
                    script_text, num_paragraphs
                )
        except Exception:
            paragraphs = _split_text_into_n_parts_fallback(script_text, num_paragraphs)

        # per-paragraph attempt cap: respect self.max_prompt_attempts if set; else infinite
        cap = getattr(self, "max_prompt_attempts", None)
        if isinstance(cap, int) and cap > 0:
            per_paragraph_max = cap
        else:
            per_paragraph_max = None  # infinite retry

        # robust extractor that tries bracketed blocks first, then heuristics
        def _extract_blocks(resp_text: str):
            if not resp_text:
                return []
            blocks = []
            try:
                # prefer an existing helper if it's available and works
                raw = (
                    extract_all_bracketed_blocks(resp_text)
                    if "extract_all_bracketed_blocks" in globals()
                    else None
                )
                if raw:
                    for b in raw:
                        if isinstance(b, str):
                            blocks.append(b.strip())
            except Exception:
                pass

            if not blocks:
                # bracket pattern (allowing newlines)
                for m in re.findall(r"\[([^\]]{3,})\]", resp_text, flags=re.DOTALL):
                    blocks.append(m.strip())

            if not blocks:
                # fallback: long-enough lines / numbered list items
                lines = [ln.strip() for ln in resp_text.splitlines() if ln.strip()]
                candidates = []
                for ln in lines:
                    # skip obvious metadata lines
                    if len(ln) < 12:
                        continue
                    if any(
                        ln.lower().startswith(pref)
                        for pref in (
                            "system:",
                            "user:",
                            "assistant:",
                            "seed:",
                            "#",
                            "reply",
                            "example",
                        )
                    ):
                        continue
                    ln2 = re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip()
                    if len(ln2) >= 12:
                        candidates.append(ln2)
                # combine short consecutive lines if they look like one prompt
                i = 0
                while i < len(candidates):
                    cur = candidates[i]
                    j = i + 1
                    while j < len(candidates) and len(cur.split()) < 10:
                        cur = cur + " " + candidates[j]
                        j += 1
                    blocks.append(cur.strip())
                    i = j

            if not blocks:
                # last resort: chunk by paragraphs
                for seg in re.split(r"(?:\n{2,}|[\r\n]+)", resp_text):
                    seg = seg.strip()
                    if len(seg.split()) >= 6:
                        blocks.append(seg)
            # normalize whitespace
            cleaned = [
                re.sub(r"\s+", " ", b).strip() for b in blocks if isinstance(b, str)
            ]
            return cleaned

        # validator to filter out meta/non-image lines (best-effort heuristics)
        def is_valid_prompt(candidate: str) -> bool:
            if not candidate or len(candidate.strip()) == 0:
                return False
            c = candidate.strip()

            # remove surrounding brackets/quotes/backticks for inspection
            c_inspect = c.strip("[]()\"'" + "`").strip()

            # reject very short candidates (too few words)
            tokens = c_inspect.split()
            if len(tokens) < 10:
                # allow slightly shorter if contains strong visual indicators (camera/lens/aspect/illustration)
                visual_keywords = (
                    "camera",
                    "lens",
                    "f/",
                    "aperture",
                    "iso",
                    "shutter",
                    "mm",
                    "aspect",
                    "16:9",
                    "9:16",
                    "2:3",
                    "photoreal",
                    "cinematic",
                    "watercolor",
                    "watercolour",
                    "illustration",
                    "render",
                    "anime",
                    "oil paint",
                    "bokeh",
                    "lighting",
                    "foreground",
                    "background",
                    "hex",
                )
                if not any(k in c_inspect.lower() for k in visual_keywords):
                    return False

            low = c_inspect.lower()

            # reject lines that are clearly a clarification request, confirmation, examples, or error messages
            reject_phrases = [
                "understood",
                "before i generate",
                "do you want",
                "which one",
                "reply",
                "please confirm",
                "need clarification",
                "do you prefer",
                "would you like",
                "do you want every prompt",
                "unable to reach",
                "error",
                "cannot reach",
                "service may be down",
                "example difference",
                "which one do you want",
                "which one",
                "select a",
                "which format",
                "do you mean",
                "clarify",
                "confirm",
                "shall i",
                "would you",
                "do you",
                "please advise",
                "which style",
            ]
            for ph in reject_phrases:
                if ph in low:
                    return False

            # reject obvious markdown/meta noise
            if any(
                tok in c_inspect
                for tok in ("**", "```", "🔍", "⚠️", "✅", "➡️", "—", "•")
            ):
                return False

            # reject if candidate appears to be an instruction rather than a description (heuristic)
            # e.g., starts with verbs like "generate", "create", "please generate"
            if re.match(
                r"^(generate|create|please generate|please create|return|output)\b", low
            ):
                return False

            # reject if contains too many question marks or is mostly a question shorter than threshold
            if c_inspect.count("?") >= 1 and len(tokens) < 20:
                # short questions are likely clarifications
                return False

            # require at least one visual/content token OR be long enough
            visual_keywords = (
                "camera",
                "lens",
                "f/",
                "aperture",
                "iso",
                "shutter",
                "mm",
                "aspect",
                "16:9",
                "9:16",
                "2:3",
                "photoreal",
                "cinematic",
                "watercolor",
                "watercolour",
                "illustration",
                "render",
                "anime",
                "oil paint",
                "bokeh",
                "lighting",
                "foreground",
                "background",
                "hex",
                "portra",
                "kodak",
                "film",
                "portrait",
                "landscape",
                "studio",
                "macro",
                "wide",
            )
            if (
                not any(k in c_inspect.lower() for k in visual_keywords)
                and len(tokens) < 15
            ):
                return False

            # finally, ensure it contains at least one noun-like token (heuristic: presence of letters)
            if not re.search(r"[A-Za-z]", c_inspect):
                return False

            return True

        # helper to build request (prefer instance builder)
        def _build_request(script_paragraph: str, theme_val: str, req_count: int):
            try:
                return self._build_image_prompt_request(
                    script_paragraph, theme_val, req_count
                )
            except Exception:
                return f"Generate {req_count} image prompts for theme {theme_val} from:\n\n{script_paragraph}"

        # ensure response folder exists
        try:
            os.makedirs(response_folder, exist_ok=True)
        except Exception:
            pass

        print(
            f"\nStarting strict per-paragraph image prompt generation: target={img_number}, batch_size={batch_size}, paragraphs={num_paragraphs}"
        )

        # MAIN: for each paragraph sequentially fill its quota fully before moving on
        for para_idx in range(num_paragraphs):
            if len(prompts) >= img_number:
                break
            remaining_total = img_number - len(prompts)
            para_quota = min(
                batch_size, remaining_total
            )  # full quota for this paragraph
            collected_for_paragraph = 0
            paragraph = paragraphs[para_idx] or script_text or ""
            paragraph_attempts = 0

            print(
                f"\n➡️ Paragraph {para_idx+1}/{num_paragraphs}: need {para_quota} prompts for this paragraph"
            )

            # keep retrying this paragraph until its quota is filled or attempts cap reached
            while collected_for_paragraph < para_quota and (
                per_paragraph_max is None or paragraph_attempts < per_paragraph_max
            ):
                paragraph_attempts += 1
                seed = random.randint(1000, 9999)
                need_now = para_quota - collected_for_paragraph
                request_count = (
                    need_now  # request the remaining quota for this paragraph
                )

                enriched_script = (
                    f"{paragraph}\n\n"
                    f"# Paragraph {para_idx+1} | Attempt {paragraph_attempts} | Seed: {seed}\n"
                    f"Generate {request_count} completely unique, creative, and ultra-detailed image prompts.\n"
                    f"Each prompt must depict a different scene, composition, camera angle, lighting, and tone.\n"
                    f"Vary the artistic style and subject matter — avoid repeating concepts, objects, or phrasing.\n"
                    f"Each prompt should explicitly mention the theme: {theme}.\n"
                    f"Return each prompt in [brackets], one per line. No explanations or extra text.\n"
                )

                print(
                    f"  🎯 Requesting {request_count} prompts (paragraph {para_idx+1}, attempt {paragraph_attempts}, seed={seed})..."
                )
                prompt_request = _build_request(enriched_script, theme, request_count)

                try:
                    resp = self.chat.send_message(
                        prompt_request,
                        timeout=timeout_per_call,
                        spinner_message=f"Generating paragraph {para_idx+1} prompts (attempt {paragraph_attempts})...",
                    )
                except Exception as e:
                    print(
                        f"  ⚠️ API error on paragraph {para_idx+1} attempt {paragraph_attempts}: {e}"
                    )
                    err_fname = f"para_{para_idx+1}_err_{paragraph_attempts}.txt"
                    try:
                        save_response(response_folder, err_fname, str(e))
                        print(
                            f"  ✅ Saved error: {os.path.join(response_folder, err_fname)}"
                        )
                    except Exception:
                        try:
                            with open(
                                os.path.join(response_folder, err_fname),
                                "w",
                                encoding="utf-8",
                            ) as f:
                                f.write(str(e))
                            print(
                                f"  ✅ Saved error fallback: {os.path.join(response_folder, err_fname)}"
                            )
                        except Exception:
                            pass
                    time.sleep(1 + random.random())
                    continue

                # normalize response safely
                resp_text = self._resp_to_text(resp)

                # extract candidate blocks (extractor expects a string)
                blocks = _extract_blocks(resp_text)

                if not blocks:
                    print(
                        f"  ⚠️ No candidate blocks parsed for paragraph {para_idx+1} attempt {paragraph_attempts}. Saving raw and retrying."
                    )
                    raw_fname = f"para_{para_idx+1}_raw_{paragraph_attempts}.txt"
                    try:
                        save_response(
                            response_folder,
                            raw_fname,
                            resp if isinstance(resp, str) else str(resp),
                        )
                        print(
                            f"  ✅ Saved raw: {os.path.join(response_folder, raw_fname)}"
                        )
                    except Exception:
                        try:
                            with open(
                                os.path.join(response_folder, raw_fname),
                                "w",
                                encoding="utf-8",
                            ) as f:
                                f.write(resp if isinstance(resp, str) else str(resp))
                            print(
                                f"  ✅ Saved raw fallback: {os.path.join(response_folder, raw_fname)}"
                            )
                        except Exception:
                            pass
                    time.sleep(0.5 + random.random() * 0.5)
                    continue

                # from blocks accept only those validated as image prompts
                added = 0
                for blk in blocks:
                    if added >= request_count:
                        break
                    candidate = blk.strip()
                    # remove surrounding bracket/quote/backtick chars safely
                    candidate = candidate.strip("[]()\"'" + "`").strip()
                    candidate = re.sub(r"\s+", " ", candidate)
                    if not candidate or len(candidate) < 8:
                        continue

                    # run validator
                    if not is_valid_prompt(candidate):
                        # skip anything that looks like a clarification, question, error, or meta text
                        continue

                    # ensure theme appended
                    if theme.lower() not in candidate.lower():
                        candidate = f"{candidate} | Theme: {theme}"

                    # dedupe globally
                    if candidate in seen_prompts:
                        continue

                    # accept
                    seen_prompts.add(candidate)
                    prompts.append(candidate)
                    added += 1
                    collected_for_paragraph += 1

                print(
                    f"  ✅ Added {added} valid prompts this attempt (collected for paragraph: {collected_for_paragraph}/{para_quota}, total: {len(prompts)}/{img_number})."
                )

                # optionally save intermediate progress to final file (only valid prompts are written)
                if save_each_batch and added > 0:
                    try:
                        # read existing final content (if any)
                        existing_text = ""
                        if os.path.exists(final_path):
                            with open(final_path, "r", encoding="utf-8") as f:
                                existing_text = f.read()
                        # append newly added valid prompts only if not already present in file
                        new_lines = []
                        for p in prompts[-added:]:
                            line = f"[{p}]\n"
                            if line not in existing_text:
                                new_lines.append(line)
                        if new_lines:
                            combined = (existing_text + "".join(new_lines)).strip()
                            try:
                                save_response(
                                    response_folder,
                                    final_fname,
                                    combined
                                    + ("\n" if not combined.endswith("\n") else ""),
                                )
                                print(
                                    f"  💾 Appended {len(new_lines)} prompts to {final_path}"
                                )
                            except Exception:
                                with open(final_path, "w", encoding="utf-8") as f:
                                    f.write(
                                        combined
                                        + ("\n" if not combined.endswith("\n") else "")
                                    )
                                print(
                                    f"  💾 Appended fallback write {len(new_lines)} prompts to {final_path}"
                                )
                    except Exception:
                        pass

                # polite delay
                time.sleep(0.4 + random.random() * 0.6)

            # if failed to fill paragraph quota and per_paragraph_max was set, warn then continue
            if (
                collected_for_paragraph < para_quota
                and per_paragraph_max is not None
                and paragraph_attempts >= per_paragraph_max
            ):
                missing = para_quota - collected_for_paragraph
                print(
                    f"⚠️ Paragraph {para_idx+1} failed to fill its quota by {missing} prompts after {per_paragraph_max} attempts."
                )
                # move to next paragraph (global target might still be achievable)
                continue

        # FINALIZE: dedupe, trim to exact img_number, and write final canonical file with only validated prompts
        final_prompts = list(dict.fromkeys(prompts))
        if len(final_prompts) > img_number:
            final_prompts = final_prompts[:img_number]

        final_content = "".join(f"[{p}]\n" for p in final_prompts)
        try:
            save_response(response_folder, final_fname, final_content)
            print(
                f"\n✅ Final saved {len(final_prompts)}/{img_number} prompts to {final_path}"
            )
        except Exception:
            try:
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(final_content)
                print(
                    f"\n✅ Final saved fallback {len(final_prompts)}/{img_number} prompts to {final_path}"
                )
            except Exception as e:
                print(f"\n⚠️ Failed to save final prompts: {e}")

        # If incomplete and infinite mode, run fill cycles until target met (keeps same strict validation)
        if len(final_prompts) < img_number and per_paragraph_max is None:
            print(
                f"🔁 Final currently {len(final_prompts)}/{img_number}. Entering fill cycles (infinite mode) until target met."
            )
            para_cycle = 0
            while len(final_prompts) < img_number:
                para_idx = para_cycle % num_paragraphs
                need_now = min(batch_size, img_number - len(final_prompts))
                paragraph = paragraphs[para_idx] or script_text or ""
                seed = random.randint(1000, 9999)
                enriched_script = (
                    f"{paragraph}\n\n"
                    f"# Extra fill cycle | Paragraph {para_idx+1} | Seed: {seed}\n"
                    f"Generate {need_now} unique ultra-detailed prompts. Each prompt in brackets on its own line.\n"
                )
                prompt_request = _build_request(enriched_script, theme, need_now)
                try:
                    resp = self.chat.send_message(
                        prompt_request,
                        timeout=timeout_per_call,
                        spinner_message="Filling missing prompts...",
                    )
                except Exception:
                    time.sleep(1 + random.random())
                    para_cycle += 1
                    continue
                resp_text = self._resp_to_text(resp)
                blocks = _extract_blocks(resp_text)
                added = 0
                for blk in blocks:
                    if added >= need_now:
                        break
                    candidate = blk.strip()
                    candidate = candidate.strip("[]()\"'" + "`").strip()
                    candidate = re.sub(r"\s+", " ", candidate)
                    if not candidate:
                        continue
                    if not is_valid_prompt(candidate):
                        continue
                    if theme.lower() not in candidate.lower():
                        candidate = f"{candidate} | Theme: {theme}"
                    if candidate in seen_prompts:
                        continue
                    seen_prompts.add(candidate)
                    final_prompts.append(candidate)
                    added += 1
                # write updated final file
                try:
                    save_response(
                        response_folder,
                        final_fname,
                        "".join(f"[{p}]\n" for p in final_prompts[:img_number]),
                    )
                except Exception:
                    try:
                        with open(final_path, "w", encoding="utf-8") as f:
                            f.write(
                                "".join(f"[{p}]\n" for p in final_prompts[:img_number])
                            )
                    except Exception:
                        pass
                para_cycle += 1
                time.sleep(0.4 + random.random() * 0.6)

            print(
                f"🎯 Fill cycles complete: {len(final_prompts[:img_number])}/{img_number}"
            )

        # Trim and final check
        if len(final_prompts) > img_number:
            final_prompts = final_prompts[:img_number]

        if len(final_prompts) < img_number:
            print(
                f"⚠️ Finished but only collected {len(final_prompts)}/{img_number} prompts (per_paragraph_max may have limited retries)."
            )

        # CLEANUP: remove intermediate files inside image_response except the final image_prompts.txt
        try:
            for fname in os.listdir(response_folder):
                fp = os.path.join(response_folder, fname)
                try:
                    if os.path.abspath(fp) == os.path.abspath(final_path):
                        continue
                    if os.path.isfile(fp):
                        os.remove(fp)
                        print(f"🧹 Removed intermediate file: {fp}")
                except Exception:
                    pass
        except Exception:
            pass

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
        resp_text = self._resp_to_text(resp)
        narr_block = extract_largest_bracketed(resp_text)
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
            "   - Adds social links and CTAs to subscribe or follow.\n"
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
            "Now generate the complete optimized metadata package."
        )

        resp = self.chat.send_message(
            prompt,
            timeout=timeout or self.chat.base_timeout,
            spinner_message="Generating YouTube metadata...",
        )
        resp_text = self._resp_to_text(resp)
        save_response("youtube_response", "youtube_metadata.txt", resp_text)
        return resp_text


# ---------------------- EXAMPLE USAGE ----------------------
if __name__ == "__main__":
    start = time.time()

    pipeline = StoryPipeline(
        local_model="mistral-small-3.1",
        local_device="cpu",
        require_local=True,
        allow_fallback=False,
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
