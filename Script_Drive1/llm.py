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
    """
    Adapter to attempt use of a local HF-style model via transformers.
    If transformers or dependencies are missing, tries to pip install them.
    If a numpy ABI error (e.g. 'numpy.dtype size changed') occurs, attempts
    a best-effort fix by reinstalling numpy (unless disabled by env var).
    """

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
        try:
            print(f"📦 pip installing {package} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except Exception as e:
            print(f"⚠️ pip install {package} failed: {e}")

    def _try_imports(self):
        """
        Try to import needed libraries; return tuple(transformers_available, numpy_ok, error_message)
        """
        try:
            import transformers  # type: ignore

            transformers_ok = True
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
        return (
            transformers_ok,
            numpy_ok,
            locals().get("transformers_err", None) or numpy_err,
        )

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
            "numpy.dtype size changed" in err_msg
            or "binary incompatibility" in err_msg.lower()
        ):
            print(
                "⚠️ Detected numpy binary-compatibility error. Attempting to reinstall numpy (best-effort)."
            )
            # Try a safe reinstall: upgrade and force-reinstall numpy to get matching wheel
            try:
                # Try to pick a commonly compatible version. If your environment requires
                # another version, set it externally and retry manually.
                target = "numpy"
                # First attempt simple upgrade
                self._pip_install(f"{target} --upgrade --force-reinstall")
                # If still fails later, user must manually fix in environment.
                return True
            except Exception as e:
                print("⚠️ numpy reinstall attempt failed:", e)
                return True
        return False

    def _ensure_transformers_and_deps(self):
        """
        Ensure transformers and a backend are installed (best-effort). May pip install.
        """
        transformers_ok, numpy_ok, err = self._try_imports()
        if not transformers_ok:
            # try to install transformers + minimal extras
            self._pip_install("transformers[torch]")
            self._pip_install("accelerate")
            self._pip_install("safetensors")
            # re-check
            transformers_ok, numpy_ok, err = self._try_imports()

        if not numpy_ok and err:
            # attempt repair for numpy ABI mismatch
            self._repair_numpy_if_needed(err)

    def _init_pipeline(self):
        """
        Initialize the HF pipeline. Wrap in try/except to surface meaningful errors.
        """
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # type: ignore

            kwargs = {"trust_remote_code": True}
            # device handling (let HF choose if not provided)
            if self.device:
                if self.device.lower() == "cpu":
                    kwargs["device_map"] = "cpu"
                elif self.device.lower() == "cuda":
                    kwargs["device_map"] = "auto"

            print(
                f"🔁 Loading local model '{self.local_model}' (this may download weights)..."
            )
            try:
                self.generator = pipeline(
                    "text-generation", model=self.local_model, **kwargs
                )
            except Exception as e:
                # fallback: try constructing tokenizer+model explicitly
                print("ℹ️ pipeline() fallback:", e)
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
                print("✅ Local generator ready.")
            else:
                print("⚠️ Local generator creation returned None.")
        except Exception as e:
            # expose the exact error (including numpy ABI problems)
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
            print(f"⚠️ {err_text}")
            attempted = self._repair_numpy_if_needed(err_text)
            if attempted:
                # reload python modules in a best-effort way
                try:
                    print("🔁 Re-importing environment after repair...")
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
                    print(f"⚠️ Retry after repair also failed: {e2}")
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
    ):
        import requests  # expected to be available in original codebase

        self.requests = requests
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        self.base_timeout = default_timeout
        self.local_model = local_model
        self.local_device = local_device
        self.local_llm: Optional[LocalLLM] = None

        if self.local_model:
            try:
                print(
                    f"🔎 Attempting to initialize local model '{self.local_model}' on device='{self.local_device}'..."
                )
                self.local_llm = LocalLLM(
                    local_model=self.local_model, device=self.local_device
                )
                if not self.local_llm.is_ready():
                    print(
                        "⚠️ Local model was specified but failed to initialize. Will fall back to remote API."
                    )
                    self.local_llm = None
            except Exception as e:
                print("⚠️ Error initializing LocalLLM:", e)
                self.local_llm = None

    def send_message(
        self,
        message: str,
        timeout: Optional[int] = None,
        retry_forever: bool = True,
        retry_on_client_errors: bool = False,
        initial_backoff: float = 1.0,
        max_backoff: float = 8.0,
        spinner_message: str = "Waiting for response...",
    ) -> str:
        """
        If a ready local LLM exists, call it synchronously and return.
        Otherwise, perform robust HTTP POST with retry/backoff (preserves infinite retry semantics).
        """
        # LOCAL PATH
        if self.local_llm is not None and self.local_llm.is_ready():
            try:
                print("➡️ Using local LLM for generation.")
                return self.local_llm.generate(
                    message, timeout=timeout or self.base_timeout
                )
            except Exception as e:
                print(
                    "⚠️ Local LLM failed during generate — falling back to remote API. Error:",
                    e,
                )
                # continue to remote fallback

        # REMOTE PATH (preserve original behavior)
        spinner = LoadingSpinner(spinner_message)
        attempt = 0
        timeout = timeout or self.base_timeout
        backoff = initial_backoff
        max_backoff = max_backoff
        start_time = time.monotonic()

        while True:
            attempt += 1
            try:
                print(f"\nAttempt #{attempt} — timeout={timeout}s — sending request.")
                spinner.start()
                resp = self.requests.post(
                    self.url,
                    headers=self.headers,
                    json={"message": message},
                    timeout=timeout,
                )
                spinner.stop()

                status = resp.status_code

                # server/rate errors -> retry
                if status == 429 or 500 <= status < 600:
                    print(
                        f"⚠️ Server/Rate error HTTP {status}. Backing off {backoff:.1f}s and retrying."
                    )
                    time.sleep(backoff + random.uniform(0, 1.0))
                    backoff = min(backoff * 2, max_backoff)
                    timeout = min(timeout + 100, 100000)
                    continue

                # client errors
                if 400 <= status < 500:
                    try:
                        payload = resp.json()
                        error_msg = payload.get("error") or json.dumps(payload)
                    except Exception:
                        error_msg = resp.text or f"HTTP {status}"
                    msg = f"HTTP {status} - {error_msg}"
                    if retry_on_client_errors:
                        print(
                            f"⚠️ Client error {msg} — retrying (retry_on_client_errors=True). Backing off {backoff:.1f}s."
                        )
                        time.sleep(backoff + random.uniform(0, 1.0))
                        backoff = min(backoff * 2, max_backoff)
                        timeout = min(timeout + 100, 100000)
                        continue
                    else:
                        raise RuntimeError(f"Fatal client error (not retried): {msg}")

                resp.raise_for_status()
                try:
                    result = resp.json()
                except json.JSONDecodeError as e:
                    print(
                        f"⚠️ Invalid JSON received (will retry): {e}. Backing off {backoff:.1f}s..."
                    )
                    time.sleep(backoff + random.uniform(0, 1.0))
                    backoff = min(backoff * 2, max_backoff)
                    timeout = min(timeout + 100, 100000)
                    continue

                # Expected original schema: {'status': 'success', 'response': '...'}
                if isinstance(result, dict) and result.get("status") == "success":
                    content = result.get("response", "") or ""
                    elapsed = int(time.monotonic() - start_time)
                    print(f"✅ Success on attempt #{attempt} (elapsed {elapsed}s).")
                    return content
                else:
                    err = (
                        result.get("error")
                        if isinstance(result, dict)
                        else f"Unexpected body: {result}"
                    )
                    print(
                        f"⚠️ API returned unexpected payload: {err}. Backing off {backoff:.1f}s and retrying..."
                    )
                    time.sleep(backoff + random.uniform(0, 1.0))
                    backoff = min(backoff * 2, max_backoff)
                    timeout = min(timeout + 100, 100000)
                    continue

            except self.requests.exceptions.Timeout as e:
                spinner.stop()
                print(
                    f"\n⚠️ Timeout on attempt #{attempt}: {e}. Backing off {backoff:.1f}s and retrying..."
                )
                time.sleep(backoff + random.uniform(0, 1.0))
                backoff = min(backoff * 2, max_backoff)
                timeout = min(timeout + 100, 100000)
                continue

            except (
                self.requests.exceptions.ConnectionError,
                self.requests.exceptions.RequestException,
            ) as e:
                spinner.stop()
                print(
                    f"\n⚠️ Network error on attempt #{attempt}: {e}. Backing off {backoff:.1f}s and retrying..."
                )
                time.sleep(backoff + random.uniform(0, 1.0))
                backoff = min(backoff * 2, max_backoff)
                timeout = min(timeout + 100, 100000)
                continue

            except RuntimeError:
                spinner.stop()
                raise

            except Exception as e:
                spinner.stop()
                print(
                    f"\n⚠️ Unexpected error on attempt #{attempt}: {e}. Backing off {backoff:.1f}s and retrying..."
                )
                time.sleep(backoff + random.uniform(0, 1.0))
                backoff = min(backoff * 2, max_backoff)
                timeout = min(timeout + 100, 100000)
                continue

            finally:
                try:
                    spinner.stop()
                except Exception:
                    pass


# ---------------------- End of replacement block ----------------------


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
        local_model: str | None = None,  # e.g. "mistral-small-3.1" or local path
        local_device: str | None = None,  # "cpu" or "cuda" or None
    ):
        # If local_model is set, ChatAPI will attempt to use LocalLLM
        self.chat = ChatAPI(
            url=api_url,
            default_timeout=default_timeout,
            local_model=local_model,
            local_device=local_device,
        )

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
        COMPLETE generate_script implementation — infinite-retry until success (no internal caps).

        Behavior summary:
        - Keeps all original features from your initial design: bracketed-only outputs in strict mode,
          strengthen/retry prompts, continuation seeded with last paragraph, model condense attempts,
          deterministic trimming fallback, fuzzy dedupe, cleaning hooks, and saving.
        - **No wall-clock or API-call caps**: the method will keep retrying until a satisfactory
          non-empty cleaned script that meets the target (within tolerance) is produced. This is
          intentional per your request. The only "cap" is success.
        - The saved artifact is EXACTLY one bracketed block and nothing else: `[<cleaned script>]`.
        - The function will not save empty content. If a generated candidate cleans to empty, it will
          keep retrying.

        WARNING: Running without caps can consume unbounded resources. Use in a controlled environment.
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
                "The bracketed block should contain ONLY the full script text (no labels, no JSON, no commentary, no metadata, no tags). "
                "There must be no additional bracketed blocks, and no leading or trailing whitespace or blank lines outside the brackets. "
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
            # deterministic conservative trimming preserving anchors
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

                # In strict mode we insist the model returns a bracketed block; otherwise we accept best candidate
                if strict:
                    if not single_block_re.match(resp):
                        print(
                            "⚠️ Strict mode: response did not contain a single bracketed block — retrying."
                        )
                        time.sleep(0.2)
                        continue
                candidate = _extract_candidate(resp)
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
                        if strict and not single_block_re.match(cond_resp):
                            print(
                                "⚠️ Strict mode: condense response not bracketed — retrying condense."
                            )
                            time.sleep(0.2)
                            continue
                        cond_inner = _extract_candidate(cond_resp)
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
                "- Maintain characters, names, facts, tone, and tense exactly as present in PREV and LAST_PARAGRAPH. If a scene or character is implied, continue that thread unless explicitly contradicted.\n"
                "- Show, don't tell: prefer sensory details, short scenes, beats, and concrete actions over long explanation. Use short+long sentences rhythmically to control pacing.\n"
                "- Callbacks, not copy: reference earlier lines or imagery with subtle callback phrases (echo words, repeated motifs, similar imagery) rather than copying whole sentences.\n"
                "- Flow & transitions: use graceful transitions between beats or scenes (one-sentence visual transitions, cut-to, or a brief descriptive line) without headings, timestamps, or labels.\n"
                "- Scene economy: each paragraph should function as a micro-scene or beat — introduce a small change, reveal, decision, or escalation that advances momentum.\n"
                "- Dialogue & tags: if characters speak, keep speaker attribution consistent with prior format. Use realistic, concise dialogue that reveals character or motive.\n"
                "- Continuity safety: never contradict established facts (names, timeline, locations, relationships). If uncertain, favor neutral phrasing that preserves continuity.\n"
                "- Length control: aim for ~{words_target} total words. If you overshoot slightly that's fine; if you undershoot, continue until the target is sensibly reached. When within ~3-7% of the target, create a satisfying mini-cliff or logical segue for the next batch.\n"
                "- Formatting: plain paragraphs only (no lists, no headers, no code). Keep natural paragraph breaks for beats. No editorial comments, no analysis, no instructions to the reader.\n"
                "- Safety & quality: keep language appropriate for a wide audience; avoid gratuitous profanity unless already present and integral to character voice.\n\n"
                "Produce exactly one bracketed script block and nothing else: output a single opening bracket [ then the entire script content followed by a single closing bracket ] — include no other characters, whitespace, newlines, headings, labels, metadata, counts, commentary, instructions, fragments of the prompt, code fences, or BOM before or after; the bracketed text must be the complete script with no explanatory notes, stage directions, or parenthetical remarks not part of the script; if the script cannot be produced, return exactly []; the response must contain absolutely nothing else.\n"
                "Deliver a single continuous piece that a human editor could paste after PREV_BEGIN and have the full script read as one complete, cinematic video script.\n"
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

            # In strict mode we accept only continuation text (model may return unbracketed continuation)
            cont_candidate = _extract_candidate(cont_resp)
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
                if strict and not single_block_re.match(gen_resp):
                    print("⚠️ Strict mode: append generation not bracketed — retrying.")
                    time.sleep(0.2)
                    continue
                gen_candidate = _extract_candidate(gen_resp)
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

                # extract candidate blocks
                blocks = _extract_blocks(resp)

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
                blocks = _extract_blocks(resp)
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
        save_response("youtube_response", "youtube_metadata.txt", resp)
        return resp


# ---------------------- EXAMPLE USAGE ----------------------
if __name__ == "__main__":
    start = time.time()

    pipeline = StoryPipeline(local_model="mistral-small-3.1", local_device="cpu")

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
