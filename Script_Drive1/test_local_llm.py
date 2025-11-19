#!/usr/bin/env python3
"""
test_local_llm.py — ultimate compatibility + candidate iteration + streaming assembly

Features:
- HF download compatibility (token= / use_auth_token=)
- Smart GGUF selection with candidate iteration (skips incompatible architectures like 'clip')
- llama-cpp-python compatibility across API variants (create/generate/callable)
- Handles generator/streaming responses from Llama.generate(...) and assembles them
- Fallback to llama.cpp `main` binary
- Prints explicit RESPONSE_VALUE and MODEL_PATH at the end
"""

from __future__ import annotations
import os
import sys
import shutil
import time
import subprocess
import inspect
import types
from typing import Any, List, Optional
from collections.abc import Iterable

# ---------- Configuration (via env overrides) ----------
HF_TOKEN: str = os.getenv("HF_TOKEN", "")  # set by GH workflow or env
REPO_ID: str = os.getenv("REPO_ID", "ggml-org/Mistral-Small-3.1-24B-Instruct-2503-GGUF")
MODEL_DEST_PATH: str = os.getenv(
    "MODEL_DEST_PATH", os.path.join(os.getcwd(), "models", "mistral-small-3.1.gguf")
)
USE_AUTH: bool = os.getenv("USE_AUTH", "true").lower() in ("1", "true", "yes")
SELECT_STRATEGY: str = os.getenv("SELECT_STRATEGY", "auto")
TEST_PROMPT: str = os.getenv(
    "TEST_PROMPT",
    "Q: What is 2 + 2?\nA:(strict answer, no jokes) and write and script with 2500 words for children story",
)
TEST_MAX_TOKENS: int = int(os.getenv("TEST_MAX_TOKENS", "64"))
TEST_TEMPERATURE: float = float(os.getenv("TEST_TEMPERATURE", "0.0"))
VERBOSE: bool = os.getenv("VERBOSE", "true").lower() in ("1", "true", "yes")
SAVE_RESPONSE_FILE: str = os.getenv(
    "SAVE_RESPONSE_FILE", ""
)  # optional path to save response
# -------------------------------------------------------


def _log(*a, **k):
    if VERBOSE:
        print(*a, **k)


# Try imports early
try:
    from huggingface_hub import list_repo_files, hf_hub_download, HfApi
except Exception as e:
    _log(
        "Please ensure huggingface_hub is installed (pip install huggingface_hub). Error:",
        e,
    )
    raise


# ---------- Helpers ----------
def ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def hf_hub_download_compat(
    repo_id: str, filename: str, token: Optional[str] = None, **kwargs
) -> str:
    """
    Call hf_hub_download and handle both token= and old use_auth_token= names.
    """
    try:
        return hf_hub_download(
            repo_id=repo_id, filename=filename, token=token, **kwargs
        )
    except TypeError:
        return hf_hub_download(
            repo_id=repo_id, filename=filename, use_auth_token=token, **kwargs
        )


def pick_gguf_from_list_simple(files: List[str], strategy: str = "auto") -> List[str]:
    """Return ordered list of .gguf candidate filenames (simple heuristics + keep order)."""
    ggufs = [f for f in files if f.lower().endswith(".gguf") or ".gguf" in f.lower()]
    if not ggufs:
        return []
    # try to prioritize q* quantized variants for low-ram systems (simple)
    if strategy and strategy.lower() in ("smallest", "auto"):
        sorted_ggufs = sorted(
            ggufs,
            key=lambda s: (
                0 if ("q4" in s.lower() or "q8" in s.lower()) else 1,
                len(s),
            ),
        )
        return sorted_ggufs
    if strategy.lower() == "largest":
        return list(reversed(ggufs))
    return ggufs


# ---------- Parsing generator/streaming chunks ----------
def extract_text_from_chunk(chunk: Any) -> str:
    try:
        if chunk is None:
            return ""
        if isinstance(chunk, dict):
            # common shapes: {'choices':[{'text': '...'}]} or {'delta': {'content': '...'}}
            if "choices" in chunk and chunk["choices"]:
                first = chunk["choices"][0]
                if isinstance(first, dict):
                    for k in ("text", "content"):
                        if k in first:
                            return str(first.get(k) or "")
                    if "delta" in first and isinstance(first["delta"], dict):
                        for k in ("content", "text"):
                            if k in first["delta"]:
                                return str(first["delta"].get(k) or "")
                return ""
            # top-level fields
            for k in ("text", "output", "content"):
                if k in chunk:
                    return str(chunk[k] or "")
            if "delta" in chunk and isinstance(chunk["delta"], dict):
                return str(chunk["delta"].get("content", "") or "")
            return ""
        if hasattr(chunk, "text"):
            try:
                return str(chunk.text or "")
            except Exception:
                pass
        if isinstance(chunk, (str, bytes)):
            return chunk.decode() if isinstance(chunk, bytes) else chunk
        return ""
    except Exception:
        return ""


def assemble_from_generator(gen: Iterable) -> str:
    parts: List[str] = []
    try:
        for chunk in gen:
            txt = extract_text_from_chunk(chunk)
            if not txt:
                # fallback: small str(chunk)
                try:
                    s = str(chunk)
                    if len(s) < 1000:
                        txt = s
                except Exception:
                    txt = ""
            if txt:
                parts.append(txt)
        return "".join(parts)
    except Exception as e:
        _log("Error while iterating generator:", e)
        try:
            return str(gen)
        except Exception:
            return ""


def parse_llama_cpp_response(resp: Any) -> str:
    try:
        if resp is None:
            return ""
        # generator
        if isinstance(resp, types.GeneratorType) or (
            isinstance(resp, Iterable)
            and not isinstance(resp, (str, bytes, dict))
            and not hasattr(resp, "choices")
        ):
            return assemble_from_generator(resp)
        if isinstance(resp, list):
            out = []
            for it in resp:
                out.append(extract_text_from_chunk(it) or str(it))
            return "".join(out)
        if isinstance(resp, dict):
            if "choices" in resp and resp["choices"]:
                c = resp["choices"][0]
                if isinstance(c, dict):
                    for k in ("text", "message", "output"):
                        if k in c:
                            if k == "message" and isinstance(c[k], dict):
                                return str(c[k].get("content", "") or "")
                            return str(c.get(k) or "")
            for k in ("text", "output", "content"):
                if k in resp:
                    return str(resp[k] or "")
            return ""
        if hasattr(resp, "choices"):
            try:
                c = resp.choices[0]
                if hasattr(c, "text"):
                    return str(c.text or "")
                if hasattr(c, "message"):
                    return str(getattr(c, "message").get("content", "") or "")
            except Exception:
                pass
        if isinstance(resp, (str, bytes)):
            return resp.decode() if isinstance(resp, bytes) else resp
        return str(resp)
    except Exception as e:
        _log("parse_llama_cpp_response error:", e)
        try:
            return str(resp)
        except Exception:
            return ""


# ---------- llama-cpp-python compatibility ----------
def llama_call_compat(
    llm: Any,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stop: Optional[List[str]] = None,
    echo: Optional[bool] = None,
):
    def map_kwargs(fn):
        sig = inspect.signature(fn)
        params = sig.parameters
        kwargs = {}
        if max_tokens is not None:
            for alt in (
                "max_tokens",
                "max_new_tokens",
                "n_predict",
                "n",
                "max_completion_tokens",
            ):
                if alt in params:
                    kwargs[alt] = max_tokens
                    break
        if temperature is not None and "temperature" in params:
            kwargs["temperature"] = temperature
        if stop is not None:
            for alt in ("stop", "stop_sequences", "stop_tokens"):
                if alt in params:
                    kwargs[alt] = stop
                    break
        if echo is not None and "echo" in params:
            kwargs["echo"] = echo
        return kwargs

    for name in ("create", "generate"):
        if hasattr(llm, name):
            fn = getattr(llm, name)
            kwargs = map_kwargs(fn)
            try:
                _log(f"Attempting llama-cpp-python method '{name}'.")
                try:
                    return fn(prompt=prompt, **kwargs)
                except TypeError:
                    return fn(prompt, **kwargs)
            except Exception as e:
                _log(f"Method '{name}' failed: {e}")

    # try callable
    try:
        _log("Attempting callable Llama(...) style.")
        try:
            return llm(prompt, max_tokens=max_tokens, temperature=temperature)
        except TypeError:
            try:
                return llm(prompt)
            except Exception as e:
                _log("callable form failed:", e)
                return None
    except Exception as e:
        _log("llama_call_compat final error:", e)
        return None


# ---------- test with llama-cpp-python (with candidate iteration) ----------
def test_with_llama_cpp_candidates(
    model_paths: List[str], prompt: str, max_tokens: int = 64, temp: float = 0.0
):
    """
    Try each provided model file path in order until one loads & returns text.
    Returns tuple (response_text_or_empty, successful_model_path_or_empty, last_error_message_or_empty)
    """
    last_error = ""
    for mp in model_paths:
        _log("Trying model file:", mp)
        try:
            from llama_cpp import Llama
        except Exception as e:
            _log("llama_cpp import failed:", e)
            return None, "", str(e)

        try:
            _log("Loading model with llama_cpp:", mp)
            llm = Llama(model_path=mp)
            _log("Model loaded successfully:", mp)
            resp = llama_call_compat(
                llm,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                stop=None,
                echo=False,
            )
            parsed = parse_llama_cpp_response(resp)
            return parsed, mp, ""
        except Exception as e:
            last_error = str(e)
            _log("Error in test_with_llama_cpp for", mp, ":", last_error)
            # If error indicates unknown architecture or incompatible model, continue to next candidate
            if (
                ("unknown model architecture" in last_error.lower())
                or ("error loading model architecture" in last_error.lower())
                or ("failed to load model" in last_error.lower())
                or ("model architecture" in last_error.lower())
            ):
                _log(
                    "Detected architecture/load issue; trying next candidate if available."
                )
                # ensure we cleanup any partially created Llama objects (best-effort)
                try:
                    # Some llama-cpp-python versions may require explicit close on object; do best-effort
                    if "llm" in locals() and hasattr(llm, "close"):
                        try:
                            llm.close()
                        except Exception:
                            pass
                        del llm
                except Exception:
                    pass
                continue
            else:
                # for other errors, also try next candidate
                continue
    return None, "", last_error


# ---------- fallback llama.cpp binary ----------
def find_llama_cpp_binary() -> Optional[str]:
    for name in ("main", "main.exe"):
        for p in os.environ.get("PATH", "").split(os.pathsep):
            candidate = os.path.join(p, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    candidate = os.path.join(os.getcwd(), "main")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def test_with_llama_cpp_binary(model_path: str, prompt: str) -> Optional[str]:
    exe = find_llama_cpp_binary()
    if not exe:
        _log("No llama.cpp main binary found.")
        return None
    cmd = [exe, "-m", model_path, "-p", prompt, "-n", str(TEST_MAX_TOKENS)]
    _log("Running llama.cpp binary:", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300
        )
        if proc.returncode != 0:
            _log("llama.cpp binary failed. stderr:", proc.stderr[:1000])
            return None
        return proc.stdout
    except Exception as e:
        _log("Error running llama.cpp binary:", e)
        return None


# ---------- main orchestration ----------
def file_nonzero(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def download_and_try(repo_id: str, select_strategy: str):
    global HF_TOKEN
    if not HF_TOKEN:
        HF_TOKEN = os.getenv("HF_TOKEN", "")

    _log("Listing files in repo:", repo_id)
    files = list_repo_files(repo_id)
    _log("Total files in repo:", len(files))
    ggufs = [f for f in files if f.lower().endswith(".gguf") or ".gguf" in f.lower()]
    if not ggufs:
        raise SystemExit("No .gguf files found in repo.")

    ordered = pick_gguf_from_list_simple(ggufs, select_strategy)
    _log("Candidate order:", ordered)

    cached_paths = []
    for fname in ordered:
        _log("Downloading candidate:", fname)
        try:
            cached = hf_hub_download_compat(
                repo_id=repo_id, filename=fname, token=(HF_TOKEN if USE_AUTH else None)
            )
            _log("hf_hub_download returned:", cached)
            # copy to a temp candidate path (keep original cached), then try loading directly from cached
            # but also maintain a canonical MODEL_DEST_PATH copy only for the candidate we ultimately succeed with.
            cached_paths.append(cached)
        except Exception as e:
            _log("Download failed for", fname, ":", e)
            continue

    if not cached_paths:
        raise SystemExit("Failed to download any candidate .gguf files.")

    # Try load from cached paths first; when success, copy to MODEL_DEST_PATH (canonical location)
    parsed_response, good_path, last_err = test_with_llama_cpp_candidates(
        cached_paths, TEST_PROMPT, max_tokens=TEST_MAX_TOKENS, temp=TEST_TEMPERATURE
    )
    if good_path:
        _log("Successful candidate:", good_path)
        ensure_parent(MODEL_DEST_PATH)
        try:
            if os.path.abspath(good_path) != os.path.abspath(MODEL_DEST_PATH):
                shutil.copy2(good_path, MODEL_DEST_PATH)
                _log("Copied successful candidate to canonical path:", MODEL_DEST_PATH)
            else:
                _log("Successful candidate already at canonical path.")
        except Exception as e:
            _log("Failed to copy candidate to MODEL_DEST_PATH:", e)
        return parsed_response, MODEL_DEST_PATH
    else:
        _log("No candidate supported by llama-cpp-python. Last error:", last_err)
        # As a fallback, still copy the first candidate to MODEL_DEST_PATH for downstream attempts (user may use other runner)
        first_cached = cached_paths[0]
        ensure_parent(MODEL_DEST_PATH)
        try:
            if os.path.abspath(first_cached) != os.path.abspath(MODEL_DEST_PATH):
                shutil.copy2(first_cached, MODEL_DEST_PATH)
                _log(
                    "Copied first candidate to canonical path (for user inspection):",
                    MODEL_DEST_PATH,
                )
        except Exception as e:
            _log("Failed to copy first candidate to MODEL_DEST_PATH:", e)
        return None, MODEL_DEST_PATH


def main():
    start = time.time()
    try:
        # If model already present, try it first (but still try candidates if it fails)
        model_present = file_nonzero(MODEL_DEST_PATH)
        if model_present:
            _log("Model file already exists at desired path:", MODEL_DEST_PATH)
            # Try that existing model first
            resp_existing, used_path_existing, _ = test_with_llama_cpp_candidates(
                [MODEL_DEST_PATH],
                TEST_PROMPT,
                max_tokens=TEST_MAX_TOKENS,
                temp=TEST_TEMPERATURE,
            )
            if resp_existing and resp_existing.strip():
                _log("Existing model at MODEL_DEST_PATH produced a response.")
                print("\n=== RESPONSE_VALUE START ===\n")
                print(resp_existing)
                print("\n=== RESPONSE_VALUE END ===\n")
                print("MODEL_PATH:", MODEL_DEST_PATH)
                sys.exit(0)
            else:
                _log("Existing model did not work; will attempt candidates from repo.")
        # Download candidates and try them
        parsed_response, final_model_path = download_and_try(REPO_ID, SELECT_STRATEGY)
        if parsed_response and parsed_response.strip():
            _log("SUCCESS: model responded.")
            _log("Preview:", parsed_response[:400])
            print("\n=== RESPONSE_VALUE START ===\n")
            print(parsed_response)
            print("\n=== RESPONSE_VALUE END ===\n")
            print("MODEL_PATH:", final_model_path)
            # optionally save
            if SAVE_RESPONSE_FILE:
                try:
                    with open(SAVE_RESPONSE_FILE, "w", encoding="utf-8") as f:
                        f.write(parsed_response)
                    _log("Saved response to:", SAVE_RESPONSE_FILE)
                except Exception as e:
                    _log("Failed to save response to file:", e)
            _log("Total time: {:.1f}s".format(time.time() - start))
            sys.exit(0)
        # else try llama.cpp binary fallback using final_model_path
        _log(
            "Primary llama-cpp-python tests failed; trying llama.cpp binary (if installed)."
        )
        out = test_with_llama_cpp_binary(final_model_path, TEST_PROMPT)
        if out and out.strip():
            _log("SUCCESS via llama.cpp binary.")
            print("\n=== RESPONSE_VALUE START ===\n")
            print(out)
            print("\n=== RESPONSE_VALUE END ===\n")
            print("MODEL_PATH:", final_model_path)
            sys.exit(0)
        _log(
            "FAIL: model did not produce a usable textual response via available runners."
        )
        print("MODEL_PATH (downloaded/copy):", final_model_path)
        sys.exit(3)
    except KeyboardInterrupt:
        _log("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        _log("Unexpected error:", e)
        sys.exit(4)


if __name__ == "__main__":
    main()
