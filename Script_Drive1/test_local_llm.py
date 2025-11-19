#!/usr/bin/env python3
"""
test_local_llm.py — ultimate compatibility + candidate iteration + robust streaming assembly

Kept features:
- HF download compatibility (token= / use_auth_token=)
- Try multiple .gguf candidates and skip incompatible archs (e.g., clip)
- llama-cpp-python compatibility across create/generate/callable
- Robustly assemble streaming/generator output into a single string
- Fallback to llama.cpp `main` binary
- Prints explicit RESPONSE_VALUE and MODEL_PATH
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

# ---------- Configuration (override with env vars) ----------
HF_TOKEN: str = os.getenv("HF_TOKEN", "")  # GH workflow should set this
REPO_ID: str = os.getenv("REPO_ID", "ggml-org/Mistral-Small-3.1-24B-Instruct-2503-GGUF")
MODEL_DEST_PATH: str = os.getenv(
    "MODEL_DEST_PATH", os.path.join(os.getcwd(), "models", "mistral-small-3.1.gguf")
)
USE_AUTH: bool = os.getenv("USE_AUTH", "true").lower() in ("1", "true", "yes")
SELECT_STRATEGY: str = os.getenv("SELECT_STRATEGY", "auto")
TEST_PROMPT: str = os.getenv("TEST_PROMPT", "Q: What is 2 + 2?\nA:")
TEST_MAX_TOKENS: int = int(os.getenv("TEST_MAX_TOKENS", "64"))
TEST_TEMPERATURE: float = float(os.getenv("TEST_TEMPERATURE", "0.0"))
VERBOSE: bool = os.getenv("VERBOSE", "true").lower() in ("1", "true", "yes")
SAVE_RESPONSE_FILE: str = os.getenv("SAVE_RESPONSE_FILE", "")
# ----------------------------------------------------------


def _log(*a, **k):
    if VERBOSE:
        print(*a, **k)


# Hugging Face hub import (fail early if not installed)
try:
    from huggingface_hub import list_repo_files, hf_hub_download, HfApi
except Exception as e:
    _log("Please install huggingface_hub (pip install huggingface_hub). Error:", e)
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
    Try both modern token= and legacy use_auth_token= signatures.
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
    ggufs = [f for f in files if f.lower().endswith(".gguf") or ".gguf" in f.lower()]
    if not ggufs:
        return []
    if strategy and strategy.lower() in ("smallest", "auto"):
        return sorted(
            ggufs,
            key=lambda s: (
                0 if ("q4" in s.lower() or "q8" in s.lower()) else 1,
                len(s),
            ),
        )
    if strategy and strategy.lower() == "largest":
        return list(reversed(ggufs))
    return ggufs


# ---------- Robust chunk-to-text extraction ----------
def extract_text_from_chunk(chunk: Any) -> str:
    """
    Attempt many safe approaches to extract human text from a chunk.
    Returns an empty string if nothing textual found (caller will fallback).
    """
    try:
        if chunk is None:
            return ""
        # If chunk already a string/bytes
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, bytes):
            try:
                return chunk.decode("utf-8", errors="ignore")
            except Exception:
                return chunk.decode(errors="ignore")
        # Common dict-like shapes returned by streaming APIs
        if isinstance(chunk, dict):
            # Top-level choices
            if (
                "choices" in chunk
                and isinstance(chunk["choices"], list)
                and chunk["choices"]
            ):
                first = chunk["choices"][0]
                if isinstance(first, dict):
                    for k in ("text", "content"):
                        if k in first and first[k]:
                            return str(first[k])
                    if "delta" in first and isinstance(first["delta"], dict):
                        for k in ("content", "text"):
                            if k in first["delta"] and first["delta"][k]:
                                return str(first["delta"][k])
                else:
                    return str(first)
            # direct fields
            for k in ("text", "content", "output"):
                if k in chunk and chunk[k]:
                    return str(chunk[k])
            if "delta" in chunk and isinstance(chunk["delta"], dict):
                for k in ("content", "text"):
                    if k in chunk["delta"] and chunk["delta"][k]:
                        return str(chunk["delta"][k])
            # fallback: try to combine any string values
            s = []
            for v in chunk.values():
                if isinstance(v, (str, bytes)):
                    s.append(v.decode() if isinstance(v, bytes) else v)
            if s:
                return "".join(s)
            return ""
        # If chunk has text-like attributes (some library objects)
        if hasattr(chunk, "text"):
            try:
                return str(chunk.text)
            except Exception:
                pass
        if hasattr(chunk, "content"):
            try:
                return str(chunk.content)
            except Exception:
                pass
        # If chunk is numeric or boolean, convert to str
        if isinstance(chunk, (int, float, bool)):
            return str(chunk)
        # If chunk is iterable (but not string), try joining subparts safely
        if isinstance(chunk, Iterable):
            parts = []
            for sub in chunk:
                try:
                    if isinstance(sub, (str, bytes)):
                        parts.append(sub.decode() if isinstance(sub, bytes) else sub)
                    else:
                        parts.append(str(sub))
                except Exception:
                    try:
                        parts.append(repr(sub))
                    except Exception:
                        pass
            joined = "".join(parts)
            if joined:
                return joined
        # Final fallback: repr
        try:
            return str(chunk)
        except Exception:
            return repr(chunk)
    except Exception:
        try:
            return repr(chunk)
        except Exception:
            return ""


def assemble_from_generator(gen: Iterable) -> str:
    """
    Iterate generator/stream and assemble textual output robustly.
    Protects against weird chunk types and iteration errors.
    """
    parts: List[str] = []
    try:
        # Some streaming generators may require repeated next() calls
        iterator = iter(gen)
        while True:
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            except TypeError:
                # If next raises TypeError, maybe generator object is not proper iterator — abort iteration.
                raise
            except Exception as e:
                _log("Iterator next() raised:", e)
                break

            try:
                txt = extract_text_from_chunk(chunk)
                if not txt:
                    # as extra fallback attempt to inspect common fields
                    try:
                        if hasattr(chunk, "get"):
                            for k in ("text", "content", "output"):
                                try:
                                    v = chunk.get(k)
                                    if v:
                                        txt = str(v)
                                        break
                                except Exception:
                                    pass
                    except Exception:
                        pass
                if not txt:
                    # ultimate fallback: repr(chunk)
                    try:
                        txt = repr(chunk)
                    except Exception:
                        txt = ""
                parts.append(txt)
            except Exception as inner:
                _log("chunk conversion error:", inner, "chunk repr:", repr(chunk))
                try:
                    parts.append(repr(chunk))
                except Exception:
                    pass
        return "".join(parts)
    except Exception as e:
        _log("Error while iterating generator:", e)
        try:
            # If some parts were gathered before failure, return them
            if parts:
                return "".join(parts)
        except Exception:
            pass
        # fallback to generator repr so caller has something
        try:
            return repr(gen)
        except Exception:
            return "<unparseable generator>"


def parse_llama_cpp_response(resp: Any) -> str:
    """
    Normalize various response shapes to a string.
    """
    try:
        if resp is None:
            return ""
        # generator (explicit)
        if isinstance(resp, types.GeneratorType):
            return assemble_from_generator(resp)
        # If it's an iterator (not string/bytes/dict), but not generator type
        if isinstance(resp, Iterable) and not isinstance(resp, (str, bytes, dict)):
            # Some versions return a generator-like object (we already handle that),
            # otherwise treat as list of chunks
            try:
                # Try to iterate safely
                parts = []
                for chunk in resp:
                    parts.append(extract_text_from_chunk(chunk) or repr(chunk))
                return "".join(parts)
            except Exception:
                # fallback to assemble_from_generator — may handle custom iterators
                try:
                    return assemble_from_generator(resp)
                except Exception:
                    return repr(resp)
        # dict responses
        if isinstance(resp, dict):
            # common shapes
            if (
                "choices" in resp
                and isinstance(resp["choices"], list)
                and resp["choices"]
            ):
                first = resp["choices"][0]
                if isinstance(first, dict):
                    for k in ("text", "content"):
                        if k in first and first[k]:
                            return str(first[k])
                    if "message" in first and isinstance(first["message"], dict):
                        return str(first["message"].get("content", "") or "")
                return str(first)
            for k in ("text", "output", "content"):
                if k in resp and resp[k]:
                    return str(resp[k])
            return ""
        # object with choices attribute
        if hasattr(resp, "choices"):
            try:
                c = resp.choices[0]
                if hasattr(c, "text"):
                    return str(c.text or "")
                if hasattr(c, "message") and isinstance(c.message, dict):
                    return str(c.message.get("content", "") or "")
            except Exception:
                pass
        # strings/bytes
        if isinstance(resp, bytes):
            return resp.decode("utf-8", errors="ignore")
        if isinstance(resp, str):
            return resp
        # fallback
        return str(resp)
    except Exception as e:
        _log("parse_llama_cpp_response error:", e)
        try:
            return str(resp)
        except Exception:
            return "<unparseable response>"


# ---------- llama-cpp-python compatibility helper ----------
def llama_call_compat(
    llm: Any,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stop: Optional[List[str]] = None,
    echo: Optional[bool] = None,
):
    """
    Try create/generate/callable forms and return whatever the underlying Llama method returns.
    """

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

    for method_name in ("create", "generate"):
        if hasattr(llm, method_name):
            fn = getattr(llm, method_name)
            kwargs = map_kwargs(fn)
            try:
                _log(f"Attempting llama-cpp-python method '{method_name}'.")
                # try keyword call first
                try:
                    return fn(prompt=prompt, **kwargs)
                except TypeError:
                    return fn(prompt, **kwargs)
            except Exception as e:
                _log(f"Method '{method_name}' failed: {e}")
    # attempt callable object
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


# ---------- Attempt load & generate for a list of candidate model paths ----------
def test_with_llama_cpp_candidates(
    model_paths: List[str], prompt: str, max_tokens: int = 64, temp: float = 0.0
):
    """
    Try each provided model file path in order until one loads & returns text.
    Returns (parsed_text_or_None, successful_path_or_empty, last_error_str_or_empty)
    """
    last_error = ""
    for mp in model_paths:
        _log("Trying model file:", mp)
        try:
            from llama_cpp import Llama
        except Exception as e:
            _log("llama_cpp import failed:", e)
            return None, "", str(e)

        llm = None
        try:
            _log("Loading model with llama_cpp:", mp)
            llm = Llama(model_path=mp)
            _log("Model loaded successfully:", mp)
            # call compat wrapper — may return generator, dict, etc.
            resp = llama_call_compat(
                llm,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                stop=None,
                echo=False,
            )
            # parse into text
            parsed = parse_llama_cpp_response(resp)
            # close llm cleanly if possible
            try:
                if hasattr(llm, "close"):
                    llm.close()
            except Exception:
                pass
            # if parsed non-empty, success
            if parsed is not None and str(parsed).strip():
                return parsed, mp, ""
            # parsed empty — record and move to next candidate
            last_error = "empty response"
            _log(
                "Candidate loaded but produced empty parsed text. Continuing to next candidate."
            )
        except Exception as e:
            last_error = str(e)
            _log("Error in test_with_llama_cpp for", mp, ":", last_error)
            # try to close if partially created
            try:
                if llm is not None and hasattr(llm, "close"):
                    try:
                        llm.close()
                    except Exception:
                        pass
                    del llm
            except Exception:
                pass
            # if architecture / load issue, try next candidate (we handle generically)
            continue
    return None, "", last_error


# ---------- fallback to llama.cpp binary ----------
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
            _log("llama.cpp binary failed. stderr:", proc.stderr[:2000])
            return None
        return proc.stdout
    except Exception as e:
        _log("Error running llama.cpp binary:", e)
        return None


# ---------- orchestrator: download candidates, try them, copy successful to MODEL_DEST_PATH ----------
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
            cached_paths.append(cached)
        except Exception as e:
            _log("Download failed for", fname, ":", e)
            continue

    if not cached_paths:
        raise SystemExit("Failed to download any candidate .gguf files.")

    # Try load from cached paths
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
        _log("No candidate produced text via llama-cpp-python. Last error:", last_err)
        # Copy first candidate to canonical path for inspection
        first_cached = cached_paths[0]
        ensure_parent(MODEL_DEST_PATH)
        try:
            if os.path.abspath(first_cached) != os.path.abspath(MODEL_DEST_PATH):
                shutil.copy2(first_cached, MODEL_DEST_PATH)
                _log("Copied first candidate to canonical path:", MODEL_DEST_PATH)
        except Exception as e:
            _log("Failed to copy first candidate to MODEL_DEST_PATH:", e)
        return None, MODEL_DEST_PATH


def main():
    start = time.time()
    try:
        model_present = file_nonzero(MODEL_DEST_PATH)
        if model_present:
            _log("Model file exists at desired path:", MODEL_DEST_PATH)
            resp_existing, used_path_existing, _ = test_with_llama_cpp_candidates(
                [MODEL_DEST_PATH],
                TEST_PROMPT,
                max_tokens=TEST_MAX_TOKENS,
                temp=TEST_TEMPERATURE,
            )
            if resp_existing and resp_existing.strip():
                _log("Existing model produced a response.")
                print("\n=== RESPONSE_VALUE START ===\n")
                print(resp_existing)
                print("\n=== RESPONSE_VALUE END ===\n")
                print("MODEL_PATH:", MODEL_DEST_PATH)
                if SAVE_RESPONSE_FILE:
                    try:
                        with open(SAVE_RESPONSE_FILE, "w", encoding="utf-8") as f:
                            f.write(resp_existing)
                    except Exception:
                        pass
                sys.exit(0)
            else:
                _log(
                    "Existing model did not produce usable response; will attempt candidates from repo."
                )

        parsed_response, final_model_path = download_and_try(REPO_ID, SELECT_STRATEGY)
        if parsed_response and str(parsed_response).strip():
            _log("SUCCESS: model responded.")
            _log("Preview (first 400 chars):", parsed_response[:400])
            print("\n=== RESPONSE_VALUE START ===\n")
            print(parsed_response)
            print("\n=== RESPONSE_VALUE END ===\n")
            print("MODEL_PATH:", final_model_path)
            if SAVE_RESPONSE_FILE:
                try:
                    with open(SAVE_RESPONSE_FILE, "w", encoding="utf-8") as f:
                        f.write(parsed_response)
                    _log("Saved response to:", SAVE_RESPONSE_FILE)
                except Exception as e:
                    _log("Failed to save response to file:", e)
            _log("Total time: {:.1f}s".format(time.time() - start))
            sys.exit(0)

        _log(
            "Primary llama-cpp-python tests failed or produced no text; trying llama.cpp binary fallback."
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
            "FAIL: model did not produce usable textual response via available runners."
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
