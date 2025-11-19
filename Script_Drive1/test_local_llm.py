#!/usr/bin/env python3
"""
test_local_llm.py — Improved: skip CLIP GGUFs and always return JSON text (not generator)

Key fixes:
- Inspect GGUF binary to detect 'general.architecture' (skip 'clip' models)
- Robustly consume generator outputs from llama-cpp-python and return assembled text in JSON
- Set safe attributes on llama instance to avoid destructor AttributeError
- Preserve HF download, candidate ordering, copying to canonical path, llama.cpp fallback
"""

from __future__ import annotations
import os
import sys
import time
import json
import shutil
import traceback
import subprocess
import inspect
import types
import re
from typing import Any, List, Optional
from collections.abc import Iterable

# -- config (env overrides)
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
REPO_ID: str = os.getenv("REPO_ID", "ggml-org/Mistral-Small-3.1-24B-Instruct-2503-GGUF")
MODEL_DEST_PATH: str = os.getenv("MODEL_DEST_PATH", os.path.join(os.getcwd(), "models", "mistral-small-3.1.gguf"))
USE_AUTH: bool = os.getenv("USE_AUTH", "true").lower() in ("1", "true", "yes")
TEST_PROMPT: str = os.getenv("TEST_PROMPT", "Q: What is 2 + 2?\nA:")
TEST_MAX_TOKENS: int = int(os.getenv("TEST_MAX_TOKENS", "64"))
TEST_TEMPERATURE: float = float(os.getenv("TEST_TEMPERATURE", "0.0"))
VERBOSE: bool = os.getenv("VERBOSE", "true").lower() in ("1", "true", "yes")

def _log(*args, **kwargs):
    if VERBOSE:
        print(*args, file=sys.stderr, **kwargs)

# huggingface_hub
try:
    from huggingface_hub import list_repo_files, hf_hub_download
except Exception as e:
    _log("Please install huggingface_hub (pip install huggingface_hub). Error:", e)
    raise

def ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def hf_hub_download_compat(repo_id: str, filename: str, token: Optional[str] = None, **kwargs) -> str:
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, token=token, **kwargs)
    except TypeError:
        return hf_hub_download(repo_id=repo_id, filename=filename, use_auth_token=token, **kwargs)

def pick_gguf_from_list_simple(files: List[str], strategy: str = "auto") -> List[str]:
    ggufs = [f for f in files if f.lower().endswith(".gguf") or ".gguf" in f.lower()]
    if not ggufs:
        return []
    # prefer smaller quantized names first (heuristic)
    return sorted(ggufs, key=lambda s: (0 if any(x in s.lower() for x in ("iq2", "ud", "q8", "q4", "q2", "mmproj")) else 1, s.lower()))

# --- inspect gguf metadata quickly to detect architecture (skip clip/vision files) ---
def guess_gguf_architecture(path: str, read_bytes: int = 256 * 1024) -> Optional[str]:
    """
    Read last `read_bytes` of GGUF file and try to extract 'general.architecture' value.
    Returns the detected architecture string (lowercased), or None if unknown.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size <= read_bytes:
                data = f.read()
            else:
                f.seek(size - read_bytes)
                data = f.read(read_bytes)
        # Try to find ASCII sequences like "general.architecture" and a value following it.
        m = re.search(rb"general\.architecture[^A-Za-z0-9_-]*([A-Za-z0-9_-]{2,40})", data, flags=re.IGNORECASE)
        if m:
            arch = m.group(1).decode("utf-8", errors="ignore").strip().lower()
            return arch
        # fallback: look for patterns like "'general.architecture': 'clip'"
        m2 = re.search(rb"general\.architecture[^']*'([^']+)'", data, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).decode("utf-8", errors="ignore").strip().lower()
        # last fallback: look for "architecture" near an ASCII token
        m3 = re.search(rb"architecture[^A-Za-z0-9_-]{0,6}([A-Za-z0-9_-]{2,40})", data, flags=re.IGNORECASE)
        if m3:
            return m3.group(1).decode("utf-8", errors="ignore").strip().lower()
        return None
    except Exception as e:
        _log("guess_gguf_architecture error:", e)
        return None

# ---------- robust chunk->text helpers ----------
def extract_text_from_chunk(chunk: Any) -> str:
    try:
        if chunk is None:
            return ""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, bytes):
            return chunk.decode("utf-8", errors="ignore")
        if isinstance(chunk, dict):
            # common shapes
            if "choices" in chunk and isinstance(chunk["choices"], list) and chunk["choices"]:
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
            for k in ("text", "content", "output"):
                if k in chunk and chunk[k]:
                    return str(chunk[k])
            if "delta" in chunk and isinstance(chunk["delta"], dict):
                for k in ("content", "text"):
                    if k in chunk["delta"] and chunk["delta"][k]:
                        return str(chunk["delta"][k])
            # fallback concatenate string/bytes values
            parts = []
            for v in chunk.values():
                if isinstance(v, (str, bytes)):
                    parts.append(v.decode() if isinstance(v, bytes) else v)
            if parts:
                return "".join(parts)
            return ""
        if hasattr(chunk, "text"):
            return str(chunk.text or "")
        if hasattr(chunk, "content"):
            return str(chunk.content or "")
        if isinstance(chunk, (int, float, bool)):
            return str(chunk)
        if isinstance(chunk, Iterable):
            collected = []
            for sub in chunk:
                if isinstance(sub, (str, bytes)):
                    collected.append(sub.decode() if isinstance(sub, bytes) else sub)
                else:
                    collected.append(str(sub))
            joined = "".join(collected)
            if joined:
                return joined
        return str(chunk)
    except Exception:
        try:
            return repr(chunk)
        except Exception:
            return ""

def consume_generator_safely(gen: Iterable) -> str:
    parts: List[str] = []
    try:
        for chunk in gen:
            parts.append(extract_text_from_chunk(chunk))
        return "".join(parts)
    except Exception as e:
        _log("Primary generator iteration failed:", e)
        try:
            it = iter(gen)
            while True:
                try:
                    chunk = next(it)
                except StopIteration:
                    break
                except Exception as inner:
                    _log("next() on generator raised:", inner)
                    break
                parts.append(extract_text_from_chunk(chunk))
            return "".join(parts)
        except Exception as e2:
            _log("Fallback iteration also failed:", e2)
            try:
                return repr(gen)
            except Exception:
                return "<unparseable generator>"

def parse_llama_cpp_response(resp: Any) -> (str, str):
    try:
        if resp is None:
            return "", ""
        if isinstance(resp, types.GeneratorType) or (isinstance(resp, Iterable) and not isinstance(resp, (str, bytes, dict)) and not hasattr(resp, "__len__")):
            assembled = consume_generator_safely(resp)
            return assembled, assembled[:500]
        if isinstance(resp, dict):
            txt = ""
            if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                first = resp["choices"][0]
                if isinstance(first, dict):
                    for k in ("text", "content"):
                        if k in first and first[k]:
                            txt = str(first[k])
                            break
                    if not txt and "message" in first and isinstance(first["message"], dict):
                        txt = str(first["message"].get("content", "") or "")
                else:
                    txt = str(first)
            for k in ("text", "output", "content"):
                if k in resp and resp[k]:
                    txt = str(resp[k])
                    break
            return txt, txt[:500]
        if isinstance(resp, bytes):
            txt = resp.decode("utf-8", errors="ignore")
            return txt, txt[:500]
        if isinstance(resp, str):
            return resp, resp[:500]
        if hasattr(resp, "text"):
            try:
                t = str(resp.text or "")
                return t, t[:500]
            except Exception:
                pass
        s = str(resp)
        return s, s[:500]
    except Exception as e:
        _log("parse_llama_cpp_response error:", e)
        try:
            return str(resp), repr(resp)[:500]
        except Exception:
            return "", ""

# ---------- llama-cpp-python compatibility call ----------
def llama_call_compat(llm: Any, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None, stop: Optional[List[str]] = None, echo: Optional[bool] = None):
    def map_kwargs(fn):
        sig = inspect.signature(fn)
        params = sig.parameters
        kwargs = {}
        if max_tokens is not None:
            for alt in ("max_tokens", "max_new_tokens", "n_predict", "n", "max_completion_tokens"):
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
                try:
                    return fn(prompt=prompt, **kwargs)
                except TypeError:
                    return fn(prompt, **kwargs)
            except Exception as e:
                _log(f"Method '{method_name}' call failed: {e}")
    try:
        _log("Attempting callable llm(...) style.")
        try:
            return llm(prompt, max_tokens=max_tokens, temperature=temperature)
        except TypeError:
            return llm(prompt)
    except Exception as e:
        _log("llama_call_compat final error:", e)
        return None

# ---------- python-binding load & generate (with architecture skip) ----------
def test_with_llama_cpp_candidates(model_paths: List[str], prompt: str, max_tokens: int = 64, temp: float = 0.0):
    last_error = ""
    for mp in model_paths:
        _log("Inspecting candidate:", mp)
        arch = guess_gguf_architecture(mp)
        _log("Detected architecture (if any):", arch)
        if arch and "clip" in arch:
            _log("Skipping candidate because architecture appears to be CLIP/vision:", arch)
            last_error = f"skipped candidate {mp} due to architecture={arch}"
            continue
        try:
            from llama_cpp import Llama
        except Exception as e:
            last_error = f"llama_cpp import failed: {e}"
            _log(last_error)
            return None, "", last_error

        llm = None
        try:
            _log("Loading model with llama_cpp:", mp)
            llm = Llama(model_path=mp)
            # avoid destructor issue in some llama-cpp builds
            try:
                if not hasattr(llm, "sampler"):
                    setattr(llm, "sampler", None)
            except Exception:
                pass

            resp = llama_call_compat(llm, prompt=prompt, max_tokens=max_tokens, temperature=temp, stop=None, echo=False)
            assembled, preview = parse_llama_cpp_response(resp)
            try:
                if hasattr(llm, "close"):
                    llm.close()
            except Exception:
                pass
            try:
                del llm
            except Exception:
                pass

            if assembled and str(assembled).strip():
                return assembled, mp, ""
            # If we got a generator-like but parse returned empty string, try explicit consume if resp is generator
            if isinstance(resp, types.GeneratorType) or hasattr(resp, "__iter__"):
                _log("Attempt extra consumption of streaming iterator...")
                try:
                    text = consume_generator_safely(resp)
                    if text and text.strip():
                        return text, mp, ""
                except Exception as extra_e:
                    _log("Extra streaming consume failed:", extra_e)

            last_error = "empty/whitespace response"
            _log("Candidate loaded but produced no usable text. Continuing.")
        except Exception as e:
            last_error = str(e)
            _log("Error in test_with_llama_cpp for", mp, ":", last_error)
            try:
                if llm is not None and hasattr(llm, "close"):
                    try:
                        llm.close()
                    except Exception:
                        pass
                    try:
                        del llm
                    except Exception:
                        pass
            except Exception:
                pass
            continue
    return None, "", last_error

# ---------- llama.cpp binary fallback ----------
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
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        if proc.returncode != 0:
            _log("llama.cpp binary failed. stderr:", proc.stderr[:2000])
            return None
        return proc.stdout
    except Exception as e:
        _log("Error running llama.cpp binary:", e)
        return None

# ---------- orchestrator ----------
def file_nonzero(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def download_and_try(repo_id: str):
    global HF_TOKEN
    if not HF_TOKEN:
        HF_TOKEN = os.getenv("HF_TOKEN", "")

    _log("Listing files in repo:", repo_id)
    files = list_repo_files(repo_id)
    _log("Total files found:", len(files))
    ggufs = [f for f in files if f.lower().endswith(".gguf") or ".gguf" in f.lower()]
    if not ggufs:
        raise SystemExit("No .gguf files found in repo.")

    ordered = pick_gguf_from_list_simple(ggufs)
    _log("Candidate order:", ordered)

    cached_paths = []
    for fname in ordered:
        _log("Downloading candidate:", fname)
        try:
            cached = hf_hub_download_compat(repo_id=repo_id, filename=fname, token=(HF_TOKEN if USE_AUTH else None))
            _log("hf_hub_download returned:", cached)
            cached_paths.append(cached)
        except Exception as e:
            _log("Download failed for", fname, ":", e)
            continue

    if not cached_paths:
        raise SystemExit("Failed to download any candidate .gguf files.")

    parsed_response, good_path, last_err = test_with_llama_cpp_candidates(cached_paths, TEST_PROMPT, max_tokens=TEST_MAX_TOKENS, temp=TEST_TEMPERATURE)
    if good_path:
        ensure_parent(MODEL_DEST_PATH)
        try:
            if os.path.abspath(good_path) != os.path.abspath(MODEL_DEST_PATH):
                shutil.copy2(good_path, MODEL_DEST_PATH)
                _log("Copied successful candidate to canonical path:", MODEL_DEST_PATH)
            else:
                _log("Successful candidate already at canonical path.")
        except Exception as e:
            _log("Failed to copy candidate to MODEL_DEST_PATH:", e)
        return parsed_response, MODEL_DEST_PATH, ""
    else:
        _log("No candidate produced text via llama-cpp-python. Last error:", last_err)
        # copy first candidate for inspection
        first_cached = cached_paths[0]
        ensure_parent(MODEL_DEST_PATH)
        try:
            if os.path.abspath(first_cached) != os.path.abspath(MODEL_DEST_PATH):
                shutil.copy2(first_cached, MODEL_DEST_PATH)
                _log("Copied first candidate to canonical path for inspection:", MODEL_DEST_PATH)
        except Exception as e:
            _log("Failed to copy first candidate to MODEL_DEST_PATH:", e)
        return None, MODEL_DEST_PATH, last_err

def main():
    start = time.time()
    out_json = {"success": False, "response": "", "model_path": "", "error": "", "preview_raw": ""}
    try:
        model_present = file_nonzero(MODEL_DEST_PATH)
        if model_present:
            _log("Model file exists at desired path:", MODEL_DEST_PATH)
            parsed, used, err = test_with_llama_cpp_candidates([MODEL_DEST_PATH], TEST_PROMPT, max_tokens=TEST_MAX_TOKENS, temp=TEST_TEMPERATURE)
            if parsed and parsed.strip():
                out_json.update({"success": True, "response": parsed, "model_path": MODEL_DEST_PATH, "error": "", "preview_raw": parsed[:500]})
                print(json.dumps(out_json, ensure_ascii=False))
                return
            else:
                _log("Existing model did not produce usable response; will attempt repo candidates.")

        parsed_response, final_model_path, last_err = download_and_try(REPO_ID)
        out_json["model_path"] = final_model_path
        if parsed_response and str(parsed_response).strip():
            out_json.update({"success": True, "response": parsed_response, "error": "", "preview_raw": parsed_response[:500]})
            print(json.dumps(out_json, ensure_ascii=False))
            _log("Total time: {:.1f}s".format(time.time() - start))
            return

        _log("Primary python binding failed or produced no text; trying llama.cpp binary fallback.")
        out = test_with_llama_cpp_binary(final_model_path, TEST_PROMPT)
        if out and out.strip():
            out_json.update({"success": True, "response": out, "error": "", "preview_raw": out[:500]})
            print(json.dumps(out_json, ensure_ascii=False))
            return

        out_json["error"] = f"Model did not produce usable textual response. last_error={last_err}"
        print(json.dumps(out_json, ensure_ascii=False))
        _log("Total time: {:.1f}s".format(time.time() - start))
        sys.exit(3)

    except KeyboardInterrupt:
        out_json["error"] = "Interrupted by user"
        print(json.dumps(out_json, ensure_ascii=False))
        sys.exit(130)
    except Exception as e:
        tb = traceback.format_exc()
        out_json["error"] = f"Unexpected error: {e}\n{tb}"
        print(json.dumps(out_json, ensure_ascii=False))
        sys.exit(4)

if __name__ == "__main__":
    main()
