#!/usr/bin/env python3
"""
test_local_llm.py (streaming/compat-ready)

Features:
- HF download compatibility (use token= or use_auth_token=)
- Smart GGUF selection from a repo
- llama-cpp-python compatibility across API variants (create/generate/callable)
- Handles generator/streaming responses from Llama.generate(...) and assembles them
- Prints RESPONSE_VALUE and a truncated preview
"""

from __future__ import annotations
import os
import sys
import shutil
import time
import subprocess
import inspect
import types
from typing import Any, List, Optional, Dict
from collections.abc import Iterable

# ---------- Configuration (via env overrides) ----------
HF_TOKEN: str = os.getenv("HF_TOKEN", "")  # set by GH workflow or env
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
# -------------------------------------------------------


def _log(*a, **k):
    if VERBOSE:
        print(*a, **k)


# Attempt imports for HF and optional helpers
try:
    from huggingface_hub import list_repo_files, hf_hub_download, HfApi
except Exception as e:
    _log("Please install huggingface_hub (pip install huggingface_hub). Error:", e)
    raise

try:
    import psutil
except Exception:
    psutil = None


# ---------- Helpers: system / selection ----------
def get_total_ram_gb() -> float:
    try:
        if psutil:
            return psutil.virtual_memory().total / (1024**3)
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024**3)
    except Exception:
        pass
    return 0.0


def parse_variant_tags(fn: str) -> List[str]:
    fnl = fn.lower()
    tags = []
    for key in ("q8", "q4", "q3", "f16", "fp16", "f32", "bf16", "gguf"):
        if key in fnl:
            tags.append(key)
    for s in ("24b", "13b", "7b", "3b", "1.3b"):
        if s in fnl:
            tags.append(s)
    return tags


def choose_gguf_from_candidates(candidates: List[str], strategy: str = "auto") -> str:
    total_ram = get_total_ram_gb()
    # simple heuristics
    categorized = [(c, parse_variant_tags(c)) for c in candidates]
    s = strategy.lower()
    if s == "first":
        return categorized[0][0]
    if s == "smallest":
        return sorted(
            categorized,
            key=lambda it: (0 if ("q4" in it[1] or "q8" in it[1]) else 1, len(it[0])),
        )[0][0]
    if s == "largest":
        return categorized[0][0]
    # auto: prefer quantized for low RAM/no GPU
    if total_ram and total_ram < 32:
        for c, tags in categorized:
            if any(t.startswith("q") for t in tags):
                return c
    return categorized[0][0]


# ---------- Hugging Face download compatibility ----------
def hf_hub_download_compat(
    repo_id: str, filename: str, token: Optional[str] = None, **kwargs
) -> str:
    """
    Call hf_hub_download trying token= first, then use_auth_token= fallback.
    """
    try:
        return hf_hub_download(
            repo_id=repo_id, filename=filename, token=token, **kwargs
        )
    except TypeError:
        return hf_hub_download(
            repo_id=repo_id, filename=filename, use_auth_token=token, **kwargs
        )


def ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def pick_gguf_from_list(files: List[str], strategy: str):
    ggufs = [f for f in files if f.lower().endswith(".gguf") or ".gguf" in f.lower()]
    if not ggufs:
        raise SystemExit("No .gguf files found.")
    # If user passed numeric index as strategy
    try:
        idx = int(strategy)
        return ggufs[idx]
    except Exception:
        pass
    # else use chooser
    return choose_gguf_from_candidates(ggufs, strategy)


def download_model_via_hf(repo_id: str, select_strategy: str) -> str:
    global HF_TOKEN
    if not HF_TOKEN:
        HF_TOKEN = os.getenv("HF_TOKEN", "")
    if USE_AUTH and not HF_TOKEN:
        _log("Warning: USE_AUTH=True but HF_TOKEN empty; anonymous download may fail.")

    _log("Listing files in repo:", repo_id)
    files = list_repo_files(repo_id)
    _log("Total files in repo:", len(files))
    ggufs = [f for f in files if f.lower().endswith(".gguf") or ".gguf" in f.lower()]
    _log("Found .gguf candidates:", ggufs)
    if not ggufs:
        raise SystemExit("No .gguf files in repo. Aborting.")

    filename = pick_gguf_from_list(ggufs, select_strategy)
    _log("Selected filename:", filename)
    _log("Downloading to HF cache using hf_hub_download ... (this may take a while)")
    cached = hf_hub_download_compat(
        repo_id=repo_id, filename=filename, token=(HF_TOKEN if USE_AUTH else None)
    )
    _log("hf_hub_download returned cached path:", cached)
    ensure_parent(MODEL_DEST_PATH)
    if os.path.abspath(cached) != os.path.abspath(MODEL_DEST_PATH):
        _log("Copying model to desired path:", MODEL_DEST_PATH)
        shutil.copy2(cached, MODEL_DEST_PATH)
    else:
        _log("Cached path already matches desired path.")
    return MODEL_DEST_PATH


# ---------- Parsing / assembling responses ----------
def extract_text_from_chunk(chunk: Any) -> str:
    """
    Attempt to extract textual content from a streamed chunk returned by llama-cpp-python.
    Handles dict shapes, objects with attributes, plain strings.
    """
    try:
        if chunk is None:
            return ""
        # common dict shapes
        if isinstance(chunk, dict):
            # chunk might be like {'choices':[{'text':'...'}], ...}
            if (
                "choices" in chunk
                and isinstance(chunk["choices"], list)
                and len(chunk["choices"]) > 0
            ):
                first = chunk["choices"][0]
                if isinstance(first, dict):
                    # streaming may yield {'text': '...'} or {'delta': {'content': '...'}}
                    if "text" in first:
                        return str(first["text"] or "")
                    if "message" in first and isinstance(first["message"], dict):
                        return str(first["message"].get("content", "") or "")
                    if "delta" in first:
                        # delta can be dict with 'role' or 'content' or 'content' nested
                        d = first["delta"]
                        if isinstance(d, dict):
                            for k in ("content", "text", "token"):
                                if k in d:
                                    return str(d[k] or "")
                            # else join all values
                            return "".join(str(v) for v in d.values())
                    # fallback: join any string values
                    return "".join(
                        str(v) for v in first.values() if isinstance(v, (str,))
                    )
                else:
                    return str(first)
            # other dict fields
            for key in ("text", "output", "content"):
                if key in chunk:
                    return str(chunk[key] or "")
            # if delta at top level
            if "delta" in chunk:
                d = chunk["delta"]
                if isinstance(d, dict):
                    return d.get("content", "") or d.get("text", "") or ""
            # fallback to stringifying the dict (last resort)
            # but avoid returning the whole dict dump per chunk - return empty then append later if nothing assembled
            return ""
        # if chunk has .get('text') like object
        if hasattr(chunk, "text"):
            try:
                return str(chunk.text or "")
            except Exception:
                pass
        # if chunk is string
        if isinstance(chunk, (str, bytes)):
            return chunk.decode() if isinstance(chunk, bytes) else chunk
        # fallback for other iterables (not recommended here)
        return ""
    except Exception:
        return ""


def assemble_from_generator(gen: Iterable) -> str:
    """
    Iterate over generator/stream and assemble a single text string from chunks.
    This handles both dict-chunks and plain string chunks.
    """
    out_parts: List[str] = []
    try:
        for chunk in gen:
            txt = extract_text_from_chunk(chunk)
            # If extraction produced nothing, as a fallback include the str(chunk) if it's short
            if not txt:
                # try simple heuristics
                try:
                    s = str(chunk)
                    if len(s) < 1000:
                        txt = s
                except Exception:
                    txt = ""
            if txt:
                out_parts.append(txt)
        return "".join(out_parts)
    except Exception as e:
        _log("Error while iterating generator:", e)
        # as last resort try to coerce gen to str
        try:
            return str(gen)
        except Exception:
            return ""


def parse_llama_cpp_response(resp: Any) -> str:
    """
    Normalize possible response shapes into a string:
     - dict with choices => extract text
     - list/obj => attempt extraction
     - generator/iterable => iterate & assemble
     - str => return
    """
    try:
        if resp is None:
            return ""
        # generator / streaming iterable (but exclude str)
        if isinstance(resp, types.GeneratorType) or (
            isinstance(resp, Iterable)
            and not isinstance(resp, (str, bytes, dict))
            and not hasattr(resp, "choices")
            and not isinstance(resp, list)
        ):
            # many llama-cpp-python streaming returns a generator object
            return assemble_from_generator(resp)
        # If it's an iterable list (non-generator), maybe it's a list of events
        if isinstance(resp, list):
            out = []
            for item in resp:
                out.append(extract_text_from_chunk(item) or str(item))
            return "".join(out)
        # dict shapes
        if isinstance(resp, dict):
            # try earlier helper
            # choice shape
            if "choices" in resp and len(resp["choices"]) > 0:
                c = resp["choices"][0]
                if isinstance(c, dict) and "text" in c:
                    return str(c["text"] or "")
                if isinstance(c, dict) and "message" in c:
                    return str(c["message"].get("content", "") or "")
                return str(c)
            if "output" in resp:
                return str(resp["output"] or "")
            # fallback: try to extract common fields
            for k in ("text", "content", "message"):
                if k in resp:
                    return str(resp[k] or "")
            return ""
        # objects with attributes
        if hasattr(resp, "choices"):
            try:
                c = resp.choices[0]
                if hasattr(c, "text"):
                    return str(c.text or "")
                if hasattr(c, "message"):
                    return str(getattr(c, "message").get("content", "") or "")
            except Exception:
                pass
        # plain string
        if isinstance(resp, (str, bytes)):
            return resp.decode() if isinstance(resp, bytes) else resp
        # last resort
        return str(resp)
    except Exception as e:
        _log("parse_llama_cpp_response error:", e)
        try:
            return str(resp)
        except Exception:
            return ""


# ---------- llama-cpp-python compatibility call ----------
def llama_call_compat(
    llm: Any,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stop: Optional[List[str]] = None,
    echo: Optional[bool] = None,
):
    """
    Attempt to call the Llama instance across API variants.
    Returns whatever the underlying call returned (could be generator, dict, str).
    """

    def map_kwargs_for(fn):
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

    # try create / generate
    for method_name in ("create", "generate"):
        if hasattr(llm, method_name):
            fn = getattr(llm, method_name)
            try:
                _log(
                    f"Attempting llama-cpp-python method '{method_name}' with compatibility mapping."
                )
                kwargs = map_kwargs_for(fn)
                # some generate/create accept prompt=..., others positional
                try:
                    return fn(prompt=prompt, **kwargs)
                except TypeError:
                    return fn(prompt, **kwargs)
            except Exception as e:
                _log(f"Method '{method_name}' failed: {e}")

    # try callable Llama object
    try:
        _log("Attempting callable Llama(...) interface.")
        try:
            return llm(prompt, max_tokens=max_tokens, temperature=temperature)
        except TypeError:
            try:
                return llm(prompt)
            except Exception as exc:
                _log("callable form failed:", exc)
                return None
    except Exception as e:
        _log("llama_call_compat final error:", e)
        return None


# ---------- test using llama-cpp-python ----------
def test_with_llama_cpp(
    model_path: str, prompt: str, max_tokens: int = 64, temp: float = 0.0
) -> Optional[str]:
    try:
        from llama_cpp import Llama
    except Exception as e:
        _log("llama_cpp not installed/importable:", e)
        return None

    try:
        _log("Loading model with llama_cpp:", model_path)
        llm = Llama(model_path=model_path)
        _log("Model loaded with llama_cpp. Calling compatibility wrapper...")
        resp = llama_call_compat(
            llm,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            stop=None,
            echo=False,
        )
        if resp is None:
            _log("llama_call_compat returned None")
            return None

        # If resp is a generator, parse it by iterating to grab the streamed pieces
        parsed = parse_llama_cpp_response(resp)
        return parsed
    except Exception as e:
        _log("Error in test_with_llama_cpp:", e)
        return None


# ---------- fallback: llama.cpp binary ----------
def find_llama_cpp_binary() -> Optional[str]:
    for name in ("main", "main.exe"):
        for p in os.environ.get("PATH", "").split(os.pathsep):
            candidate = os.path.join(p, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    # try current dir
    for name in ("main", "main.exe"):
        candidate = os.path.join(os.getcwd(), name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def test_with_llama_cpp_binary(model_path: str, prompt: str) -> Optional[str]:
    exe = find_llama_cpp_binary()
    if not exe:
        _log("No llama.cpp main binary found on PATH/current dir.")
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
def file_is_present_and_nonzero(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def main():
    start = time.time()
    try:
        model_path = MODEL_DEST_PATH
        if not file_is_present_and_nonzero(model_path):
            _log("Model not present. Beginning download...")
            model_path = download_model_via_hf(REPO_ID, SELECT_STRATEGY)
            _log("Downloaded/copied model to:", model_path)
        else:
            _log("Model already present at:", model_path)

        if not file_is_present_and_nonzero(model_path):
            _log("Model missing or zero-sized after download. Exiting.")
            sys.exit(2)

        _log("Attempting inference test using llama-cpp-python (preferred).")
        response_text = test_with_llama_cpp(
            model_path, TEST_PROMPT, max_tokens=TEST_MAX_TOKENS, temp=TEST_TEMPERATURE
        )
        if response_text and response_text.strip():
            _log("SUCCESS: model responded via llama-cpp-python.")
            _log("Preview (first 400 chars):")
            _log(response_text[:400])
            # Print the full value explicitly labeled as requested
            print("\n=== RESPONSE_VALUE START ===\n")
            print(response_text)
            print("\n=== RESPONSE_VALUE END ===\n")
            _log("Total time: {:.1f}s".format(time.time() - start))
            sys.exit(0)

        _log(
            "Primary llama-cpp-python test failed or produced no text. Trying llama.cpp binary fallback."
        )
        response_text2 = test_with_llama_cpp_binary(model_path, TEST_PROMPT)
        if response_text2 and response_text2.strip():
            _log("SUCCESS: model responded via llama.cpp binary.")
            _log("Preview (first 400 chars):")
            _log(response_text2[:400])
            print("\n=== RESPONSE_VALUE START ===\n")
            print(response_text2)
            print("\n=== RESPONSE_VALUE END ===\n")
            _log("Total time: {:.1f}s".format(time.time() - start))
            sys.exit(0)

        _log("FAIL: model did not produce a usable textual response.")
        sys.exit(3)
    except KeyboardInterrupt:
        _log("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        _log("Unexpected error:", e)
        sys.exit(4)


if __name__ == "__main__":
    main()
