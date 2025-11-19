#!/usr/bin/env python3
"""
test_local_llm.py — Defensive loader that returns JSON (not generator objects)

Features:
- Download .gguf candidates from HF repo (supports token= or use_auth_token=)
- Try multiple candidates until one loads & yields text
- Robustly consume generator streaming outputs from llama-cpp-python
- Return a single JSON object to stdout (success/response/model_path/error/preview_raw)
- Attempt llama.cpp binary fallback if python binding fails to produce text
- Keeps verbose logs to stderr while printing only JSON to stdout
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
from typing import Any, List, Optional
from collections.abc import Iterable

# ---------- Configuration (override via env) ----------
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
REPO_ID: str = os.getenv("REPO_ID", "ggml-org/Mistral-Small-3.1-24B-Instruct-2503-GGUF")
MODEL_DEST_PATH: str = os.getenv(
    "MODEL_DEST_PATH", os.path.join(os.getcwd(), "models", "mistral-small-3.1.gguf")
)
USE_AUTH: bool = os.getenv("USE_AUTH", "true").lower() in ("1", "true", "yes")
SELECT_STRATEGY: str = os.getenv("SELECT_STRATEGY", "auto")
TEST_PROMPT: str = os.getenv(
    "TEST_PROMPT",
    "Q: What is 2 + 2?\nA:(strictly answer no jokes) and write a script with 3000 words a children story about a robot learning to love.\n",
)
TEST_MAX_TOKENS: int = int(os.getenv("TEST_MAX_TOKENS", "64"))
TEST_TEMPERATURE: float = float(os.getenv("TEST_TEMPERATURE", "0.0"))
VERBOSE: bool = os.getenv("VERBOSE", "true").lower() in ("1", "true", "yes")
SAVE_RESPONSE_FILE: str = os.getenv("SAVE_RESPONSE_FILE", "")
# ------------------------------------------------------


def _log(*args, **kwargs):
    # verbose logs go to stderr so stdout can be clean JSON
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


def hf_hub_download_compat(
    repo_id: str, filename: str, token: Optional[str] = None, **kwargs
) -> str:
    """
    Call hf_hub_download using modern or legacy arg names.
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


# ---------- robust chunk->text ----------
def extract_text_from_chunk(chunk: Any) -> str:
    try:
        if chunk is None:
            return ""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, bytes):
            return chunk.decode("utf-8", errors="ignore")
        if isinstance(chunk, dict):
            # Common shapes from llama-cpp streaming
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
            for k in ("text", "content", "output"):
                if k in chunk and chunk[k]:
                    return str(chunk[k])
            if "delta" in chunk and isinstance(chunk["delta"], dict):
                for k in ("content", "text"):
                    if k in chunk["delta"] and chunk["delta"][k]:
                        return str(chunk["delta"][k])
            # fallback combine string-like dict values
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
        # If iterable but not string/bytes/dict
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
        # final try
        return str(chunk)
    except Exception:
        try:
            return repr(chunk)
        except Exception:
            return ""


def consume_generator_safely(gen: Iterable) -> str:
    parts: List[str] = []
    try:
        # Primary approach: iterate normally
        for i, chunk in enumerate(gen):
            try:
                parts.append(extract_text_from_chunk(chunk))
            except Exception as echunk:
                _log("chunk extraction error:", echunk, "chunk repr:", repr(chunk))
                try:
                    parts.append(repr(chunk))
                except Exception:
                    parts.append("")
        return "".join(parts)
    except Exception as e:
        _log("Primary generator iteration failed:", e)
        # Fallback: try manual next loop
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
                try:
                    parts.append(extract_text_from_chunk(chunk))
                except Exception:
                    try:
                        parts.append(repr(chunk))
                    except Exception:
                        parts.append("")
            return "".join(parts)
        except Exception as e2:
            _log("Fallback iterator approach failed:", e2)
            # Last resort: return repr(gen)
            try:
                return repr(gen)
            except Exception:
                return "<unparseable generator>"


def parse_llama_cpp_response(resp: Any) -> (str, str):
    """
    Return (assembled_text, preview_raw)
    """
    try:
        if resp is None:
            return "", ""
        if isinstance(resp, types.GeneratorType) or (
            isinstance(resp, Iterable)
            and not isinstance(resp, (str, bytes, dict))
            and not hasattr(resp, "__len__")
        ):
            # streaming generator/iterator
            assembled = consume_generator_safely(resp)
            preview = assembled[:500]
            return assembled, preview
        if isinstance(resp, dict):
            # try common fields
            txt = ""
            if (
                "choices" in resp
                and isinstance(resp["choices"], list)
                and resp["choices"]
            ):
                first = resp["choices"][0]
                if isinstance(first, dict):
                    for k in ("text", "content"):
                        if k in first and first[k]:
                            txt = str(first[k])
                            break
                    if (
                        not txt
                        and "message" in first
                        and isinstance(first["message"], dict)
                    ):
                        txt = str(first["message"].get("content", "") or "")
                else:
                    txt = str(first)
            for k in ("text", "output", "content"):
                if k in resp and resp[k]:
                    txt = str(resp[k])
                    break
            preview = txt[:500] if isinstance(txt, str) else repr(txt)[:500]
            return txt, preview
        if isinstance(resp, bytes):
            txt = resp.decode("utf-8", errors="ignore")
            return txt, txt[:500]
        if isinstance(resp, str):
            return resp, resp[:500]
        # generic object: try attributes
        if hasattr(resp, "text"):
            try:
                t = str(resp.text or "")
                return t, t[:500]
            except Exception:
                pass
        # Finally, coerce to str
        s = str(resp)
        return s, s[:500]
    except Exception as e:
        _log("parse_llama_cpp_response error:", e)
        try:
            return str(resp), repr(resp)[:500]
        except Exception:
            return "", ""


# ---------- llama-cpp-python compatibility call ----------
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
    # attempt callable object
    try:
        _log("Attempting callable llm(...) style.")
        try:
            return llm(prompt, max_tokens=max_tokens, temperature=temperature)
        except TypeError:
            return llm(prompt)
    except Exception as e:
        _log("llama_call_compat final error:", e)
        return None


# ---------- attempt python-binding load & generate ----------
def test_with_llama_cpp_candidates(
    model_paths: List[str], prompt: str, max_tokens: int = 64, temp: float = 0.0
):
    last_error = ""
    for mp in model_paths:
        _log("Trying model file:", mp)
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
            # defensively set sampler attr to avoid destructor AttributeError in some versions
            try:
                if not hasattr(llm, "sampler"):
                    setattr(llm, "sampler", None)
            except Exception:
                pass

            # call compat wrapper
            resp = llama_call_compat(
                llm,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                stop=None,
                echo=False,
            )
            assembled, preview = parse_llama_cpp_response(resp)
            # try close
            try:
                if hasattr(llm, "close"):
                    llm.close()
            except Exception:
                pass
            # ensure deletion
            try:
                del llm
            except Exception:
                pass

            if assembled and str(assembled).strip():
                return assembled, mp, ""
            # If assembled empty but preview indicates generator or something, attempt special handling:
            if preview and (
                "<generator" in preview or isinstance(resp, types.GeneratorType)
            ):
                # If resp was generator object that we couldn't fully handle earlier, try to consume again defensively:
                try:
                    if isinstance(resp, types.GeneratorType) or hasattr(
                        resp, "__iter__"
                    ):
                        _log(
                            "Attempting additional safe consume of streaming generator/iterator..."
                        )
                        text = consume_generator_safely(resp)
                        if text and text.strip():
                            return text, mp, ""
                except Exception as ee:
                    _log("Extra consumption attempt failed:", ee)

            last_error = "empty/whitespace response"
            _log("Candidate loaded but produced empty parsed text. Continuing.")
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


# ---------- orchestrator ----------
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

    parsed_response, good_path, last_err = test_with_llama_cpp_candidates(
        cached_paths, TEST_PROMPT, max_tokens=TEST_MAX_TOKENS, temp=TEST_TEMPERATURE
    )
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
        first_cached = cached_paths[0]
        ensure_parent(MODEL_DEST_PATH)
        try:
            if os.path.abspath(first_cached) != os.path.abspath(MODEL_DEST_PATH):
                shutil.copy2(first_cached, MODEL_DEST_PATH)
                _log(
                    "Copied first candidate to canonical path for inspection:",
                    MODEL_DEST_PATH,
                )
        except Exception as e:
            _log("Failed to copy first candidate to MODEL_DEST_PATH:", e)
        return None, MODEL_DEST_PATH, last_err


def main():
    start = time.time()
    out_json = {
        "success": False,
        "response": "",
        "model_path": "",
        "error": "",
        "preview_raw": "",
    }
    try:
        model_present = file_nonzero(MODEL_DEST_PATH)
        if model_present:
            _log("Model file exists at desired path:", MODEL_DEST_PATH)
            parsed, used, err = test_with_llama_cpp_candidates(
                [MODEL_DEST_PATH],
                TEST_PROMPT,
                max_tokens=TEST_MAX_TOKENS,
                temp=TEST_TEMPERATURE,
            )
            if parsed and parsed.strip():
                out_json.update(
                    {
                        "success": True,
                        "response": parsed,
                        "model_path": MODEL_DEST_PATH,
                        "error": "",
                        "preview_raw": parsed[:500],
                    }
                )
                print(json.dumps(out_json, ensure_ascii=False))
                if SAVE_RESPONSE_FILE:
                    try:
                        with open(SAVE_RESPONSE_FILE, "w", encoding="utf-8") as f:
                            f.write(parsed)
                    except Exception:
                        pass
                return
            else:
                _log(
                    "Existing model did not produce usable response; will attempt repo candidates."
                )

        parsed_response, final_model_path, last_err = download_and_try(
            REPO_ID, SELECT_STRATEGY
        )
        out_json["model_path"] = final_model_path
        if parsed_response and str(parsed_response).strip():
            out_json.update(
                {
                    "success": True,
                    "response": parsed_response,
                    "error": "",
                    "preview_raw": parsed_response[:500],
                }
            )
            print(json.dumps(out_json, ensure_ascii=False))
            if SAVE_RESPONSE_FILE:
                try:
                    with open(SAVE_RESPONSE_FILE, "w", encoding="utf-8") as f:
                        f.write(parsed_response)
                except Exception:
                    pass
            _log("Total time: {:.1f}s".format(time.time() - start))
            return

        # try binary fallback
        _log(
            "Primary python binding failed or produced no text; trying llama.cpp binary fallback."
        )
        out = test_with_llama_cpp_binary(final_model_path, TEST_PROMPT)
        if out and out.strip():
            out_json.update(
                {
                    "success": True,
                    "response": out,
                    "error": "",
                    "preview_raw": out[:500],
                }
            )
            print(json.dumps(out_json, ensure_ascii=False))
            return

        out_json["error"] = (
            f"Model did not produce usable textual response. last_error={last_err}"
        )
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
