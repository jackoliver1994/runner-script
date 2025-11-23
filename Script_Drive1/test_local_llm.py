#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import time
import shutil
import subprocess
from typing import List, Optional, Tuple, Dict, Any, Union
import inspect

# ----------------- CONFIG (read from env; override here if needed) -----------------
HF_TOKEN: str = os.getenv("HF_TOKEN", "")  # Provided by GitHub workflow (hf_key)
REPO_ID: str = os.getenv("REPO_ID", "Qwen/Qwen1.5-7B-Chat-GGUF")
MODEL_DEST_PATH: str = os.getenv(
    "MODEL_DEST_PATH", os.path.join(os.getcwd(), "models", "Qwen1.5-7B-Chat.gguf")
)
USE_AUTH: bool = os.getenv("USE_AUTH", "true").lower() in ("1", "true", "yes")
SELECT_STRATEGY: str = os.getenv(
    "SELECT_STRATEGY", "auto"
).lower()  # auto, first, smallest, largest
TEST_PROMPT: str = os.getenv(
    "TEST_PROMPT",
    "Write me a script for a 10 min video for children(stritly answer with the script)\n",
)
TEST_MAX_TOKENS: int = int(os.getenv("TEST_MAX_TOKENS", "30000"))
TEST_TEMPERATURE: float = float(os.getenv("TEST_TEMPERATURE", "0.0"))
VERBOSE: bool = os.getenv("VERBOSE", "true").lower() in ("1", "true", "yes")
POST_SUCCESS_CMD: Optional[str] = os.getenv("POST_SUCCESS_CMD", None)
# -----------------------------------------------------------------------------------


def _log(*a, **k):
    if VERBOSE:
        print(*a, **k, flush=True)


# Try importing required HF helpers early
try:
    from huggingface_hub import list_repo_files, hf_hub_download, HfApi
except Exception as e:
    _log("Please ensure huggingface_hub is installed (pip install huggingface_hub).")
    raise

# Optional helper modules (not fatal)
try:
    import psutil
except Exception:
    psutil = None

# Try to import torch for GPU detection (optional)
try:
    import torch
except Exception:
    torch = None

# ----------------- System detection -----------------


def get_total_ram_gb() -> float:
    try:
        if psutil:
            return psutil.virtual_memory().total / (1024**3)
        # fallback for *nix
        if hasattr(os, "sysconf"):
            if (
                "SC_PAGE_SIZE" in os.sysconf_names
                and "SC_PHYS_PAGES" in os.sysconf_names
            ):
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                return (pages * page_size) / (1024**3)
    except Exception:
        pass
    return 0.0


def detect_cpu_flags() -> List[str]:
    flags = []
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read().lower()
            # common flags to check
            for flag in ("avx512", "avx2", "avx", "sse4_2", "fma"):
                if flag in txt:
                    flags.append(flag)
    except Exception:
        pass
    return flags


def detect_gpu_info() -> Optional[Dict[str, float]]:
    """
    returns {'vram_gb': float, 'cuda': True/False} or None if no GPU detected
    """
    try:
        if torch is not None and torch.cuda.is_available():
            # only first device
            prop = torch.cuda.get_device_properties(0)
            vram = prop.total_memory / (1024**3)
            return {"vram_gb": float(vram), "cuda": True}
    except Exception:
        pass

    # Try nvidia-smi if available
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        if proc.returncode == 0:
            val = proc.stdout.strip().splitlines()[0]
            vram_gb = float(val) / 1024.0
            return {"vram_gb": vram_gb, "cuda": True}
    except Exception:
        pass

    return None


# ----------------- Variant selection heuristics -----------------


def parse_variant_tags(filename: str) -> List[str]:
    """Simple tag extractor from filename."""
    fn = filename.lower()
    tags = []
    for key in (
        "q8",
        "q4",
        "q3",
        "q2",
        "q5",
        "q6",
        "q7",
        "q8_0",
        "q4_0",
        "q4_k",
        "q8_0",
        "q5_k",
        "f16",
        "fp16",
        "f32",
        "bf16",
        "gguf",
        "safetensors",
        "cpu",
        "cuda",
        "cuda-rt",
    ):
        if key in fn:
            tags.append(key)
    # also detect sizes like '24b' or '13b'
    for s in ("24b", "13b", "7b", "3b", "2.7b", "1.3b"):
        if s in fn:
            tags.append(s)
    return tags


def _try_get_local_size(path: str) -> Optional[int]:
    """Return file size in bytes if path exists locally and is accessible, else None."""
    try:
        if path.startswith("http://") or path.startswith("https://"):
            return None
        return os.path.getsize(path)
    except Exception:
        return None


def _is_quantized_name(name: str) -> bool:
    """Heuristic: file names containing q4, q8, q5, quant, 'int' often indicate quantized models."""
    n = name.lower()
    # common patterns: q4_0, q8_0, q4, q8, 4bit, 8bit, quantized, quant, int8, int4
    patterns = [r"\bq\d", r"\b\dbit\b", r"quant", r"int8", r"int4", r"8bit", r"4bit"]
    return any(re.search(p, n) for p in patterns)


def _available_ram_bytes() -> Optional[int]:
    """Try to get available RAM in bytes; returns None if psutil not available."""
    try:
        import psutil

        return psutil.virtual_memory().available
    except Exception:
        return None


def _try_get_gpu_mem_bytes() -> Optional[int]:
    """Try to detect the total GPU memory (bytes) on the main GPU. Return None if not detectable."""
    # Try pynvml first, then torch.cuda
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return int(meminfo.total)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            # total memory of device 0
            props = torch.cuda.get_device_properties(0)
            return int(props.total_memory)
    except Exception:
        pass
    return None


def choose_gguf_from_candidates(
    candidates: List[str],
    strategy: Union[str, int] = "auto",
    verbose: bool = False,
) -> str:
    """
    Choose one .gguf candidate from a list according to strategy.

    candidates: list of filenames, paths or URLs (strings)
    strategy: one of:
        - integer index -> return candidates[index]
        - 'first' -> first candidate
        - 'smallest' -> by local file size if available, else by quantized heuristics
        - 'largest' -> by local file size if available, else by non-quantized heuristics
        - 'auto' -> uses system resources heuristics (GPU memory > threshold or RAM) to
                    prefer full (non-quantized) model or quantized model
        - filename string -> matched against basename or full candidate entry (case-insensitive)
    Returns selected candidate string.
    Raises SystemExit if no valid .gguf candidates or index out of range.
    """
    if not candidates:
        raise SystemExit("No candidates provided to choose_gguf_from_candidates().")

    # Filter to .gguf-like entries (be tolerant: could be '...gguf' or contain '.gguf')
    ggufs = [c for c in candidates if c and ".gguf" in c.lower()]
    if not ggufs:
        raise SystemExit("No .gguf files found in candidates.")

    # Normalize strategy string if applicable
    if isinstance(strategy, str):
        strategy_lower = strategy.strip().lower()
    else:
        strategy_lower = None

    # 1) Direct filename/full-string match (case-insensitive)
    if isinstance(strategy, str) and strategy_lower:
        for c in ggufs:
            if (
                os.path.basename(c).lower() == strategy_lower
                or c.lower() == strategy_lower
            ):
                if verbose:
                    print(f"[chooser] Direct filename match -> {c}")
                return c

    # 2) Integer index
    if isinstance(strategy, int):
        idx = strategy
        if idx < 0 or idx >= len(ggufs):
            raise SystemExit(f"Index strategy out of range: {idx} (len={len(ggufs)})")
        if verbose:
            print(f"[chooser] Index strategy -> {idx} -> {ggufs[idx]}")
        return ggufs[idx]

    # 3) 'first' strategy
    if isinstance(strategy, str) and strategy_lower in (
        "first",
        "first-found",
        "first_found",
    ):
        if verbose:
            print(f"[chooser] First strategy -> {ggufs[0]}")
        return ggufs[0]

    # Helper: gather sizes (if available) and quant flag
    sized_list = []
    for c in ggufs:
        size = _try_get_local_size(c)  # may be None for URLs or missing files
        quant = _is_quantized_name(os.path.basename(c))
        sized_list.append({"path": c, "size": size, "quant": quant})

    # 4) 'smallest' strategy
    if isinstance(strategy, str) and strategy_lower == "smallest":
        # Prefer smallest by available size; if no sizes, prefer quantized names
        sizes = [s["size"] for s in sized_list if s["size"] is not None]
        if sizes:
            # pick candidate with smallest size (local)
            pick = min(
                (s for s in sized_list if s["size"] is not None),
                key=lambda x: x["size"],
            )
            if verbose:
                print(
                    f"[chooser] Smallest-by-size -> {pick['path']} ({pick['size']} bytes)"
                )
            return pick["path"]
        # fallback: choose quantized name if present
        for s in sized_list:
            if s["quant"]:
                if verbose:
                    print(
                        f"[chooser] Smallest heuristic fallback -> {s['path']} (quantized by name)"
                    )
                return s["path"]
        if verbose:
            print(f"[chooser] Smallest fallback -> {sized_list[0]['path']}")
        return sized_list[0]["path"]

    # 5) 'largest' strategy
    if isinstance(strategy, str) and strategy_lower == "largest":
        sizes = [s["size"] for s in sized_list if s["size"] is not None]
        if sizes:
            pick = max(
                (s for s in sized_list if s["size"] is not None),
                key=lambda x: x["size"],
            )
            if verbose:
                print(
                    f"[chooser] Largest-by-size -> {pick['path']} ({pick['size']} bytes)"
                )
            return pick["path"]
        # fallback: prefer non-quantized names
        nonq = [s for s in sized_list if not s["quant"]]
        if nonq:
            if verbose:
                print(
                    f"[chooser] Largest heuristic fallback -> {nonq[0]['path']} (non-quantized by name)"
                )
            return nonq[0]["path"]
        if verbose:
            print(f"[chooser] Largest fallback -> {sized_list[0]['path']}")
        return sized_list[0]["path"]

    # 6) 'auto' strategy (default)
    # Try to determine if system has a capable GPU or much RAM. If resources are constrained, prefer quantized.
    if strategy is None or (isinstance(strategy, str) and strategy_lower == "auto"):
        gpu_mem = _try_get_gpu_mem_bytes()
        ram_avail = _available_ram_bytes()
        if verbose:
            print(
                f"[chooser] Auto heuristics: gpu_mem={gpu_mem}, ram_avail={ram_avail}"
            )

        # Use thresholds (bytes): GPU >= 12 GiB -> prefer full (non-quantized); else prefer quantized
        GB = 1024**3
        gpu_threshold = 12 * GB
        ram_threshold = (
            24 * GB
        )  # arbitrary: if machine has lots of RAM, can pick larger models

        prefer_quant = True
        if gpu_mem is not None:
            prefer_quant = gpu_mem < gpu_threshold
            if verbose:
                print(f"[chooser] prefer_quant (based on GPU) = {prefer_quant}")
        elif ram_avail is not None:
            # if lots of RAM, don't prefer quant
            prefer_quant = ram_avail < ram_threshold
            if verbose:
                print(f"[chooser] prefer_quant (based on RAM) = {prefer_quant}")
        else:
            # No reliable system info: make a conservative choice using filenames:
            # if any candidate looks quantized -> prefer quant, else prefer non-quant
            has_quant = any(s["quant"] for s in sized_list)
            has_nonq = any(not s["quant"] for s in sized_list)
            prefer_quant = has_quant and not has_nonq
            if verbose:
                print(f"[chooser] prefer_quant (fallback by names) = {prefer_quant}")

        # If sizes are available, use sizes combined with quant flag
        sizes_known = any(s["size"] is not None for s in sized_list)
        if sizes_known:
            # If prefer_quant -> choose smallest quantized; else choose largest non-quantized (or largest overall)
            if prefer_quant:
                quant_candidates = [
                    s for s in sized_list if s["quant"] and s["size"] is not None
                ]
                if quant_candidates:
                    pick = min(quant_candidates, key=lambda x: x["size"])
                    if verbose:
                        print(f"[chooser] Auto (prefer quant) -> {pick['path']}")
                    return pick["path"]
                # otherwise fall back to smallest by size
                pick = min(
                    (s for s in sized_list if s["size"] is not None),
                    key=lambda x: x["size"],
                )
                if verbose:
                    print(f"[chooser] Auto fallback smallest -> {pick['path']}")
                return pick["path"]
            else:
                nonq_candidates = [
                    s for s in sized_list if (not s["quant"]) and s["size"] is not None
                ]
                if nonq_candidates:
                    pick = max(nonq_candidates, key=lambda x: x["size"])
                    if verbose:
                        print(f"[chooser] Auto (prefer full) -> {pick['path']}")
                    return pick["path"]
                # otherwise pick largest overall
                pick = max(
                    (s for s in sized_list if s["size"] is not None),
                    key=lambda x: x["size"],
                )
                if verbose:
                    print(f"[chooser] Auto fallback largest -> {pick['path']}")
                return pick["path"]
        else:
            # No sizes known. Use name heuristics:
            if prefer_quant:
                for s in sized_list:
                    if s["quant"]:
                        if verbose:
                            print(
                                f"[chooser] Auto (no sizes) choose quant by name -> {s['path']}"
                            )
                        return s["path"]
                if verbose:
                    print(
                        f"[chooser] Auto (no sizes) no quant names; pick first -> {sized_list[0]['path']}"
                    )
                return sized_list[0]["path"]
            else:
                for s in sized_list:
                    if not s["quant"]:
                        if verbose:
                            print(
                                f"[chooser] Auto (no sizes) choose non-quant by name -> {s['path']}"
                            )
                        return s["path"]
                # else fallback
                if verbose:
                    print(
                        f"[chooser] Auto (no sizes) fallback -> {sized_list[0]['path']}"
                    )
                return sized_list[0]["path"]

    # If strategy didn't match any known commands, raise
    raise SystemExit(
        f"Unknown choose strategy: {strategy!r}. Valid: int, 'first', 'smallest', 'largest', 'auto', or filename."
    )


# ----------------- HF download helpers -----------------


def hf_hub_download_compat(
    repo_id: str, filename: str, token: Optional[str] = None, **kwargs
) -> str:
    try:
        return hf_hub_download(
            repo_id=repo_id, filename=filename, token=token, **kwargs
        )
    except TypeError:
        return hf_hub_download(
            repo_id=repo_id, filename=filename, use_auth_token=token, **kwargs
        )


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def pick_gguf_from_list(files: List[str], strategy) -> str:
    candidates = [
        f for f in files if f.lower().endswith(".gguf") or ".gguf" in f.lower()
    ]
    if not candidates:
        raise SystemExit("No .gguf files found in the repo listing.")
    if isinstance(strategy, int):
        return candidates[strategy]
    s = str(strategy).lower()
    if s in ("first", "smallest", "largest"):
        if s == "first":
            return candidates[0]
        if s == "smallest":
            heur = sorted(
                candidates,
                key=lambda fn: (
                    (
                        0
                        if (
                            "q8" in fn.lower()
                            or "q4" in fn.lower()
                            or "f16" in fn.lower()
                        )
                        else 1
                    ),
                    len(fn),
                ),
            )
            return heur[0]
        if s == "largest":
            try:
                api = HfApi()
                info = api.repo_info(REPO_ID, token=HF_TOKEN or None)
                siblings = getattr(info, "siblings", None)
                if siblings:
                    gg = [
                        x
                        for x in siblings
                        if (
                            x.rfilename.lower().endswith(".gguf")
                            or ".gguf" in x.rfilename.lower()
                        )
                    ]
                    if gg:
                        gg.sort(key=lambda x: getattr(x, "size", 0), reverse=True)
                        return gg[0].rfilename
            except Exception:
                pass
            return candidates[0]

    return choose_gguf_from_candidates(candidates, REPO_ID, strategy or "auto")


def download_model_via_hf(repo_id: str, select_strategy) -> str:
    global HF_TOKEN
    if not HF_TOKEN:
        HF_TOKEN = os.getenv("HF_TOKEN", "")
    if USE_AUTH and not HF_TOKEN:
        _log(
            "Warning: USE_AUTH=True but HF_TOKEN not set. Anonymous downloads may fail."
        )

    _log("Listing files in repo:", repo_id)
    files = list_repo_files(repo_id)
    _log("Total files in repo:", len(files))
    ggufs = [f for f in files if f.lower().endswith(".gguf") or ".gguf" in f.lower()]
    _log("Found .gguf candidates:", ggufs)
    if not ggufs:
        raise SystemExit("No .gguf files found in the repo. Aborting.")

    filename = pick_gguf_from_list(ggufs, select_strategy)
    _log("Selected filename:", filename)

    _log("Downloading to HF cache using hf_hub_download ... (this may take a while)")
    cached = hf_hub_download_compat(
        repo_id=repo_id,
        filename=filename,
        token=(HF_TOKEN if USE_AUTH else None),
    )
    _log("hf_hub_download returned cached path:", cached)

    ensure_parent(MODEL_DEST_PATH)
    if os.path.abspath(cached) != os.path.abspath(MODEL_DEST_PATH):
        _log("Copying model to desired path:", MODEL_DEST_PATH)
        shutil.copy2(cached, MODEL_DEST_PATH)
    else:
        _log("Cached path already matches desired path.")

    return MODEL_DEST_PATH


# ----------------- Inference test helpers (FIXED parsing + streaming handling) -----------------


def _extract_text(obj: Any) -> str:
    """Robustly extract textual content from many common LLM response shapes.

    Handles:
    - plain strings/bytes
    - dicts with keys: text, content, message, delta, output, data
    - OpenAI/Chat-style {'choices': [{'message': {'content': ...}}]}
    - streaming chunks like {'choices':[{'delta':{'content':'...'}}]}
    - nested lists
    Returns concatenated text (may include newlines).
    """

    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, list):
        return "".join(_extract_text(x) for x in obj)
    if isinstance(obj, dict):
        # Priority: direct text/content fields
        for k in ("text", "content", "answer", "response", "output"):
            if k in obj and obj[k] is not None:
                return _extract_text(obj[k])
        # message -> content
        if "message" in obj and obj["message"] is not None:
            # message may be a dict with 'content' or string
            return _extract_text(obj["message"])
        # delta commonly used in streaming
        if "delta" in obj and obj["delta"] is not None:
            return _extract_text(obj["delta"])
        # choices: concatenate each choice
        if "choices" in obj and obj["choices"] is not None:
            parts = []
            try:
                for choice in obj["choices"]:
                    parts.append(_extract_text(choice))
                return "".join(parts)
            except Exception:
                # fallback to stringifying
                return str(obj["choices"]) if obj["choices"] is not None else ""
        # If none matched, try concatenating values (preserve order) to avoid losing text
        try:
            vals = []
            for v in obj.values():
                vals.append(_extract_text(v))
            return "".join(vals)
        except Exception:
            return str(obj)
    # fallback
    try:
        return str(obj)
    except Exception:
        return ""


def parse_llama_cpp_response(resp: Any) -> str:
    """
    Try multiple common response shapes from llama-cpp-python and related wrappers.
    This version uses _extract_text to preserve multi-chunk/streamed responses and nested
    message/delta shapes.
    """
    try:
        return _extract_text(resp)
    except Exception:
        try:
            return str(resp)
        except Exception:
            return "<unparseable response>"


def _attempt_method_calls(
    llm, prompt: str, max_tokens: int, temp: float
) -> Optional[str]:
    """
    Try several method call shapes in order. Returns text or None.
    Logs attempted signatures for debugging.
    """
    attempts = []

    def try_call(func, *args, **kwargs):
        name = getattr(func, "__name__", repr(func))
        attempts.append((name, args, kwargs))
        try:
            out = func(*args, **kwargs)

            # If out is a generator/iterator (streaming), gather and attempt to extract text
            if hasattr(out, "__iter__") and not isinstance(out, (str, bytes, dict)):
                collected = []
                try:
                    for chunk in out:
                        # Each chunk may be a dict-like or simple string
                        collected.append(
                            _extract_text(chunk)
                            if not isinstance(chunk, str)
                            else chunk
                        )
                except TypeError:
                    # not actually iterable
                    pass
                except Exception:
                    # fallback: stringify each chunk
                    try:
                        collected = [str(x) for x in out]
                    except Exception:
                        pass
                out = "".join(collected)

            # If out is a dict/list or other composite type, extract textual content
            if not isinstance(out, (str, bytes)):
                out = _extract_text(out)

            return out
        except TypeError as te:
            _log(f"TypeError calling {name}: {te}")
            return te
        except Exception as e:
            _log(f"Exception calling {name}: {e}")
            return e

    # Candidate call shapes (order matters)
    # 1) create(prompt=..., max_tokens=..., temperature=...)
    if hasattr(llm, "create"):
        _log("Attempting llm.create with various arg names...")
        # try many plausible kwarg combos
        cand_kwargs = [
            {"prompt": prompt, "max_tokens": max_tokens, "temperature": temp},
            {"prompt": prompt, "max_new_tokens": max_tokens, "temperature": temp},
            {"prompt": prompt, "n_predict": max_tokens, "temperature": temp},
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temp,
            },
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_new_tokens": max_tokens,
                "temperature": temp,
            },
            {"prompt": prompt},  # minimal
        ]
        for kw in cand_kwargs:
            res = try_call(llm.create, **kw)
            if not isinstance(res, (TypeError, Exception)) and res is not None:
                _log("llm.create succeeded with kwargs:", kw.keys())
                return parse_llama_cpp_response(res)

    # 2) generate(prompt, ...) positional-first style
    if hasattr(llm, "generate"):
        _log("Attempting llm.generate with various arg names/positions...")
        try_order = []
        # try positional first
        try_order.append(
            ("positional", (prompt,), {"max_tokens": max_tokens, "temperature": temp})
        )
        try_order.append(
            (
                "kwargs_max_tokens",
                (),
                {"prompt": prompt, "max_tokens": max_tokens, "temperature": temp},
            )
        )
        try_order.append(
            (
                "kwargs_max_new_tokens",
                (),
                {"prompt": prompt, "max_new_tokens": max_tokens, "temperature": temp},
            )
        )
        try_order.append(
            (
                "kwargs_n_predict",
                (),
                {"prompt": prompt, "n_predict": max_tokens, "temperature": temp},
            )
        )
        try_order.append(("prompt_only", (), {"prompt": prompt}))
        for tag, args, kw in try_order:
            res = try_call(llm.generate, *args, **kw)
            if not isinstance(res, (TypeError, Exception)) and res is not None:
                _log("llm.generate succeeded with attempt:", tag)
                return parse_llama_cpp_response(res)

    # 3) Try calling the instance directly: llm(prompt) or llm(prompt=..)
    _log("Attempting to call llm as callable...")
    try:
        res = try_call(llm, prompt)
        if not isinstance(res, (TypeError, Exception)) and res is not None:
            _log("llm(...) callable succeeded (positional).")
            return parse_llama_cpp_response(res)
    except Exception as e:
        _log("callable attempt raised:", e)

    try:
        res = try_call(llm, prompt=prompt, max_tokens=max_tokens, temperature=temp)
        if not isinstance(res, (TypeError, Exception)) and res is not None:
            _log("llm(...) callable succeeded (kwargs).")
            return parse_llama_cpp_response(res)
    except Exception as e:
        _log("callable kwargs attempt raised:", e)

    # If none worked, log all attempts for debugging
    _log("All llama-cpp call attempts failed. Attempts log (truncated):")
    for a in attempts:
        _log(
            " -",
            a[0],
            "args:",
            a[1],
            "kwargs keys:",
            list(a[2].keys()) if isinstance(a[2], dict) else a[2],
        )
    return None


def test_with_llama_cpp(
    model_path: str, prompt: str, max_tokens: int = 64, temp: float = 0.0
) -> Optional[str]:
    """
    Try to use llama-cpp-python (llama_cpp package) to load the gguf and get a response.
    Robustly handles multiple API shapes and kwarg names.
    """
    try:
        from llama_cpp import Llama
    except Exception as e:
        _log("llama_cpp not installed or failed import:", e)
        return None

    try:
        _log("Loading model with llama_cpp:", model_path)
        llm = Llama(model_path=model_path)
        _log("Model loaded with llama_cpp. Trying to create/generate response...")
        text = _attempt_method_calls(llm, prompt, max_tokens, temp)
        if text is not None:
            _log("llama-cpp call produced output (full):")
            _log(str(text))
            return str(text)
        _log(
            "No usable response from llama-cpp-python instance after trying known call shapes."
        )
        return None
    except Exception as e:
        _log("Error loading model / running inference with llama_cpp:", e)
        return None


def find_llama_cpp_main_executable() -> Optional[str]:
    candidates = ["main", "main.exe"]
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for p in paths:
        for c in candidates:
            fp = os.path.join(p, c)
            if os.path.isfile(fp) and os.access(fp, os.X_OK):
                return fp
    for c in candidates:
        for root in (os.getcwd(), os.path.dirname(os.getcwd())):
            fp = os.path.join(root, c)
            if os.path.isfile(fp) and os.access(fp, os.X_OK):
                return fp
    return None


def test_with_llama_cpp_binary(
    model_path: str, prompt: str, exec_path: Optional[str] = None
) -> Optional[str]:
    if exec_path is None:
        exec_path = find_llama_cpp_main_executable()
    if exec_path is None:
        _log("No llama.cpp 'main' executable found on PATH or current dir.")
        return None

    _log("Using llama.cpp binary:", exec_path)
    cmd = [exec_path, "-m", model_path, "-p", prompt, "-n", str(TEST_MAX_TOKENS)]
    try:
        _log("Running:", " ".join(cmd))
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300
        )
        if proc.returncode != 0:
            _log(
                "llama.cpp binary returned non-zero code. stderr (full):",
                proc.stderr,
            )
            return None
        out = proc.stdout
        _log("llama.cpp binary output (full):")
        _log(out)
        return out
    except Exception as e:
        _log("Error running llama.cpp binary:", e)
        return None


# ----------------- Orchestration -----------------


def file_is_present_and_nonzero(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def run_post_success(cmd: str):
    if not cmd:
        return
    _log("Running post-success command:", cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
        _log("Post-success command finished.")
    except Exception as e:
        _log("Post-success command failed:", e)


def main():
    start = time.time()
    try:
        model_path = MODEL_DEST_PATH
        # If the model is already there and non-zero, skip download
        if file_is_present_and_nonzero(model_path):
            _log("Model file already exists at desired path:", model_path)
        else:
            _log("Model not present. Beginning download...")
            model_path = download_model_via_hf(REPO_ID, SELECT_STRATEGY)
            _log("Downloaded/copied model to:", model_path)

        if not file_is_present_and_nonzero(model_path):
            _log("Model file missing or zero bytes after download. Aborting test.")
            sys.exit(2)

        _log("Attempting inference test using llama-cpp-python (preferred).")
        resp = test_with_llama_cpp(
            model_path, TEST_PROMPT, max_tokens=TEST_MAX_TOKENS, temp=TEST_TEMPERATURE
        )
        if resp and len(str(resp).strip()) > 0:
            _log("SUCCESS: model responded via llama-cpp-python.")
            _log("Response (full):")
            _log(str(resp))
            if POST_SUCCESS_CMD:
                run_post_success(POST_SUCCESS_CMD)
            _log("Total time: {:.1f}s".format(time.time() - start))
            sys.exit(0)

        _log(
            "Primary test failed or llama-cpp-python not available. Trying external llama.cpp binary fallback."
        )
        resp2 = test_with_llama_cpp_binary(model_path, TEST_PROMPT)
        if resp2 and len(str(resp2).strip()) > 0:
            _log("SUCCESS: model responded via llama.cpp binary.")
            _log("Response (full):")
            _log(str(resp2))
            if POST_SUCCESS_CMD:
                run_post_success(POST_SUCCESS_CMD)
            _log("Total time: {:.1f}s".format(time.time() - start))
            sys.exit(0)

        _log("FAIL: model did not produce a usable response with available runners.")
        _log("Hints:")
        _log(" - Install llama-cpp-python: pip install llama-cpp-python")
        _log(" - Or build/run llama.cpp `main` binary and ensure it's on PATH")
        _log(
            " - If the model is too large for your machine (RAM), try a quantized Q8_0 or f16 variant."
        )
        sys.exit(3)

    except KeyboardInterrupt:
        _log("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        _log("Unexpected error:", e)
        sys.exit(4)


if __name__ == "__main__":
    main()
