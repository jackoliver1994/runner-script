#!/usr/bin/env python3
"""
test_local_llm.py

- Auto-detects system resources (RAM, CPU flags, GPU)
- Picks best-fitting .gguf in a HF repo (heuristics)
- Downloads model via huggingface_hub (HF_TOKEN from env)
- Copies to MODEL_DEST_PATH (set by env or default)
- Attempts to run a small test prompt using:
    1) llama-cpp-python (preferred, with API-variant fallbacks)
    2) llama.cpp 'main' binary (fallback)
- Exit codes:
    0  -> success (model responded)
    2  -> model missing or zero bytes after download
    3  -> model did not produce a usable response
    4  -> unexpected error
    130-> interrupted by user
"""

from __future__ import annotations
import os
import sys
import time
import shutil
import subprocess
from typing import List, Optional, Tuple, Dict

# ----------------- CONFIG (read from env; override here if needed) -----------------
HF_TOKEN: str = os.getenv("HF_TOKEN", "")  # Provided by GitHub workflow (hf_key)
REPO_ID: str = os.getenv("REPO_ID", "ggml-org/Mistral-Small-3.1-24B-Instruct-2503-GGUF")
MODEL_DEST_PATH: str = os.getenv(
    "MODEL_DEST_PATH", os.path.join(os.getcwd(), "models", "mistral-small-3.1.gguf")
)
USE_AUTH: bool = os.getenv("USE_AUTH", "true").lower() in ("1", "true", "yes")
SELECT_STRATEGY: str = os.getenv("SELECT_STRATEGY", "auto").lower()
TEST_PROMPT: str = os.getenv(
    "TEST_PROMPT",
    "Q: What is 2 + 2?\nA:(stritly answer with correct answer) and write me 3000 word script for children story about a brave little toaster.\n",
)
TEST_MAX_TOKENS: int = int(os.getenv("TEST_MAX_TOKENS", "64"))
TEST_TEMPERATURE: float = float(os.getenv("TEST_TEMPERATURE", "0.0"))
VERBOSE: bool = os.getenv("VERBOSE", "true").lower() in ("1", "true", "yes")
POST_SUCCESS_CMD: Optional[str] = os.getenv("POST_SUCCESS_CMD", None)
# -----------------------------------------------------------------------------------


def _log(*a, **k):
    if VERBOSE:
        print(*a, **k)


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


def choose_gguf_from_candidates(
    candidates: List[str], repo_id: str, strategy: str = "auto"
) -> str:
    """
    Strategy 'auto' picks based on:
      - GPU with VRAM -> prefer fp16/f16/float variants
      - Low system RAM (<=16GB) -> prefer q8*, q4_* smallest
      - High RAM (>=64GB) -> prefer largest (non-quantized) if available
    You can still override via SELECT_STRATEGY env to 'first', 'smallest', 'largest'.
    """
    if not candidates:
        raise SystemExit("No .gguf candidates provided to chooser.")

    total_ram = get_total_ram_gb()
    cpu_flags = detect_cpu_flags()
    gpu = detect_gpu_info()
    _log(f"System: RAM={total_ram:.1f}GB, CPU flags={cpu_flags}, GPU={gpu}")

    # quick categorization of candidates
    categorized = []
    api = None
    try:
        api = HfApi()
        info = api.repo_info(repo_id, token=HF_TOKEN if USE_AUTH else None)
        siblings = getattr(info, "siblings", None)
    except Exception:
        siblings = None

    # Attempt to get file sizes when possible (for 'smallest'/'largest' decisions)
    size_map = {}
    if siblings:
        for s in siblings:
            name = getattr(s, "rfilename", None)
            if name and (name in candidates):
                size_map[name] = getattr(s, "size", 0)

    for c in candidates:
        tags = parse_variant_tags(c)
        sz = size_map.get(c, None)
        categorized.append((c, tags, sz))

    # If explicit strategies requested
    s = strategy.lower()
    if s in ("first", "smallest", "largest"):
        if s == "first":
            return categorized[0][0]
        if s == "smallest":
            # prefer q* or f16 small names then file size
            prioritized = sorted(
                categorized,
                key=lambda it: (
                    (
                        0
                        if any(
                            (t.startswith("q") or "q4" in t or "q8" in t) for t in it[1]
                        )
                        else 1
                    ),
                    it[2] or 1e12,
                ),
            )
            return prioritized[0][0]
        if s == "largest":
            # choose by file size if available else first
            sorted_by_size = sorted(categorized, key=lambda it: -(it[2] or 0))
            return sorted_by_size[0][0]

    # AUTO strategy
    # If GPU with >=12GB VRAM: prefer f16/fp16 or non-quantized
    if gpu and gpu.get("vram_gb", 0) >= 12:
        # prefer f16 variants
        for name, tags, _ in categorized:
            if any(t in ("f16", "fp16", "f32", "bf16") for t in tags):
                return name
        # else pick the largest by size if available
        if size_map:
            return max(categorized, key=lambda it: it[2] or 0)[0]
        return categorized[0][0]

    # If low RAM (<16GB) or no GPU -> prefer q8/q4 variants (smallest)
    if total_ram and total_ram < 16 or not gpu:
        # look for q4/q8 variants
        q_candidates = [
            it for it in categorized if any(t.startswith("q") for t in it[1])
        ]
        if q_candidates:
            # choose smallest file size if possible or shortest name
            if size_map:
                return min(q_candidates, key=lambda it: it[2] or 1e12)[0]
            return sorted(q_candidates, key=lambda it: len(it[0]))[0]
        # fallback: choose smallest by file size if available
        if size_map:
            return min(categorized, key=lambda it: it[2] or 1e12)[0]
        return categorized[0][0]

    # Default fallback: pick first
    return categorized[0][0]


# ----------------- HF download helpers -----------------


def hf_hub_download_compat(
    repo_id: str, filename: str, token: Optional[str] = None, **kwargs
) -> str:
    """
    Call hf_hub_download using the correct kwarg name for the installed huggingface_hub:
      - newer versions expect token=<token>
      - older versions expect use_auth_token=<token>
    Falls back automatically.
    """
    try:
        # try the modern signature first
        return hf_hub_download(
            repo_id=repo_id, filename=filename, token=token, **kwargs
        )
    except TypeError:
        # older huggingface_hub versions used use_auth_token
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
        # Use earlier logic but simpler
        if s == "first":
            return candidates[0]
        if s == "smallest":
            # simple: prefer filenames containing q8 or q4
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

    # else 'auto' or unknown -> use smarter chooser
    return choose_gguf_from_candidates(candidates, REPO_ID, strategy or "auto")


def download_model_via_hf(repo_id: str, select_strategy) -> str:
    """
    Uses hf_hub_download to download into HF cache and returns final path (copied to MODEL_DEST_PATH).
    """
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


# ----------------- Inference test helpers -----------------


def parse_llama_cpp_response(resp) -> str:
    """Try common response shapes from llama-cpp-python."""
    try:
        # older/newer shapes
        if isinstance(resp, dict):
            if "choices" in resp and len(resp["choices"]) > 0:
                c = resp["choices"][0]
                if isinstance(c, dict):
                    if "text" in c:
                        return str(c["text"])
                    if "message" in c and isinstance(c["message"], dict):
                        return str(c["message"].get("content", ""))
                return str(c)
        return str(resp)
    except Exception:
        return str(resp)


def test_with_llama_cpp(
    model_path: str, prompt: str, max_tokens: int = 64, temp: float = 0.0
) -> Optional[str]:
    """
    Try to use llama-cpp-python (llama_cpp package) to load the gguf and get a response.
    Handles a few API variants / fallbacks.
    """
    try:
        from llama_cpp import Llama
    except Exception as e:
        _log("llama_cpp not installed or failed import:", e)
        return None

    try:
        _log("Loading model with llama_cpp:", model_path)
        # instantiate Llama; the library will load the GGUF
        llm = Llama(model_path=model_path)
        _log("Model loaded with llama_cpp. Trying to create/generate response...")
        # Try several API variants:
        try:
            resp = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temp)
            text = parse_llama_cpp_response(resp)
            _log("llama_cpp.create success (truncated):", text[:300])
            return text
        except AttributeError:
            _log("llama_cpp instance has no method 'create'. Trying 'generate' ...")
            try:
                resp = llm.generate(prompt, max_tokens=max_tokens, temperature=temp)
                # some builds return a generator; others return object
                if isinstance(resp, dict) and "choices" in resp:
                    text = parse_llama_cpp_response(resp)
                else:
                    text = str(resp)
                _log("llama_cpp.generate success (truncated):", text[:300])
                return text
            except Exception as e2:
                _log("llama_cpp.generate failed:", e2)
                # try call as function (some older wrappers)
                try:
                    resp = llm(prompt)
                    text = parse_llama_cpp_response(resp)
                    _log(
                        "llama_cpp callable produced response (truncated):", text[:300]
                    )
                    return text
                except Exception as e3:
                    _log("llama_cpp callable failed:", e3)
                    return None
        except Exception as e:
            _log("llama_cpp.create returned error:", e)
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
                "llama.cpp binary returned non-zero code. stderr (truncated):",
                proc.stderr[:1000],
            )
            return None
        out = proc.stdout.strip()
        _log("llama.cpp binary output (truncated):", out[:800])
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
            _log("Response (first 400 chars):")
            _log(str(resp)[:400])
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
            _log("Response (first 400 chars):")
            _log(str(resp2)[:400])
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
