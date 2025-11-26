#!/usr/bin/env python3
"""
merged_local_runner.py

Orchestrator script that:
  1) uses the HF download / selection logic from `test_local_llm.py`
     to ensure a model is available on disk (keeps all original behavior).
  2) uses `LocalLLM` and `StoryPipeline` from `local_llm.py`
     (keeps the entire feature set of local_llm intact by importing it).

This file does not modify the original modules; it imports and orchestrates them.
"""

from __future__ import annotations
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

# Ensure the directory containing the uploaded files is on sys.path.
# Adjust this if your files are in another folder.
UPLOAD_DIR = "/mnt/data"
if UPLOAD_DIR not in sys.path:
    sys.path.insert(0, UPLOAD_DIR)

# Import the two modules we merged (these are the original uploaded files).
# They contain all their original functions/classes (download_model_via_hf, LocalLLM, StoryPipeline, ...)
try:
    import test_local_llm as downloader  # provides download_model_via_hf, MODEL_DEST_PATH, etc.
except Exception as e:
    raise RuntimeError(
        "Failed to import test_local_llm module from /mnt/data. Make sure test_local_llm.py exists."
    ) from e

try:
    import local_llm as llm_mod  # provides LocalLLM and StoryPipeline
except Exception as e:
    raise RuntimeError(
        "Failed to import local_llm module from /mnt/data. Make sure local_llm.py exists."
    ) from e


def file_is_present_and_nonzero(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def ensure_model_present(
    model_dest_path: str,
    repo_id: str,
    select_strategy: str,
    use_auth: bool = True,
) -> str:
    """
    Ensure model exists at model_dest_path. If not, call test_local_llm.download_model_via_hf
    which follows the original download logic. Returns the final path to the .gguf
    model (inside the created model folder).
    """
    model_path = model_dest_path

    if file_is_present_and_nonzero(model_path):
        downloader._log("Model file already exists at desired path:", model_path)
        # If the model_path is a gguf inside a folder, return it.
        return model_path

    downloader._log(
        "Model not present. Beginning HF download via test_local_llm logic..."
    )
    final_model_path = downloader.download_model_via_hf(repo_id, select_strategy)
    downloader._log("download_model_via_hf returned:", final_model_path)

    if not file_is_present_and_nonzero(final_model_path):
        raise RuntimeError(
            f"Downloaded model path {final_model_path} missing or zero bytes after download."
        )
    return final_model_path


def run_inference_tests(model_path: str, prompt: str, max_tokens: int, temp: float):
    """
    Run the inference tests provided by test_local_llm: try python llama-cpp first,
    then the llama.cpp binary fallback. Mirrors original script behavior.
    """
    downloader._log("Attempting inference test using llama-cpp-python (preferred).")
    resp = None
    try:
        resp = downloader.test_with_llama_cpp(
            model_path, prompt, max_tokens=max_tokens, temp=temp
        )
    except Exception as e:
        downloader._log("test_with_llama_cpp raised:", e)

    if resp and len(str(resp).strip()) > 0:
        downloader._log("SUCCESS: model responded via llama-cpp-python.")
        downloader._log("Response (preview):")
        downloader._log(str(resp)[:800])
        if getattr(downloader, "POST_SUCCESS_CMD", None):
            downloader.run_post_success(downloader.POST_SUCCESS_CMD)
        return True

    downloader._log(
        "Primary test failed or llama-cpp-python not available. Trying external llama.cpp binary fallback."
    )
    try:
        resp2 = downloader.test_with_llama_cpp_binary(
            model_path, downloader.TEST_PROMPT
        )
    except Exception as e:
        downloader._log("test_with_llama_cpp_binary raised:", e)
        resp2 = None

    if resp2 and len(str(resp2).strip()) > 0:
        downloader._log("SUCCESS: model responded via llama.cpp binary.")
        downloader._log("Response (preview):")
        downloader._log(str(resp2)[:800])
        if getattr(downloader, "POST_SUCCESS_CMD", None):
            downloader.run_post_success(downloader.POST_SUCCESS_CMD)
        return True

    downloader._log(
        "FAIL: model did not produce a usable response with available runners."
    )
    return False


def run_story_pipeline_example(model_folder: str, timeout: int = 120):
    """
    Instantiate StoryPipeline using the downloaded model folder and run an example
    generation to demonstrate LocalLLM integration. This uses the full StoryPipeline
    and LocalLLM functionality unchanged.
    """
    downloader._log("Instantiating StoryPipeline with model_dir:", model_folder)
    # model_folder should be a directory containing the gguf and other artifacts
    pipeline = llm_mod.StoryPipeline(default_timeout=10, model_dir=model_folder)

    downloader._log(
        "Generating script (this may take a while depending on model/backend)..."
    )
    try:
        script = pipeline.generate_script(
            niche="Children's bedtime",
            person="",
            timing_minutes=10,
            timeout=timeout,
            topic="The Little Cloud Painter",
        )
        downloader._log("\n--- Script (preview) ---")
        downloader._log(script[:800])
    except Exception as e:
        downloader._log("StoryPipeline.generate_script failed:", e)
        raise

    # Optionally create image prompts, narration, metadata (keeps original behavior)
    try:
        image_prompts = pipeline.generate_image_prompts(
            script_text=script,
            theme="watercolor, children's book, whimsical",
            img_number=10,
            batch_size=5,
            timeout_per_call=60,
        )
        downloader._log(f"Generated {len(image_prompts)} image prompts (sample):")
        for p in image_prompts[:3]:
            downloader._log(f"[{p}]")
    except Exception as e:
        downloader._log("generate_image_prompts failed (continuing):", e)

    try:
        narration_text = pipeline.generate_narration(script, timing_minutes=10)
        downloader._log("Narration generated and saved.")
    except Exception as e:
        downloader._log("generate_narration failed (continuing):", e)

    try:
        yt_meta = pipeline.generate_youtube_metadata(script, timing_minutes=10)
        downloader._log("YouTube metadata generated and saved.")
    except Exception as e:
        downloader._log("generate_youtube_metadata failed (continuing):", e)


def main(argv: Optional[list] = None):
    p = argparse.ArgumentParser(
        description="Merged runner: download model (test_local_llm) then use LocalLLM/StoryPipeline (local_llm)."
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Skip HF download; assume model already present.",
    )
    p.add_argument(
        "--run-tests",
        action="store_true",
        help="Run inference tests (llama-cpp & llama.cpp binary fallback) after download.",
    )
    p.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run StoryPipeline example generation after download.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout (in seconds) for pipeline.generate_script.",
    )
    args = p.parse_args(argv)

    start = time.time()

    # Read config values from downloader module (preserves behavior)
    MODEL_DEST_PATH = getattr(downloader, "MODEL_DEST_PATH", None)
    REPO_ID = getattr(downloader, "REPO_ID", None)
    SELECT_STRATEGY = getattr(downloader, "SELECT_STRATEGY", "auto")
    TEST_PROMPT = getattr(downloader, "TEST_PROMPT", "Write me a script.")
    TEST_MAX_TOKENS = getattr(downloader, "TEST_MAX_TOKENS", 1024)
    TEST_TEMPERATURE = getattr(downloader, "TEST_TEMPERATURE", 0.0)

    if MODEL_DEST_PATH is None:
        raise RuntimeError("MODEL_DEST_PATH not defined in test_local_llm module.")

    # Ensure model present (download unless --no-download)
    final_model_path = MODEL_DEST_PATH
    try:
        if not args.no_download:
            final_model_path = ensure_model_present(
                MODEL_DEST_PATH,
                REPO_ID,
                SELECT_STRATEGY,
                use_auth=getattr(downloader, "USE_AUTH", True),
            )
        else:
            downloader._log(
                "--no-download passed; skipping download. Using MODEL_DEST_PATH as-is."
            )
            if not file_is_present_and_nonzero(final_model_path):
                raise RuntimeError(
                    f"--no-download passed but model file at MODEL_DEST_PATH ({final_model_path}) is missing or zero bytes."
                )
    except Exception as e:
        downloader._log("Model fetch failed:", e)
        raise

    # model_folder is the directory containing the gguf and artifacts (LocalLLM expects a folder)
    model_folder = str(Path(final_model_path).parent)

    downloader._log("Final model path:", final_model_path)
    downloader._log("Model folder (for LocalLLM):", model_folder)
    downloader._log("Elapsed until download:", f"{time.time()-start:.1f}s")

    # Optionally run the inference tests from test_local_llm
    if args.run_tests:
        success = run_inference_tests(
            final_model_path, TEST_PROMPT, TEST_MAX_TOKENS, TEST_TEMPERATURE
        )
        if not success:
            downloader._log(
                "Inference tests failed. You can still try StoryPipeline, but it may not run if backends are missing."
            )

    # Optionally run the StoryPipeline example (this uses LocalLLM inside)
    if args.run_pipeline:
        run_story_pipeline_example(model_folder, timeout=args.timeout)

    downloader._log("Total time: {:.1f}s".format(time.time() - start))


if __name__ == "__main__":
    main()
