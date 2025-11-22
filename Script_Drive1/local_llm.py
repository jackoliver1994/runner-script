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

import time
import sys
import os
import re
import difflib
from pathlib import Path
from typing import Optional, List
from threading import Thread, Event
from datetime import datetime
from transformers import AutoTokenizer


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


# ---------------------- LOCAL LLM BACKEND (context-aware + chunked/unlimited generation) ----------------------
class LocalLLM:
    """
    Robust LocalLLM (drop-in):
     - auto-detects model files
     - attempts to raise n_ctx based on model metadata (progressive fallback)
     - configures threading (OMP_NUM_THREADS / n_threads) for llama-cpp if available
     - supports chunked 'unlimited' generation by auto-continuing until stop
     - keeps the same send_message(...) signature used by the rest of the code
    Usage (existing code unchanged):
       local = LocalLLM(model_dir="/home/runner/work/runner-script/runner-script/models")
       resp = local.send_message(prompt, timeout=120, max_new_tokens=1024, unlimited=True)
    """

    MODEL_EXTENSIONS = [
        ".gguf",
        ".ggml",
        ".bin",
        ".pt",
        ".pth",
        ".safetensors",
        ".model",
    ]

    def __init__(
        self,
        model_dir: str = "/home/runner/work/runner-script/runner-script/models",
        default_timeout: int = 10,
    ):
        self.model_dir = str(model_dir)
        self.base_timeout = default_timeout
        self.backend = None
        self.model_path = None
        self._impl = None
        self._n_ctx = None
        self._avg_token_char = 4
        self._detect_and_init()

    def _call_with_alarm_timeout(
        self, fn, *args, timeout: Optional[int] = None, **kwargs
    ):
        """
        Try to run fn(*args, **kwargs) but raise TimeoutError if it doesn't return
        within `timeout` seconds. Uses signal.alarm on Unix (works only in main thread).
        On Windows (os.name == 'nt') or if signal isn't available, it will just call fn.
        This is a pragmatic guard to fail fast when a backend call hangs.
        """
        if not timeout or timeout <= 0:
            return fn(*args, **kwargs)

        if os.name == "nt":
            # signal.alarm not available on Windows — call directly (no hard timeout)
            return fn(*args, **kwargs)

        import signal

        def _handler(signum, frame):
            raise TimeoutError("generation call timed out")

        old_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, int(timeout))
            return fn(*args, **kwargs)
        finally:
            # clear alarm and restore handler
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
            signal.signal(signal.SIGALRM, old_handler)

    def _ensure_tokenizer_for_model(self):
        """
        Ensure we have a cached tokenizer for the local model.
        Uses self.model (HF id) or self.model_path. Fails silently and leaves
        self._hf_tokenizer = None if transformers is not available.
        """
        if getattr(self, "_hf_tokenizer", None) is not None:
            return
        self._hf_tokenizer = None
        model_id = getattr(self, "model", None) or getattr(self, "model_path", None)
        if model_id is None:
            return
        if AutoTokenizer is None:
            # transformers not installed or import failed
            return
        try:
            # don't crash if tokenizer download fails
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as e:
            print(f"LocalLLM: tokenizer load failed for {model_id}: {e}")
            self._hf_tokenizer = None

    def compute_dynamic_max_new_tokens(
        self, prompt: str, safety_factor: float = 0.80, hard_cap: Optional[int] = None
    ) -> int:
        """
        Compute a safe max_new_tokens based on:
        - runtime context length (self._n_ctx if detected, else 131072)
        - tokenized prompt length (using HF tokenizer if available)
        - safety_factor (0 < safety_factor <= 1) to leave headroom for kv cache
        - optional hard_cap to never exceed a specific value
        Returns an integer >= 1.
        """
        # 1) get model context
        n_ctx = getattr(self, "_n_ctx", None) or 131072

        # 2) ensure tokenizer cached
        self._ensure_tokenizer_for_model()

        prompt_tokens = None
        if getattr(self, "_hf_tokenizer", None) is not None:
            try:
                prompt_tokens = len(self._hf_tokenizer(prompt)["input_ids"])
            except Exception as e:
                # fallback to estimate
                print(f"LocalLLM: tokenizer call failed, falling back to estimate: {e}")
                prompt_tokens = None

        if prompt_tokens is None:
            # rough fallback: estimate ~4 characters per token for english text
            prompt_tokens = max(1, len(prompt) // 4)

        # compute raw available tokens
        available = max(0, n_ctx - prompt_tokens)

        # apply safety factor so we don't fill the entire context (avoid OOM / kv growth)
        safe_allowed = int(available * float(max(0.01, min(1.0, safety_factor))))

        # enforce a minimum and optional hard cap
        MIN_NEW_TOKENS = 8
        if hard_cap is not None:
            safe_allowed = min(safe_allowed, int(hard_cap))
        safe_allowed = max(MIN_NEW_TOKENS, safe_allowed)

        # As an extra conservative guard, clamp to a reasonable ceiling for local runs if not explicitly requested
        # (you can change/remove this if you want to allow full 128k generations)
        LOCAL_CONSERVATIVE_CEILING = (
            getattr(self, "local_max_new_tokens_ceiling", None) or 65536
        )
        safe_allowed = min(safe_allowed, int(LOCAL_CONSERVATIVE_CEILING))

        return safe_allowed

    # ---------------- discover model file ----------------
    def _find_model_file(self) -> Optional[str]:
        p = Path(self.model_dir)
        if not p.exists():
            return None
        for ext in self.MODEL_EXTENSIONS:
            items = list(p.rglob(f"*{ext}"))
            if items:
                items_sorted = sorted(
                    items, key=lambda x: x.stat().st_size, reverse=True
                )
                return str(items_sorted[0])
        allfiles = [f for f in p.rglob("*") if f.is_file()]
        if allfiles:
            return str(
                sorted(allfiles, key=lambda x: x.stat().st_size, reverse=True)[0]
            )
        return None

    # ---------------- init backends (try to set n_ctx progressively) ----------------
    def _detect_and_init(self):
        model_file = self._find_model_file()
        if not model_file:
            raise RuntimeError(f"No model files found under {self.model_dir}.")
        self.model_path = model_file

        # prefer llama-cpp-python (for gguf/ggml)
        try:
            from llama_cpp import Llama  # type: ignore

            print(f"LocalLLM: attempting llama-cpp-python init with {self.model_path}")
            cpu_count = os.cpu_count() or 1
            os.environ.setdefault("OMP_NUM_THREADS", str(max(1, cpu_count - 1)))

            # Try a single, safe Llama init (no insane n_ctx values).
            try:
                self._impl = Llama(model_path=self.model_path, n_threads=cpu_count)
            except TypeError:
                # older llama-cpp-python might not accept n_threads
                self._impl = Llama(model_path=self.model_path)

            self._n_ctx = self._detect_n_ctx()
            print(f"LocalLLM: llama-cpp init succeeded -> runtime n_ctx={self._n_ctx}")
            self.backend = "llama_cpp"
            return
        except Exception as e:
            print(f"LocalLLM: llama-cpp-python not available or init failed: {e}")

        # try gpt4all
        try:
            from gpt4all import GPT4All  # type: ignore

            print(f"LocalLLM: attempting gpt4all init with {self.model_path}")
            try:
                self._impl = GPT4All(model=self.model_path)
                self.backend = "gpt4all"
                self._n_ctx = self._detect_n_ctx() or 2048
                return
            except Exception as e:
                print(f"LocalLLM: gpt4all init failed: {e}")
        except Exception:
            pass

        # try transformers pipeline fallback
        try:
            from transformers import pipeline  # type: ignore

            print(f"LocalLLM: attempting transformers pipeline with {self.model_path}")
            try:
                gen = pipeline(
                    "text-generation", model=self.model_path, device_map=None
                )
                self._impl = gen
                self.backend = "transformers"
                self._n_ctx = self._detect_n_ctx() or 2048
                return
            except Exception as e:
                print(f"LocalLLM: transformers init failed: {e}")
        except Exception:
            pass

        raise RuntimeError(
            "LocalLLM: could not initialize any local backend. Install llama-cpp-python, gpt4all, or transformers+torch."
        )

    # ---------------- utility token counting + context detection ----------------
    def _detect_n_ctx(self) -> int:
        if not self._impl:
            return 512
        # common fields
        for attr in (
            "n_ctx",
            "context_size",
            "context_length",
            "n_ctx_max",
            "n_ctx_train",
        ):
            val = getattr(self._impl, attr, None)
            if isinstance(val, int) and val > 0:
                return int(val)
        # sometimes llama-cpp has model.meta or model.model_metadata
        try:
            meta = getattr(self._impl, "model", None)
            if meta is not None:
                # attempt various introspections
                for maybe in ("n_ctx", "context_length", "metadata", "info"):
                    m = getattr(meta, maybe, None)
                    if isinstance(m, int) and m > 0:
                        return int(m)
                # if metadata dict exists
                md = getattr(meta, "metadata", None) or getattr(
                    meta, "model_meta", None
                )
                if isinstance(md, dict):
                    val = (
                        md.get("llama.context_length")
                        or md.get("context_length")
                        or md.get("n_ctx")
                    )
                    if val:
                        try:
                            return int(val)
                        except Exception:
                            pass
        except Exception:
            pass
        # final fallback
        return 512

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            # llama-cpp tokenize if available
            if hasattr(self._impl, "tokenize"):
                try:
                    toks = self._impl.tokenize(text)
                except Exception:
                    toks = self._impl.tokenize(text.encode("utf-8"))
                return len(toks)
        except Exception:
            pass
        return max(1, int(len(text) / self._avg_token_char))

    # ---------------- prompt trimming strategy (keep recent context) ----------------
    def _trim_prompt_to_fit(self, prompt: str, max_new_tokens: int) -> str:
        n_ctx = self._n_ctx or 512
        safety = 8
        allowed = n_ctx - max_new_tokens - safety
        if allowed <= 0:
            # if impossible, reduce max_new_tokens heuristically (caller should pick smaller chunk)
            allowed = max(32, n_ctx // 8)
        # if prompt already fits, return
        if self._count_tokens(prompt) <= allowed:
            return prompt
        # trim by paragraphs first
        parts = [p for p in prompt.split("\n\n") if p.strip() != ""]
        if not parts:
            # hard trim
            approx_chars = allowed * self._avg_token_char
            return prompt[-int(approx_chars) :]
        while parts:
            joined = "\n\n".join(parts)
            if self._count_tokens(joined) <= allowed:
                return joined
            parts.pop(0)
        # fallback hard trim
        approx_chars = allowed * self._avg_token_char
        return prompt[-int(approx_chars) :]

    # ---------------- public send_message (preserves retry loop & signature) ----------------
    def send_message(
        self,
        message: str,
        timeout: Optional[int] = None,
        retry_forever: bool = True,
        retry_on_client_errors: bool = False,
        initial_backoff: float = 1.0,
        max_backoff: float = 8.0,
        spinner_message: str = "Waiting for response.",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        n_batch: int = 1,
        unlimited: bool = False,
        chunk_size: int = 512,
        max_retries: Optional[int] = None,  # NEW: allow caller to limit retries
        safety_factor: float = 0.80,
        hard_cap: Optional[int] = None,
    ) -> str:
        """
        Generate text from the available backend. Keeps retry/backoff behavior but replaces
        the previous placeholder body with a working implementation.
        """
        timeout = timeout or getattr(self, "base_timeout", None) or 30
        attempt = 0
        backoff = initial_backoff
        spinner = LoadingSpinner(spinner_message)

        # inside send_message, near start:
        if max_new_tokens is None:
            # compute dynamically based on prompt & model
            # use explicit safety_factor and hard_cap params (defaults provided in signature)
            max_new_tokens = self.compute_dynamic_max_new_tokens(
                message, safety_factor=safety_factor, hard_cap=hard_cap
            )
            print(
                f"LocalLLM: computed dynamic max_new_tokens={max_new_tokens} (safety_factor={safety_factor})"
            )

        # convert None => infinite when retry_forever True; otherwise cap = max_retries or 5
        if retry_forever:
            cap = None
        else:
            cap = max_retries if isinstance(max_retries, int) and max_retries > 0 else 5

        while True:
            attempt += 1
            try:
                spinner.message = f"{spinner_message} (attempt {attempt})"
                spinner.start()

                # Unlimited (chunked) generation path
                if unlimited:
                    out = self._generate_unlimited(
                        prompt=message,
                        timeout=timeout,
                        chunk_size=chunk_size,
                        temperature=temperature,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        n_batch=n_batch,
                    )
                    spinner.stop()
                    return (out or "").strip()

                # One-shot generation path
                if self.backend == "llama_cpp":
                    text = self._send_llama_cpp(
                        prompt=message,
                        timeout=timeout,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        n_batch=n_batch,
                    )
                elif self.backend == "gpt4all":
                    text = self._send_gpt4all(
                        prompt=message, timeout=timeout, max_new_tokens=max_new_tokens
                    )
                elif self.backend == "transformers":
                    text = self._send_transformers(
                        prompt=message,
                        timeout=timeout,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                else:
                    spinner.stop()
                    raise RuntimeError(
                        "LocalLLM: no backend configured for send_message"
                    )

                spinner.stop()
                return (text or "").strip()

            except Exception as e:
                spinner.stop()
                print(f"⚠️ send_message failed on generation attempt {attempt}: {e}")

                # enforce hard cap if set and not retry_forever
                if cap is not None and attempt >= cap:
                    raise RuntimeError(
                        f"send_message: reached max retries ({cap}). Last error: {e}"
                    )

                # some errors (403 / client) shouldn't be retried unless explicitly allowed
                if (
                    "403" in str(e) or "client error" in str(e).lower()
                ) and not retry_on_client_errors:
                    raise

                # backoff and retry
                time.sleep(min(backoff, max_backoff))
                backoff = min(backoff * 2, max_backoff)
                # loop continues (may be infinite if cap is None)

    # --------------- chunked/unlimited generation helper ---------------
    def _generate_unlimited(
        self,
        prompt: str,
        timeout: int,
        chunk_size: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        n_batch: int,
    ) -> str:
        """
        Generate by chunks:
        1. generate chunk_size tokens
        2. append to output
        3. build a continuation prompt using the trailing context (keep <= n_ctx)
        4. repeat until model returns an end-of-text signal or a safety iteration cap
        This helps produce arbitrarily long outputs while respecting model n_ctx.
        """
        n_ctx = self._n_ctx or 512
        max_iterations = (
            200  # cap to avoid infinite runaway (user can increase if desired)
        )
        output = ""
        current_prompt = prompt
        for iteration in range(max_iterations):
            # send one chunk
            chunk = None
            try:
                if self.backend == "llama_cpp":
                    print(
                        f"LocalLLM: calling _send_llama_cpp (iteration {iteration+1}/{max_iterations})"
                    )
                    chunk = self._call_with_alarm_timeout(
                        self._send_llama_cpp,
                        current_prompt,
                        timeout,
                        chunk_size,
                        temperature,
                        top_p,
                        repeat_penalty,
                        n_batch,
                        timeout=timeout,
                    )
                    print(
                        f"LocalLLM: _send_llama_cpp returned {len(chunk) if chunk is not None else 'None'} characters"
                    )
                elif self.backend == "gpt4all":
                    print(
                        f"LocalLLM: calling _send_gpt4all (iteration {iteration+1}/{max_iterations})"
                    )
                    chunk = self._call_with_alarm_timeout(
                        self._send_gpt4all,
                        current_prompt,
                        timeout,
                        chunk_size,
                        timeout=timeout,
                    )
                    print(
                        f"LocalLLM: _send_gpt4all returned {len(chunk) if chunk is not None else 'None'} characters"
                    )
                elif self.backend == "transformers":
                    print(
                        f"LocalLLM: calling _send_transformers (iteration {iteration+1}/{max_iterations})"
                    )
                    chunk = self._call_with_alarm_timeout(
                        self._send_transformers,
                        current_prompt,
                        timeout,
                        chunk_size,
                        temperature,
                        top_p,
                        timeout=timeout,
                    )
                    print(
                        f"LocalLLM: _send_transformers returned {len(chunk) if chunk is not None else 'None'} characters"
                    )
                else:
                    raise RuntimeError(
                        "LocalLLM: unsupported backend for unlimited generation"
                    )
            except TimeoutError:
                # timed out — surface an exception so callers can retry or abort
                raise RuntimeError(
                    f"LocalLLM: backend generation timed out after {timeout} seconds"
                )
            except Exception as e:
                # show what failed, then raise so send_message will handle retry/backoff
                print(f"LocalLLM: backend generation raised an exception: {e}")
                raise

            if not chunk:
                break
            # append chunk to output
            output += chunk
            # Trim chunk and detect common end signals or repeated outputs
            chunk_stripped = chunk.strip()
            if not chunk_stripped:
                # model emitted nothing -> stop
                break

            # If the chunk ends with explicit end-of-text marker, stop.
            if chunk_stripped.endswith("") or chunk_stripped.endswith("[end]"):
                break

            # If generation is repeating the same short fragment many times, stop to avoid runaway loops.
            recent_tail = output[-512:] if len(output) > 512 else output
            if recent_tail.endswith(chunk_stripped) and len(chunk_stripped) < 64:
                # repeating a small fragment -> break to avoid infinite loops
                break

            # prepare next prompt: keep last (n_ctx // 2) tokens of combined prompt+output to retain context
            combined = (prompt + "\n" + output)[
                -(n_ctx * self._avg_token_char * 2) :
            ]  # approx chars
            # Trim to token budget for next chunk: ensure tokens(combined) + chunk_size <= n_ctx
            next_prompt = self._trim_prompt_to_fit(combined, chunk_size)
            current_prompt = next_prompt
            # small sleep to let resources stabilize
            time.sleep(0.05)
        return output

    # --------------- backend-specific generation functions ---------------
    def _send_llama_cpp(
        self,
        prompt: str,
        timeout: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        n_batch: int,
    ) -> str:
        llm = self._impl

        def _extract_text(resp):
            if resp is None:
                return ""
            if isinstance(resp, str):
                return resp
            if isinstance(resp, list):
                for it in resp:
                    if isinstance(it, dict) and "generated_text" in it:
                        return it["generated_text"]
                return str(resp[0]) if resp else ""
            if isinstance(resp, dict):
                if "choices" in resp:
                    parts = []
                    for c in resp["choices"]:
                        if isinstance(c, dict):
                            if "text" in c:
                                parts.append(c.get("text", ""))
                            elif "message" in c and isinstance(c["message"], dict):
                                parts.append(c["message"].get("content", ""))
                    return "".join(parts)
                if "text" in resp:
                    return resp["text"]
                if "content" in resp:
                    return resp["content"]
            return str(resp)

        # ensure prompt fits
        prompt = self._trim_prompt_to_fit(prompt, max_new_tokens)

        # try .create
        try:
            if hasattr(llm, "create"):
                try:
                    # modern llama-cpp-python: supports stream and callback; here call non-streaming
                    resp = llm.create(
                        prompt=prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    return _extract_text(resp).strip()
                except Exception as e:
                    print(f"LocalLLM: .create failed: {e}")
        except Exception:
            pass

        # try other APIs
        try:
            if hasattr(llm, "create_completion"):
                try:
                    resp = llm.create_completion(
                        prompt=prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    return _extract_text(resp).strip()
                except Exception as e:
                    print(f"LocalLLM: .create_completion failed: {e}")
        except Exception:
            pass

        try:
            if hasattr(llm, "create_chat_completion"):
                try:
                    msgs = [{"role": "user", "content": prompt}]
                    resp = llm.create_chat_completion(
                        messages=msgs,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    return _extract_text(resp).strip()
                except Exception as e:
                    print(f"LocalLLM: .create_chat_completion failed: {e}")
        except Exception:
            pass

        try:
            if callable(getattr(llm, "__call__", None)):
                try:
                    resp = llm(
                        prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    return _extract_text(resp).strip()
                except Exception as e:
                    print(f"LocalLLM: __call__ failed: {e}")
        except Exception:
            pass

        # low-level generate/tokenize/detokenize
        try:
            if (
                hasattr(llm, "generate")
                and hasattr(llm, "tokenize")
                and hasattr(llm, "detokenize")
            ):
                try:
                    try:
                        tokens = llm.tokenize(prompt)
                    except Exception:
                        tokens = llm.tokenize(prompt.encode("utf-8"))
                    out_tokens = []
                    # some generate signatures accept max_tokens or n_predict
                    try:
                        for t in llm.generate(
                            tokens,
                            max_tokens=max_new_tokens,
                            n_batch=n_batch,
                            top_p=top_p,
                            temp=temperature,
                        ):
                            out_tokens.append(int(t))
                    except TypeError:
                        for t in llm.generate(tokens):
                            out_tokens.append(int(t))
                    text = llm.detokenize(out_tokens)
                    return text.strip()
                except Exception as e:
                    print(f"LocalLLM: generate/tokenize failed: {e}")
        except Exception:
            pass

        raise RuntimeError(
            "LocalLLM: no usable llama-cpp generation method found or context issue."
        )

    def _send_gpt4all(self, prompt: str, timeout: int, max_new_tokens: int) -> str:
        try:
            # try common signatures
            try:
                out = self._impl.generate(
                    prompt, max_tokens=max_new_tokens, streaming=False
                )
            except TypeError:
                out = self._impl.generate(
                    prompt, n_predict=max_new_tokens, streaming=False
                )
            if isinstance(out, dict) and "text" in out:
                return out["text"]
            if hasattr(self._impl, "response"):
                return str(self._impl.response)
            if hasattr(self._impl, "last_text"):
                return str(self._impl.last_text)
            return str(out)
        except Exception as e:
            raise RuntimeError(f"LocalLLM: gpt4all error: {e}")

    def _send_transformers(
        self,
        prompt: str,
        timeout: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        try:
            out = self._impl(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
            )
            if isinstance(out, list) and out:
                return out[0].get("generated_text", "") or out[0].get("text", "")
            return str(out)
        except Exception as e:
            raise RuntimeError(f"LocalLLM: transformers error: {e}")


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
    blocks = extract_all_bracketed_blocks(text)
    if not blocks:
        return None
    largest = max(blocks, key=lambda s: len(s))
    return largest.strip()


def extract_all_bracketed_blocks(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r"\[([\s\S]*?)\]", text, flags=re.DOTALL)]


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
        default_timeout: int = 10,
        model_dir: str = "/home/runner/work/runner-script/runner-script/models",
    ):
        self.chat = LocalLLM(model_dir=model_dir, default_timeout=default_timeout)

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
        # Short, robust driver that preserves all original behavior but enforces
        # deterministic bracket extraction and saves the final script to disk.
        words_target = timing_minutes * words_per_minute
        tolerance = max(1, int(words_target * 0.01))
        prompt = self._build_script_prompt(
            niche=niche,
            person=person,
            timing_minutes=timing_minutes,
            words_per_minute=words_per_minute,
            topic=topic,
        )
        timeout = timeout or getattr(self.chat, "base_timeout", None) or 30

        single_block_re = re.compile(r"^\s*\[[\s\S]*\]\s*$", flags=re.DOTALL)

        def _word_count(text: str) -> int:
            if not text:
                return 0
            return len(re.findall(r"\b[\w'-]+\b", text))

        attempt = 0
        accumulated = ""
        # Primary infinite-retry loop (keeps the original 'retry until success' behavior)
        while True:
            attempt += 1
            try:
                resp = self.chat.send_message(
                    prompt,
                    timeout=timeout,
                    unlimited=True,
                    retry_forever=False,
                    max_retries=5,
                    spinner_message=f"Generating script (attempt {attempt})",
                )
            except Exception as e:
                print(
                    f"⚠️ generate_script: send_message failed on attempt {attempt}: {e}"
                )
                time.sleep(0.5)
                continue

            # If strict mode requires a single bracketed block, prefer exact match
            if strict and not single_block_re.match(resp):
                print(
                    "⚠️ Strict mode: model didn't return single bracketed block; trying best candidate."
                )
                candidate = extract_largest_bracketed(resp) or resp
            else:
                candidate = extract_largest_bracketed(resp) or resp

            cleaned_candidate = clean_script_text(candidate) or candidate
            # Avoid duplicate appends
            if cleaned_candidate and cleaned_candidate not in accumulated:
                accumulated = (accumulated + "\n\n" + cleaned_candidate).strip()

            wc = _word_count(accumulated)
            print(
                f"ℹ️ script words so far: {wc}/{words_target} (tolerance ±{tolerance})"
            )

            # If too long, do a deterministic trim (preserve beginning & end, remove middle)
            if wc > words_target + tolerance:
                # heuristic trim: keep first 60% and last 40% and then re-clean
                words = re.findall(r"\S+", accumulated)
                keep_first = int(len(words) * 0.60)
                trimmed = " ".join(
                    words[:keep_first] + words[-int(len(words) * 0.40) :]
                )
                accumulated = clean_script_text(trimmed) or trimmed
                wc = _word_count(accumulated)

            # Finalize when reached or very near target
            if wc >= words_target or (attempt >= max_attempts and wc > 0):
                # Ensure bracketed single-block format when saving
                final_block = extract_largest_bracketed(accumulated) or accumulated
                final_block = clean_script_text(final_block) or final_block
                # Guarantee single pair of brackets in the saved file
                saved_text = f"[{final_block.strip()}]"
                # save to the exact path the user expects
                save_response(
                    "generated_complete_script",
                    "generated_complete_script.txt",
                    saved_text,
                )
                print(
                    f"✅ Script generation complete (attempts={attempt}, words={wc}). Saved to generated_complete_script/generated_complete_script.txt"
                )
                return final_block.strip()

            # otherwise adapt and ask for continuation seeded with the last paragraph
            # choose last ~2 paragraphs as continuation seed
            paras = [p for p in re.split(r"\n{2,}|\r\n{2,}", accumulated) if p.strip()]
            seed = "\n\n".join(paras[-2:]) if paras else accumulated
            cont_prompt = (
                "Continue the narrative below without repeating previous paragraphs. "
                "Output exactly one bracketed block containing only the new section text.\n\n"
                f"PREV_BEGIN\n{seed}\nPREV_END\n"
                f"Aim to add approximately {max(8, words_target - _word_count(accumulated))} words."
            )
            try:
                cont_resp = self.chat.send_message(
                    cont_prompt,
                    timeout=timeout,
                    unlimited=True,
                    retry_forever=False,
                    max_retries=5,
                    spinner_message="Requesting continuation...",
                )
            except Exception as e:
                print(f"⚠️ generate_script: continuation call failed: {e}")
                time.sleep(0.4)
                continue

            cont_candidate = extract_largest_bracketed(cont_resp) or cont_resp
            cont_clean = clean_script_text(cont_candidate) or cont_candidate
            # append with de-duplication
            if cont_clean and cont_clean not in accumulated:
                accumulated = (accumulated + "\n\n" + cont_clean).strip()

            # tiny backoff to keep local LLM stable
            time.sleep(0.05)

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
        img_number: int = 100,
        batch_size: int = 20,
        timeout_per_call: int = 60,
        response_folder: str = "image_response",
        per_paragraph_max: Optional[int] = 6,
    ) -> List[str]:
        """
        Generate 'img_number' image prompts (one [bracketed] prompt per line).
        Saves to image_response/image_prompts.txt and returns list[str] (without outer brackets).
        """
        import random

        def _extract_blocks(raw: str) -> List[str]:
            return [
                m.strip() for m in re.findall(r"\[([\s\S]*?)\]", raw, flags=re.DOTALL)
            ]

        prompts: List[str] = []
        paragraphs = [
            p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", script_text) if p.strip()
        ]
        if not paragraphs:
            paragraphs = [script_text.strip()]

        # iterate paragraphs and request batches until quota met
        for idx, paragraph in enumerate(paragraphs):
            if len(prompts) >= img_number:
                break
            attempts = 0
            while len(prompts) < img_number and (
                per_paragraph_max is None or attempts < per_paragraph_max
            ):
                attempts += 1
                need = img_number - len(prompts)
                req_count = min(batch_size, need)
                seed = random.randint(1000, 9999)
                request_prompt = (
                    f"{paragraph}\n\nGenerate {req_count} unique, ultra-detailed image prompts. "
                    f"Each prompt must be returned in [brackets], one per line, and must end with '| Theme: {theme}'. "
                    "Do NOT return extra commentary."
                )
                try:
                    resp = self.chat.send_message(
                        request_prompt,
                        timeout=timeout_per_call,
                        unlimited=True,
                        retry_forever=False,
                        max_retries=5,
                        spinner_message=f"Generating image prompts (para {idx+1} attempt {attempts})",
                    )
                except Exception as e:
                    print(
                        f"  ⚠️ image prompt API error (para {idx+1} attempt {attempts}): {e}"
                    )
                    time.sleep(0.5)
                    continue

                blocks = _extract_blocks(resp)
                if not blocks:
                    # fallback: split lines and filter short ones
                    blocks = [
                        l.strip("[] \"'`")
                        for l in resp.splitlines()
                        if len(l.strip()) > 10
                    ]

                for b in blocks:
                    b = re.sub(r"\s+", " ", b).strip()
                    if len(prompts) >= img_number:
                        break
                    if not b or len(b) < 12:
                        continue
                    # ensure theme appended
                    if "| Theme:" not in b:
                        b = f"{b} | Theme: {theme}"
                    if b not in prompts:
                        prompts.append(b)

                # small jitter to avoid overloading local backend
                time.sleep(0.05)

        # save to image_response as one bracketed prompt per line
        try:
            os.makedirs(response_folder, exist_ok=True)
            with open(
                os.path.join(response_folder, "image_prompts.txt"),
                "w",
                encoding="utf-8",
            ) as fh:
                for p in prompts:
                    fh.write(f"[{p}]\n")
            print(
                f"✅ Saved {len(prompts)} image prompts to {os.path.join(response_folder, 'image_prompts.txt')}"
            )
        except Exception as e:
            print(f"⚠️ Failed to save image prompts: {e}")

        return prompts

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
        words_target = timing_minutes * words_per_minute
        prompt = (
            "Using the script below, write a single, polished third-person narration derived directly from it, suitable for Coqui TTS (Jenny). "
            f"Length: produce approximately {words_target} words (aim for 90%-110%), matching the {timing_minutes}-minute length. "
            "Preserve punctuation and natural pauses. Output exactly one bracketed block beginning with 'narrate_prompt' label on the first line."
            "Script follows: ===\n"
            f"{script_text}\n"
            "===\nGenerate now."
        )
        resp = self.chat.send_message(
            prompt,
            timeout=timeout or self.chat.base_timeout,
            unlimited=True,
            retry_forever=False,
            max_retries=5,
            spinner_message="Generating narration.",
        )
        narr_block = extract_largest_bracketed(resp)
        if narr_block:
            if narr_block.lower().startswith(bracket_label.lower()):
                parts = narr_block.split("\n", 1)
                narr_content = parts[1].strip() if len(parts) > 1 else ""
            else:
                narr_content = narr_block.strip()
        else:
            blocks = extract_all_bracketed_blocks(resp)
            narr_content = max(blocks, key=len) if blocks else resp.strip()
            print(
                "⚠️ Narration bracket not found exactly as requested — using best available content."
            )

        narr_content = escape_for_coqui_tts(narr_content)
        save_response("narration_response", "narrate_prompt.txt", f"[{narr_content}]")
        print("✅ Narration saved to narration_response/narrate_prompt.txt")
        return narr_content

    def generate_youtube_metadata(
        self, script_text: str, timing_minutes: int = 10, timeout: Optional[int] = None
    ) -> str:
        prompt = (
            f"Act as a professional YouTube growth strategist and SEO copywriter. "
            f"Your task is to generate a *complete, optimized YouTube metadata package* for a {timing_minutes}-minute video. "
            "Base everything strictly on the script provided below.\n\n"
            "The output must include labeled sections: TITLE, SUBTITLE, DESCRIPTION, TAGS, CHAPTERS, and TRANSCRIPT where appropriate. "
            "Return the metadata as plain text. Script follows:\n\n"
            f"{script_text}\n\nGenerate now."
        )
        resp = self.chat.send_message(
            prompt,
            timeout=timeout or self.chat.base_timeout,
            unlimited=True,
            retry_forever=False,
            max_retries=5,
            spinner_message="Generating YouTube metadata.",
        )
        # save verbatim so you can review/edit before uploading
        save_response("youtube_response", "youtube_metadata.txt", resp)
        print("✅ YouTube metadata saved to youtube_response/youtube_metadata.txt")
        return resp


# ---------------------- EXAMPLE USAGE ----------------------
if __name__ == "__main__":
    start = time.time()

    pipeline = StoryPipeline(
        default_timeout=10,
        model_dir="/home/runner/work/runner-script/runner-script/models",
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
