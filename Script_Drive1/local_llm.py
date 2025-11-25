#!/usr/bin/env python3
"""
story_pipeline.py — compact prompts version

This file retains all original features:
- StoryPipeline with generate_script, generate_image_prompts, generate_narration, generate_youtube_metadata
- LocalLLM auto-detects model files (gguf, ggml, safetensors, etc.)
- dynamic token / context handling and chunked "unlimited" generation
- infinite retry behavior where requested (generate_script, image prompt batching)
- model_dir fixed to "/home/runner/work/runner-script/runner-script/models"
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

# optional tokenizer import (best-effort)
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# ---------------------- LOADING SPINNER ----------------------
class LoadingSpinner:
    def __init__(self, message: str = "Waiting for response."):
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


# ---------------------- LOCAL LLM BACKEND ----------------------
class LocalLLM:
    """
    Robust LocalLLM:
    - detects models under model_dir (default path set by StoryPipeline)
    - tries llama-cpp, gpt4all, transformers fallbacks
    - dynamic max_new_tokens computed from model n_ctx and prompt
    - supports unlimited (chunked) generation
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
        self.model_path = str(model_dir)
        self._impl = None
        self._n_ctx = None
        self._avg_token_char = 4
        self._hf_tokenizer = None
        self._detect_and_init()

    # --------------- helper: find model ---------------
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

    # --------------- init backends ---------------
    def _detect_and_init(self):
        model_file = self._find_model_file()
        if not model_file:
            raise RuntimeError(f"No model files found under {self.model_dir}.")
        self.model_path = model_file

        # try llama-cpp-python
        try:
            from llama_cpp import Llama  # type: ignore

            cpu_count = os.cpu_count() or 1
            os.environ.setdefault("OMP_NUM_THREADS", str(max(1, cpu_count - 1)))
            try:
                self._impl = Llama(model_path=self.model_path, n_threads=cpu_count)
            except TypeError:
                self._impl = Llama(model_path=self.model_path)
            self.backend = "llama_cpp"
            self._n_ctx = self._detect_n_ctx()
            print(f"LocalLLM: llama-cpp init succeeded -> runtime n_ctx={self._n_ctx}")
            return
        except Exception as e:
            print(f"LocalLLM: llama-cpp init failed: {e}")

        # try gpt4all
        try:
            from gpt4all import GPT4All  # type: ignore

            try:
                self._impl = GPT4All(model=self.model_path)
                self.backend = "gpt4all"
                self._n_ctx = self._detect_n_ctx() or 2048
                return
            except Exception as e:
                print(f"LocalLLM: gpt4all init failed: {e}")
        except Exception:
            pass

        # transformers fallback
        try:
            from transformers import pipeline  # type: ignore

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
            "LocalLLM: could not initialize a compatible local backend. Install llama-cpp-python, gpt4all, or transformers+torch."
        )

    # --------------- tokenizer helper ---------------
    def _ensure_tokenizer_for_model(self):
        if self._hf_tokenizer is not None or AutoTokenizer is None:
            return
        model_id = getattr(self, "model", None) or getattr(self, "model_path", None)
        if not model_id:
            return
        try:
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as e:
            print(f"LocalLLM: tokenizer load failed for {model_id}: {e}")
            self._hf_tokenizer = None

    # --------------- detect n_ctx ---------------
    def _detect_n_ctx(self) -> int:
        if not self._impl:
            return 512
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
        # try introspection into impl.model metadata
        try:
            meta = getattr(self._impl, "model", None)
            if meta is not None:
                for maybe in ("n_ctx", "context_length", "metadata", "info"):
                    m = getattr(meta, maybe, None)
                    if isinstance(m, int) and m > 0:
                        return int(m)
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
        return 512

    # --------------- token counting ---------------
    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            if hasattr(self._impl, "tokenize"):
                try:
                    toks = self._impl.tokenize(text)
                except Exception:
                    toks = self._impl.tokenize(text.encode("utf-8"))
                return len(toks)
        except Exception:
            pass
        return max(1, int(len(text) / self._avg_token_char))

    # --------------- prompt trimming ---------------
    def _trim_prompt_to_fit(self, prompt: str, max_new_tokens: int) -> str:
        n_ctx = self._n_ctx or 512
        safety = 8
        allowed = n_ctx - max_new_tokens - safety
        if allowed <= 0:
            allowed = max(32, n_ctx // 8)
        if self._count_tokens(prompt) <= allowed:
            return prompt
        parts = [p for p in prompt.split("\n\n") if p.strip() != ""]
        if not parts:
            approx_chars = allowed * self._avg_token_char
            return prompt[-int(approx_chars) :]
        while parts:
            joined = "\n\n".join(parts)
            if self._count_tokens(joined) <= allowed:
                return joined
            parts.pop(0)
        approx_chars = allowed * self._avg_token_char
        return prompt[-int(approx_chars) :]

    # --------------- dynamic max_new_tokens ---------------
    def compute_dynamic_max_new_tokens(
        self, prompt: str, safety_factor: float = 0.80, hard_cap: Optional[int] = None
    ) -> int:
        n_ctx = getattr(self, "_n_ctx", None) or 131072
        self._ensure_tokenizer_for_model()
        prompt_tokens = None
        if getattr(self, "_hf_tokenizer", None) is not None:
            try:
                prompt_tokens = len(self._hf_tokenizer(prompt)["input_ids"])
            except Exception as e:
                print(f"LocalLLM: tokenizer call failed, fallback estimate: {e}")
                prompt_tokens = None
        if prompt_tokens is None:
            prompt_tokens = max(1, len(prompt) // 4)
        available = max(0, n_ctx - prompt_tokens)
        safe_allowed = int(available * float(max(0.01, min(1.0, safety_factor))))
        MIN_NEW_TOKENS = 8
        if hard_cap is not None:
            safe_allowed = min(safe_allowed, int(hard_cap))
        safe_allowed = max(MIN_NEW_TOKENS, safe_allowed)
        LOCAL_CONSERVATIVE_CEILING = (
            getattr(self, "local_max_new_tokens_ceiling", None) or 65536
        )
        safe_allowed = min(safe_allowed, int(LOCAL_CONSERVATIVE_CEILING))
        return safe_allowed

    # --------------- call-with-timeout helper (Unix) ---------------
    def _call_with_alarm_timeout(
        self, fn, *args, timeout: Optional[int] = None, **kwargs
    ):
        if not timeout or timeout <= 0:
            return fn(*args, **kwargs)
        if os.name == "nt":
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
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
            signal.signal(signal.SIGALRM, old_handler)

    # --------------- send_message (public API) ---------------
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
        max_retries: Optional[int] = None,
        safety_factor: float = 0.80,
        hard_cap: Optional[int] = None,
    ) -> str:
        timeout = timeout or getattr(self, "base_timeout", None) or 30
        attempt = 0
        backoff = initial_backoff
        spinner = LoadingSpinner(spinner_message)

        if max_new_tokens is None:
            max_new_tokens = self.compute_dynamic_max_new_tokens(
                message, safety_factor=safety_factor, hard_cap=hard_cap
            )
            print(f"LocalLLM: computed dynamic max_new_tokens={max_new_tokens}")

        cap = (
            None
            if retry_forever
            else (
                max_retries if isinstance(max_retries, int) and max_retries > 0 else 5
            )
        )

        while True:
            attempt += 1
            try:
                spinner.message = f"{spinner_message} (attempt {attempt})"
                spinner.start()

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
                print(f"⚠️ send_message failed (attempt {attempt}): {e}")
                if cap is not None and attempt >= cap:
                    raise RuntimeError(
                        f"send_message: reached max retries ({cap}). Last error: {e}"
                    )
                if (
                    "403" in str(e) or "client error" in str(e).lower()
                ) and not retry_on_client_errors:
                    raise
                time.sleep(min(backoff, max_backoff))
                backoff = min(backoff * 2, max_backoff)
                # loop continues (infinite if cap is None)

    # --------------- unlimited (chunked) generation ---------------
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
        n_ctx = self._n_ctx or 512
        max_iterations = 200
        output = ""
        current_prompt = prompt
        for iteration in range(max_iterations):
            chunk = None
            try:
                if self.backend == "llama_cpp":
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
                elif self.backend == "gpt4all":
                    chunk = self._call_with_alarm_timeout(
                        self._send_gpt4all,
                        current_prompt,
                        timeout,
                        chunk_size,
                        timeout=timeout,
                    )
                elif self.backend == "transformers":
                    chunk = self._call_with_alarm_timeout(
                        self._send_transformers,
                        current_prompt,
                        timeout,
                        chunk_size,
                        temperature,
                        top_p,
                        timeout=timeout,
                    )
                else:
                    raise RuntimeError(
                        "LocalLLM: unsupported backend for unlimited generation"
                    )
            except TimeoutError:
                raise RuntimeError(
                    f"LocalLLM: backend timed out after {timeout} seconds"
                )
            except Exception as e:
                print(f"LocalLLM: backend error during unlimited generation: {e}")
                raise

            if not chunk:
                break
            output += chunk
            chunk_stripped = chunk.strip()
            if not chunk_stripped:
                break
            if chunk_stripped.endswith("") or chunk_stripped.endswith("[end]"):
                break
            recent_tail = output[-512:] if len(output) > 512 else output
            if recent_tail.endswith(chunk_stripped) and len(chunk_stripped) < 64:
                break
            combined = (prompt + "\n" + output)[-(n_ctx * self._avg_token_char * 2) :]
            next_prompt = self._trim_prompt_to_fit(combined, chunk_size)
            current_prompt = next_prompt
            time.sleep(0.05)
        return output

    # --------------- backend-specific sends ---------------
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

        prompt = self._trim_prompt_to_fit(prompt, max_new_tokens)

        # try common llama-cpp interfaces
        try:
            if hasattr(llm, "create"):
                resp = llm.create(
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return _extract_text(resp).strip()
        except Exception as e:
            print(f"LocalLLM: .create failed: {e}")

        try:
            if hasattr(llm, "create_completion"):
                resp = llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return _extract_text(resp).strip()
        except Exception as e:
            print(f"LocalLLM: .create_completion failed: {e}")

        try:
            if hasattr(llm, "create_chat_completion"):
                msgs = [{"role": "user", "content": prompt}]
                resp = llm.create_chat_completion(
                    messages=msgs, max_tokens=max_new_tokens, temperature=temperature
                )
                return _extract_text(resp).strip()
        except Exception as e:
            print(f"LocalLLM: .create_chat_completion failed: {e}")

        try:
            if callable(getattr(llm, "__call__", None)):
                resp = llm(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return _extract_text(resp).strip()
        except Exception as e:
            print(f"LocalLLM: __call__ failed: {e}")

        try:
            if (
                hasattr(llm, "generate")
                and hasattr(llm, "tokenize")
                and hasattr(llm, "detokenize")
            ):
                try:
                    tokens = llm.tokenize(prompt)
                except Exception:
                    tokens = llm.tokenize(prompt.encode("utf-8"))
                out_tokens = []
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

        raise RuntimeError("LocalLLM: no usable llama-cpp generation method found.")

    def _send_gpt4all(self, prompt: str, timeout: int, max_new_tokens: int) -> str:
        try:
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
    if seconds or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    if show_ms and milliseconds:
        parts.append(f"{milliseconds} ms")
    human = _nice_join(parts)
    current_time = datetime.now().strftime("%I:%M:%S %p")
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
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"__(.*?)__", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\*(.*?)\*", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"_(.*?)_", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"(?im)^\s*(INT|EXT|INT/EXT|INT\.|EXT\.).*$", "", s)
    s = re.sub(r'(?m)^\s*[A-Za-z\s0-9\-\–\—\'"“”&.,]{1,}:\s*', "", s)
    s = re.sub(r"(?m)^\s*:\s*$", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\bNARRATOR\b", "", s, flags=re.IGNORECASE)
    s = s.replace("*", "")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
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

    # compacted script prompt: keeps rules but with shorter text
    def _build_script_prompt(
        self,
        niche: str,
        person: str,
        timing_minutes: int,
        words_per_minute: int = 250,
        topic: str = "",
    ) -> str:
        words_target = timing_minutes * words_per_minute
        p = str(person).strip()
        t = str(topic).strip()
        if p:
            person_section = f"Center on {p}. Tell their arc: origins, struggles, turning points, breakthroughs, legacy. Include 3–7 verifiable facts."
        else:
            if t:
                person_section = f"Write an original third-person story for niche '{niche}' and topic '{t}'. Create a strong protagonist and arc."
            else:
                person_section = f"Write an original third-person story for niche '{niche}'. Create a protagonist and arc."

        prompt = (
            "You are an expert cinematic storyteller.\n\n"
            f"{person_section}\n\n"
            f"Write a cinematic long-form script (~{words_target} words, about {timing_minutes} minutes at {words_per_minute} wpm). Aim for ±1% if possible.\n"
            "Structure naturally: Hook; Origins; Turning points & conflicts; Breakthrough; Legacy + 3 takeaways; Closing line.\n"
            "Style: show not tell; sensory detail; tension-release rhythm; no large repeated blocks.\n\n"
            "OUTPUT RULE (CRITICAL): Output exactly ONE bracketed block and nothing else: the response must be a single '[' then the full script then a single ']'. No extra text or metadata outside the brackets. If you cannot produce it, return exactly [].\n"
            f"Produce approximately {words_target} words.\n"
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
        words_target = timing_minutes * words_per_minute
        tolerance = max(1, int(words_target * 0.01))
        base_prompt = self._build_script_prompt(
            niche, person, timing_minutes, words_per_minute, topic
        )
        timeout = timeout or getattr(self.chat, "base_timeout", None)

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
            kept, normals = [], []
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

        def _strengthen_prompt(
            base_prompt: str, previous_words: Optional[int], attempt_no: int
        ) -> str:
            extra = f"\n\nSTRICT: output exactly one bracketed block only. Target words: {words_target} (±{tolerance})."
            if previous_words is not None:
                diff = words_target - previous_words
                if diff > 0:
                    extra += " Extend naturally."
                elif diff < 0:
                    extra += " Condense while preserving beats."
            extra += f" Attempt #{attempt_no}."
            return base_prompt + extra

        def _extract_candidate(resp: str) -> str:
            try:
                brs = re.findall(r"\[[\s\S]*?\]", resp)
                if brs:
                    return max(brs, key=len)[1:-1].strip()
            except Exception:
                pass
            return resp.strip()

        def _heuristic_trim_to_target(text: str, target_words: int) -> str:
            paras = _get_paragraphs(text)
            if not paras:
                return text
            # conservative trim by removing low-value sentences while preserving first and last
            protect_first = 1
            protect_last = 1 if len(paras) > 1 else 0
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
                    flat.append(re.sub(r"[^\w\s]", "", s or "").lower())
                    loc.append((pi, si))
            n = len(flat)
            if n == 0:
                return text
            # remove sentences least similar to others until under target
            current_text = text
            current_wc = _word_count(current_text)
            removable = []
            for idx, (pi, si) in enumerate(loc):
                if pi < protect_first or pi >= len(paras) - protect_last:
                    continue
                sent_text = para_sents[pi][si]
                removable.append((len(sent_text.split()), pi, si, sent_text))
            removable.sort(key=lambda x: x[0], reverse=True)
            for _, pi, si, stext in removable:
                if current_wc <= target_words:
                    break
                if len(para_sents[pi]) <= 1:
                    continue
                para_sents[pi][si] = ""
                new_paras = []
                for sents in para_sents:
                    sents_clean = [s for s in sents if s and s.strip()]
                    if sents_clean:
                        new_paras.append(" ".join(sents_clean))
                current_text = "\n\n".join(new_paras).strip()
                current_wc = _word_count(current_text)
            if _word_count(current_text) > target_words:
                words = re.findall(r"\S+", current_text)
                current_text = " ".join(words[:target_words])
            return current_text.strip()

        def _finalize_and_save(text: str) -> Optional[str]:
            final_text = _remove_exact_and_fuzzy_duplicates(text, fuzzy_threshold=0.92)
            if final_text.startswith("[") and final_text.endswith("]"):
                final_text = final_text[1:-1].strip()
            final_text = final_text.strip()
            if not final_text:
                print("⚠️ Final text empty after cleaning — retrying.")
                return None
            save_response(
                "generated_complete_script",
                "generated_complete_script.txt",
                f"[{final_text}]",
            )
            return final_text

        accumulated = ""
        attempt = 0

        while True:
            attempt += 1
            if not accumulated:
                req_prompt = (
                    base_prompt
                    if attempt == 1
                    else _strengthen_prompt(base_prompt, None, attempt)
                )
                try:
                    resp = self.chat.send_message(
                        req_prompt,
                        timeout=timeout,
                        unlimited=True,
                        retry_forever=False,
                        max_retries=5,
                        spinner_message=f"Generating initial script (attempt {attempt})...",
                    )
                except Exception as e:
                    print(f"⚠️ send_message failed: {e}")
                    time.sleep(0.2)
                    continue
                if strict:
                    if not single_block_re.match(resp):
                        print("⚠️ Strict mode: non-bracketed response — retrying.")
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
                wc = _word_count(cleaned_candidate)
                print(
                    f"[Initial] After cleaning: {wc} words; Target: {words_target} ±{tolerance}"
                )

                if wc > words_target + tolerance:
                    condensed_candidate = cleaned_candidate
                    condense_tries = min(6, max(1, max_attempts - attempt))
                    prev_long = wc
                    for ct in range(condense_tries):
                        condense_prompt = f"Condense the script below to {words_target} words (±{tolerance}) while preserving structure and meaning. Output a single bracketed block only.\n\nPREV_BEGIN\n{condensed_candidate}\nPREV_END\n"
                        try:
                            cond_resp = self.chat.send_message(
                                condense_prompt,
                                timeout=timeout,
                                unlimited=True,
                                retry_forever=False,
                                max_retries=5,
                                spinner_message=f"Condensing ({ct+1}/{condense_tries})...",
                            )
                        except Exception as e:
                            print(f"⚠️ Condense failed: {e}")
                            break
                        if strict and not single_block_re.match(cond_resp):
                            print(
                                "⚠️ Strict condense response not bracketed — retrying condense."
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
                        cond_wc = _word_count(cond_clean)
                        print(f"[Condense {ct+1}] {cond_wc} words")
                        if abs(cond_wc - words_target) <= tolerance:
                            accumulated = cond_clean
                            break
                        if cond_wc < prev_long:
                            condensed_candidate = cond_clean
                            prev_long = cond_wc
                            time.sleep(0.2)
                            continue
                        break
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
                    accumulated = cleaned_candidate

                acc_wc = _word_count(accumulated)
                if abs(acc_wc - words_target) <= tolerance:
                    res = _finalize_and_save(accumulated)
                    if res is not None:
                        return res
                    else:
                        accumulated = ""
                        continue
                continue

            # continuation phase
            acc_wc = _word_count(accumulated)
            remaining = max(0, words_target - acc_wc)
            last_para = (
                _get_paragraphs(accumulated)[-1] if _get_paragraphs(accumulated) else ""
            )
            cont_prompt = f"Continue the script below seamlessly to add about {remaining} words and reach ~{words_target} words. Do NOT repeat the last paragraph. Output only the continuation text (no brackets).\n\nPREV_BEGIN\n{accumulated}\nPREV_END\n\nLAST_PARAGRAPH_BEGIN\n{last_para}\nLAST_PARAGRAPH_END\n"
            try:
                cont_resp = self.chat.send_message(
                    cont_prompt,
                    timeout=timeout,
                    unlimited=True,
                    retry_forever=False,
                    max_retries=5,
                    spinner_message=f"Continuation attempt (overall attempt {attempt})...",
                )
            except Exception as e:
                print(f"⚠️ Continuation send_message failed: {e}")
                time.sleep(0.2)
                continue
            cont_candidate = _extract_candidate(cont_resp)
            try:
                cont_clean = clean_script_text(cont_candidate) or cont_candidate
            except Exception:
                cont_clean = cont_candidate

            combined = (accumulated.rstrip() + "\n\n" + cont_clean.strip()).strip()
            combined = _remove_exact_and_fuzzy_duplicates(
                combined, fuzzy_threshold=0.90
            )
            new_wc = _word_count(combined)
            print(f"[After append] {new_wc} words (was {acc_wc})")
            if new_wc <= acc_wc:
                print(
                    "⚠️ Continuation produced no progress; regenerating with stronger prompt."
                )
                base_prompt = _strengthen_prompt(base_prompt, acc_wc, attempt + 1)
                gen_prompt = f"Produce a new script chunk to add about {remaining} words. Output one bracketed block only.\n\nPREV_BEGIN\n{accumulated}\nPREV_END\n"
                try:
                    gen_resp = self.chat.send_message(
                        gen_prompt,
                        timeout=timeout,
                        unlimited=True,
                        retry_forever=False,
                        max_retries=5,
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
                new_wc2 = _word_count(new_combined)
                print(f"[After regeneration append] {new_wc2} words")
                if new_wc2 > acc_wc:
                    accumulated = new_combined
                    continue
                else:
                    time.sleep(0.2)
                    continue

            accumulated = combined
            if _word_count(accumulated) >= words_target:
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
                    accumulated = ""
                    continue
            time.sleep(0.05)

        raise RuntimeError("generate_script failed to terminate normally.")

    # --------------- split helper ---------------
    def _split_text_into_n_parts(self, text: str, n: int) -> List[str]:
        import re, math

        text = (text or "").strip()
        if n <= 1 or not text:
            return [text] + [""] * (n - 1) if n > 0 else []
        sentences = re.split(r"(?<=[\.\?!])\s+", text)
        if not sentences:
            return [""] * n
        words_per_sentence = [len(re.findall(r"\S+", s)) for s in sentences]
        total_words = sum(words_per_sentence)
        if total_words == 0:
            return [""] * n
        target = math.ceil(total_words / n)
        parts, current_sentences, current_count, sentences_idx = [], [], 0, 0
        while sentences_idx < len(sentences) and len(parts) < n - 1:
            w = words_per_sentence[sentences_idx]
            s = sentences[sentences_idx]
            if current_count + w > target and current_count > 0:
                parts.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_count = 0
                continue
            current_sentences.append(s)
            current_count += w
            sentences_idx += 1
        tail = []
        while sentences_idx < len(sentences):
            tail.append(sentences[sentences_idx])
            sentences_idx += 1
        if current_sentences:
            parts.append(" ".join(current_sentences).strip())
        if tail:
            parts.append(" ".join(tail).strip())
        while len(parts) < n:
            parts.append("")
        if len(parts) > n:
            parts = parts[: n - 1] + [" ".join(parts[n - 1 :]).strip()]
        return parts

    # --------------- image prompt builder ---------------
    def _build_image_prompt_request(
        self, script_paragraph: str, theme: str, req_count: int
    ) -> str:
        # compact, instruction-forward builder used by generate_image_prompts
        body = (
            "Generate {n} unique, detailed image prompts for the scene below. One prompt per line, wrapped in [ ].\n"
            "Each prompt: include composition, camera/angle, lighting, mood, materials, style, aspect ratio. End with: | Theme: {theme}\n\n"
            "Scene:\n{scene}"
        ).format(n=req_count, theme=theme, scene=script_paragraph.strip())
        return body

    def generate_image_prompts(
        self,
        script_text: str,
        theme: str,
        img_number: int = 50,
        batch_size: int = 5,
        timeout_per_call: Optional[int] = None,
        save_each_batch: bool = True,
    ) -> List[str]:
        import random, math

        if img_number <= 0:
            return []
        timeout_per_call = timeout_per_call or min(
            120, getattr(self.chat, "base_timeout", 120)
        )
        prompts: List[str] = []
        response_folder = "image_response"
        final_fname = "image_prompts.txt"
        os.makedirs(response_folder, exist_ok=True)

        num_paragraphs = max(1, math.ceil(img_number / max(1, batch_size)))
        try:
            paragraphs = self._split_text_into_n_parts(script_text, num_paragraphs)
            if not paragraphs or len(paragraphs) < num_paragraphs:
                paragraphs = self._split_text_into_n_parts(script_text, num_paragraphs)
        except Exception:
            paragraphs = self._split_text_into_n_parts(script_text, num_paragraphs)

        cap = getattr(self, "max_prompt_attempts", None)
        per_paragraph_max = cap if isinstance(cap, int) and cap > 0 else None

        def _extract_blocks(resp_text: str):
            if not resp_text:
                return []
            blocks = []
            brs = extract_all_bracketed_blocks(resp_text)
            if brs:
                blocks.extend(brs)
            if not blocks:
                for m in re.findall(r"\[([^\]]{3,})\]", resp_text, flags=re.DOTALL):
                    blocks.append(m.strip())
            if not blocks:
                lines = [ln.strip() for ln in resp_text.splitlines() if ln.strip()]
                candidates = []
                for ln in lines:
                    if len(ln) < 12:
                        continue
                    ln2 = re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip()
                    if len(ln2) >= 12:
                        candidates.append(ln2)
                i = 0
                while i < len(candidates):
                    cur = candidates[i]
                    j = i + 1
                    while j < len(candidates) and len(cur.split()) < 10:
                        cur = cur + " " + candidates[j]
                        j += 1
                    blocks.append(cur.strip())
                    i = j
            cleaned = [
                re.sub(r"\s+", " ", b).strip() for b in blocks if isinstance(b, str)
            ]
            return cleaned

        def is_valid_prompt(candidate: str) -> bool:
            if not candidate or len(candidate.strip()) == 0:
                return False
            c_inspect = candidate.strip("[]()\"'`").strip()
            tokens = c_inspect.split()
            if len(tokens) < 10:
                visual_keywords = (
                    "camera",
                    "lens",
                    "f/",
                    "aperture",
                    "iso",
                    "shutter",
                    "mm",
                    "aspect",
                    "photoreal",
                    "cinematic",
                    "watercolor",
                    "illustration",
                    "render",
                    "anime",
                    "bokeh",
                    "lighting",
                )
                if not any(k in c_inspect.lower() for k in visual_keywords):
                    return False
            if re.match(
                r"^(generate|create|please generate|please create|return|output)\b",
                c_inspect.lower(),
            ):
                return False
            if c_inspect.count("?") >= 1 and len(tokens) < 20:
                return False
            if not re.search(r"[A-Za-z]", c_inspect):
                return False
            return True

        final_path = os.path.join(response_folder, final_fname)
        print(
            f"\nStarting image prompt generation: target={img_number}, batch_size={batch_size}, paragraphs={num_paragraphs}"
        )

        for para_idx in range(num_paragraphs):
            if len(prompts) >= img_number:
                break
            remaining_total = img_number - len(prompts)
            para_quota = min(batch_size, remaining_total)
            collected_for_paragraph = 0
            paragraph = paragraphs[para_idx] or script_text or ""
            paragraph_attempts = 0
            print(
                f"\n➡️ Paragraph {para_idx+1}/{num_paragraphs}: need {para_quota} prompts"
            )

            while collected_for_paragraph < para_quota and (
                per_paragraph_max is None or paragraph_attempts < per_paragraph_max
            ):
                paragraph_attempts += 1
                seed = random.randint(1000, 9999)
                need_now = para_quota - collected_for_paragraph
                request_count = need_now
                enriched_script = (
                    f"{paragraph}\n\n# Paragraph {para_idx+1} | Attempt {paragraph_attempts} | Seed: {seed}\n"
                    f"Generate {request_count} unique, detailed image prompts. Each in [brackets]. Include the theme: {theme}."
                )
                prompt_request = self._build_image_prompt_request(
                    paragraph, theme, request_count
                )
                try:
                    resp = self.chat.send_message(
                        prompt_request,
                        timeout=timeout_per_call,
                        unlimited=True,
                        retry_forever=False,
                        max_retries=5,
                        spinner_message=f"Generating paragraph {para_idx+1} prompts (attempt {paragraph_attempts})...",
                    )
                except Exception as e:
                    print(f"  ⚠️ API error: {e}")
                    err_fname = f"para_{para_idx+1}_err_{paragraph_attempts}.txt"
                    try:
                        save_response(response_folder, err_fname, str(e))
                        print(
                            f"  ✅ Saved error: {os.path.join(response_folder, err_fname)}"
                        )
                    except Exception:
                        pass
                    time.sleep(0.2)
                    continue
                blocks = _extract_blocks(resp)
                added = 0
                for b in blocks:
                    if len(prompts) >= img_number:
                        break
                    cand = b.strip()
                    if not cand.startswith("["):
                        cand = f"[{cand}]"
                    content = cand.strip("[] \n\t")
                    if not is_valid_prompt(content):
                        continue
                    if content in prompts:
                        continue
                    prompts.append(content)
                    added += 1
                    collected_for_paragraph += 1
                print(
                    f"  Collected {added} prompts this attempt (total now {len(prompts)})"
                )
                # Save batch progress occasionally
                if save_each_batch and added > 0:
                    try:
                        save_response(
                            response_folder,
                            final_fname,
                            "\n".join(f"[{p}]" for p in prompts),
                        )
                    except Exception:
                        pass
                if collected_for_paragraph < para_quota:
                    time.sleep(0.2)
                    continue

        # final save (only prompts, one per line in brackets)
        try:
            save_response(
                response_folder, final_fname, "\n".join(f"[{p}]" for p in prompts)
            )
        except Exception:
            pass
        return prompts

    # --------------- narration generator ---------------
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
            "Create a single third-person narration for TTS from the script below.\n"
            f"Target ~{words_target} words (±10%). Preserve punctuation and natural pauses. Output exactly one bracketed block starting with the literal label '{bracket_label}' then a newline and narration text. Example: [narrate_prompt\nText...]\n\n"
            "Script:\n===\n"
            f"{script_text}\n"
            "===\n"
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
            print("⚠️ Narration bracket not found — using best available content.")
        narr_content = escape_for_coqui_tts(narr_content)
        save_response("narration_response", "narrate_prompt.txt", f"[{narr_content}]")
        return narr_content

    # --------------- youtube metadata ---------------
    def generate_youtube_metadata(
        self, script_text: str, timing_minutes: int = 10, timeout: Optional[int] = None
    ) -> str:
        prompt = (
            f"Act as a YouTube SEO copywriter. From the script below produce a complete metadata package: TITLE (<=90 chars), DESCRIPTION (with tags, CTA), and TAGS (comma-separated). Use script content only.\n\nScript:\n==={script_text}\n===\n"
            "Return clearly labeled sections. Keep output concise."
        )
        resp = self.chat.send_message(
            prompt,
            timeout=timeout or self.chat.base_timeout,
            unlimited=True,
            retry_forever=False,
            max_retries=5,
            spinner_message="Generating YouTube metadata.",
        )
        save_response("youtube_response", "youtube_metadata.txt", resp)
        return resp


# ---------------------- EXAMPLE USAGE ----------------------
if __name__ == "__main__":
    start = time.time()
    pipeline = StoryPipeline(
        default_timeout=10,
        model_dir="/home/runner/work/runner-script/runner-script/models",
    )

    # generate script (will retry until success per method behavior)
    script = pipeline.generate_script(
        niche="Children's bedtime",
        person="",
        timing_minutes=10,
        timeout=120,
        topic="The Little Cloud Painter",
    )
    print("\n--- Script (preview) ---")
    print(script[:400])

    # generate image prompts in batches
    image_prompts = pipeline.generate_image_prompts(
        script_text=script,
        theme="watercolor, children's book, whimsical",
        img_number=50,
        batch_size=10,
        timeout_per_call=100,
    )
    print(f"\nGenerated {len(image_prompts)} image prompts (sample):")
    for p in image_prompts[:5]:
        print(f"[{p}]")

    # narration
    narration_text = pipeline.generate_narration(script, timing_minutes=10)
    print("\nNarration saved.")

    # youtube metadata
    yt_meta = pipeline.generate_youtube_metadata(script, timing_minutes=10)
    print("\nYouTube metadata saved.")

    end = time.time()
    log_execution_time(start, end)
