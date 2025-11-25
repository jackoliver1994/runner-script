#!/usr/bin/env python3
# compacted / functionally equivalent version of local_llm.py
# preserves all features; trimmed comments & consolidation for size/readability

import os, sys, re, time, difflib, random, math
from pathlib import Path
from threading import Thread, Event
from datetime import datetime
from typing import Optional, List

# try import for tokenizer if available
try:
    from transformers import AutoTokenizer, pipeline
except Exception:
    AutoTokenizer = None
    pipeline = None


# -------------------- Model token helper --------------------
class ModelTokenBudget:
    DEFAULT_CHAR_PER_TOKEN = 4.0
    MODEL_OVERRIDES = {
        "gpt": 4.0,
        "gpt-j": 4.0,
        "gpt2": 4.0,
        "llama": 4.0,
        "gguf": 4.0,
        "ggml": 4.0,
        "mistral": 3.5,
        "bloom": 4.5,
        "opt": 4.0,
        "vicuna": 4.0,
    }

    def __init__(self, model_id: Optional[str] = None, n_ctx_default: int = 4096):
        self.model_id = model_id or ""
        self.n_ctx_default = n_ctx_default
        self._tok = None
        self._cpt = None
        self._try_load()

    def _try_load(self):
        if not self.model_id or AutoTokenizer is None:
            return
        try:
            if os.path.isdir(self.model_id):
                self._tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            else:
                self._tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        except Exception:
            self._tok = None

    def _infer(self):
        if self._cpt is not None:
            return self._cpt
        if self._tok:
            self._cpt = None
            return None
        mid = (self.model_id or "").lower()
        for k, v in self.MODEL_OVERRIDES.items():
            if k in mid:
                self._cpt = v
                return v
        self._cpt = self.DEFAULT_CHAR_PER_TOKEN
        return self._cpt

    def count_prompt_tokens(self, prompt: str) -> int:
        if not prompt:
            return 0
        if self._tok:
            try:
                return len(self._tok.encode(prompt, add_special_tokens=False))
            except Exception:
                pass
        cpt = self._infer() or self.DEFAULT_CHAR_PER_TOKEN
        return max(0, int(len(prompt) / cpt))

    def compute_max_new_tokens(
        self,
        prompt: str,
        n_ctx: Optional[int] = None,
        safety_factor: float = 0.8,
        requested_max_new: Optional[int] = None,
        min_new_tokens: int = 64,
        local_ceiling: Optional[int] = 16384,
    ) -> int:
        n_ctx = n_ctx or self.n_ctx_default
        pt = self.count_prompt_tokens(prompt)
        avail = max(0, int((n_ctx - pt) * safety_factor))
        if requested_max_new is not None:
            chosen = min(requested_max_new, avail)
        else:
            chosen = min(avail, int(n_ctx * (1.0 - safety_factor)))
        chosen = max(chosen, min_new_tokens)
        if local_ceiling:
            chosen = min(chosen, local_ceiling)
        return int(chosen)


# -------------------- spinner --------------------
class LoadingSpinner:
    def __init__(self, msg="Waiting..."):
        self.msg = msg
        self.chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.stop_event = Event()
        self.thread = None

    def spin(self):
        while not self.stop_event.is_set():
            for c in self.chars:
                sys.stdout.write("\r" + f"{self.msg} {c}")
                sys.stdout.flush()
                time.sleep(0.08)
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


# -------------------- Local LLM backend --------------------
class LocalLLM:
    MODEL_EXTS = [".gguf", ".ggml", ".bin", ".pt", ".pth", ".safetensors", ".model"]

    def __init__(self, model_dir: str = "models", default_timeout: int = 10):
        self.model_dir = str(model_dir)
        self.base_timeout = default_timeout
        self.backend = None
        self.model_path = None
        self._impl = None
        self._n_ctx = None
        self._avg_token_char = 4
        self._detect_and_init()

    def _find_model(self) -> Optional[str]:
        p = Path(self.model_dir)
        if not p.exists():
            return None
        for ext in self.MODEL_EXTS:
            items = list(p.rglob(f"*{ext}"))
            if items:
                items_sorted = sorted(
                    items, key=lambda x: x.stat().st_size, reverse=True
                )
                return str(items_sorted[0])
        allf = [f for f in p.rglob("*") if f.is_file()]
        if allf:
            return str(sorted(allf, key=lambda x: x.stat().st_size, reverse=True)[0])
        return None

    def _detect_and_init(self):
        model_file = self._find_model()
        if not model_file:
            raise RuntimeError(f"No model under {self.model_dir}")
        self.model_path = model_file
        p = Path(self.model_path)
        self.model = str(self.model_path) if p.exists() and p.is_dir() else None

        # try llama_cpp
        try:
            from llama_cpp import Llama

            cpu = os.cpu_count() or 1
            os.environ.setdefault("OMP_NUM_THREADS", str(max(1, cpu - 1)))
            try:
                self._impl = Llama(model_path=self.model_path, n_threads=cpu)
            except TypeError:
                self._impl = Llama(model_path=self.model_path)
            self._n_ctx = self._detect_n_ctx()
            self.backend = "llama_cpp"
            return
        except Exception as e:
            pass

        # try gpt4all
        try:
            from gpt4all import GPT4All

            try:
                self._impl = GPT4All(model=self.model_path)
                self.backend = "gpt4all"
                self._n_ctx = self._detect_n_ctx() or 2048
                return
            except Exception:
                pass
        except Exception:
            pass

        # try transformers pipeline
        try:
            if pipeline is not None:
                try:
                    gen = pipeline(
                        "text-generation", model=self.model_path, device_map=None
                    )
                    self._impl = gen
                    self.backend = "transformers"
                    self._n_ctx = self._detect_n_ctx() or 2048
                    return
                except Exception:
                    pass
        except Exception:
            pass

        raise RuntimeError(
            "LocalLLM: cannot init any backend (install llama-cpp-python, gpt4all, or transformers+torch)"
        )

    def _detect_n_ctx(self) -> int:
        impl = self._impl
        if not impl:
            return 512
        for a in ("n_ctx", "context_size", "context_length", "n_ctx_max"):
            v = getattr(impl, a, None)
            if isinstance(v, int) and v > 0:
                return int(v)
        try:
            meta = getattr(impl, "model", None)
            if meta is not None:
                for maybe in ("n_ctx", "context_length"):
                    v = getattr(meta, maybe, None)
                    if isinstance(v, int) and v > 0:
                        return int(v)
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
                        return int(val)
        except Exception:
            pass
        return 512

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

    def _trim_prompt_to_fit(self, prompt: str, max_new_tokens: int) -> str:
        try:
            n_ctx = int(getattr(self, "_n_ctx", 512) or 512)
            margin = 8
            max_new_tokens = max(1, int(max_new_tokens or 64))
            prompt_toks = self._count_tokens(prompt)
            allowed = max(0, n_ctx - max_new_tokens - margin)
            if prompt_toks <= allowed:
                return prompt
            approx_cpt = getattr(self, "_avg_token_char", 4.0)
            approx_allowed_chars = int(allowed * approx_cpt * 1.1)
            if approx_allowed_chars <= 0:
                return prompt[-2048:]
            return prompt[-approx_allowed_chars:].lstrip()
        except Exception:
            return prompt[-2000:]

    def compute_dynamic_max_new_tokens(
        self,
        prompt: str,
        safety_factor: float = 0.8,
        hard_cap: Optional[int] = None,
        requested_max_new: Optional[int] = None,
        min_new_tokens: int = 64,
        local_ceiling: Optional[int] = None,
        n_ctx: Optional[int] = None,
    ) -> int:
        if n_ctx is None:
            n_ctx = getattr(self, "_n_ctx", None) or 131072
        if local_ceiling is None:
            local_ceiling = getattr(self, "local_max_new_tokens_ceiling", None) or 65536
        model_id = None
        for attr in ("model", "model_path", "model_dir", "model_file"):
            if getattr(self, attr, None):
                model_id = getattr(self, attr)
                break
        try:
            b = ModelTokenBudget(model_id, n_ctx_default=n_ctx)
            comp = b.compute_max_new_tokens(
                prompt,
                n_ctx=n_ctx,
                safety_factor=safety_factor,
                requested_max_new=requested_max_new or hard_cap,
                min_new_tokens=min_new_tokens,
                local_ceiling=local_ceiling,
            )
        except Exception:
            approx_cpt = getattr(self, "_avg_token_char", 4.0)
            prompt_tokens = max(0, int(len(prompt) / approx_cpt))
            available = max(
                0, int((n_ctx - prompt_tokens) * max(0.01, min(1.0, safety_factor)))
            )
            comp = max(min_new_tokens, min(int(local_ceiling), available))
            if requested_max_new is not None:
                comp = min(comp, int(requested_max_new))
        comp = int(comp)
        if comp < int(min_new_tokens):
            comp = int(min_new_tokens)
        if local_ceiling:
            comp = min(comp, int(local_ceiling))
        if hard_cap is not None:
            comp = min(comp, int(hard_cap))
        return comp

    # wrapper to optionally use signal alarm on POSIX
    def _call_with_alarm_timeout(
        self, fn, *args, timeout: Optional[int] = None, **kwargs
    ):
        if not timeout or timeout <= 0:
            return fn(*args, **kwargs)
        if os.name == "nt":
            return fn(*args, **kwargs)
        import signal

        def _handler(signum, frame):
            raise TimeoutError("timed out")

        old = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, int(timeout))
            return fn(*args, **kwargs)
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
            signal.signal(signal.SIGALRM, old)

    # public send_message signature preserved
    def send_message(
        self,
        message: str,
        timeout: Optional[int] = None,
        retry_forever: bool = True,
        retry_on_client_errors: bool = False,
        initial_backoff: float = 1.0,
        max_backoff: float = 8.0,
        spinner_message: str = "Waiting",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        n_batch: int = 1,
        unlimited: bool = False,
        chunk_size: int = 512,
        max_retries: Optional[int] = None,
        safety_factor: float = 0.8,
        hard_cap: Optional[int] = None,
    ) -> str:

        timeout = timeout or getattr(self, "base_timeout", 30)
        attempt = 0
        backoff = initial_backoff
        spinner = LoadingSpinner(spinner_message)
        if max_new_tokens is None:
            max_new_tokens = self.compute_dynamic_max_new_tokens(
                message,
                safety_factor=safety_factor,
                hard_cap=hard_cap,
                requested_max_new=2048,
                min_new_tokens=128,
                n_ctx=getattr(self, "_n_ctx", None) or 8192,
            )
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
                        message,
                        timeout,
                        chunk_size,
                        temperature,
                        top_p,
                        repeat_penalty,
                        n_batch,
                    )
                    spinner.stop()
                    return (out or "").strip()

                if self.backend == "llama_cpp":
                    text = self._send_llama_cpp(
                        message,
                        timeout,
                        max_new_tokens,
                        temperature,
                        top_p,
                        repeat_penalty,
                        n_batch,
                    )
                elif self.backend == "gpt4all":
                    text = self._send_gpt4all(message, timeout, max_new_tokens)
                elif self.backend == "transformers":
                    text = self._send_transformers(
                        message, timeout, max_new_tokens, temperature, top_p
                    )
                else:
                    spinner.stop()
                    raise RuntimeError("no backend")
                spinner.stop()
                return (text or "").strip()
            except Exception as e:
                spinner.stop()
                print(f"⚠️ send_message failed attempt {attempt}: {e}")
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

    # chunked unlimited generation
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
        max_iter = 200
        out = ""
        cur_prompt = prompt
        for i in range(max_iter):
            try:
                if self.backend == "llama_cpp":
                    chunk = self._call_with_alarm_timeout(
                        self._send_llama_cpp,
                        cur_prompt,
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
                        cur_prompt,
                        timeout,
                        chunk_size,
                        timeout=timeout,
                    )
                elif self.backend == "transformers":
                    chunk = self._call_with_alarm_timeout(
                        self._send_transformers,
                        cur_prompt,
                        timeout,
                        chunk_size,
                        temperature,
                        top_p,
                        timeout=timeout,
                    )
                else:
                    raise RuntimeError("unsupported backend")
            except TimeoutError:
                raise RuntimeError(f"backend timed out after {timeout}s")
            except Exception as e:
                print(f"backend error: {e}")
                raise
            if not chunk:
                break
            out += chunk
            s = chunk.strip()
            if not s:
                break
            if s.endswith("") or s.endswith("[end]"):
                break
            recent = out[-512:] if len(out) > 512 else out
            if recent.endswith(s) and len(s) < 64:
                break
            combined = (prompt + "\n" + out)[-(n_ctx * self._avg_token_char * 2) :]
            cur_prompt = self._trim_prompt_to_fit(combined, chunk_size)
            time.sleep(0.05)
        return out

    # llama-cpp adapter (tries many method signatures)
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

        def ex(r):
            if r is None:
                return ""
            if isinstance(r, str):
                return r
            if isinstance(r, list):
                for it in r:
                    if isinstance(it, dict) and "generated_text" in it:
                        return it["generated_text"]
                return str(r[0]) if r else ""
            if isinstance(r, dict):
                if "choices" in r:
                    out = []
                    for c in r["choices"]:
                        if isinstance(c, dict):
                            if "text" in c:
                                out.append(c.get("text", ""))
                            elif "message" in c and isinstance(c["message"], dict):
                                out.append(c["message"].get("content", ""))
                    return "".join(out)
                if "text" in r:
                    return r["text"]
                if "content" in r:
                    return r["content"]
            return str(r)

        prompt = self._trim_prompt_to_fit(prompt, max_new_tokens)
        try:
            if hasattr(llm, "create"):
                try:
                    resp = llm.create(
                        prompt=prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    return ex(resp).strip()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(llm, "create_completion"):
                try:
                    resp = llm.create_completion(
                        prompt=prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    return ex(resp).strip()
                except Exception:
                    pass
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
                    return ex(resp).strip()
                except Exception:
                    pass
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
                    return ex(resp).strip()
                except Exception:
                    pass
        except Exception:
            pass
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
                except Exception:
                    pass
        except Exception:
            pass
        raise RuntimeError("no usable llama-cpp method found")

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
            raise RuntimeError(f"gpt4all error: {e}")

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
            raise RuntimeError(f"transformers error: {e}")


# -------------------- utilities --------------------
def _nice_join(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def log_execution_time(s, t, show_ms: bool = False):
    elapsed = max(0.0, t - s)
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    sec = int(elapsed % 60)
    ms = int((elapsed - int(elapsed)) * 1000)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h!=1 else ''}")
    if m:
        parts.append(f"{m} minute{'s' if m!=1 else ''}")
    if sec or not parts:
        parts.append(f"{sec} second{'s' if sec!=1 else ''}")
    if show_ms and ms:
        parts.append(f"{ms} ms")
    human = _nice_join(parts)
    print(
        f"[{s} to {t} now {datetime.now().strftime('%I:%M:%S %p')}] Execution completed in {human}."
    )


def save_response(folder, fname, content):
    os.makedirs(folder, exist_ok=True)
    fp = os.path.join(folder, fname)
    with open(fp, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Saved: {fp}")


def extract_largest_bracketed(text: str) -> Optional[str]:
    m = re.findall(r"\[([^\]]+)\]", text, flags=re.DOTALL)
    if not m:
        return None
    return max(m, key=len).strip()


def extract_all_bracketed_blocks(text: str) -> List[str]:
    return [x.strip() for x in re.findall(r"\[([^\]]+)\]", text, flags=re.DOTALL)]


def escape_for_coqui_tts(text: str) -> str:
    t = text.replace("\\", "\\\\").replace('"', '\\"')
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"([.,!?;:])([^\s])", r"\1 \2", t)
    t = re.sub(r"--", "—", t)
    if not re.search(r"[.!?…]$", t):
        t += "."
    return t


def clean_script_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(
        r"\*\*(.*?)\*\*|\_\_(.*?)\_\_|\*(.*?)\*|_(.*?)_",
        lambda m: (m.group(1) or m.group(2) or m.group(3) or m.group(4) or ""),
        s,
        flags=re.DOTALL,
    )
    s = re.sub(r"(?im)^\s*(INT|EXT|INT/EXT|INT\.|EXT\.).*$", "", s)
    s = re.sub(r'(?m)^\s*[A-Za-z\s0-9\-\–\—\'"“”&.,]{1,}:\s*', "", s)
    s = re.sub(r"(?m)^\s*:\s*$", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\bNARRATOR\b", "", s, flags=re.IGNORECASE)
    s = s.replace("*", "")
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    s = re.sub(r'([.!?])\s*([A-Z0-9"\'])', r"\1 \2", s)
    return s.strip()


# -------------------- Story pipeline --------------------
class StoryPipeline:
    def __init__(self, default_timeout: int = 10, model_dir: str = "models"):
        self.chat = LocalLLM(model_dir=model_dir, default_timeout=default_timeout)

    def _build_script_prompt(
        self,
        niche: str,
        person: str,
        timing_minutes: int,
        words_per_minute: int = 250,
        topic: str = "",
    ) -> str:
        words_target = timing_minutes * words_per_minute
        p = person.strip()
        t = topic.strip()
        if p:
            if t:
                person_section = f"Center the narrative on {p} within the context of the topic '{t}'. Tell their story with emotional depth: origins, struggles, turning points, breakthroughs, legacy. Weave 3-7 verifiable facts naturally."
            else:
                person_section = f"Center the narrative on {p}. Tell their story with emotional depth: origins, struggles, turning points, breakthroughs, legacy. Weave 3-7 verifiable facts naturally."
        else:
            if t:
                person_section = f"Do not center on a real person. Write an original third-person story relevant to niche '{niche}' and topic '{t}'. Create a protagonist and trace origins, conflict, transformation, and legacy."
            else:
                person_section = f"Do not center on a real person. Write an original third-person story relevant to niche '{niche}'. Create a protagonist and trace origins, conflict, transformation, and legacy."

        prompt = (
            "You are an expert cinematic storyteller and YouTube scriptwriter.\n\n"
            f"{person_section}\n\n"
            f"Write a cinematic long-form storytelling script (~{words_target} words, ~{timing_minutes} minutes). Aim for 90%-110% tolerance; prefer within ±1%.\n\n"
            "Structure: Hook; Act1 Origins; Act2 Turning Points & Conflicts; Act3 Breakthrough & Mastery; Act4 Legacy & Lessons; Closing Line.\n\n"
            "Style: Show, don't tell; tension-release rhythm; quotes/internal thoughts; avoid repetition; maintain authenticity.\n\n"
            "OUTPUT RULE (CRITICAL): OUTPUT EXACTLY ONE PAIR OF SQUARE BRACKETS AND NOTHING ELSE: a single [ ... ] containing ONLY the full script text. No extra text outside the brackets. If impossible, return exactly [].\n\n"
            "Preserve facts and beats. Generate now."
        )
        return prompt

    # generate_script kept feature parity: strict bracket enforcement, condense attempts, continuation, dedupe, cleaning, saving
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
        tol = max(1, int(words_target * 0.01))
        base_prompt = self._build_script_prompt(
            niche, person, timing_minutes, words_per_minute, topic
        )
        timeout = timeout or getattr(self.chat, "base_timeout", None)
        single_block_re = re.compile(r"^\s*\[[\s\S]*\]\s*$", flags=re.DOTALL)

        def wc(text):
            return len(re.findall(r"\w+", text or ""))

        def paras(text):
            if not text:
                return []
            paras = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", text) if p.strip()]
            if not paras:
                paras = [p.strip() for p in re.split(r"\n|\r\n", text) if p.strip()]
            return paras

        def normalize(s):
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s or "").lower()).strip()

        def remove_dups(text, fuzzy_threshold=0.90):
            ps = paras(text)
            if not ps:
                return text
            kept = []
            normals = []
            for p in ps:
                npn = normalize(p)
                if not npn:
                    continue
                dup = False
                if npn in normals:
                    dup = True
                else:
                    for k in normals:
                        if len(npn) < 40 or len(k) < 40:
                            continue
                        if (
                            difflib.SequenceMatcher(None, npn, k).ratio()
                            >= fuzzy_threshold
                        ):
                            dup = True
                            break
                if not dup:
                    kept.append(p)
                    normals.append(npn)
            return "\n\n".join(kept).strip()

        def strengthen_prompt(base, prev_words, attempt_no):
            extra = (
                "\n\nIMPORTANT: Output EXACTLY ONE bracketed block and NOTHING ELSE. "
                f"Make the script contain exactly {words_target} words if possible (±{tol}). "
                "Do not output any comments or metadata outside the brackets."
            )
            if prev_words is not None:
                diff = words_target - prev_words
                if diff > 0:
                    extra += " Extend naturally to reach the target."
                elif diff < 0:
                    extra += " Condense tightly preserving beats."
            extra += f" Attempt #{attempt_no}."
            return base + extra

        def extract_candidate(resp):
            try:
                brs = re.findall(r"\[[\s\S]*?\]", resp)
                if brs:
                    return max(brs, key=len)[1:-1].strip()
            except Exception:
                pass
            return resp.strip()

        def heuristic_trim(text, target):
            ps = paras(text)
            if not ps:
                return text
            protect_first = min(1, len(ps))
            protect_last = min(1, len(ps) - protect_first) if len(ps) > 1 else 0
            para_sents = [
                [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", p) if s.strip()]
                or [p.strip()]
                for p in ps
            ]
            flat = []
            loc = []
            for pi, sents in enumerate(para_sents):
                for si, s in enumerate(sents):
                    flat.append(normalize(s))
                    loc.append((pi, si))
            n = len(flat)
            if n == 0:
                return text
            scores = [0.0] * n
            for i in range(n):
                si = flat[i]
                if not si or len(si) < 20:
                    scores[i] = 0.0
                    continue
                tot = 0.0
                cnt = 0
                for j in range(n):
                    if i == j:
                        continue
                    sj = flat[j]
                    if not sj:
                        continue
                    tot += difflib.SequenceMatcher(None, si, sj).ratio()
                    cnt += 1
                scores[i] = (tot / cnt) if cnt else 0.0
            removable = []
            for idx, (pi, si) in enumerate(loc):
                if pi < protect_first or pi >= len(ps) - protect_last:
                    continue
                sent_text = para_sents[pi][si]
                wc_sent = len(re.findall(r"\w+", sent_text))
                removable.append((scores[idx], wc_sent, pi, si, sent_text))
            removable.sort(key=lambda x: (x[0], x[1]), reverse=True)
            current_text = text
            current_wc = wc(current_text)
            removals_by_para = {}
            for score, wc_sent, pi, si, st in removable:
                if current_wc <= target:
                    break
                if len(para_sents[pi]) <= 1:
                    continue
                already = removals_by_para.get(pi, 0)
                if (already + 1) / len(para_sents[pi]) > 0.6:
                    continue
                para_sents[pi][si] = ""
                removals_by_para[pi] = already + 1
                newp = [
                    " ".join([s for s in psent if s and s.strip()])
                    for psent in para_sents
                ]
                current_text = "\n\n".join([p for p in newp if p.strip()]).strip()
                current_wc = wc(current_text)
                if current_wc <= target:
                    break
            if wc(current_text) > target:
                ps_now = paras(current_text)
                cand_idxs = [
                    i
                    for i in range(len(ps_now))
                    if i >= protect_first and i < len(ps_now) - protect_last
                ]
                cand_idxs_sorted = sorted(cand_idxs, key=lambda i: wc(ps_now[i]))
                for i in cand_idxs_sorted:
                    if wc(current_text) <= target:
                        break
                    ps_now[i] = ""
                    current_text = "\n\n".join([p for p in ps_now if p.strip()]).strip()
            if wc(current_text) > target:
                words = re.findall(r"\S+", current_text)
                paras_now = paras(current_text)
                cum = []
                total = 0
                for p in paras_now:
                    wc_p = wc(p)
                    cum.append((total, total + wc_p))
                    total += wc_p
                last_protected_idx = (
                    max(0, len(paras_now) - protect_last)
                    if protect_last > 0
                    else len(paras_now)
                )
                keep_last_start = (
                    cum[last_protected_idx][0] if last_protected_idx < len(cum) else 0
                )
                allowable = max(0, keep_last_start + 3)
                if target <= allowable:
                    truncated = " ".join(words[:target])
                else:
                    truncated = " ".join(words[: max(target, allowable)])
                return truncated.strip()
            return current_text.strip()

        accumulated = ""
        attempt = 0

        def finalize_and_save(text):
            final_text = remove_dups(text, fuzzy_threshold=0.92)
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

        while True:
            attempt += 1
            # initial generation
            if not accumulated:
                req_prompt = (
                    base_prompt
                    if attempt == 1
                    else strengthen_prompt(base_prompt, None, attempt)
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
                if strict and not single_block_re.match(resp):
                    print("⚠️ Strict mode: not a single bracketed block — retrying.")
                    time.sleep(0.2)
                    continue
                candidate = extract_candidate(resp)
                cleaned = clean_script_text(candidate) or candidate
                cleaned = remove_dups(cleaned, fuzzy_threshold=0.90)
                wc0 = wc(cleaned)
                print(
                    f"[Initial] After cleaning: {wc0} words; Remaining to target: {words_target - wc0} ({words_target}±{tol})"
                )
                if wc0 > words_target + tol:
                    condensed_candidate = cleaned
                    prev_long = wc0
                    cond_tries = min(6, max(1, max_attempts - attempt))
                    for ct in range(cond_tries):
                        condense_prompt = "Condense the script below to exactly {0} words (±{1}) preserving structure, voice, and facts. Output exactly one bracketed block and nothing else.\n\nPREVIOUS_SCRIPT_BEGIN\n{2}\nPREVIOUS_SCRIPT_END\n".format(
                            words_target, tol, condensed_candidate
                        )
                        try:
                            cond_resp = self.chat.send_message(
                                condense_prompt,
                                timeout=timeout,
                                unlimited=True,
                                retry_forever=False,
                                max_retries=5,
                                spinner_message=f"Condensing (try {ct+1}/{cond_tries})...",
                            )
                        except Exception as e:
                            print(f"⚠️ Condense failed: {e}")
                            break
                        if strict and not single_block_re.match(cond_resp):
                            print(
                                "⚠️ Strict mode: condense response not bracketed — retrying condense."
                            )
                            time.sleep(0.2)
                            continue
                        inner = extract_candidate(cond_resp)
                        try:
                            cclean = clean_script_text(inner) or inner
                        except:
                            cclean = inner
                        cclean = remove_dups(cclean, fuzzy_threshold=0.90)
                        cwc = wc(cclean)
                        print(f"[Condense {ct+1}] After cleaning: {cwc} words")
                        if abs(cwc - words_target) <= tol:
                            accumulated = cclean
                            break
                        if cwc < prev_long:
                            condensed_candidate = cclean
                            prev_long = cwc
                            time.sleep(0.2)
                            continue
                        break
                    if not accumulated:
                        trimmed = heuristic_trim(condensed_candidate, words_target)
                        try:
                            trimmed_clean = clean_script_text(trimmed) or trimmed
                        except:
                            trimmed_clean = trimmed
                        accumulated = remove_dups(trimmed_clean, fuzzy_threshold=0.90)
                else:
                    accumulated = cleaned

                if (
                    abs(len(re.findall(r"\w+", accumulated or "")) - words_target)
                    <= tol
                ):
                    res = finalize_and_save(accumulated)
                    if res is not None:
                        return res
                    accumulated = ""
                    continue
                continue

            # continuation
            acc_wc = wc(accumulated)
            remaining = max(0, words_target - acc_wc)
            last_para = paras(accumulated)[-1] if paras(accumulated) else ""
            cont_prompt = (
                "Continue the script below seamlessly so final combined script reaches ~{0} words (add ~{1} words). Output ONLY the continuation text (no brackets). "
                "Do NOT repeat the last paragraph; maintain voice, pacing, continuity.\n\nPREV_BEGIN\n{2}\nPREV_END\n\nLAST_PARAGRAPH_BEGIN\n{3}\nLAST_PARAGRAPH_END\n".format(
                    words_target, remaining, accumulated, last_para
                )
            )
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
            cont_candidate = extract_candidate(cont_resp)
            try:
                cont_clean = clean_script_text(cont_candidate) or cont_candidate
            except:
                cont_clean = cont_candidate
            combined = (accumulated.rstrip() + "\n\n" + cont_clean.strip()).strip()
            combined = remove_dups(combined, fuzzy_threshold=0.90)
            new_wc = len(re.findall(r"\w+", combined or ""))
            print(
                f"[After append] After cleaning: {new_wc} words; Remaining: {words_target - new_wc}"
            )
            if new_wc <= acc_wc:
                print(
                    "⚠️ Continuation produced no net progress; trying stronger append."
                )
                base_prompt = strengthen_prompt(base_prompt, acc_wc, attempt + 1)
                gen_prompt = "Produce a new script section continuing the narrative below, not repeating existing paragraphs. Aim to add about {0} words. Output EXACTLY ONE bracketed block.".format(
                    remaining
                )
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
                    print(f"⚠️ Append generation failed: {e}")
                    time.sleep(0.2)
                    continue
                if strict and not single_block_re.match(gen_resp):
                    print("⚠️ Strict mode: append generation not bracketed — retrying.")
                    time.sleep(0.2)
                    continue
                gen_candidate = extract_candidate(gen_resp)
                try:
                    gen_clean = clean_script_text(gen_candidate) or gen_candidate
                except:
                    gen_clean = gen_candidate
                new_combined = (
                    accumulated.rstrip() + "\n\n" + gen_clean.strip()
                ).strip()
                new_combined = remove_dups(new_combined, fuzzy_threshold=0.90)
                new_wc2 = len(re.findall(r"\w+", new_combined or ""))
                print(f"[After regeneration append] {new_wc2} words")
                if new_wc2 > acc_wc:
                    accumulated = new_combined
                    continue
                time.sleep(0.2)
                continue

            accumulated = combined
            if len(re.findall(r"\w+", accumulated or "")) >= words_target:
                if len(re.findall(r"\w+", accumulated or "")) > words_target + tol:
                    trimmed = heuristic_trim(accumulated, words_target)
                    try:
                        trimmed_clean = clean_script_text(trimmed) or trimmed
                    except:
                        trimmed_clean = trimmed
                    accumulated = remove_dups(trimmed_clean, fuzzy_threshold=0.92)
                res = finalize_and_save(accumulated)
                if res is not None:
                    return res
                accumulated = ""
                continue
            time.sleep(0.05)

        raise RuntimeError("generate_script failed")

    # simple helper that splits text into n parts while preserving sentence boundaries
    def _split_text_into_n_parts(self, text: str, n: int) -> List[str]:
        text = (text or "").strip()
        if n <= 1 or not text:
            return [text] + [""] * (n - 1) if n > 0 else []
        sentences = re.split(r"(?<=[\.\?!])\s+", text)
        if not sentences:
            return [""] * n
        words_per_sentence = [len(re.findall(r"\S+", s)) for s in sentences]
        total = sum(words_per_sentence)
        if total == 0:
            return [""] * n
        target = math.ceil(total / n)
        parts = []
        cur = []
        cur_count = 0
        idx = 0
        while idx < len(sentences) and len(parts) < n - 1:
            w = words_per_sentence[idx]
            s = sentences[idx]
            if cur_count + w > target and cur_count > 0:
                parts.append(" ".join(cur).strip())
                cur = []
                cur_count = 0
                continue
            cur.append(s)
            cur_count += w
            idx += 1
        tail = []
        while idx < len(sentences):
            tail.append(sentences[idx])
            idx += 1
        if cur:
            parts.append(" ".join(cur).strip())
        if tail:
            parts.append(" ".join(tail).strip())
        while len(parts) < n:
            parts.append("")
        if len(parts) > n:
            parts = parts[: n - 1] + [" ".join(parts[n - 1 :]).strip()]
        return parts

    # condensed but preserving behavior & interface for image prompts, narration, youtube metadata
    def _build_image_prompt_request(
        self, script_paragraph: str, theme: str, req_count: int
    ) -> str:
        body = f"""Generate {req_count} ultra-detailed image prompts from the paragraph below. Include camera, lighting, composition, material/texture, mood, style, aspect ratio, and '{theme}'. Output one prompt per line wrapped in [brackets].\n\nSCRIPT:\n{script_paragraph.strip()}\n"""
        return body.strip()

    def generate_image_prompts(
        self,
        script_text: str,
        theme: str,
        img_number: int = 50,
        batch_size: int = 5,
        timeout_per_call: Optional[int] = None,
        save_each_batch: bool = True,
    ) -> List[str]:
        # strict per-paragraph batching: each paragraph must fill its batch quota before moving on
        paragraphs = [
            p.strip()
            for p in re.split(r"\n{2,}", (script_text or "").strip())
            if p.strip()
        ]
        num_paragraphs = max(1, len(paragraphs))
        prompts = []
        response_folder = "image_response"
        per_paragraph_max = None
        print(
            f"\nStarting strict per-paragraph image prompt generation: target={img_number}, batch_size={batch_size}, paragraphs={num_paragraphs}"
        )
        for para_idx in range(num_paragraphs):
            if len(prompts) >= img_number:
                break
            remaining_total = img_number - len(prompts)
            para_quota = min(batch_size, remaining_total)
            collected = 0
            para = (
                paragraphs[para_idx]
                if para_idx < len(paragraphs)
                else script_text or ""
            )
            attempts = 0
            while collected < para_quota and (
                per_paragraph_max is None or attempts < per_paragraph_max
            ):
                attempts += 1
                seed = random.randint(1000, 9999)
                need_now = para_quota - collected
                request_count = need_now
                enriched = f"{para}\n\n# Paragraph {para_idx+1} Attempt {attempts} Seed:{seed}\nGenerate {request_count} unique image prompts. Each prompt must be wrapped in [brackets] one per line.\nTheme: {theme}"
                prompt_request = self._build_image_prompt_request(
                    enriched, theme, request_count
                )
                try:
                    resp = self.chat.send_message(
                        prompt_request,
                        timeout=timeout_per_call,
                        unlimited=True,
                        retry_forever=False,
                        max_retries=5,
                        spinner_message=f"Generating paragraph {para_idx+1} prompts (attempt {attempts})",
                    )
                except Exception as e:
                    print(
                        f"  ⚠️ API error paragraph {para_idx+1} attempt {attempts}: {e}"
                    )
                    save_response(
                        response_folder, f"para_{para_idx+1}_err_{attempts}.txt", str(e)
                    )
                    time.sleep(0.2)
                    continue
                # extract bracketed prompts
                lines = re.findall(r"\[([^\]]+)\]", resp)
                added = 0
                for L in lines:
                    s = L.strip()
                    if s and s not in prompts:
                        prompts.append(s)
                        added += 1
                        collected += 1
                        if len(prompts) >= img_number or collected >= para_quota:
                            break
                if added == 0:
                    # if nothing new, try again a few times then break
                    print(
                        f"  ⚠️ No new prompts returned for paragraph {para_idx+1} attempt {attempts}."
                    )
                    time.sleep(0.2)
                    if attempts > 6:
                        break
            # end paragraph loop
        save_response(
            "image_response",
            "image_prompts.txt",
            "\n".join([f"[{p}]" for p in prompts]),
        )
        return prompts

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
            f"Using the script below, write one polished third-person narration for Coqui TTS. Length ~{words_target} words (90%-110%). "
            "Output exactly ONE bracketed block beginning with 'narrate_prompt' then a newline and the narration text. Script follows:\n===\n"
            f"{script_text}\n===\nGenerate now."
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
        return narr_content

    def generate_youtube_metadata(
        self, script_text: str, timing_minutes: int = 10, timeout: Optional[int] = None
    ) -> str:
        prompt = f"Act as a YouTube growth strategist. Generate a complete optimized YouTube metadata package for a {timing_minutes}-minute video based on the script below. Include TITLE (<=90 chars), DESCRIPTION (3-5 short paragraphs + hashtags), 15-30 TAGS, 3 thumbnail headline options, and 5 video chapters with timestamps. Output as plain text. Script:\n\n{script_text}"
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


# -------------------- example usage --------------------
if __name__ == "__main__":
    s = time.time()
    pipe = StoryPipeline(default_timeout=10, model_dir="models")
    script = pipe.generate_script(
        niche="Preschool-early-elementary children",
        person="",
        timing_minutes=10,
        timeout=100,
        topic="The Little Cloud Painter",
    )
    print("\n--- Script (first 400 chars) ---")
    print(script[:400])
    images = pipe.generate_image_prompts(
        script_text=script,
        theme="water color illustrations, children's book, whimsical",
        img_number=50,
        batch_size=10,
        timeout_per_call=100,
    )
    print(f"\nGenerated {len(images)} image prompts (sample):")
    for p in images[:5]:
        print(f"[{p}]")
    narration = pipe.generate_narration(script, timing_minutes=10)
    print("\nNarration saved and ready for TTS")
    yt_meta = pipe.generate_youtube_metadata(script, timing_minutes=10)
    print("\nYouTube metadata saved.")
    e = time.time()
    log_execution_time(s, e)
