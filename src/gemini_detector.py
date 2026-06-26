import asyncio
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import errors as gemini_errors
from google.genai import types

logger = logging.getLogger("GeminiDetector")

# ---------------------------------------------------------------------------
# Env‑var defaults
# ---------------------------------------------------------------------------
_ENV_RPS = float(os.environ.get("GEMINI_MAX_RPM", "15")) / 60.0
_ENV_BURST = int(os.environ.get("GEMINI_BURST", "1"))
_ENV_COOLDOWN = float(os.environ.get("GEMINI_COOLDOWN_SECONDS", "15"))
_ENV_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "5"))
_ENV_BASE_DELAY = float(os.environ.get("GEMINI_BASE_DELAY", "2.0"))
_ENV_MAX_DELAY = float(os.environ.get("GEMINI_MAX_DELAY", "30.0"))
_ENV_CACHE_TTL = int(os.environ.get("GEMINI_CACHE_TTL", "300"))
_ENV_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "60"))
_ENV_CONCURRENCY = int(os.environ.get("GEMINI_MAX_CONCURRENCY", "2"))
_ENV_FALLBACK_MODEL = os.environ.get("OLLAMA_FALLBACK_MODEL", "llama3.2:1b")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RateLimitConfig:
    requests_per_second: float = _ENV_RPS
    burst: int = _ENV_BURST
    cooldown_seconds: float = _ENV_COOLDOWN


@dataclass
class RetryConfig:
    max_retries: int = _ENV_MAX_RETRIES
    base_delay_seconds: float = _ENV_BASE_DELAY
    max_delay_seconds: float = _ENV_MAX_DELAY


@dataclass
class CacheConfig:
    ttl_seconds: float = _ENV_CACHE_TTL


# ---------------------------------------------------------------------------
# Token‑bucket rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, config: RateLimitConfig):
        self._config = config
        self._tokens = float(config.burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._config.burst, self._tokens + elapsed * self._config.requests_per_second)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._config.requests_per_second
            self._tokens = 0.0
            self._last_refill = now + wait
        await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Per‑API‑key shared state
# ---------------------------------------------------------------------------
@dataclass
class _CooldownState:
    until: float = 0.0
    lock: "asyncio.Lock" = None

    def __post_init__(self):
        if self.lock is None:
            self.lock = asyncio.Lock()


@dataclass
class _SharedState:
    rate_limiter: RateLimiter
    cooldown: _CooldownState = field(default_factory=_CooldownState)
    cache: dict = field(default_factory=dict)


_shared_states: dict[str, _SharedState] = {}


def _key_for(api_key: Optional[str]) -> str:
    if not api_key:
        return "__default__"
    return hashlib.sha256(api_key.encode()).hexdigest()


def _get_shared(api_key: Optional[str], rl_config: RateLimitConfig) -> _SharedState:
    k = _key_for(api_key)
    if k not in _shared_states:
        _shared_states[k] = _SharedState(rate_limiter=RateLimiter(rl_config))
    return _shared_states[k]


# ---------------------------------------------------------------------------
# Prompt cache (shared across all keys)
# ---------------------------------------------------------------------------
_prompt_cache: dict[str, tuple[float, str]] = {}
_cache_config = CacheConfig()


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:::{prompt}".encode()).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    entry = _prompt_cache.get(key)
    if entry is None:
        return None
    expires_at, text = entry
    if time.monotonic() > expires_at:
        del _prompt_cache[key]
        return None
    return text


def _cache_set(key: str, text: str):
    _prompt_cache[key] = (time.monotonic() + _cache_config.ttl_seconds, text)
    if len(_prompt_cache) > 5000:
        now = time.monotonic()
        stale = [k for k, (exp, _) in _prompt_cache.items() if now > exp]
        for k in stale:
            del _prompt_cache[k]


# ---------------------------------------------------------------------------
# GeminiDetector
# ---------------------------------------------------------------------------
class GeminiDetector:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        max_concurrency: int = _ENV_CONCURRENCY,
        timeout: float = _ENV_TIMEOUT,
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self._timeout = timeout
        rl_config = rate_limit_config or RateLimitConfig()
        self._retry_config = retry_config or RetryConfig()
        self._shared = _get_shared(api_key, rl_config)
        self._client = genai.Client(api_key=api_key)

        logger.info(
            "GeminiDetector initialized | model=%s concurrency=%d timeout=%.1fs "
            "rps=%.2f burst=%d cooldown=%.1f retries=%d cache_ttl=%ds",
            model_name, max_concurrency, timeout,
            rl_config.requests_per_second, rl_config.burst,
            rl_config.cooldown_seconds, self._retry_config.max_retries,
            _cache_config.ttl_seconds,
        )

    # -- rate limiter helpers ------------------------------------------------

    async def _enter_cooldown(self):
        async with self._shared.cooldown.lock:
            self._shared.cooldown.until = time.monotonic() + self._shared.rate_limiter._config.cooldown_seconds

    async def _wait_cooldown(self):
        async with self._shared.cooldown.lock:
            remaining = self._shared.cooldown.until - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)

    # -- core API call -------------------------------------------------------

    async def _call_gemini(self, prompt: str) -> str:
        await self._shared.rate_limiter.acquire()
        await self._wait_cooldown()

        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                ),
                timeout=self._timeout,
            )
            return response.text
        except gemini_errors.ClientError as e:
            if e.code == 429:
                await self._enter_cooldown()
            raise
        except gemini_errors.ServerError:
            raise

    # -- retry loop ----------------------------------------------------------

    async def _call_with_retry(self, prompt: str, context: str = "") -> str:
        ck = _cache_key(self.model_name, prompt)
        cached = _cache_get(ck)
        if cached is not None:
            return cached

        last_error: Optional[Exception] = None

        for attempt in range(self._retry_config.max_retries):
            try:
                text = await self._call_gemini(prompt)
                _cache_set(ck, text)
                return text
            except gemini_errors.ClientError as e:
                last_error = e
                if e.code == 429:
                    msg = (e.message or "").lower()
                    is_quota = any(x in msg for x in [
                        "quota", "exhausted", "daily", "limit reached",
                        "resource_exhausted", "insufficient",
                    ])
                    if is_quota:
                        logger.error(
                            "Daily quota exhausted [%s] — not retrying: %s",
                            context, e.message,
                        )
                        raise

                    if attempt < self._retry_config.max_retries - 1:
                        delay = self._backoff(attempt, rate_limit=True)
                        logger.warning(
                            "Rate limited [%s] retry=%d/%d waiting=%.1fs",
                            context, attempt + 1, self._retry_config.max_retries, delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    logger.error(
                        "Rate limit retries exhausted [%s] after %d attempts: %s",
                        context, self._retry_config.max_retries, e.message,
                    )
                    raise

                logger.error(
                    "Client error %d [%s] non-retryable: %s",
                    e.code, context, e.message,
                )
                raise
            except gemini_errors.ServerError as e:
                last_error = e
                if attempt < self._retry_config.max_retries - 1:
                    delay = self._backoff(attempt)
                    logger.warning(
                        "Server error %d [%s] retry=%d/%d waiting=%.1fs",
                        e.code, context, attempt + 1, self._retry_config.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "Server error retries exhausted [%s] after %d attempts: %s",
                    context, self._retry_config.max_retries, e.message,
                )
                raise

        raise RuntimeError(
            f"Unreachable: _call_with_retry for [{context}]: {last_error}"
        )

    # -- backoff -------------------------------------------------------------

    def _backoff(self, attempt: int, rate_limit: bool = False) -> float:
        base = self._retry_config.base_delay_seconds
        if rate_limit:
            base *= 3.0
        delay = min(base ** attempt, self._retry_config.max_delay_seconds)
        return delay * random.uniform(0.5, 1.5)

    # -- public methods ------------------------------------------------------

    async def analyze_batch(self, logs, mode="auto", progress_callback=None, chunk_percentage=10, total_lines=0):
        if mode == "line-by-line":
            return await self._analyze_line_by_line(logs, progress_callback)
        elif mode == "intensive":
            return await self._analyze_chunked(logs, chunk_size=20, progress_callback=progress_callback)
        elif mode == "chunking":
            if total_lines == 0:
                total_lines = len(logs)
            chunk_size = max(10, int(total_lines * chunk_percentage / 100))
            return await self._analyze_chunked(logs, chunk_size=chunk_size, progress_callback=progress_callback)
        else:
            return await self._analyze_chunked(logs, chunk_size=10, progress_callback=progress_callback)

    async def explain_batch(self, logs, mode="auto", progress_callback=None, chunk_percentage=10):
        chunks = [logs[i:i + 10] for i in range(0, len(logs), 10)]
        total_chunks = len(chunks)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0

        async def explain_chunk(chunk):
            nonlocal completed
            async with semaphore:
                chunk_text = "\n".join([f"Line {ln}: {c}" for ln, c in chunk])
                prompt = (
                    "The following log lines have been flagged as anomalous by an ML anomaly detection system.\n"
                    "For each line, explain why it is unusual or suspicious in a system log context.\n"
                    "Respond in JSON as a list of objects, each with 'line' (integer) and 'reason' (string).\n"
                    "\n"
                    f"Logs:\n{chunk_text}"
                )
                chunk_results = []
                try:
                    text = await self._call_with_retry(
                        prompt, context=f"explain lines {chunk[0][0]}-{chunk[-1][0]}",
                    )
                    batch = json.loads(text)
                    if isinstance(batch, dict) and "anomalies" in batch:
                        batch = batch["anomalies"]
                    elif isinstance(batch, dict):
                        batch = [batch]
                    if not isinstance(batch, list):
                        raise ValueError(f"Expected list, got {type(batch).__name__}")
                    for item in batch:
                        line_num = item.get("line")
                        content = next((c for ln, c in chunk if ln == line_num), "Unknown")
                        chunk_results.append({
                            "line": line_num, "content": content,
                            "anomaly": True, "reason": item.get("reason", "Anomalous log line"),
                        })
                except Exception as e:
                    logger.error("Failed to explain chunk lines %d-%d: %s", chunk[0][0], chunk[-1][0], e)
                    for ln, c in chunk:
                        chunk_results.append({
                            "line": ln, "content": c,
                            "anomaly": True, "reason": f"Explanation failed: {str(e)}",
                        })
                completed += 1
                if progress_callback:
                    await progress_callback(completed, total_chunks)
                return chunk_results

        tasks = [explain_chunk(chunk) for chunk in chunks]
        all_results = await asyncio.gather(*tasks)
        results = []
        for cr in all_results:
            results.extend(cr)
        results.sort(key=lambda r: r["line"])
        return results

    # -- internal helpers ----------------------------------------------------

    async def _analyze_line_by_line(self, logs, progress_callback=None):
        total = len(logs)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0

        async def process_one(line_num, content):
            nonlocal completed
            async with semaphore:
                prompt = (
                    "Analyze the following log line for anomalies.\n"
                    "Is it an anomaly? Respond in JSON format with 'anomaly' (boolean) and 'reason' (string).\n"
                    f"Log: {content}"
                )
                result = {"line": line_num, "content": content, "anomaly": False, "reason": "Analysis failed"}
                try:
                    text = await self._call_with_retry(prompt, context=f"line {line_num}")
                    data = json.loads(text)
                    result = {
                        "line": line_num, "content": content,
                        "anomaly": data.get("anomaly", False),
                        "reason": data.get("reason", "N/A"),
                    }
                except Exception as e:
                    logger.error("Failed to analyze line %d: %s", line_num, e)
                completed += 1
                if progress_callback:
                    await progress_callback(completed, total)
                return result

        tasks = [process_one(ln, c) for ln, c in logs]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda r: r["line"])
        return results

    async def _analyze_chunked(self, logs, chunk_size=10, progress_callback=None):
        chunks = [logs[i:i + chunk_size] for i in range(0, len(logs), chunk_size)]
        total_chunks = len(chunks)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0

        async def process_chunk(chunk):
            nonlocal completed
            async with semaphore:
                chunk_line_nums = [ln for ln, _ in chunk]
                chunk_text = "\n".join([f"Line {ln}: {c}" for ln, c in chunk])
                prompt = (
                    "Analyze the following batch of logs for anomalies.\n"
                    "Identify any lines that are unusual, indicate errors, or show suspicious patterns.\n"
                    "Respond in JSON as a list of objects, each containing 'line' (integer), "
                    "'anomaly' (boolean), and 'reason' (string).\n"
                    "Only include lines that ARE anomalies.\n"
                    "\n"
                    f"Logs:\n{chunk_text}"
                )
                chunk_results = []
                try:
                    text = await self._call_with_retry(
                        prompt, context=f"chunk lines {chunk[0][0]}-{chunk[-1][0]}",
                    )
                    batch = json.loads(text)
                    if isinstance(batch, dict) and "anomalies" in batch:
                        batch = batch["anomalies"]
                    elif isinstance(batch, dict):
                        batch = [batch]
                    if not isinstance(batch, list):
                        raise ValueError(f"Expected list, got {type(batch).__name__}")
                    for res in batch:
                        if res.get("anomaly") and res.get("line") in chunk_line_nums:
                            content = next((c for ln, c in chunk if ln == res["line"]), "Unknown")
                            chunk_results.append({
                                "line": res["line"], "content": content,
                                "anomaly": True, "reason": res.get("reason", "N/A"),
                            })
                except Exception as e:
                    logger.error("Failed to analyze chunk lines %d-%d: %s", chunk[0][0], chunk[-1][0], e)
                    for ln, c in chunk:
                        chunk_results.append({
                            "line": ln, "content": c,
                            "anomaly": False, "reason": f"Analysis failed: {str(e)}",
                        })
                completed += 1
                if progress_callback:
                    await progress_callback(completed, total_chunks)
                return chunk_results

        tasks = [process_chunk(chunk) for chunk in chunks]
        all_results = await asyncio.gather(*tasks)
        results = []
        for cr in all_results:
            results.extend(cr)
        results.sort(key=lambda r: r["line"])
        return results


# ---------------------------------------------------------------------------
# Standalone helpers used by app.py (API key validation with caching)
# ---------------------------------------------------------------------------
_validation_cache: dict[str, tuple[float, list[str]]] = {}
_validation_cooldowns: dict[str, float] = {}


async def validate_api_key(google_api_key: str) -> dict:
    """Validate a Gemini API key, returning available models.
    
    Cached per key for 5 minutes so page refreshes don't call the API.
    Uses the same shared rate limiter so validation doesn't steal
    capacity from analysis calls for the same key.
    """
    k = _key_for(google_api_key)

    now = time.monotonic()

    # Respect cooldown from previous rate-limit
    cd = _validation_cooldowns.get(k, 0.0)
    if now < cd:
        raise ValueError("API key validation temporarily rate-limited. Wait a moment and try again.")

    # Cache hit
    cached = _validation_cache.get(k)
    if cached and now < cached[0]:
        return {"status": "success", "models": cached[1]}

    # Rate-limit via shared limiter
    rl_config = RateLimitConfig()
    shared = _get_shared(google_api_key, rl_config)
    await shared.rate_limiter.acquire()

    client = genai.Client(api_key=google_api_key)
    try:
        models = await asyncio.wait_for(
            client.aio.models.list(),
            timeout=15,
        )
        gemini_models = [m.name for m in models if m.name.startswith("models/gemini")]
        _validation_cache[k] = (time.monotonic() + 300, gemini_models)
        return {"status": "success", "models": gemini_models}
    except gemini_errors.ClientError as e:
        if e.code == 429:
            _validation_cooldowns[k] = time.monotonic() + 30.0
            logger.warning("Validation rate-limited, cooling down for 30s")
        raise ValueError(f"{e.message or str(e)}")
    except Exception as e:
        raise ValueError(f"Could not validate API key: {e}")
