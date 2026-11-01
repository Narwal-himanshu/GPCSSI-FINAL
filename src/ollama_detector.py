import asyncio
import logging
import random
import time

import ollama

from .llm_utils import anomalies_for_chunk, explanations_for_chunk, failed_chunk, parse_llm_list

logger = logging.getLogger("OllamaDetector")

_TRANSIENT_MARKERS = ("timeout", "timed out", "connection", "refused", "reset", "eagain", "unavailable")


class RateLimiter:
    """Token bucket rate limiter — prevents overwhelming Ollama with concurrent requests."""
    def __init__(self, requests_per_second: float = 4.0, burst: int = 6):
        self.rate = requests_per_second
        self.burst = burst
        self.tokens = burst
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            wait = (1 - self.tokens) / self.rate
            self.tokens = 0
            self.last_refill = now + wait
        await asyncio.sleep(wait)


class OllamaDetector:
    def __init__(self, model_name="llama3.2:1b", max_concurrency=2, ollama_options=None,
                 host=None, timeout=120.0, max_retries=3):
        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.ollama_options = ollama_options or {
            "num_gpu": -1,
            "num_ctx": 4096,
        }
        self.client = ollama.AsyncClient(host=host)
        self.rate_limiter = RateLimiter(requests_per_second=4.0, burst=6)
        logger.info(
            "OllamaDetector initialized with model: %s, concurrency: %d, options: %s",
            model_name, max_concurrency, self.ollama_options,
        )

    async def analyze_batch(self, logs, mode="auto", progress_callback=None, chunk_percentage=10, total_lines=0):
        if mode == "line-by-line":
            return await self._analyze_line_by_line(logs, progress_callback)
        if mode == "intensive":
            return await self._analyze_chunked(logs, chunk_size=20, progress_callback=progress_callback)
        if not total_lines:
            total_lines = len(logs)
        pct = chunk_percentage if mode == "chunking" else 10
        chunk_size = max(10, int(total_lines * pct / 100))
        return await self._analyze_chunked(logs, chunk_size=chunk_size, progress_callback=progress_callback)

    async def explain_batch(self, logs, mode="auto", progress_callback=None, chunk_percentage=10):
        prompt_header = (
            "The following log lines have been flagged as anomalous by an ML anomaly detection system.\n"
            "For each line, explain why it is unusual or suspicious in a system log context.\n"
            "Respond in JSON as a list of objects, each with 'line' (integer) and 'reason' (string).\n"
            "\n"
            "Logs:\n"
        )

        async def explain_chunk(chunk):
            chunk_text = "\n".join(f"Line {ln}: {c}" for ln, c in chunk)
            try:
                context = f"lines {chunk[0][0]}-{chunk[-1][0]}"
                items = await self._generate_list(prompt_header + chunk_text, context=context)
                return explanations_for_chunk(chunk, items)
            except Exception as e:
                logger.error("Error explaining chunk starting at line %d: %s", chunk[0][0], e)
                return failed_chunk(chunk, e, anomaly=True)

        return await self._run_chunks(logs, 250, explain_chunk, progress_callback)

    async def _call_ollama(self, prompt):
        await self.rate_limiter.acquire()
        try:
            response = await asyncio.wait_for(
                self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    format="json",
                    options=self.ollama_options,
                    keep_alive=-1,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s") from None
        return response["response"]

    async def _generate_list(self, prompt, context=""):
        """Call Ollama and parse a JSON list, retrying transient failures with backoff."""
        for attempt in range(self.max_retries):
            try:
                return parse_llm_list(await self._call_ollama(prompt))
            except Exception as e:
                is_transient = any(x in str(e).lower() for x in _TRANSIENT_MARKERS)
                if not is_transient or attempt == self.max_retries - 1:
                    raise
                delay = (2 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning("Retrying %s (attempt %d/%d) in %.1fs: %s",
                               context, attempt + 2, self.max_retries, delay, e)
                await asyncio.sleep(delay)

    async def _run_chunks(self, logs, chunk_size, worker, progress_callback):
        chunks = [logs[i:i + chunk_size] for i in range(0, len(logs), chunk_size)]
        total_chunks = len(chunks)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0

        async def run(chunk):
            nonlocal completed
            async with semaphore:
                chunk_results = await worker(chunk)
            completed += 1
            if progress_callback:
                await progress_callback(completed, total_chunks)
            return chunk_results

        all_results = await asyncio.gather(*(run(chunk) for chunk in chunks))
        results = [r for chunk_results in all_results for r in chunk_results]
        results.sort(key=lambda r: r["line"])
        return results

    async def _analyze_line_by_line(self, logs, progress_callback=None):
        async def process_one(chunk):
            line_num, content = chunk[0]
            prompt = (
                "Analyze the following log line for anomalies.\n"
                "Is it an anomaly? Respond in JSON format with 'anomaly' (boolean) and 'reason' (string).\n"
                f"Log: {content}"
            )
            try:
                items = await self._generate_list(prompt, context=f"line {line_num}")
                data = items[0] if items else {}
                return [{
                    "line": line_num,
                    "content": content,
                    "anomaly": bool(data.get("anomaly", False)),
                    "reason": str(data.get("reason", "N/A")),
                }]
            except Exception as e:
                logger.error("Error analyzing line %d: %s", line_num, e)
                return failed_chunk(chunk, e, anomaly=False)

        return await self._run_chunks(logs, 1, process_one, progress_callback)

    async def _analyze_chunked(self, logs, chunk_size=10, progress_callback=None):
        prompt_header = (
            "Analyze the following batch of logs for anomalies.\n"
            "Identify any lines that are unusual, indicate errors, or show suspicious patterns.\n"
            "Respond in JSON format as a list of objects, each containing 'line' (integer), "
            "'anomaly' (boolean), and 'reason' (string).\n"
            "Only include lines that ARE anomalies.\n"
            "\n"
            "Logs:\n"
        )

        async def process_chunk(chunk):
            chunk_text = "\n".join(f"Line {ln}: {c}" for ln, c in chunk)
            try:
                items = await self._generate_list(prompt_header + chunk_text, context=f"chunk at line {chunk[0][0]}")
                return anomalies_for_chunk(chunk, items)
            except Exception as e:
                logger.error("Error analyzing chunk starting at line %d: %s", chunk[0][0], e)
                return failed_chunk(chunk, e, anomaly=False)

        return await self._run_chunks(logs, chunk_size, process_chunk, progress_callback)
