import ollama
import logging
import json
import asyncio
import random
import time

logger = logging.getLogger("OllamaDetector")


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
    def __init__(self, model_name="llama3.2:1b", max_concurrency=2, ollama_options=None):
        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.ollama_options = ollama_options or {
            "num_gpu": -1,
            "num_ctx": 4096,
        }
        self.rate_limiter = RateLimiter(requests_per_second=4.0, burst=6)
        logger.info(
            f"OllamaDetector initialized with model: {model_name}, "
            f"concurrency: {max_concurrency}, options: {self.ollama_options}"
        )

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
            if total_lines == 0:
                total_lines = len(logs)
            chunk_size = max(10, int(total_lines * 10 / 100))
            return await self._analyze_chunked(logs, chunk_size=chunk_size, progress_callback=progress_callback)

    async def explain_batch(self, logs, mode="auto", progress_callback=None, chunk_percentage=10):
        chunks = [logs[i:i + 10] for i in range(0, len(logs), 10)]
        total_chunks = len(chunks)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0

        async def explain_chunk(chunk):
            nonlocal completed
            async with semaphore:
                chunk_text = "\n".join([f"Line {ln}: {c}" for ln, c in chunk])

                prompt = f"""The following log lines have been flagged as anomalous by an ML anomaly detection system.
For each line, explain why it is unusual or suspicious in a system log context.
Respond in JSON as a list of objects, each with 'line' (integer) and 'reason' (string).

Logs:
{chunk_text}"""

                chunk_results = []
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = await self._call_ollama(prompt)
                        batch = json.loads(response['response'])
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
                                "line": line_num,
                                "content": content,
                                "anomaly": True,
                                "reason": item.get("reason", "Anomalous log line")
                            })
                        break
                    except Exception as e:
                        is_transient = any(x in str(e).lower() for x in [
                            "timeout", "connection", "refused",
                            "reset", "eagain", "unavailable",
                        ])
                        if is_transient and attempt < max_retries - 1:
                            delay = (2 ** attempt) + random.uniform(0.5, 1.5)
                            logger.warning(f"Retrying chunk (attempt {attempt+2}/{max_retries}) in {delay:.1f}s: {e}")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"Error explaining chunk: {e}")
                            for ln, c in chunk:
                                chunk_results.append({
                                    "line": ln, "content": c,
                                    "anomaly": True,
                                    "reason": f"Failed to generate explanation: {str(e)}"
                                })
                            break

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

    async def _call_ollama(self, prompt, timeout=120):
        await self.rate_limiter.acquire()
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    ollama.generate,
                    model=self.model_name,
                    prompt=prompt,
                    format="json",
                    options=self.ollama_options,
                    keep_alive=-1,
                ),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Ollama request timed out after {timeout}s")

    async def _analyze_line_by_line(self, logs, progress_callback=None):
        total = len(logs)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0

        async def process_one(line_num, content):
            nonlocal completed
            async with semaphore:
                prompt = f"""Analyze the following log line for anomalies.
                Is it an anomaly? Respond in JSON format with 'anomaly' (boolean) and 'reason' (string).
                Log: {content}"""

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = await self._call_ollama(prompt)
                        data = json.loads(response['response'])
                        result = {
                            "line": line_num,
                            "content": content,
                            "anomaly": data.get("anomaly", False),
                            "reason": data.get("reason", "N/A")
                        }
                        break
                    except Exception as e:
                        is_transient = any(x in str(e).lower() for x in [
                            "timeout", "connection", "refused",
                            "reset", "eagain", "unavailable",
                        ])
                        if is_transient and attempt < max_retries - 1:
                            delay = (2 ** attempt) + random.uniform(0.5, 1.5)
                            logger.warning(f"Retrying line {line_num} (attempt {attempt+2}/{max_retries}) in {delay:.1f}s: {e}")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"Error analyzing line {line_num}: {e}")
                            result = {
                                "line": line_num,
                                "content": content,
                                "anomaly": False,
                                "reason": "Analysis failed"
                            }
                            break

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

                prompt = f"""Analyze the following batch of logs for anomalies.
                Identify any lines that are unusual, indicate errors, or show suspicious patterns.
                Respond in JSON format as a list of objects, each containing 'line' (integer), 'anomaly' (boolean), and 'reason' (string).
                Only include lines that ARE anomalies.

                Logs:
                {chunk_text}"""

                chunk_results = []
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = await self._call_ollama(prompt)
                        batch_results = json.loads(response['response'])
                        if isinstance(batch_results, dict) and "anomalies" in batch_results:
                            batch_results = batch_results["anomalies"]
                        elif isinstance(batch_results, dict):
                            batch_results = [batch_results]

                        if not isinstance(batch_results, list):
                            raise ValueError(f"Expected list response, got {type(batch_results).__name__}")

                        for res in batch_results:
                            if res.get("anomaly") and res.get("line") in chunk_line_nums:
                                content = next((c for ln, c in chunk if ln == res["line"]), "Unknown")
                                chunk_results.append({
                                    "line": res["line"],
                                    "content": content,
                                    "anomaly": True,
                                    "reason": res.get("reason", "N/A")
                                })
                        break
                    except Exception as e:
                        is_transient = any(x in str(e).lower() for x in [
                            "timeout", "connection", "refused",
                            "reset", "eagain", "unavailable",
                        ])
                        if is_transient and attempt < max_retries - 1:
                            delay = (2 ** attempt) + random.uniform(0.5, 1.5)
                            logger.warning(f"Retrying chunk at line {chunk[0][0]} (attempt {attempt+2}/{max_retries}) in {delay:.1f}s: {e}")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"Error analyzing chunk starting at line {chunk[0][0]}: {e}")
                            for ln, c in chunk:
                                chunk_results.append({
                                    "line": ln,
                                    "content": c,
                                    "anomaly": False,
                                    "reason": f"Analysis failed: {str(e)}"
                                })
                            break

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
