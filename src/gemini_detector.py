from google import genai
from google.genai import types
import logging
import json
import asyncio
import os
import re
import random
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeminiDetector")


class RateLimiter:
    def __init__(self, requests_per_second: float = 0.25, burst: int = 1):
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


class GeminiDetector:
    def __init__(self, model_name="gemini-2.0-flash", api_key=None, max_concurrency=2):
        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.client = genai.Client(api_key=api_key)
        self.rate_limiter = RateLimiter(requests_per_second=0.25, burst=1)
        self._cooldown_until = 0.0
        self._cooldown_lock = asyncio.Lock()

        logger.info(
            f"GeminiDetector initialized with model: {model_name}, "
            f"concurrency: {max_concurrency}"
        )

    async def _call_gemini(self, prompt, timeout=60):
        await self.rate_limiter.acquire()

        async with self._cooldown_lock:
            now = time.monotonic()
            if now < self._cooldown_until:
                wait = self._cooldown_until - now
                logger.warning(f"Global cooldown active, waiting {wait:.1f}s")
                await asyncio.sleep(wait)

        try:
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                ),
                timeout=timeout
            )
            return response
        except Exception as e:
            error_str = str(e)
            is_rate_limit = any(x in error_str for x in [
                "429", "503", "Too Many Requests", "RESOURCE_EXHAUSTED",
                "rate_limit", "quota", "RATE_LIMIT",
            ])
            if is_rate_limit:
                async with self._cooldown_lock:
                    self._cooldown_until = time.monotonic() + 10.0
                    logger.warning(f"Rate limit hit. Setting global cooldown for 10s")
            raise

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

                prompt = f"""The following log lines have been flagged as anomalous by an ML anomaly detection system.
For each line, explain why it is unusual or suspicious in a system log context.
Respond in JSON as a list of objects, each with 'line' (integer) and 'reason' (string).

Logs:
{chunk_text}"""

                chunk_results = []
                max_retries = 8
                base_delay = 5

                for attempt in range(max_retries):
                    try:
                        response = await self._call_gemini(prompt)
                        batch = json.loads(response.text)
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
                        error_str = str(e)
                        is_rate_limit = any(x in error_str for x in [
                            "429", "503", "Too Many Requests", "RESOURCE_EXHAUSTED",
                            "rate_limit", "quota", "RATE_LIMIT",
                        ])
                        if is_rate_limit and attempt < max_retries - 1:
                            delay = min((base_delay ** attempt) + random.uniform(1, 5), 120.0)
                            logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}). Retrying in {delay:.2f}s...")
                            await asyncio.sleep(delay)
                        elif attempt < max_retries - 1:
                            delay = (2 ** attempt) + random.uniform(0.5, 2)
                            logger.warning(f"Transient error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"Max retries ({max_retries}) reached for chunk: {e}")
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

                max_retries = 8
                base_delay = 5

                for attempt in range(max_retries):
                    try:
                        response = await self._call_gemini(prompt)
                        data = json.loads(response.text)
                        result = {
                            "line": line_num,
                            "content": content,
                            "anomaly": data.get("anomaly", False),
                            "reason": data.get("reason", "N/A")
                        }
                        break
                    except Exception as e:
                        error_str = str(e)
                        is_rate_limit = any(x in error_str for x in [
                            "429", "503", "Too Many Requests", "RESOURCE_EXHAUSTED",
                            "rate_limit", "quota", "RATE_LIMIT",
                        ])
                        if is_rate_limit and attempt < max_retries - 1:
                            delay = min((base_delay ** attempt) + random.uniform(1, 5), 120.0)
                            logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}) on line {line_num}. Retrying in {delay:.2f}s...")
                            await asyncio.sleep(delay)
                        elif attempt < max_retries - 1:
                            delay = (2 ** attempt) + random.uniform(0.5, 2)
                            logger.warning(f"Transient error on line {line_num} (attempt {attempt+1}/{max_retries}): {e}")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"Max retries ({max_retries}) reached for line {line_num}: {e}")
                            result = {
                                "line": line_num,
                                "content": content,
                                "anomaly": False,
                                "reason": f"Analysis failed: {str(e)}"
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
                max_retries = 8
                base_delay = 5

                for attempt in range(max_retries):
                    try:
                        response = await self._call_gemini(prompt)
                        batch_results = json.loads(response.text)
                        if isinstance(batch_results, dict) and "anomalies" in batch_results:
                            batch_results = batch_results["anomalies"]
                        elif isinstance(batch_results, dict):
                            batch_results = [batch_results]

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
                        error_str = str(e)
                        is_rate_limit = any(x in error_str for x in [
                            "429", "503", "Too Many Requests", "RESOURCE_EXHAUSTED",
                            "rate_limit", "quota", "RATE_LIMIT",
                        ])
                        if is_rate_limit and attempt < max_retries - 1:
                            delay = min((base_delay ** attempt) + random.uniform(1, 5), 120.0)
                            logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}) on chunk at line {chunk[0][0]}. Retrying in {delay:.2f}s...")
                            await asyncio.sleep(delay)
                        elif attempt < max_retries - 1:
                            delay = (2 ** attempt) + random.uniform(0.5, 2)
                            logger.warning(f"Transient error on chunk at line {chunk[0][0]} (attempt {attempt+1}/{max_retries}): {e}")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"Max retries ({max_retries}) reached for chunk at line {chunk[0][0]}: {e}")
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
