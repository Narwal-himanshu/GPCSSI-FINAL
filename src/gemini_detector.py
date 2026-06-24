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
    """Token bucket rate limiter to prevent API burst rate limit hits."""
    def __init__(self, requests_per_second: float = 0.5, burst: int = 2):
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
        self.rate_limiter = RateLimiter(requests_per_second=0.5, burst=2)

        logger.info(
            f"GeminiDetector initialized with model: {model_name}, "
            f"concurrency: {max_concurrency}"
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
            return await self._analyze_chunked(logs, chunk_size=10, progress_callback=progress_callback)

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
                base_delay = 4

                for attempt in range(max_retries):
                    await self.rate_limiter.acquire()
                    try:
                        response = await asyncio.to_thread(
                            self.client.models.generate_content,
                            model=self.model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                            ),
                        )
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
                        if "429" in error_str or "503" in error_str or "Too Many Requests" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                            if attempt < max_retries - 1:
                                match = re.search(r'retry\s+(in|after)\s+([0-9.]+)', error_str, re.IGNORECASE)
                                if match:
                                    delay = float(match.group(2)) + random.uniform(0.5, 2.0)
                                elif "Please retry in" in error_str:
                                    match = re.search(r'Please retry in ([0-9.]+)s', error_str)
                                    delay = float(match.group(1)) + random.uniform(0.5, 2.0)
                                else:
                                    delay = min((base_delay ** attempt) + random.uniform(1, 3), 60.0)

                                logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}) on line {line_num}. Retrying in {delay:.2f}s...")
                                await asyncio.sleep(delay)
                            else:
                                logger.error(f"Max retries ({max_retries}) reached for line {line_num}: {e}")
                                result = {
                                    "line": line_num,
                                    "content": content,
                                    "anomaly": False,
                                    "reason": f"Analysis failed (rate limit exhausted): {str(e)}"
                                }
                                break
                        else:
                            logger.error(f"Error analyzing line {line_num}: {e}")
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
                base_delay = 4

                for attempt in range(max_retries):
                    await self.rate_limiter.acquire()
                    try:
                        response = await asyncio.to_thread(
                            self.client.models.generate_content,
                            model=self.model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                            ),
                        )
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
                        if "429" in error_str or "503" in error_str or "Too Many Requests" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                            if attempt < max_retries - 1:
                                match = re.search(r'retry\s+(in|after)\s+([0-9.]+)', error_str, re.IGNORECASE)
                                if match:
                                    delay = float(match.group(2)) + random.uniform(0.5, 2.0)
                                elif "Please retry in" in error_str:
                                    match = re.search(r'Please retry in ([0-9.]+)s', error_str)
                                    delay = float(match.group(1)) + random.uniform(0.5, 2.0)
                                else:
                                    delay = min((base_delay ** attempt) + random.uniform(1, 3), 60.0)

                                logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}) on chunk at line {chunk[0][0]}. Retrying in {delay:.2f}s...")
                                await asyncio.sleep(delay)
                            else:
                                logger.error(f"Max retries ({max_retries}) reached for chunk at line {chunk[0][0]}: {e}")
                                for ln, c in chunk:
                                    chunk_results.append({
                                        "line": ln,
                                        "content": c,
                                        "anomaly": False,
                                        "reason": f"Analysis failed (rate limit exhausted): {str(e)}"
                                    })
                                break
                        else:
                            logger.error(f"Error analyzing chunk starting at line {chunk[0][0]}: {e}")
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
