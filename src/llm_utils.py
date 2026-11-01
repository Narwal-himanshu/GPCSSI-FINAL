"""Helpers shared by the Ollama and Gemini detectors for parsing LLM output."""

import json


def parse_llm_list(text: str) -> list[dict]:
    """Parse a JSON response from an LLM into a list of result objects.

    Models are asked for a JSON list but frequently wrap it in an object
    (``{"anomalies": [...]}``, ``{"results": [...]}``) or return a single object.
    """
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("anomalies", "results", "lines", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data).__name__}")
    return [item for item in data if isinstance(item, dict)]


def _as_line_number(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def anomalies_for_chunk(chunk: list[tuple[int, str]], items: list[dict]) -> list[dict]:
    """Keep only items the model marked as anomalies that refer to lines in ``chunk``."""
    contents = dict(chunk)
    results = {}
    for item in items:
        line_num = _as_line_number(item.get("line"))
        if line_num in contents and item.get("anomaly") and line_num not in results:
            results[line_num] = {
                "line": line_num,
                "content": contents[line_num],
                "anomaly": True,
                "reason": str(item.get("reason") or "N/A"),
            }
    return list(results.values())


def explanations_for_chunk(chunk: list[tuple[int, str]], items: list[dict]) -> list[dict]:
    """Attach an explanation to every line in ``chunk``.

    Every line passed in was flagged by the ML pre-filter, so each one must appear
    in the output exactly once, even if the model skipped it or invented line numbers.
    """
    reasons = {}
    for item in items:
        line_num = _as_line_number(item.get("line"))
        if line_num is not None and line_num not in reasons and item.get("reason"):
            reasons[line_num] = str(item["reason"])
    return [
        {
            "line": ln,
            "content": content,
            "anomaly": True,
            "reason": reasons.get(ln, "Flagged by ML pre-filter (no LLM explanation returned)"),
        }
        for ln, content in chunk
    ]


def failed_chunk(chunk: list[tuple[int, str]], error: Exception, anomaly: bool) -> list[dict]:
    """Results for a chunk whose LLM call failed."""
    prefix = "Explanation failed" if anomaly else "Analysis failed"
    return [
        {"line": ln, "content": c, "anomaly": anomaly, "reason": f"{prefix}: {error}"}
        for ln, c in chunk
    ]
