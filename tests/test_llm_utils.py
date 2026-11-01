import json

import pytest

from src.llm_utils import anomalies_for_chunk, explanations_for_chunk, failed_chunk, parse_llm_list

CHUNK = [(10, "line ten"), (11, "line eleven"), (12, "line twelve")]


@pytest.mark.parametrize("payload", [
    [{"line": 10}],
    {"anomalies": [{"line": 10}]},
    {"results": [{"line": 10}]},
    {"line": 10},
])
def test_parse_llm_list_shapes(payload):
    assert parse_llm_list(json.dumps(payload)) == [{"line": 10}]


def test_parse_llm_list_rejects_scalars():
    with pytest.raises(ValueError):
        parse_llm_list("42")


def test_parse_llm_list_drops_non_objects():
    assert parse_llm_list('[{"line": 1}, "junk", 3]') == [{"line": 1}]


def test_anomalies_for_chunk_filters_unknown_lines_and_non_anomalies():
    items = [
        {"line": 10, "anomaly": True, "reason": "bad"},
        {"line": "11", "anomaly": False, "reason": "fine"},
        {"line": 99, "anomaly": True, "reason": "hallucinated"},
        {"line": 12, "anomaly": True},
        {"line": 10, "anomaly": True, "reason": "duplicate"},
    ]
    results = anomalies_for_chunk(CHUNK, items)
    assert [(r["line"], r["content"], r["reason"]) for r in results] == [
        (10, "line ten", "bad"),
        (12, "line twelve", "N/A"),
    ]


def test_explanations_cover_every_line():
    items = [{"line": "11", "reason": "odd"}, {"line": 99, "reason": "hallucinated"}]
    results = explanations_for_chunk(CHUNK, items)
    assert [r["line"] for r in results] == [10, 11, 12]
    assert results[1]["reason"] == "odd"
    assert all(r["anomaly"] for r in results)
    assert "no LLM explanation" in results[0]["reason"]


def test_failed_chunk():
    results = failed_chunk(CHUNK, RuntimeError("boom"), anomaly=False)
    assert len(results) == 3
    assert results[0]["reason"] == "Analysis failed: boom"
    assert not results[0]["anomaly"]
