import numpy as np
import pytest

from src.prefilter import LogPrefilter


def _normal_lines(n=400):
    return [
        f'10.0.0.{i % 250} - - [21/Jun/2026:17:03:{i % 60:02d} +0000] "GET /index.html HTTP/1.1" 200 {1000 + i}'
        for i in range(n)
    ]


def test_fit_requires_lines():
    with pytest.raises(ValueError):
        LogPrefilter().fit([])


def test_filter_requires_fit():
    with pytest.raises(RuntimeError):
        LogPrefilter().filter([(1, "x")])


def test_flags_injected_anomalies():
    lines = _normal_lines()
    anomalies = [
        "kernel: Out of memory: Kill process 4242 (java) score 987 or sacrifice child!!!",
        "FATAL ERROR: SEGFAULT at 0xDEADBEEF in module libcrypto.so {{{ stack corrupted }}}",
    ]
    lines[100] = anomalies[0]
    lines[300] = anomalies[1]

    pf = LogPrefilter(seed=0)
    pf.fit(lines, verbose=False)
    flagged = {ln for ln, _, _ in pf.filter(list(enumerate(lines, 1)))}

    assert 101 in flagged
    assert 301 in flagged
    # Roughly the top (100 - percentile)% of lines are flagged.
    assert len(flagged) <= len(lines) * 0.2


def test_seed_makes_results_reproducible():
    lines = _normal_lines()
    lines[50] = "ERROR something completely different happened here!!"
    logs = list(enumerate(lines, 1))

    a = LogPrefilter(seed=7)
    a.fit(lines, verbose=False)
    b = LogPrefilter(seed=7)
    b.fit(lines, verbose=False)
    assert a.filter(logs) == b.filter(logs)


def test_vectorised_scores_match_reference():
    lines = _normal_lines(200)
    pf = LogPrefilter(seed=1)
    pf.fit(lines, verbose=False)
    vectors = (pf._extract_features_batch(lines) - pf.mean) / pf.std
    fast = pf._compute_scores(vectors)
    slow = np.array([pf._compute_score(v) for v in vectors])
    np.testing.assert_allclose(fast, slow, rtol=1e-3, atol=1e-4)


def test_progress_callback_reaches_100():
    seen = []
    LogPrefilter(seed=0).fit(_normal_lines(100), verbose=False,
                             progress_callback=lambda c, t, m=None: seen.append(c))
    assert seen[-1] == 100
