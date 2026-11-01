# ML pre-filter

The pre-filter (`src/prefilter.py`, class `LogPrefilter`) is an unsupervised anomaly detector. It learns the shape of "normal" lines in the uploaded file and flags the ones that do not fit. No training data or labels are needed: a fresh model is trained on every upload.

The approach comes from [AICoE/log-anomaly-detector](https://github.com/AICoE/log-anomaly-detector), which uses Word2Vec embeddings and a Self-Organizing Map (SOM). This implementation keeps the SOM but replaces Word2Vec with cheap character-level and statistical features, so it needs only NumPy.

## 1. Features

Each line becomes a 49-dimensional vector:

| Group | Dimensions | Features |
|---|---|---|
| Size | 3 | character count, word count, mean word length |
| Character-class ratios | 8 | digits, uppercase, special characters, spaces, punctuation (`.:;!?`), brackets, slashes/dashes, hex letters (a–f) |
| Letter frequencies | 26 | frequency of each letter a–z (case-insensitive) |
| Digit frequencies | 10 | frequency of each digit 0–9 |
| Other | 2 | unique-word ratio, double-quote ratio |

Ratios and frequencies are divided by the line length. All features are then z-score normalised using the mean and standard deviation of the uploaded file.

## 2. Self-Organizing Map

- A `PREFILTER_MAP_SIZE × PREFILTER_MAP_SIZE` grid of nodes (default 12 × 12), each a 49-dimensional weight vector, is initialised randomly. The RNG is seeded with `PREFILTER_SEED` so results are reproducible.
- For `min(5000, max(800, 15 × n_lines))` iterations, the trainer takes the next line, finds its **best-matching unit** (BMU, the nearest node) and pulls the BMU and its neighbours towards the line:

  ```
  w ← w + η(t) · h(t) · (x − w)
  ```

  where the learning rate `η(t) = 0.5 · exp(−t / (T/2))`, and `h(t)` is a Gaussian neighbourhood whose width `σ(t) = σ₀ · exp(−t / (T/3))` shrinks over time.
- The trainer counts how often each node is a BMU (`bmu_hits`).

## 3. Scoring and threshold

For a line `x` with BMU `b`:

```
score(x) = ‖x − w_b‖² / max(√hits(b), 0.01)
```

The first factor is the **quantisation error**: how far the line is from anything the map learned. The second boosts lines that land on rarely used nodes. Scores are computed for all lines at once with matrix operations, using `‖x−w‖² = ‖x‖² − 2x·w + ‖w‖²` in batches of 4,096.

The threshold is the `PREFILTER_PERCENTILE`-th percentile of the training scores (default 85), so roughly the top 15% most unusual lines are flagged.

## 4. Tuning

| Goal | Change |
|---|---|
| Flag fewer lines (higher precision) | Raise `PREFILTER_PERCENTILE` (e.g. 95) |
| Flag more lines (higher recall) | Lower `PREFILTER_PERCENTILE` (e.g. 75) |
| Very heterogeneous logs | Increase `PREFILTER_MAP_SIZE` (e.g. 16) |
| Different result on every run | Set `PREFILTER_SEED` to an empty value |

## 5. Performance

On a laptop CPU, training and scoring the 25,567-line `data/Linux.log` takes about a second. Feature extraction is pure Python and is the main cost on very large files.

## 6. Limitations

- Features capture *shape* (length, character mix), not meaning. A malicious line that looks structurally normal will not be flagged. That is what the LLM stage is for.
- Because the threshold is a percentile, a fixed share of lines is always flagged, even in a perfectly healthy file.
- Every upload retrains from scratch. Nothing is learned across files.

## Further reading

- [`reference/ml-core-math.txt`](reference/ml-core-math.txt): the original detailed mathematical write-up. It describes 44 features; the current code extracts 49.
- [`reference/prefilter-metrics.txt`](reference/prefilter-metrics.txt): AUROC, PR-AUC and precision/recall measured on synthetic Apache logs. It was measured with a 10 × 10 map; the default is now 12 × 12.
