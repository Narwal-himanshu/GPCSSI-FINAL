import numpy as np
import math
import logging

logger = logging.getLogger("Prefilter")


class LogPrefilter:
    """
    Lightweight log anomaly pre-filter using text feature encoding
    and Self-Organizing Map (SOM) for unsupervised anomaly detection.

    Mirrors the approach from AICoE/log-anomaly-detector (Word2Vec + SOM)
    but uses only numpy with character-level and statistical features
    for text vectorization.

    Anomaly scoring combines:
    - Distance from line vector to its closest SOM node (quantization error)
    - Rarity of the BMU (how few training lines map to that node)

    Uses percentile-based thresholding to flag the most unusual lines.
    """

    def __init__(self, map_size=12, flag_percentile=85, learning_rate=0.5,
                 sigma=None, n_iterations=None):
        self.map_size = map_size
        self.flag_percentile = flag_percentile
        self.learning_rate = learning_rate
        self.sigma = sigma or (map_size / 2.0)
        self.n_iterations = n_iterations
        self.som = None
        self.input_dim = None
        self.bmu_hits = None
        self.threshold = None
        self.mean = None
        self.std = None

    def _extract_features(self, text):
        features = []
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        safe = max(char_count, 1)

        features.append(char_count)
        features.append(word_count)
        features.append(np.mean([len(w) for w in words]) if words else 0)

        digit_count = sum(1 for c in text if c.isdigit())
        upper_count = sum(1 for c in text if c.isupper())
        space_count = text.count(' ')
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        punct_count = sum(1 for c in text if c in '.:;!?')
        bracket_count = sum(1 for c in text if c in '[]{}()<>')
        slash_count = sum(1 for c in text if c in '/\\-')
        hex_chars = sum(1 for c in text if c in 'abcdefABCDEF' and c.isalpha())

        features.append(digit_count / safe)
        features.append(upper_count / safe)
        features.append(special_count / safe)
        features.append(space_count / safe)
        features.append(punct_count / safe)
        features.append(bracket_count / safe)
        features.append(slash_count / safe)
        features.append(hex_chars / safe)

        text_lower = text.lower()
        for c in 'abcdefghijklmnopqrstuvwxyz':
            features.append(text_lower.count(c) / safe)
        for d in '0123456789':
            features.append(text.count(d) / safe)

        features.append(len(set(words)) / max(word_count, 1))
        features.append(sum(1 for c in text if c == '"') / safe)

        return np.array(features, dtype=np.float32)

    def _extract_features_batch(self, texts):
        return np.array([self._extract_features(t) for t in texts])

    def _initialize_som(self, input_dim):
        self.som = np.random.randn(self.map_size, self.map_size, input_dim).astype(np.float32)
        self.input_dim = input_dim
        self.bmu_hits = np.zeros((self.map_size, self.map_size), dtype=np.int32)

    def _find_bmu(self, vector):
        distances = np.sum((self.som - vector) ** 2, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def _get_neighborhood(self, bmu, iteration, total_iterations):
        sigma_t = self.sigma * math.exp(-iteration / (total_iterations / 3))
        lr_t = self.learning_rate * math.exp(-iteration / (total_iterations / 2))
        x = np.arange(self.map_size)
        y = np.arange(self.map_size)
        xx, yy = np.meshgrid(x, y)
        dist = np.sqrt((xx - bmu[0]) ** 2 + (yy - bmu[1]) ** 2)
        h = np.exp(-dist ** 2 / (2 * max(sigma_t ** 2, 1e-8)))
        return h, lr_t

    def fit(self, lines, verbose=True, progress_callback=None):
        if not lines:
            raise ValueError("No log lines provided for training")

        vectors = self._extract_features_batch(lines)
        if progress_callback:
            progress_callback(5, 100, "Extracting features")

        self.mean = np.mean(vectors, axis=0)
        self.std = np.std(vectors, axis=0)
        self.std[self.std == 0] = 1.0
        vectors = (vectors - self.mean) / self.std

        self.input_dim = vectors.shape[1]
        self._initialize_som(self.input_dim)

        n_samples = vectors.shape[0]
        total_iters = self.n_iterations or min(5000, max(800, 15 * n_samples))
        self.n_iterations = total_iters

        logger.info(f"SOM training: {n_samples} samples, {total_iters} iterations, "
                     f"grid {self.map_size}x{self.map_size}")

        report_every = max(1, total_iters // 20)
        for i in range(total_iters):
            idx = i % n_samples
            vector = vectors[idx]
            bmu = self._find_bmu(vector)
            h, lr = self._get_neighborhood(bmu, i, total_iters)
            self.som += lr * h[:, :, np.newaxis] * (vector[np.newaxis, np.newaxis, :] - self.som)
            self.bmu_hits[bmu[0], bmu[1]] += 1

            if (i + 1) % report_every == 0:
                pct = int((i + 1) / total_iters * 85) + 5
                if progress_callback:
                    progress_callback(pct, 100, f"SOM training {i+1}/{total_iters}")
                if verbose:
                    logger.info(f"  SOM: {i + 1}/{total_iters}")

        if progress_callback:
            progress_callback(90, 100, "Scoring lines")

        combined = np.array([self._compute_score(v) for v in vectors])
        self.threshold = float(np.percentile(combined, self.flag_percentile))

        if progress_callback:
            progress_callback(100, 100, "Done")

        n_anomalies = int(np.sum(combined > self.threshold))
        logger.info(f"Prefilter trained: flagged {n_anomalies}/{n_samples} "
                     f"(top {100 - self.flag_percentile}%, threshold={self.threshold:.4f})")

    def _compute_score(self, vector):
        distances = np.sum((self.som - vector) ** 2, axis=2)
        bmu_flat = np.argmin(distances)
        bmu = np.unravel_index(bmu_flat, distances.shape)
        min_dist = float(np.min(distances))
        hits = self.bmu_hits[bmu[0], bmu[1]]
        freq_penalty = 1.0 / max(math.sqrt(hits), 0.01)
        return min_dist * freq_penalty

    def filter(self, logs):
        if self.som is None:
            raise RuntimeError("Prefilter not fitted. Call fit() first.")

        texts = [content for _, content in logs]
        vectors = self._extract_features_batch(texts)
        vectors = (vectors - self.mean) / self.std

        anomalous = []
        for i, (line_num, content) in enumerate(logs):
            score = self._compute_score(vectors[i])
            if score > self.threshold:
                anomalous.append((line_num, content, float(score)))

        logger.info(f"Prefilter flagged {len(anomalous)}/{len(logs)} lines as anomalous")
        return anomalous
