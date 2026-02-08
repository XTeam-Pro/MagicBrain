from __future__ import annotations
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def normalize_probs(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    if s <= 0:
        return np.ones_like(p) / float(len(p))
    return p / s

def sparsify_topm(x: np.ndarray, m: int) -> np.ndarray:
    if m <= 0:
        return np.zeros_like(x)
    if m >= x.shape[0]:
        return x.copy()
    idx = np.argpartition(x, -m)[-m:]
    y = np.zeros_like(x)
    y[idx] = x[idx]
    return y

def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))
