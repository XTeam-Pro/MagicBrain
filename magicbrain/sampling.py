from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .brain import TextBrain

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def normalize_probs(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    if s <= 0:
        return np.ones_like(p) / float(len(p))
    return p / s

def apply_sampling_filters(probs: np.ndarray, temperature: float = 0.8, top_k: int = 18, top_p: float = 0.92) -> np.ndarray:
    p = probs.astype(np.float64)

    if temperature and temperature != 1.0:
        p = p ** (1.0 / float(temperature))
        p = normalize_probs(p)

    if top_k and 0 < top_k < len(p):
        idx = np.argpartition(p, -top_k)[-top_k:]
        mask = np.zeros_like(p)
        mask[idx] = p[idx]
        p = normalize_probs(mask)

    if top_p and 0.0 < top_p < 1.0:
        order = np.argsort(-p)
        sp = p[order]
        csum = np.cumsum(sp)
        keep = csum <= top_p
        if not np.any(keep):
            keep[0] = True
        k = int(np.where(keep)[0][-1] + 1)
        keep_idx = order[:k]
        mask = np.zeros_like(p)
        mask[keep_idx] = p[keep_idx]
        p = normalize_probs(mask)

    return p

def sample(
    brain: TextBrain,
    stoi: dict,
    itos: dict,
    seed: str,
    n: int = 700,
    temperature: float = 0.75,
    top_k: int = 18,
    top_p: float = 0.92,
):
    brain.reset_state()
    seed_chars = [ch for ch in seed if ch in stoi]
    if not seed_chars:
        if stoi:
            seed_chars = [next(iter(stoi.keys()))]
        else:
            return ""

    for ch in seed_chars[:-1]:
        brain.forward(stoi[ch])

    x = stoi[seed_chars[-1]]
    out = list(seed_chars)

    for _ in range(n):
        probs = brain.forward(x)
        p = apply_sampling_filters(probs, temperature=temperature, top_k=top_k, top_p=top_p)
        x = int(brain.rng.choice(len(p), p=p))
        out.append(itos[x])

    return "".join(out)
