from __future__ import annotations
import numpy as np

def build_graph(N: int, K: int, p_long: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = rng.random((N, 3), dtype=np.float32)

    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2).astype(np.float32)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, K, axis=1)[:, :K]

    src = np.repeat(np.arange(N, dtype=np.int32), K)
    dst = nn.reshape(-1).astype(np.int32)

    n_long = int(src.shape[0] * p_long)
    if n_long > 0:
        long_src = rng.integers(0, N, size=n_long, dtype=np.int32)
        long_dst = rng.integers(0, N, size=n_long, dtype=np.int32)
        src = np.concatenate([src, long_src])
        dst = np.concatenate([dst, long_dst])

    dist = np.linalg.norm(pos[src] - pos[dst], axis=1).astype(np.float32)
    delay = np.clip((dist * 6).astype(np.int32) + 1, 1, 5)

    idx_by_delay = [np.array([], dtype=np.int32)]
    for d in range(1, 6):
        idx_by_delay.append(np.where(delay == d)[0].astype(np.int32))

    return pos, src, dst, delay, idx_by_delay
