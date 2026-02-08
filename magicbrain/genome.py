from __future__ import annotations
import numpy as np

def decode_genome(genome: str) -> dict:
    g = np.array([ord(c) - 48 for c in genome if c in "0123"], dtype=np.int32)
    if len(g) < 24:
        g = np.pad(g, (0, 24 - len(g)), constant_values=1)

    def b4(i: int, n: int) -> int:
        x = 0
        for k in range(n):
            x = x * 4 + int(g[(i + k) % len(g)])
        return x

    N = 256 + 64 * b4(0, 2)
    K = 8 + b4(2, 1) * 4
    p_long = 0.02 + 0.02 * b4(3, 1)

    lr = 0.0005 + 0.0005 * b4(4, 1)

    k_active = max(48, int(N * (0.04 + 0.01 * b4(5, 1))))

    trace_fast_decay = 0.92 + 0.02 * b4(6, 1)
    trace_slow_decay = 0.985 + 0.003 * b4(10, 1)

    homeo = 0.001 + 0.001 * b4(7, 1)
    buf_decay = 0.92 + 0.02 * b4(12, 1)

    seed = b4(8, 4)

    alpha = 0.25 + 0.15 * b4(13, 1)
    beta = 0.05 + 0.05 * b4(14, 1)

    p_inhib = 0.10 + 0.05 * b4(15, 1)

    dopamine_gain = 0.8 + 0.4 * b4(16, 1)
    dopamine_bias = -0.2 + 0.2 * b4(17, 1)

    cons_eps = 0.0005 + 0.0005 * b4(18, 1)
    w_fast_decay = 0.9990 + 0.0003 * b4(19, 1)

    prune_every = 800 + 200 * b4(20, 1)
    prune_frac = 0.02 + 0.01 * b4(21, 1)
    rewire_frac = 0.50 + 0.10 * b4(22, 1)

    return dict(
        N=N,
        K=K,
        p_long=p_long,
        lr=lr,
        k_active=k_active,
        trace_fast_decay=trace_fast_decay,
        trace_slow_decay=trace_slow_decay,
        homeo=homeo,
        buf_decay=buf_decay,
        seed=seed,
        alpha=alpha,
        beta=beta,
        p_inhib=p_inhib,
        dopamine_gain=dopamine_gain,
        dopamine_bias=dopamine_bias,
        cons_eps=cons_eps,
        w_fast_decay=w_fast_decay,
        prune_every=prune_every,
        prune_frac=prune_frac,
        rewire_frac=rewire_frac,
    )
