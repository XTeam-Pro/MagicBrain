from __future__ import annotations
import time
import numpy as np
from ..brain import TextBrain

def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos

def train_loop(
    brain: TextBrain,
    text: str,
    stoi: dict,
    steps: int = 80000,
    print_every: int = 5000
):
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    if len(ids) < 2:
        raise ValueError("Text too short.")

    print(f"Stats: N={brain.N}, K={brain.K}")
    print(f"Target Active: {brain.p['k_active']} ({brain.target_rate*100:.1f}%)")
    print(f"Sensory Fanout: {brain.sens_fanout}")
    print(
        f"alpha={brain.alpha:.2f}, beta={brain.beta:.2f}, m_fast={brain.m_fast}, m_slow={brain.m_slow} | "
        f"p_inhib={brain.p['p_inhib']:.2f} | cons_eps={brain.p['cons_eps']:.4f} | prune_every={brain.p['prune_every']}"
    )

    losses = []
    n = len(ids) - 1
    t0 = time.time()
    
    start_step = brain.step

    for i_step in range(steps):
        # We use brain.step to track global progress, but i_step for loop control
        # The data index depends on brain.step or just loop?
        # Let's just use sequential sampling from text based on step
        
        idx = (start_step + i_step) % n
        x = int(ids[idx])
        y = int(ids[idx + 1])

        probs = brain.forward(x)
        loss = brain.learn(y, probs)
        losses.append(loss)

        if (i_step + 1) % print_every == 0:
            dt = time.time() - t0
            avg_loss = float(np.mean(losses[-print_every:]))
            print(
                f"Step {brain.step}: Loss={avg_loss:.4f} | "
                f"DA={brain.dopamine:.3f} | AvgTheta={brain.avg_theta():.3f} | |W|={brain.mean_abs_w():.3f} | "
                f"Rate~{brain.firing_rate():.3f} | {dt:.1f}s"
            )
            t0 = time.time()

    return losses
