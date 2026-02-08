from __future__ import annotations
import numpy as np
from ..brain import TextBrain
from .text_task import build_vocab
from ..sampling import sample

def benchmark_self_repair(
    brain: TextBrain,
    text: str,
    stoi: dict,
    itos: dict,
    eval_steps: int = 2000,
    recovery_steps: int = 8000,
    damage_frac: float = 0.2,
    report_every: int = 1000,
):
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    n = len(ids) - 1
    
    print(f"\nRunning self-repair benchmark (damage={damage_frac:.2f})...")

    def eval_loss(steps: int) -> float:
        losses = []
        # Use a fixed offset for eval to be consistent? 
        # Or just random/sequential. Let's use current brain step.
        start_idx = brain.step % n
        
        # Temporarily disable learning? 
        # TextBrain doesn't have a 'train' flag, but forward() doesn't learn.
        # learn() updates weights. So just forward() loop.
        
        brain.reset_state() # Reset state before eval to be clean?
        # Usually eval starts with some warmup.
        
        curr_x = int(ids[start_idx])
        
        # Warmup
        for _ in range(50):
            brain.forward(curr_x)
            start_idx = (start_idx + 1) % n
            curr_x = int(ids[start_idx])
            
        for _ in range(steps):
            probs = brain.forward(curr_x)
            target = int(ids[(start_idx + 1) % n])
            
            p = float(probs[target])
            losses.append(float(-np.log(p + 1e-9)))
            
            start_idx = (start_idx + 1) % n
            curr_x = target
            
        return float(np.mean(losses))

    def train_steps(steps: int):
        curr_idx = brain.step % n
        for _ in range(steps):
            x = int(ids[curr_idx])
            y = int(ids[(curr_idx + 1) % n])
            probs = brain.forward(x)
            brain.learn(y, probs)
            curr_idx = (curr_idx + 1) % n

    # 1. Pre-damage eval
    pre_eval = eval_loss(eval_steps)
    
    # 2. Damage
    brain.damage_edges(damage_frac)
    
    # 3. Post-damage eval
    post_damage_eval = eval_loss(eval_steps)
    
    # 4. Recovery
    recovery_curve = []
    remaining = int(recovery_steps)
    while remaining > 0:
        chunk = min(int(report_every), remaining)
        train_steps(chunk)
        remaining -= chunk
        recovery_curve.append(eval_loss(eval_steps))

    print(
        f"Self-repair(eval): pre={pre_eval:.4f} | post_damage={post_damage_eval:.4f} | "
        f"final={recovery_curve[-1]:.4f}"
    )

    if recovery_curve:
        parts = [f"{(i+1)*report_every}:{v:.4f}" for i, v in enumerate(recovery_curve)]
        print("Recovery curve:")
        print("  " + " | ".join(parts))

    seed_text = "To be, or not to be"
    generated = sample(brain, stoi, itos, seed=seed_text, n=300, temperature=0.75)
    print("-" * 60)
    print(generated)
    print("-" * 60)
