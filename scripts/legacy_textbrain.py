# textbrain_v5_magicbrain_mvp.py
# Stable sparse "DNA -> 3D recurrent graph -> spiking -> local plasticity" + MagicBrain MVP:
# - E/I neuron types
# - dopamine-gated local plasticity
# - fast/slow weights with consolidation
# - structural pruning + rewiring
# - self-repair benchmark

from __future__ import annotations

import time
import numpy as np


# -----------------------------
# Helpers
# -----------------------------
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


# -----------------------------
# 1) DNA (base-4 genome) -> phenotype
# -----------------------------
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


# -----------------------------
# 2) Build 3D sparse graph + delays
# -----------------------------
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


# -----------------------------
# 3) Vocab
# -----------------------------
def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


# -----------------------------
# 4) Brain
# -----------------------------
class TextBrain:
    def __init__(self, genome: str, vocab_size: int):
        self.p = decode_genome(genome)
        self.N = int(self.p["N"])
        self.K = int(self.p["K"])
        self.vocab_size = int(vocab_size)

        self.rng = np.random.default_rng(int(self.p["seed"]))
        self.pos, self.src, self.dst, self.delay, self.idx_by_delay = build_graph(
            self.N, self.K, float(self.p["p_long"]), int(self.p["seed"])
        )

        self.is_inhib = (self.rng.random(self.N) < float(self.p["p_inhib"])).astype(np.bool_)

        self.w_slow = self.rng.normal(0, 0.03, size=self.src.shape[0]).astype(np.float32)
        self.w_fast = np.zeros_like(self.w_slow)

        self._enforce_ei_signs()

        self.a = np.zeros(self.N, dtype=np.float32)
        self.theta = np.zeros(self.N, dtype=np.float32)
        self.trace_fast = np.zeros(self.N, dtype=np.float32)
        self.trace_slow = np.zeros(self.N, dtype=np.float32)
        self.buffers = [np.zeros(self.N, dtype=np.float32) for _ in range(6)]

        self.R = self.rng.normal(0, 0.12, size=(self.N, vocab_size)).astype(np.float32)
        self.b = np.zeros(vocab_size, dtype=np.float32)

        self.loss_ema = float(np.log(max(2, vocab_size)))

        self.recur_scale = float(1.0 / np.sqrt(float(self.K)))
        self.target_rate = float(self.p["k_active"]) / float(self.N)

        self.sens_fanout = int(max(10, min(16, int(self.p["k_active"]) // 4)))
        perm = self.rng.permutation(self.N).astype(np.int32)
        needed = vocab_size * self.sens_fanout
        sens_flat = perm[:needed] if needed <= self.N else np.resize(perm, needed)
        self.sens_idx = sens_flat.reshape(vocab_size, self.sens_fanout)

        self.noise_std = 0.01
        self.inp_gain = 1.0

        self.alpha = float(self.p["alpha"])
        self.beta = float(self.p["beta"])

        self.fast_clip = 1.5
        self.slow_clip = 2.0

        self.m_fast = min(self.N, 4 * int(self.p["k_active"]))
        self.m_slow = min(self.N, 8 * int(self.p["k_active"]))

        self.lr_out_mul = 0.6
        self.lr_rec_mul = 1.0
        self.max_R_update = 0.02
        self.max_b_update = 0.02

        self.state_sum_target = float(self.p["k_active"])

        self.dopamine = 0.0
        self.ach = 0.0
        self.serotonin = 0.0

        self.step = 0

    def _enforce_ei_signs(self):
        inhib_src = self.is_inhib[self.src]
        if np.any(inhib_src):
            self.w_slow[inhib_src] = -np.abs(self.w_slow[inhib_src])
            self.w_fast[inhib_src] = -np.abs(self.w_fast[inhib_src])

    def _effective_w(self) -> np.ndarray:
        return (self.w_slow + self.w_fast).astype(np.float32)

    def reset_state(self):
        self.a.fill(0)
        self.trace_fast.fill(0)
        self.trace_slow.fill(0)
        for b in self.buffers:
            b.fill(0)

    def compute_state(self) -> np.ndarray:
        tf = np.clip(self.trace_fast, 0.0, self.fast_clip)
        ts = np.clip(self.trace_slow, 0.0, self.slow_clip)

        tf = sparsify_topm(tf, self.m_fast)
        ts = sparsify_topm(ts, self.m_slow)

        state = (self.a + self.alpha * tf + self.beta * ts).astype(np.float32)

        s = float(np.sum(state))
        if s > 1e-6:
            state *= (self.state_sum_target / s)
        return state

    def _update_modulators(self, loss: float):
        adv = float(self.loss_ema - loss)
        self.loss_ema = 0.995 * self.loss_ema + 0.005 * loss

        gain = float(self.p["dopamine_gain"])
        bias = float(self.p["dopamine_bias"])
        self.dopamine = sigmoid(gain * adv + bias)

        return adv

    def forward(self, token_id: int) -> np.ndarray:
        self.step += 1

        delayed_now = self.buffers[1].copy()
        for d in range(1, 5):
            self.buffers[d] = self.buffers[d + 1]
        self.buffers[5].fill(0)

        bd = float(self.p["buf_decay"])
        for d in range(1, 6):
            self.buffers[d] *= bd

        x = delayed_now - self.theta
        if self.noise_std > 0:
            x += self.rng.normal(0, self.noise_std, size=self.N).astype(np.float32)

        a = np.zeros(self.N, dtype=np.float32)
        sidx = self.sens_idx[token_id]
        a[sidx] = 1.0

        remaining = int(self.p["k_active"] - self.sens_fanout)
        if remaining > 0:
            x2 = x.copy()
            x2[sidx] = -1e9
            x[sidx] += self.inp_gain

            idx = np.argpartition(x2, -remaining)[-remaining:]
            pos_idx = idx[x2[idx] > 0]
            if pos_idx.size >= remaining:
                idx = pos_idx[:remaining]
            a[idx] = 1.0

        self.a = a

        fd = float(self.p["trace_fast_decay"])
        sd = float(self.p["trace_slow_decay"])

        self.trace_fast = fd * self.trace_fast + self.a
        np.clip(self.trace_fast, 0.0, self.fast_clip, out=self.trace_fast)

        self.trace_slow = sd * self.trace_slow + 0.15 * self.trace_fast
        np.clip(self.trace_slow, 0.0, self.slow_clip, out=self.trace_slow)

        rs = self.recur_scale
        w_eff = self._effective_w()
        for d in range(1, 6):
            idxe = self.idx_by_delay[d]
            if idxe.size == 0:
                continue
            contrib = (self.a[self.src[idxe]] * w_eff[idxe] * rs).astype(np.float32)
            np.add.at(self.buffers[d], self.dst[idxe], contrib)
            np.clip(self.buffers[d], -5.0, 5.0, out=self.buffers[d])

        state = self.compute_state()
        logits = (state @ self.R + self.b).astype(np.float32)
        return softmax(logits)

    def _consolidate(self):
        eps = float(self.p["cons_eps"])
        if eps <= 0:
            return
        self.w_slow = ((1.0 - eps) * self.w_slow + eps * self.w_fast).astype(np.float32)

    def _decay_fast(self):
        d = float(self.p["w_fast_decay"])
        self.w_fast *= d

    def _prune_and_rewire(self):
        prune_frac = float(self.p["prune_frac"])
        if prune_frac <= 0:
            return

        E = int(self.src.shape[0])
        n_prune = int(E * prune_frac)
        if n_prune <= 0:
            return

        w_eff = self._effective_w()
        absw = np.abs(w_eff)
        prune_idx = np.argpartition(absw, n_prune)[:n_prune]

        n_rewire = int(n_prune * float(self.p["rewire_frac"]))
        if n_rewire > 0:
            rewire_sel = prune_idx[:n_rewire]
            self.src[rewire_sel] = self.rng.integers(0, self.N, size=n_rewire, dtype=np.int32)
            self.dst[rewire_sel] = self.rng.integers(0, self.N, size=n_rewire, dtype=np.int32)
            dist = np.linalg.norm(self.pos[self.src[rewire_sel]] - self.pos[self.dst[rewire_sel]], axis=1).astype(np.float32)
            self.delay[rewire_sel] = np.clip((dist * 6).astype(np.int32) + 1, 1, 5)

            self.w_slow[rewire_sel] = self.rng.normal(0, 0.01, size=n_rewire).astype(np.float32)
            self.w_fast[rewire_sel] = 0.0

        zero_sel = prune_idx[n_rewire:]
        if zero_sel.size > 0:
            self.w_slow[zero_sel] = 0.0
            self.w_fast[zero_sel] = 0.0

        self._enforce_ei_signs()

        idx_by_delay = [np.array([], dtype=np.int32)]
        for d in range(1, 6):
            idx_by_delay.append(np.where(self.delay == d)[0].astype(np.int32))
        self.idx_by_delay = idx_by_delay

    def damage_edges(self, frac: float = 0.2):
        frac = clamp(float(frac), 0.0, 1.0)
        E = int(self.src.shape[0])
        n = int(E * frac)
        if n <= 0:
            return
        idx = self.rng.choice(E, size=n, replace=False)
        self.w_slow[idx] = 0.0
        self.w_fast[idx] = 0.0

    def learn(self, target_id: int, probs: np.ndarray) -> float:
        p = float(probs[target_id])
        loss = float(-np.log(p + 1e-9))

        adv = self._update_modulators(loss)

        lr = float(self.p["lr"])
        lr_out = lr * self.lr_out_mul
        lr_rec = lr * self.lr_rec_mul

        grad = probs.copy()
        grad[target_id] -= 1.0

        state = self.compute_state()

        dR = (lr_out * (state[:, None] * grad[None, :])).astype(np.float32)
        np.clip(dR, -self.max_R_update, self.max_R_update, out=dR)
        self.R -= dR

        db = (lr_out * grad).astype(np.float32)
        np.clip(db, -self.max_b_update, self.max_b_update, out=db)
        self.b -= db

        np.clip(self.R, -1.0, 1.0, out=self.R)

        pre = self.trace_fast[self.src]
        post = self.a[self.dst]

        dW = (lr_rec * self.dopamine * adv * pre * post).astype(np.float32)
        np.clip(dW, -0.02, 0.02, out=dW)
        self.w_fast += dW

        self._enforce_ei_signs()

        np.clip(self.w_fast, -0.5, 0.5, out=self.w_fast)
        np.clip(self.w_slow, -0.5, 0.5, out=self.w_slow)

        self._decay_fast()
        self._consolidate()

        self.theta += (float(self.p["homeo"]) * (self.a - self.target_rate)).astype(np.float32)

        prune_every = int(self.p["prune_every"])
        if prune_every > 0 and (self.step % prune_every) == 0:
            self._prune_and_rewire()

        return loss

    def avg_theta(self) -> float:
        return float(np.mean(self.theta))

    def mean_abs_w(self) -> float:
        return float(np.mean(np.abs(self._effective_w())))

    def firing_rate(self) -> float:
        return float(np.mean(self.a))


# -----------------------------
# Training
# -----------------------------
def train_on_text(genome: str, text: str, steps: int = 80000, print_every: int = 5000):
    stoi, itos = build_vocab(text)
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    if len(ids) < 2:
        raise ValueError("Text too short.")

    brain = TextBrain(genome, vocab_size=len(stoi))
    baseline = float(np.log(len(stoi)))

    print(f"Stats: N={brain.N}, K={brain.K}")
    print(f"Target Active: {brain.p['k_active']} ({brain.target_rate*100:.1f}%)")
    print(f"Sensory Fanout: {brain.sens_fanout}")
    print(f"Vocab size: {len(stoi)} | random baseline ~ log(V)={baseline:.3f}")
    print(
        f"alpha={brain.alpha:.2f}, beta={brain.beta:.2f}, m_fast={brain.m_fast}, m_slow={brain.m_slow} | "
        f"p_inhib={brain.p['p_inhib']:.2f} | cons_eps={brain.p['cons_eps']:.4f} | prune_every={brain.p['prune_every']}"
    )

    losses = []
    n = len(ids) - 1
    t0 = time.time()

    for step in range(steps):
        i = step % n
        x = int(ids[i])
        y = int(ids[i + 1])

        probs = brain.forward(x)
        loss = brain.learn(y, probs)
        losses.append(loss)

        if (step + 1) % print_every == 0:
            dt = time.time() - t0
            avg_loss = float(np.mean(losses[-print_every:]))
            print(
                f"Step {step+1}: Loss={avg_loss:.4f} | "
                f"DA={brain.dopamine:.3f} | AvgTheta={brain.avg_theta():.3f} | |W|={brain.mean_abs_w():.3f} | "
                f"Rate~{brain.firing_rate():.3f} | {dt:.1f}s"
            )
            t0 = time.time()

    return brain, stoi, itos


# -----------------------------
# Sampling
# -----------------------------
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
        seed_chars = [next(iter(stoi.keys()))]

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


# -----------------------------
# Benchmarks
# -----------------------------
def benchmark_self_repair(
    genome: str,
    text: str,
    pre_train_steps: int = 8000,
    eval_steps: int = 2000,
    recovery_steps: int = 8000,
    damage_frac: float = 0.2,
    report_every: int = 1000,
):
    stoi, itos = build_vocab(text)
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    if len(ids) < 2:
        raise ValueError("Text too short.")

    brain = TextBrain(genome, vocab_size=len(stoi))
    n = len(ids) - 1

    def eval_loss(steps: int) -> float:
        losses = []
        for step in range(steps):
            i = step % n
            x = int(ids[i])
            y = int(ids[i + 1])
            probs = brain.forward(x)
            p = float(probs[y])
            losses.append(float(-np.log(p + 1e-9)))
        return float(np.mean(losses))

    def train_steps(steps: int):
        for step in range(steps):
            i = step % n
            x = int(ids[i])
            y = int(ids[i + 1])
            probs = brain.forward(x)
            brain.learn(y, probs)

    train_steps(pre_train_steps)

    pre_eval = eval_loss(eval_steps)

    brain.damage_edges(damage_frac)

    post_damage_eval = eval_loss(eval_steps)

    recovery_curve = []
    remaining = int(recovery_steps)
    while remaining > 0:
        chunk = min(int(report_every), remaining)
        train_steps(chunk)
        remaining -= chunk
        recovery_curve.append(eval_loss(eval_steps))

    print(
        f"Self-repair(eval): pre={pre_eval:.4f} | post_damage={post_damage_eval:.4f} | "
        f"final={recovery_curve[-1]:.4f} | damage_frac={damage_frac:.2f}"
    )

    if recovery_curve:
        parts = [f"{(i+1)*report_every}:{v:.4f}" for i, v in enumerate(recovery_curve)]
        print("Recovery curve:")
        print("  " + " | ".join(parts))

    seed = "To be, or not to be"
    generated = sample(brain, stoi, itos, seed=seed, n=300, temperature=0.75, top_k=18, top_p=0.92)
    print("-" * 60)
    print(generated)
    print("-" * 60)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    text = """
From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:

But thou contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thy self thy foe, to thy sweet self too cruel:

Thou that art now the world's fresh ornament,
And only herald to the gaudy spring,
Within thine own bud buriest thy content,
And tender churl mak'st waste in niggarding:

Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.

To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
    """ * 40

    text = " ".join(text.split())

    genome = "30321031103321200112332100123"

    print(f"Initializing Brain with genome: {genome}...")
    p = decode_genome(genome)
    print(
        f"Phenotype: N={p['N']}, K={p['K']}, LR={p['lr']:.4f}, Active={p['k_active']}, "
        f"BufDecay={p['buf_decay']:.2f}, alpha={p['alpha']:.2f}, beta={p['beta']:.2f}, pI={p['p_inhib']:.2f}"
    )

    print("\nStarting training...")
    brain, stoi, itos = train_on_text(genome, text, steps=120000, print_every=5000)

    print("\nTraining done. Generating text...")
    seed = "To be, or not to be"
    generated = sample(brain, stoi, itos, seed=seed, n=500, temperature=0.75, top_k=18, top_p=0.92)

    print("-" * 60)
    print(generated)
    print("-" * 60)

    print("\nRunning self-repair benchmark...")
    benchmark_self_repair(
        genome,
        text,
        pre_train_steps=12000,
        eval_steps=2000,
        recovery_steps=12000,
        damage_frac=0.2,
        report_every=2000,
    )
