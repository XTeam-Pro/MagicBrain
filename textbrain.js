/**
 * textbrain_v5_magicbrain_mvp.js
 *
 * Перевод на JavaScript (Node.js)
 * Stable sparse "DNA -> 3D recurrent graph -> spiking -> local plasticity" + MagicBrain MVP
 *
 * Для запуска: node textbrain.js
 */

const util = require('util');

// -----------------------------
// Math & Numpy Helpers
// -----------------------------

class Random {
    constructor(seed) {
        // Простой LCG генератор для воспроизводимости (как seeded numpy)
        this.state = seed ? seed : Math.floor(Math.random() * 1e9);
    }

    next() {
        this.state = (this.state * 1664525 + 1013904223) % 4294967296;
        return this.state;
    }

    nextFloat() {
        return this.next() / 4294967296;
    }

    // Box-Muller transform для нормального распределения
    normal(mean = 0, std = 1) {
        let u = 0, v = 0;
        while (u === 0) u = this.nextFloat();
        while (v === 0) v = this.nextFloat();
        const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        return z * std + mean;
    }

    integers(min, max, size = null) {
        if (size === null) {
            return Math.floor(this.nextFloat() * (max - min)) + min;
        }
        const res = new Int32Array(size);
        for (let i = 0; i < size; i++) {
            res[i] = Math.floor(this.nextFloat() * (max - min)) + min;
        }
        return res;
    }

    randomArray(size) {
        const res = new Float32Array(size);
        for (let i = 0; i < size; i++) res[i] = this.nextFloat();
        return res;
    }

    choice(n, p) {
        // Выбор индекса с учетом вероятностей
        const r = this.nextFloat();
        let acc = 0.0;
        for (let i = 0; i < n; i++) {
            acc += p[i];
            if (r < acc) return i;
        }
        return n - 1;
    }
    
    permutation(n) {
        const arr = new Int32Array(n);
        for(let i=0; i<n; i++) arr[i] = i;
        for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(this.nextFloat() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
    }
}

function softmax(x) {
    let max = -Infinity;
    for (let i = 0; i < x.length; i++) if (x[i] > max) max = x[i];
    
    const e = new Float32Array(x.length);
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
        e[i] = Math.exp(x[i] - max);
        sum += e[i];
    }
    
    const out = new Float32Array(x.length);
    const div = sum + 1e-9;
    for (let i = 0; i < x.length; i++) out[i] = e[i] / div;
    return out;
}

function normalizeProbs(p) {
    let sum = 0;
    for (let v of p) sum += v;
    const out = new Float32Array(p.length);
    if (sum <= 0) {
        const val = 1.0 / p.length;
        for(let i=0; i<p.length; i++) out[i] = val;
        return out;
    }
    for(let i=0; i<p.length; i++) out[i] = p[i] / sum;
    return out;
}

function sparsifyTopM(x, m) {
    if (m <= 0) return new Float32Array(x.length);
    if (m >= x.length) return x.slice(); // copy

    // Находим индексы топ-m элементов
    // Так как нет argpartition, делаем через сортировку индексов
    const indices = new Int32Array(x.length);
    for(let i=0; i<x.length; i++) indices[i] = i;
    
    // Сортировка по убыванию значений x
    indices.sort((a, b) => x[b] - x[a]);
    
    const y = new Float32Array(x.length); // заполнено нулями
    for (let k = 0; k < m; k++) {
        const idx = indices[k];
        y[idx] = x[idx];
    }
    return y;
}

function sigmoid(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

function clamp(x, lo, hi) {
    return Math.min(hi, Math.max(lo, x));
}

// -----------------------------
// 1) DNA (base-4 genome) -> phenotype
// -----------------------------
function decodeGenome(genome) {
    const validChars = "0123";
    const g = [];
    for (let char of genome) {
        if (validChars.includes(char)) {
            g.push(char.charCodeAt(0) - 48);
        }
    }
    
    // Pad if needed
    while (g.length < 24) {
        g.push(1);
    }

    function b4(i, n) {
        let x = 0;
        for (let k = 0; k < n; k++) {
            x = x * 4 + g[(i + k) % g.length];
        }
        return x;
    }

    return {
        N: 256 + 64 * b4(0, 2),
        K: 8 + b4(2, 1) * 4,
        p_long: 0.02 + 0.02 * b4(3, 1),
        lr: 0.0005 + 0.0005 * b4(4, 1),
        k_active: Math.max(48, Math.floor((256 + 64 * b4(0, 2)) * (0.04 + 0.01 * b4(5, 1)))),
        trace_fast_decay: 0.92 + 0.02 * b4(6, 1),
        trace_slow_decay: 0.985 + 0.003 * b4(10, 1),
        homeo: 0.001 + 0.001 * b4(7, 1),
        buf_decay: 0.92 + 0.02 * b4(12, 1),
        seed: b4(8, 4),
        alpha: 0.25 + 0.15 * b4(13, 1),
        beta: 0.05 + 0.05 * b4(14, 1),
        p_inhib: 0.10 + 0.05 * b4(15, 1),
        dopamine_gain: 0.8 + 0.4 * b4(16, 1),
        dopamine_bias: -0.2 + 0.2 * b4(17, 1),
        cons_eps: 0.0005 + 0.0005 * b4(18, 1),
        w_fast_decay: 0.9990 + 0.0003 * b4(19, 1),
        prune_every: 800 + 200 * b4(20, 1),
        prune_frac: 0.02 + 0.01 * b4(21, 1),
        rewire_frac: 0.50 + 0.10 * b4(22, 1),
    };
}

// -----------------------------
// 2) Build 3D sparse graph + delays
// -----------------------------
function buildGraph(N, K, p_long, seed = 0) {
    const rng = new Random(seed);
    const pos = new Float32Array(N * 3);
    for (let i = 0; i < N * 3; i++) pos[i] = rng.nextFloat();

    // Эвклидово расстояние и K-NN
    // Для JS реализуем упрощенно: полный перебор расстояний для каждого узла
    // Это O(N^2), но для N ~ 500-1000 это терпимо в V8
    
    let srcList = [];
    let dstList = [];

    for (let i = 0; i < N; i++) {
        const dists = [];
        const xi = pos[i*3], yi = pos[i*3+1], zi = pos[i*3+2];
        
        for (let j = 0; j < N; j++) {
            if (i === j) {
                dists.push({idx: j, d2: Infinity});
                continue;
            }
            const xj = pos[j*3], yj = pos[j*3+1], zj = pos[j*3+2];
            const d2 = (xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2;
            dists.push({idx: j, d2: d2});
        }
        
        // Сортируем и берем K ближайших
        dists.sort((a, b) => a.d2 - b.d2);
        
        for (let k = 0; k < K; k++) {
            srcList.push(i);
            dstList.push(dists[k].idx);
        }
    }

    // Long range connections
    const n_long = Math.floor(srcList.length * p_long); // srcList.length is basically N*K
    if (n_long > 0) {
        const long_src = rng.integers(0, N, n_long);
        const long_dst = rng.integers(0, N, n_long);
        for(let v of long_src) srcList.push(v);
        for(let v of long_dst) dstList.push(v);
    }

    const E = srcList.length;
    const src = new Int32Array(srcList);
    const dst = new Int32Array(dstList);
    const delay = new Int32Array(E);

    // Calculate delays based on distance
    for (let i = 0; i < E; i++) {
        const u = src[i];
        const v = dst[i];
        const dist = Math.sqrt(
            (pos[u*3] - pos[v*3])**2 + 
            (pos[u*3+1] - pos[v*3+1])**2 + 
            (pos[u*3+2] - pos[v*3+2])**2
        );
        let d = Math.floor(dist * 6) + 1;
        if (d < 1) d = 1;
        if (d > 5) d = 5;
        delay[i] = d;
    }

    // Index by delay для быстрого доступа
    const idx_by_delay = [[], [], [], [], [], []]; // 0 unused, 1..5
    for (let i = 0; i < E; i++) {
        idx_by_delay[delay[i]].push(i);
    }
    // Convert to TypedArrays for performance
    const idx_by_delay_typed = idx_by_delay.map(arr => new Int32Array(arr));

    return { pos, src, dst, delay, idx_by_delay: idx_by_delay_typed };
}

// -----------------------------
// 3) Vocab
// -----------------------------
function buildVocab(text) {
    const unique = new Set(text.split(''));
    const chars = Array.from(unique).sort();
    const stoi = {};
    const itos = {};
    chars.forEach((c, i) => {
        stoi[c] = i;
        itos[i] = c;
    });
    return { stoi, itos, size: chars.length };
}

// -----------------------------
// 4) Brain
// -----------------------------
class TextBrain {
    constructor(genome, vocabSize) {
        this.p = decodeGenome(genome);
        this.N = this.p.N;
        this.K = this.p.K;
        this.vocabSize = vocabSize;

        this.rng = new Random(this.p.seed);

        const graph = buildGraph(this.N, this.K, this.p.p_long, this.p.seed);
        this.pos = graph.pos;
        this.src = graph.src;
        this.dst = graph.dst;
        this.delay = graph.delay;
        this.idx_by_delay = graph.idx_by_delay;
        this.E = this.src.length;

        // Is Inhibitory?
        this.is_inhib = new Uint8Array(this.N); // bool as 0/1
        for(let i=0; i<this.N; i++) {
            this.is_inhib[i] = (this.rng.nextFloat() < this.p.p_inhib) ? 1 : 0;
        }

        // Weights
        this.w_slow = new Float32Array(this.E);
        this.w_fast = new Float32Array(this.E);
        for(let i=0; i<this.E; i++) {
            this.w_slow[i] = this.rng.normal(0, 0.03);
            this.w_fast[i] = 0.0;
        }

        this._enforceEiSigns();

        // State
        this.a = new Float32Array(this.N);
        this.theta = new Float32Array(this.N);
        this.trace_fast = new Float32Array(this.N);
        this.trace_slow = new Float32Array(this.N);
        
        // Buffers 1..5. Index 0 unused
        this.buffers = [];
        for(let i=0; i<6; i++) this.buffers.push(new Float32Array(this.N));

        // Readout
        // R is dense [N, Vocab]. Flattened to 1D array size N*Vocab
        this.R = new Float32Array(this.N * vocabSize);
        for(let i=0; i<this.R.length; i++) this.R[i] = this.rng.normal(0, 0.12);
        
        this.b = new Float32Array(vocabSize);

        this.loss_ema = Math.log(Math.max(2, vocabSize));

        this.recur_scale = 1.0 / Math.sqrt(this.K);
        this.target_rate = this.p.k_active / this.N;

        // Sensory connection
        this.sens_fanout = Math.max(10, Math.min(16, Math.floor(this.p.k_active / 4)));
        const perm = this.rng.permutation(this.N);
        const needed = vocabSize * this.sens_fanout;
        
        // sens_idx[token_id][k] -> flattened logic
        // We will store as 2D array equivalent: flattened Int32Array
        this.sens_idx = new Int32Array(needed);
        for(let i=0; i<needed; i++) {
            this.sens_idx[i] = perm[i % this.N];
        }

        this.noise_std = 0.01;
        this.inp_gain = 1.0;

        this.alpha = this.p.alpha;
        this.beta = this.p.beta;

        this.fast_clip = 1.5;
        this.slow_clip = 2.0;

        this.m_fast = Math.min(this.N, 4 * this.p.k_active);
        this.m_slow = Math.min(this.N, 8 * this.p.k_active);

        this.lr_out_mul = 0.6;
        this.lr_rec_mul = 1.0;
        this.max_R_update = 0.02;
        this.max_b_update = 0.02;

        this.state_sum_target = this.p.k_active;

        this.dopamine = 0.0;
        this.step = 0;
    }

    _enforceEiSigns() {
        for(let i=0; i<this.E; i++) {
            const u = this.src[i];
            if (this.is_inhib[u]) {
                this.w_slow[i] = -Math.abs(this.w_slow[i]);
                this.w_fast[i] = -Math.abs(this.w_fast[i]);
            }
        }
    }

    _effectiveW(out) {
        // Записывает сумму весов в переданный массив out (если передан) или создает новый
        const res = out || new Float32Array(this.E);
        for(let i=0; i<this.E; i++) {
            res[i] = this.w_slow[i] + this.w_fast[i];
        }
        return res;
    }

    resetState() {
        this.a.fill(0);
        this.trace_fast.fill(0);
        this.trace_slow.fill(0);
        for(let b of this.buffers) b.fill(0);
    }

    computeState() {
        // TF copy & clip
        const tf = new Float32Array(this.N);
        for(let i=0; i<this.N; i++) tf[i] = clamp(this.trace_fast[i], 0.0, this.fast_clip);
        
        // TS copy & clip
        const ts = new Float32Array(this.N);
        for(let i=0; i<this.N; i++) ts[i] = clamp(this.trace_slow[i], 0.0, this.slow_clip);

        const tf_sparse = sparsifyTopM(tf, this.m_fast);
        const ts_sparse = sparsifyTopM(ts, this.m_slow);

        const state = new Float32Array(this.N);
        let s = 0;
        for(let i=0; i<this.N; i++) {
            state[i] = this.a[i] + this.alpha * tf_sparse[i] + this.beta * ts_sparse[i];
            s += state[i];
        }

        if (s > 1e-6) {
            const factor = this.state_sum_target / s;
            for(let i=0; i<this.N; i++) state[i] *= factor;
        }
        return state;
    }

    _updateModulators(loss) {
        const adv = this.loss_ema - loss;
        this.loss_ema = 0.995 * this.loss_ema + 0.005 * loss;

        const gain = this.p.dopamine_gain;
        const bias = this.p.dopamine_bias;
        this.dopamine = sigmoid(gain * adv + bias);
        return adv;
    }

    forward(tokenId) {
        this.step++;

        // Shift buffers
        // buffers[1] is "now"
        const delayed_now = this.buffers[1]; // reference to Float32Array
        
        // Rotate buffers logic: discard buf 1, move 2->1, 3->2, etc. New buf for 5.
        // Or simpler: just shift references and clear the last one.
        const reusedBuf = this.buffers[1]; 
        for(let d=1; d<5; d++) {
            this.buffers[d] = this.buffers[d+1];
        }
        this.buffers[5] = reusedBuf;
        this.buffers[5].fill(0);

        const bd = this.p.buf_decay;
        for(let d=1; d<=5; d++) {
            const buf = this.buffers[d];
            for(let i=0; i<this.N; i++) buf[i] *= bd;
        }

        // Calculate input X
        const x = new Float32Array(this.N);
        for(let i=0; i<this.N; i++) {
            let val = delayed_now[i] - this.theta[i];
            if (this.noise_std > 0) {
                val += this.rng.normal(0, this.noise_std);
            }
            x[i] = val;
        }

        // Sensory Input
        const new_a = new Float32Array(this.N);
        const startIdx = tokenId * this.sens_fanout;
        const endIdx = startIdx + this.sens_fanout;
        
        // Indices in x that are sensory
        const sensIndices = []; 
        for(let i=startIdx; i<endIdx; i++) {
            const idx = this.sens_idx[i];
            new_a[idx] = 1.0;
            sensIndices.push(idx);
        }

        const remaining = Math.floor(this.p.k_active - this.sens_fanout);
        if (remaining > 0) {
            // x2 logic: mask sensory nodes to very low, boost sensory in x
            const x2 = x.slice(); 
            for(let idx of sensIndices) {
                x2[idx] = -1e9;
                x[idx] += this.inp_gain; // Boost sensory in original x (for dynamics? python script logic checks out)
            }
            // Actually python script boosts x[sidx] THEN does argpartition on x2 (where sidx is -inf)
            // So sensory neurons are forced ON by new_a=1.0, and removed from competition for remaining spots.
            
            // Find top 'remaining' in x2
            // Quick approach: sort indices
            const indices = new Int32Array(this.N);
            for(let i=0; i<this.N; i++) indices[i] = i;
            indices.sort((a,b) => x2[b] - x2[a]);

            let count = 0;
            for(let i=0; i<this.N && count < remaining; i++) {
                const idx = indices[i];
                if (x2[idx] > 0) {
                    new_a[idx] = 1.0;
                    count++;
                }
            }
        }
        
        this.a = new_a;

        // Traces
        const fd = this.p.trace_fast_decay;
        const sd = this.p.trace_slow_decay;
        for(let i=0; i<this.N; i++) {
            this.trace_fast[i] = fd * this.trace_fast[i] + this.a[i];
            this.trace_fast[i] = clamp(this.trace_fast[i], 0.0, this.fast_clip);
            
            this.trace_slow[i] = sd * this.trace_slow[i] + 0.15 * this.trace_fast[i];
            this.trace_slow[i] = clamp(this.trace_slow[i], 0.0, this.slow_clip);
        }

        // Recurrent Spreading
        const rs = this.recur_scale;
        const w_eff = this._effectiveW();
        
        // Python: np.add.at(buffers[d], dst[idxe], a[src[idxe]] * w * rs)
        for(let d=1; d<=5; d++) {
            const indices = this.idx_by_delay[d];
            const buf = this.buffers[d];
            if (indices.length === 0) continue;
            
            for(let i=0; i<indices.length; i++) {
                const edgeIdx = indices[i];
                const u = this.src[edgeIdx];
                const v = this.dst[edgeIdx];
                const val = this.a[u] * w_eff[edgeIdx] * rs;
                buf[v] += val;
            }
            // Clip buffer
            for(let i=0; i<this.N; i++) buf[i] = clamp(buf[i], -5.0, 5.0);
        }

        const state = this.computeState();
        
        // Logits: state @ R + b
        // R is [N, V] stored flat. state is [N]. Result [V].
        const logits = new Float32Array(this.vocabSize);
        for(let j=0; j<this.vocabSize; j++) {
            let sum = this.b[j];
            for(let i=0; i<this.N; i++) {
                sum += state[i] * this.R[i * this.vocabSize + j];
            }
            logits[j] = sum;
        }

        return softmax(logits);
    }

    _consolidate() {
        const eps = this.p.cons_eps;
        if (eps <= 0) return;
        for(let i=0; i<this.E; i++) {
            this.w_slow[i] = (1.0 - eps) * this.w_slow[i] + eps * this.w_fast[i];
        }
    }

    _decayFast() {
        const d = this.p.w_fast_decay;
        for(let i=0; i<this.E; i++) this.w_fast[i] *= d;
    }

    _pruneAndRewire() {
        const prune_frac = this.p.prune_frac;
        if (prune_frac <= 0) return;

        const n_prune = Math.floor(this.E * prune_frac);
        if (n_prune <= 0) return;

        const w_eff = this._effectiveW();
        // abs values
        const absw = new Float32Array(this.E);
        for(let i=0; i<this.E; i++) absw[i] = Math.abs(w_eff[i]);

        // Find smallest n_prune indices
        const indices = new Int32Array(this.E);
        for(let i=0; i<this.E; i++) indices[i] = i;
        indices.sort((a,b) => absw[a] - absw[b]); // ascending

        const prune_idx = indices.slice(0, n_prune);
        const n_rewire = Math.floor(n_prune * this.p.rewire_frac);

        // Rewire
        if (n_rewire > 0) {
            for(let k=0; k<n_rewire; k++) {
                const idx = prune_idx[k];
                // Random new src/dst
                const u = this.rng.integers(0, this.N);
                const v = this.rng.integers(0, this.N);
                this.src[idx] = u;
                this.dst[idx] = v;
                
                // Recalc delay
                const dist = Math.sqrt(
                    (this.pos[u*3] - this.pos[v*3])**2 + 
                    (this.pos[u*3+1] - this.pos[v*3+1])**2 + 
                    (this.pos[u*3+2] - this.pos[v*3+2])**2
                );
                let d = Math.floor(dist * 6) + 1;
                this.delay[idx] = clamp(d, 1, 5);

                this.w_slow[idx] = this.rng.normal(0, 0.01);
                this.w_fast[idx] = 0.0;
            }
        }

        // Zero out the rest
        for(let k=n_rewire; k<n_prune; k++) {
            const idx = prune_idx[k];
            this.w_slow[idx] = 0.0;
            this.w_fast[idx] = 0.0;
        }

        this._enforceEiSigns();

        // Rebuild idx_by_delay
        // Clear
        for(let d=1; d<=5; d++) this.idx_by_delay[d] = []; // Temp use JS array then convert back if needed, or just iterate fully to rebuild
        // Re-scanning entire array to rebuild indices
        const new_idx_map = [[], [], [], [], [], []];
        for(let i=0; i<this.E; i++) {
            new_idx_map[this.delay[i]].push(i);
        }
        for(let d=1; d<=5; d++) {
            this.idx_by_delay[d] = new Int32Array(new_idx_map[d]);
        }
    }

    damageEdges(frac = 0.2) {
        frac = clamp(frac, 0.0, 1.0);
        const n = Math.floor(this.E * frac);
        if (n <= 0) return;
        
        const idxs = this.rng.permutation(this.E);
        for(let i=0; i<n; i++) {
            const idx = idxs[i];
            this.w_slow[idx] = 0.0;
            this.w_fast[idx] = 0.0;
        }
    }

    learn(targetId, probs) {
        const p = probs[targetId];
        const loss = -Math.log(p + 1e-9);

        const adv = this._updateModulators(loss);

        const lr = this.p.lr;
        const lr_out = lr * this.lr_out_mul;
        const lr_rec = lr * this.lr_rec_mul;

        // Gradient for Readout
        // grad = probs - 1 at target
        const grad = probs.slice();
        grad[targetId] -= 1.0;

        const state = this.computeState();

        // dR = state[i] * grad[j] * lr_out
        // R is [N, V]
        for(let i=0; i<this.N; i++) {
            const s_val = state[i];
            if (s_val === 0) continue; 
            for(let j=0; j<this.vocabSize; j++) {
                let delta = s_val * grad[j] * lr_out;
                delta = clamp(delta, -this.max_R_update, this.max_R_update);
                const idx = i * this.vocabSize + j;
                this.R[idx] -= delta;
                this.R[idx] = clamp(this.R[idx], -1.0, 1.0);
            }
        }

        // db
        for(let j=0; j<this.vocabSize; j++) {
            let delta = grad[j] * lr_out;
            delta = clamp(delta, -this.max_b_update, this.max_b_update);
            this.b[j] -= delta;
        }

        // dW Recurrent
        // pre = trace_fast[src], post = a[dst]
        // dW = lr_rec * dopamine * adv * pre * post
        const factor = lr_rec * this.dopamine * adv;
        
        // Loop over all edges
        for(let i=0; i<this.E; i++) {
            const u = this.src[i];
            const v = this.dst[i];
            const pre = this.trace_fast[u];
            const post = this.a[v];
            
            if (pre === 0 || post === 0) continue;

            let dW = factor * pre * post;
            dW = clamp(dW, -0.02, 0.02);
            this.w_fast[i] += dW;
        }

        this._enforceEiSigns();

        for(let i=0; i<this.E; i++) {
            this.w_fast[i] = clamp(this.w_fast[i], -0.5, 0.5);
            this.w_slow[i] = clamp(this.w_slow[i], -0.5, 0.5);
        }

        this._decayFast();
        this._consolidate();

        // Homeostasis
        const homeo = this.p.homeo;
        for(let i=0; i<this.N; i++) {
            this.theta[i] += homeo * (this.a[i] - this.target_rate);
        }

        const prune_every = this.p.prune_every;
        if (prune_every > 0 && (this.step % prune_every === 0)) {
            this._pruneAndRewire();
        }

        return loss;
    }

    avgTheta() {
        let sum = 0;
        for(let v of this.theta) sum += v;
        return sum / this.N;
    }

    meanAbsW() {
        const w = this._effectiveW();
        let sum = 0;
        for(let v of w) sum += Math.abs(v);
        return sum / this.E;
    }

    firingRate() {
        let sum = 0;
        for(let v of this.a) sum += v;
        return sum / this.N;
    }
}


// -----------------------------
// Training
// -----------------------------
function trainOnText(genome, text, steps = 80000, printEvery = 5000) {
    const vocab = buildVocab(text);
    const stoi = vocab.stoi;
    const itos = vocab.itos;
    
    // Encode
    const ids = new Int32Array(text.length);
    for(let i=0; i<text.length; i++) ids[i] = stoi[text[i]];

    if (ids.length < 2) throw new Error("Text too short");

    const brain = new TextBrain(genome, vocab.size);
    const baseline = Math.log(vocab.size);

    console.log(`Stats: N=${brain.N}, K=${brain.K}`);
    console.log(`Target Active: ${brain.p.k_active} (${(brain.target_rate*100).toFixed(1)}%)`);
    console.log(`Sensory Fanout: ${brain.sens_fanout}`);
    console.log(`Vocab size: ${vocab.size} | random baseline ~ log(V)=${baseline.toFixed(3)}`);
    console.log(`alpha=${brain.alpha.toFixed(2)}, beta=${brain.beta.toFixed(2)}, m_fast=${brain.m_fast}, m_slow=${brain.m_slow}`);

    let losses = [];
    const n = ids.length - 1;
    let t0 = Date.now();

    for (let step = 0; step < steps; step++) {
        const i = step % n;
        const x = ids[i];
        const y = ids[i+1];

        const probs = brain.forward(x);
        const loss = brain.learn(y, probs);
        losses.push(loss);

        if ((step + 1) % printEvery === 0) {
            const dt = (Date.now() - t0) / 1000;
            // Mean of last printEvery losses
            let sumL = 0;
            for(let k=losses.length - printEvery; k<losses.length; k++) sumL += losses[k];
            const avgLoss = sumL / printEvery;
            
            console.log(
                `Step ${step+1}: Loss=${avgLoss.toFixed(4)} | ` +
                `DA=${brain.dopamine.toFixed(3)} | AvgTheta=${brain.avgTheta().toFixed(3)} | |W|=${brain.meanAbsW().toFixed(3)} | ` +
                `Rate~${brain.firingRate().toFixed(3)} | ${dt.toFixed(1)}s`
            );
            t0 = Date.now();
        }
    }
    
    return { brain, stoi, itos };
}

// -----------------------------
// Sampling
// -----------------------------
function applySamplingFilters(probs, temperature=0.8, topK=18, topP=0.92) {
    let p = Array.from(probs); // convert to standard array for easier manipulation

    // Temp
    if (temperature && temperature !== 1.0) {
        for(let i=0; i<p.length; i++) p[i] = Math.pow(p[i], 1.0/temperature);
        p = Array.from(normalizeProbs(p));
    }

    // Top K
    if (topK && topK > 0 && topK < p.length) {
        const indexed = p.map((val, idx) => ({val, idx}));
        indexed.sort((a,b) => b.val - a.val);
        const topIndices = new Set(indexed.slice(0, topK).map(o => o.idx));
        for(let i=0; i<p.length; i++) {
            if (!topIndices.has(i)) p[i] = 0;
        }
        p = Array.from(normalizeProbs(p));
    }

    // Top P (Nucleus)
    if (topP && topP > 0.0 && topP < 1.0) {
        const indexed = p.map((val, idx) => ({val, idx}));
        indexed.sort((a,b) => b.val - a.val);
        
        let csum = 0;
        let cutIdx = indexed.length - 1;
        for(let i=0; i<indexed.length; i++) {
            csum += indexed[i].val;
            if (csum > topP) {
                cutIdx = i;
                break;
            }
        }
        // cutIdx includes the element that pushed over topP
        const keepIndices = new Set(indexed.slice(0, cutIdx + 1).map(o => o.idx));
        for(let i=0; i<p.length; i++) {
            if (!keepIndices.has(i)) p[i] = 0;
        }
        p = Array.from(normalizeProbs(p));
    }

    return p;
}

function sample(brain, stoi, itos, seedStr, n=700, temperature=0.75, topK=18, topP=0.92) {
    brain.resetState();
    let seedChars = [];
    for(let c of seedStr) {
        if (stoi.hasOwnProperty(c)) seedChars.push(c);
    }
    if (seedChars.length === 0) seedChars.push(Object.keys(stoi)[0]);

    // Priming
    for(let i=0; i<seedChars.length - 1; i++) {
        brain.forward(stoi[seedChars[i]]);
    }

    let x = stoi[seedChars[seedChars.length - 1]];
    let out = [...seedChars];

    for(let i=0; i<n; i++) {
        const probs = brain.forward(x);
        const filtered = applySamplingFilters(probs, temperature, topK, topP);
        
        // Choice
        x = brain.rng.choice(filtered.length, filtered);
        out.push(itos[x]);
    }
    return out.join("");
}

// -----------------------------
// Benchmarks
// -----------------------------
function benchmarkSelfRepair(genome, text, preTrainSteps=8000, evalSteps=2000, recoverySteps=8000, damageFrac=0.2, reportEvery=1000) {
    const vocab = buildVocab(text);
    const stoi = vocab.stoi;
    const itos = vocab.itos;
    const ids = new Int32Array(text.length);
    for(let i=0; i<text.length; i++) ids[i] = stoi[text[i]];

    if (ids.length < 2) throw new Error("Text too short");

    const brain = new TextBrain(genome, vocab.size);
    const n = ids.length - 1;

    const evalLoss = (steps) => {
        let losses = [];
        for(let step=0; step<steps; step++) {
            const i = step % n;
            const x = ids[i];
            const y = ids[i+1];
            const probs = brain.forward(x);
            const p = probs[y];
            losses.push(-Math.log(p + 1e-9));
        }
        let sum = 0; 
        for(let l of losses) sum += l;
        return sum / losses.length;
    };

    const trainSteps = (steps) => {
        for(let step=0; step<steps; step++) {
            const i = step % n;
            const x = ids[i];
            const y = ids[i+1];
            const probs = brain.forward(x);
            brain.learn(y, probs);
        }
    };

    trainSteps(preTrainSteps);
    const preEval = evalLoss(evalSteps);

    brain.damageEdges(damageFrac);
    const postDamageEval = evalLoss(evalSteps);

    const recoveryCurve = [];
    let remaining = recoverySteps;
    while (remaining > 0) {
        const chunk = Math.min(reportEvery, remaining);
        trainSteps(chunk);
        remaining -= chunk;
        recoveryCurve.push(evalLoss(evalSteps));
    }

    console.log(
        `Self-repair(eval): pre=${preEval.toFixed(4)} | post_damage=${postDamageEval.toFixed(4)} | ` +
        `final=${recoveryCurve[recoveryCurve.length - 1].toFixed(4)} | damage_frac=${damageFrac.toFixed(2)}`
    );

    if (recoveryCurve.length > 0) {
        const parts = recoveryCurve.map((v, i) => `${(i+1)*reportEvery}:${v.toFixed(4)}`);
        console.log("Recovery curve:");
        console.log("  " + parts.join(" | "));
    }

    const seed = "To be, or not to be";
    const generated = sample(brain, stoi, itos, seed, 300);
    console.log("-".repeat(60));
    console.log(generated);
    console.log("-".repeat(60));
}

// -----------------------------
// Main
// -----------------------------
function main() {
    let text = `
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
    `.repeat(40);

    // Normalize spacing
    text = text.replace(/\s+/g, ' ').trim();

    const genome = "30321031103321200112332100123";

    console.log(`Initializing Brain with genome: ${genome}...`);
    const p = decodeGenome(genome);
    console.log(
        `Phenotype: N=${p.N}, K=${p.K}, LR=${p.lr.toFixed(4)}, Active=${p.k_active}, ` +
        `BufDecay=${p.buf_decay.toFixed(2)}, alpha=${p.alpha.toFixed(2)}, beta=${p.beta.toFixed(2)}, pI=${p.p_inhib.toFixed(2)}`
    );

    console.log("\nStarting training...");
    const {brain, stoi, itos} = trainOnText(genome, text, 120000, 5000);

    console.log("\nTraining done. Generating text...");
    const seed = "To be, or not to be";
    const generated = sample(brain, stoi, itos, seed, 500);

    console.log("-".repeat(60));
    console.log(generated);
    console.log("-".repeat(60));

    console.log("\nRunning self-repair benchmark...");
    benchmarkSelfRepair(
        genome,
        text,
        12000, // pre_train
        2000,  // eval
        12000, // recovery
        0.2,   // damage
        2000   // report
    );
}

if (require.main === module) {
    main();
}