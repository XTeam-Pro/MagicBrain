package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"
)

// -----------------------------
// Helpers & Math
// -----------------------------

func softmax(x []float32) []float32 {
	maxVal := float32(-math.MaxFloat32)
	for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}

	e := make([]float32, len(x))
	sum := float32(0.0)
	for i, v := range x {
		e[i] = float32(math.Exp(float64(v - maxVal)))
		sum += e[i]
	}

	out := make([]float32, len(x))
	invSum := 1.0 / (sum + 1e-9)
	for i := range e {
		out[i] = e[i] * invSum
	}
	return out
}

func normalizeProbs(p []float32) []float32 {
	sum := float32(0.0)
	for _, v := range p {
		sum += v
	}
	out := make([]float32, len(p))
	if sum <= 0 {
		val := 1.0 / float32(len(p))
		for i := range out {
			out[i] = val
		}
		return out
	}
	invSum := 1.0 / sum
	for i := range p {
		out[i] = p[i] * invSum
	}
	return out
}

// Helper struct for sorting indices by value
type idxVal struct {
	Idx int
	Val float32
}

func sparsifyTopM(x []float32, m int) []float32 {
	if m <= 0 {
		return make([]float32, len(x))
	}
	if m >= len(x) {
		cp := make([]float32, len(x))
		copy(cp, x)
		return cp
	}

	// Create helper to sort
	pairs := make([]idxVal, len(x))
	for i, v := range x {
		pairs[i] = idxVal{i, v}
	}

	// Partial sort or full sort (using full sort for simplicity in Go)
	// Sort descending
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Val > pairs[j].Val
	})

	y := make([]float32, len(x))
	// Only keep top m
	for i := 0; i < m; i++ {
		idx := pairs[i].Idx
		y[idx] = x[idx]
	}
	return y
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func clamp(x, lo, hi float32) float32 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

// -----------------------------
// 1) DNA (base-4 genome) -> phenotype
// -----------------------------

type BrainParams struct {
	N              int
	K              int
	PLong          float32
	LR             float32
	KActive        int
	TraceFastDecay float32
	TraceSlowDecay float32
	Homeo          float32
	BufDecay       float32
	Seed           int64
	Alpha          float32
	Beta           float32
	PInhib         float32
	DopamineGain   float32
	DopamineBias   float32
	ConsEps        float32
	WFastDecay     float32
	PruneEvery     int
	PruneFrac      float32
	RewireFrac     float32
}

func decodeGenome(genome string) BrainParams {
	// Filter chars
	var g []int
	for _, r := range genome {
		if r >= '0' && r <= '3' {
			g = append(g, int(r-'0'))
		}
	}
	// Pad
	if len(g) < 24 {
		padding := make([]int, 24-len(g))
		for i := range padding {
			padding[i] = 1
		}
		g = append(g, padding...)
	}

	b4 := func(i, n int) int {
		x := 0
		for k := 0; k < n; k++ {
			x = x*4 + g[(i+k)%len(g)]
		}
		return x
	}

	N := 256 + 64*b4(0, 2)
	K := 8 + b4(2, 1)*4
	pLong := 0.02 + 0.02*float32(b4(3, 1))

	lr := 0.0005 + 0.0005*float32(b4(4, 1))

	kActive := int(float32(N) * (0.04 + 0.01*float32(b4(5, 1))))
	if kActive < 48 {
		kActive = 48
	}

	traceFastDecay := 0.92 + 0.02*float32(b4(6, 1))
	traceSlowDecay := 0.985 + 0.003*float32(b4(10, 1))

	homeo := 0.001 + 0.001*float32(b4(7, 1))
	bufDecay := 0.92 + 0.02*float32(b4(12, 1))

	seed := int64(b4(8, 4))

	alpha := 0.25 + 0.15*float32(b4(13, 1))
	beta := 0.05 + 0.05*float32(b4(14, 1))

	pInhib := 0.10 + 0.05*float32(b4(15, 1))

	dopamineGain := 0.8 + 0.4*float32(b4(16, 1))
	dopamineBias := -0.2 + 0.2*float32(b4(17, 1))

	consEps := 0.0005 + 0.0005*float32(b4(18, 1))
	wFastDecay := 0.9990 + 0.0003*float32(b4(19, 1))

	pruneEvery := 800 + 200*b4(20, 1)
	pruneFrac := 0.02 + 0.01*float32(b4(21, 1))
	rewireFrac := 0.50 + 0.10*float32(b4(22, 1))

	return BrainParams{
		N: N, K: K, PLong: pLong, LR: lr, KActive: kActive,
		TraceFastDecay: traceFastDecay, TraceSlowDecay: traceSlowDecay,
		Homeo: homeo, BufDecay: bufDecay, Seed: seed, Alpha: alpha, Beta: beta,
		PInhib: pInhib, DopamineGain: dopamineGain, DopamineBias: dopamineBias,
		ConsEps: consEps, WFastDecay: wFastDecay, PruneEvery: pruneEvery,
		PruneFrac: pruneFrac, RewireFrac: rewireFrac,
	}
}

// -----------------------------
// 2) Build 3D sparse graph + delays
// -----------------------------

type Point3 struct{ x, y, z float32 }

func (p Point3) DistSq(o Point3) float32 {
	return (p.x-o.x)*(p.x-o.x) + (p.y-o.y)*(p.y-o.y) + (p.z-o.z)*(p.z-o.z)
}

func buildGraph(N, K int, pLong float32, seed int64) ([]Point3, []int, []int, []int32, [][]int) {
	rng := rand.New(rand.NewSource(seed))

	pos := make([]Point3, N)
	for i := 0; i < N; i++ {
		pos[i] = Point3{
			x: float32(rng.Float64()),
			y: float32(rng.Float64()),
			z: float32(rng.Float64()),
		}
	}

	// Naive KNN
	var src []int
	var dst []int

	type distIdx struct {
		d2  float32
		idx int
	}

	for i := 0; i < N; i++ {
		dists := make([]distIdx, 0, N)
		for j := 0; j < N; j++ {
			if i == j {
				continue
			}
			dists = append(dists, distIdx{d2: pos[i].DistSq(pos[j]), idx: j})
		}
		// Sort by distance
		sort.Slice(dists, func(a, b int) bool {
			return dists[a].d2 < dists[b].d2
		})

		// Take K
		for k := 0; k < K && k < len(dists); k++ {
			src = append(src, i)
			dst = append(dst, dists[k].idx)
		}
	}

	// Long range
	nLong := int(float32(len(src)) * pLong)
	for i := 0; i < nLong; i++ {
		src = append(src, rng.Intn(N))
		dst = append(dst, rng.Intn(N))
	}

	// Delays
	delay := make([]int32, len(src))
	for i := range src {
		d := float32(math.Sqrt(float64(pos[src[i]].DistSq(pos[dst[i]]))))
		val := int32(d*6) + 1
		if val < 1 {
			val = 1
		}
		if val > 5 {
			val = 5
		}
		delay[i] = val
	}

	idxByDelay := make([][]int, 6)
	for i, d := range delay {
		idxByDelay[d] = append(idxByDelay[d], i)
	}

	return pos, src, dst, delay, idxByDelay
}

// -----------------------------
// 3) Vocab
// -----------------------------

func buildVocab(text string) (map[rune]int, map[int]rune) {
	charsMap := make(map[rune]bool)
	for _, r := range text {
		charsMap[r] = true
	}
	var chars []rune
	for r := range charsMap {
		chars = append(chars, r)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })

	stoi := make(map[rune]int)
	itos := make(map[int]rune)
	for i, c := range chars {
		stoi[c] = i
		itos[i] = c
	}
	return stoi, itos
}

// -----------------------------
// 4) Brain
// -----------------------------

type TextBrain struct {
	P           BrainParams
	N, K        int
	VocabSize   int
	Rng         *rand.Rand
	Pos         []Point3
	Src, Dst    []int
	Delay       []int32
	IdxByDelay  [][]int
	IsInhib     []bool
	WSlow       []float32
	WFast       []float32
	A           []float32
	Theta       []float32
	TraceFast   []float32
	TraceSlow   []float32
	Buffers     [6][]float32
	R           []float32 // Flattened [N * VocabSize]
	B           []float32 // [VocabSize]
	LossEma     float32
	RecurScale  float32
	TargetRate  float32
	SensFanout  int
	SensIdx     [][]int // [VocabSize][SensFanout]
	NoiseStd    float32
	InpGain     float32
	Alpha, Beta float32
	FastClip    float32
	SlowClip    float32
	MFast       int
	MSlow       int
	LrOutMul    float32
	LrRecMul    float32
	MaxRUpdate  float32
	MaxBUpdate  float32
	StateSumTgt float32
	Dopamine    float32
	Step        int
}

func NewTextBrain(genome string, vocabSize int) *TextBrain {
	p := decodeGenome(genome)
	rng := rand.New(rand.NewSource(p.Seed))

	pos, src, dst, delay, idxByDelay := buildGraph(p.N, p.K, p.PLong, p.Seed)
	N := p.N
	E := len(src)

	isInhib := make([]bool, N)
	for i := 0; i < N; i++ {
		isInhib[i] = rng.Float32() < p.PInhib
	}

	wSlow := make([]float32, E)
	wFast := make([]float32, E)
	for i := 0; i < E; i++ {
		wSlow[i] = float32(rng.NormFloat64()) * 0.03
	}

	tb := &TextBrain{
		P:           p,
		N:           N,
		K:           p.K,
		VocabSize:   vocabSize,
		Rng:         rng,
		Pos:         pos,
		Src:         src,
		Dst:         dst,
		Delay:       delay,
		IdxByDelay:  idxByDelay,
		IsInhib:     isInhib,
		WSlow:       wSlow,
		WFast:       wFast,
		A:           make([]float32, N),
		Theta:       make([]float32, N),
		TraceFast:   make([]float32, N),
		TraceSlow:   make([]float32, N),
		R:           make([]float32, N*vocabSize),
		B:           make([]float32, vocabSize),
		LossEma:     float32(math.Log(math.Max(2, float64(vocabSize)))),
		RecurScale:  1.0 / float32(math.Sqrt(float64(p.K))),
		TargetRate:  float32(p.KActive) / float32(N),
		NoiseStd:    0.01,
		InpGain:     1.0,
		Alpha:       p.Alpha,
		Beta:        p.Beta,
		FastClip:    1.5,
		SlowClip:    2.0,
		MFast:       min(N, 4*p.KActive),
		MSlow:       min(N, 8*p.KActive),
		LrOutMul:    0.6,
		LrRecMul:    1.0,
		MaxRUpdate:  0.02,
		MaxBUpdate:  0.02,
		StateSumTgt: float32(p.KActive),
	}

	for i := 0; i < 6; i++ {
		tb.Buffers[i] = make([]float32, N)
	}

	// Initialize R
	for i := range tb.R {
		tb.R[i] = float32(rng.NormFloat64()) * 0.12
	}

	tb.enforceEiSigns()

	// Sensory Fanout
	tb.SensFanout = max(10, min(16, p.KActive/4))
	needed := vocabSize * tb.SensFanout
	perm := rng.Perm(N)
	sensFlat := make([]int, needed)
	for i := 0; i < needed; i++ {
		sensFlat[i] = perm[i%N]
	}

	tb.SensIdx = make([][]int, vocabSize)
	for i := 0; i < vocabSize; i++ {
		tb.SensIdx[i] = sensFlat[i*tb.SensFanout : (i+1)*tb.SensFanout]
	}

	return tb
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (tb *TextBrain) enforceEiSigns() {
	for i, srcIdx := range tb.Src {
		if tb.IsInhib[srcIdx] {
			tb.WSlow[i] = -float32(math.Abs(float64(tb.WSlow[i])))
			tb.WFast[i] = -float32(math.Abs(float64(tb.WFast[i])))
		}
	}
}

func (tb *TextBrain) effectiveW() []float32 {
	w := make([]float32, len(tb.WSlow))
	for i := range w {
		w[i] = tb.WSlow[i] + tb.WFast[i]
	}
	return w
}

func (tb *TextBrain) ResetState() {
	for i := range tb.A {
		tb.A[i] = 0
		tb.TraceFast[i] = 0
		tb.TraceSlow[i] = 0
	}
	for i := 0; i < 6; i++ {
		for j := range tb.Buffers[i] {
			tb.Buffers[i][j] = 0
		}
	}
}

func (tb *TextBrain) computeState() []float32 {
	tf := make([]float32, tb.N)
	ts := make([]float32, tb.N)
	for i := 0; i < tb.N; i++ {
		tf[i] = clamp(tb.TraceFast[i], 0.0, tb.FastClip)
		ts[i] = clamp(tb.TraceSlow[i], 0.0, tb.SlowClip)
	}
	tf = sparsifyTopM(tf, tb.MFast)
	ts = sparsifyTopM(ts, tb.MSlow)

	state := make([]float32, tb.N)
	sum := float32(0.0)
	for i := 0; i < tb.N; i++ {
		state[i] = tb.A[i] + tb.Alpha*tf[i] + tb.Beta*ts[i]
		sum += state[i]
	}

	if sum > 1e-6 {
		factor := tb.StateSumTgt / sum
		for i := range state {
			state[i] *= factor
		}
	}
	return state
}

func (tb *TextBrain) updateModulators(loss float32) float32 {
	adv := tb.LossEma - loss
	tb.LossEma = 0.995*tb.LossEma + 0.005*loss
	tb.Dopamine = sigmoid(tb.P.DopamineGain*adv + tb.P.DopamineBias)
	return adv
}

func (tb *TextBrain) Forward(tokenId int) []float32 {
	tb.Step++

	// Buffer management
	// In Python: buffers[1] is current, shift others down, clear last
	delayedNow := make([]float32, tb.N)
	copy(delayedNow, tb.Buffers[1])

	for d := 1; d < 5; d++ {
		copy(tb.Buffers[d], tb.Buffers[d+1])
	}
	for i := range tb.Buffers[5] {
		tb.Buffers[5][i] = 0
	}

	bd := tb.P.BufDecay
	for d := 1; d <= 5; d++ {
		for i := range tb.Buffers[d] {
			tb.Buffers[d][i] *= bd
		}
	}

	x := make([]float32, tb.N)
	for i := 0; i < tb.N; i++ {
		val := delayedNow[i] - tb.Theta[i]
		if tb.NoiseStd > 0 {
			val += float32(tb.Rng.NormFloat64()) * tb.NoiseStd
		}
		x[i] = val
	}

	// Sensory Input
	sidx := tb.SensIdx[tokenId]
	a := make([]float32, tb.N)
	for _, idx := range sidx {
		a[idx] = 1.0
	}

	remaining := tb.P.KActive - tb.SensFanout
	if remaining > 0 {
		x2 := make([]float32, tb.N)
		copy(x2, x)
		for _, idx := range sidx {
			x2[idx] = -1e9
			x[idx] += tb.InpGain
		}

		// ArgPartition equivalent (Top K)
		pairs := make([]idxVal, tb.N)
		for i, v := range x2 {
			pairs[i] = idxVal{i, v}
		}
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].Val > pairs[j].Val
		})

		count := 0
		for i := 0; i < len(pairs); i++ {
			if count >= remaining {
				break
			}
			idx := pairs[i].Idx
			if x2[idx] > 0 {
				a[idx] = 1.0
				count++
			}
		}
	}
	tb.A = a

	// Traces
	fd := tb.P.TraceFastDecay
	sd := tb.P.TraceSlowDecay
	for i := 0; i < tb.N; i++ {
		tb.TraceFast[i] = fd*tb.TraceFast[i] + tb.A[i]
		tb.TraceFast[i] = clamp(tb.TraceFast[i], 0.0, tb.FastClip)

		tb.TraceSlow[i] = sd*tb.TraceSlow[i] + 0.15*tb.TraceFast[i]
		tb.TraceSlow[i] = clamp(tb.TraceSlow[i], 0.0, tb.SlowClip)
	}

	// Recur
	rs := tb.RecurScale
	wEff := tb.effectiveW()
	for d := 1; d <= 5; d++ {
		indices := tb.IdxByDelay[d]
		if len(indices) == 0 {
			continue
		}
		for _, edgeIdx := range indices {
			src := tb.Src[edgeIdx]
			dst := tb.Dst[edgeIdx]
			contrib := tb.A[src] * wEff[edgeIdx] * rs
			tb.Buffers[d][dst] += contrib
		}
		// Clip buffer
		for i := range tb.Buffers[d] {
			tb.Buffers[d][i] = clamp(tb.Buffers[d][i], -5.0, 5.0)
		}
	}

	// Readout
	state := tb.computeState()
	logits := make([]float32, tb.VocabSize)
	// Matrix mul: logits = state @ R + b
	// R is N rows * Vocab cols
	for j := 0; j < tb.VocabSize; j++ {
		sum := tb.B[j]
		for i := 0; i < tb.N; i++ {
			sum += state[i] * tb.R[i*tb.VocabSize+j]
		}
		logits[j] = sum
	}

	return softmax(logits)
}

func (tb *TextBrain) consolidate() {
	eps := tb.P.ConsEps
	if eps <= 0 {
		return
	}
	for i := range tb.WSlow {
		tb.WSlow[i] = (1.0-eps)*tb.WSlow[i] + eps*tb.WFast[i]
	}
}

func (tb *TextBrain) decayFast() {
	d := tb.P.WFastDecay
	for i := range tb.WFast {
		tb.WFast[i] *= d
	}
}

func (tb *TextBrain) pruneAndRewire() {
	if tb.P.PruneFrac <= 0 {
		return
	}
	nPrune := int(float32(len(tb.Src)) * tb.P.PruneFrac)
	if nPrune <= 0 {
		return
	}

	wEff := tb.effectiveW()
	pairs := make([]idxVal, len(wEff))
	for i, v := range wEff {
		pairs[i] = idxVal{i, float32(math.Abs(float64(v)))}
	}
	// Sort ascending (smallest first)
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Val < pairs[j].Val
	})

	pruneIdx := make([]int, nPrune)
	for i := 0; i < nPrune; i++ {
		pruneIdx[i] = pairs[i].Idx
	}

	nRewire := int(float32(nPrune) * tb.P.RewireFrac)

	// Rewire
	for k := 0; k < nRewire; k++ {
		idx := pruneIdx[k]
		newSrc := tb.Rng.Intn(tb.N)
		newDst := tb.Rng.Intn(tb.N)
		tb.Src[idx] = newSrc
		tb.Dst[idx] = newDst

		dist := tb.Pos[newSrc].DistSq(tb.Pos[newDst])
		d := int32(math.Sqrt(float64(dist))*6) + 1
		if d < 1 {
			d = 1
		}
		if d > 5 {
			d = 5
		}
		tb.Delay[idx] = d

		tb.WSlow[idx] = float32(tb.Rng.NormFloat64()) * 0.01
		tb.WFast[idx] = 0.0
	}

	// Zero remaining
	for k := nRewire; k < nPrune; k++ {
		idx := pruneIdx[k]
		tb.WSlow[idx] = 0.0
		tb.WFast[idx] = 0.0
	}

	tb.enforceEiSigns()

	// Rebuild idxByDelay
	tb.IdxByDelay = make([][]int, 6)
	for i, d := range tb.Delay {
		tb.IdxByDelay[d] = append(tb.IdxByDelay[d], i)
	}
}

func (tb *TextBrain) DamageEdges(frac float32) {
	frac = clamp(frac, 0.0, 1.0)
	E := len(tb.Src)
	n := int(float32(E) * frac)
	if n <= 0 {
		return
	}
	perm := tb.Rng.Perm(E)
	for i := 0; i < n; i++ {
		idx := perm[i]
		tb.WSlow[idx] = 0.0
		tb.WFast[idx] = 0.0
	}
}

func (tb *TextBrain) Learn(targetId int, probs []float32) float32 {
	p := probs[targetId]
	loss := -float32(math.Log(float64(p + 1e-9)))

	adv := tb.updateModulators(loss)

	lr := tb.P.LR
	lrOut := lr * tb.LrOutMul
	lrRec := lr * tb.LrRecMul

	grad := make([]float32, len(probs))
	copy(grad, probs)
	grad[targetId] -= 1.0

	state := tb.computeState()

	// dR
	for i := 0; i < tb.N; i++ {
		s := state[i]
		if s == 0 {
			continue
		}
		for j := 0; j < tb.VocabSize; j++ {
			dR := lrOut * s * grad[j]
			dR = clamp(dR, -tb.MaxRUpdate, tb.MaxRUpdate)
			idx := i*tb.VocabSize + j
			tb.R[idx] -= dR
			tb.R[idx] = clamp(tb.R[idx], -1.0, 1.0)
		}
	}

	// db
	for j := 0; j < tb.VocabSize; j++ {
		db := lrOut * grad[j]
		db = clamp(db, -tb.MaxBUpdate, tb.MaxBUpdate)
		tb.B[j] -= db
	}

	// Recurrent Update
	// dW = lrRec * dopamine * adv * pre * post
	commonFactor := lrRec * tb.Dopamine * adv
	for i := range tb.Src {
		src := tb.Src[i]
		dst := tb.Dst[i]
		pre := tb.TraceFast[src]
		post := tb.A[dst]

		dW := commonFactor * pre * post
		dW = clamp(dW, -0.02, 0.02)
		tb.WFast[i] += dW
	}

	tb.enforceEiSigns()

	for i := range tb.WFast {
		tb.WFast[i] = clamp(tb.WFast[i], -0.5, 0.5)
		tb.WSlow[i] = clamp(tb.WSlow[i], -0.5, 0.5)
	}

	tb.decayFast()
	tb.consolidate()

	// Homeostasis
	for i := 0; i < tb.N; i++ {
		tb.Theta[i] += tb.P.Homeo * (tb.A[i] - tb.TargetRate)
	}

	if tb.P.PruneEvery > 0 && (tb.Step%tb.P.PruneEvery) == 0 {
		tb.pruneAndRewire()
	}

	return loss
}

// Stats
func (tb *TextBrain) AvgTheta() float32 {
	sum := float32(0)
	for _, v := range tb.Theta {
		sum += v
	}
	return sum / float32(tb.N)
}

func (tb *TextBrain) MeanAbsW() float32 {
	w := tb.effectiveW()
	sum := float32(0)
	for _, v := range w {
		sum += float32(math.Abs(float64(v)))
	}
	return sum / float32(len(w))
}

func (tb *TextBrain) FiringRate() float32 {
	sum := float32(0)
	for _, v := range tb.A {
		sum += v
	}
	return sum / float32(tb.N)
}

// -----------------------------
// Training
// -----------------------------

func trainOnText(genome, text string, steps, printEvery int) (*TextBrain, map[rune]int, map[int]rune) {
	stoi, itos := buildVocab(text)
	runes := []rune(text)
	ids := make([]int, len(runes))
	for i, r := range runes {
		ids[i] = stoi[r]
	}

	brain := NewTextBrain(genome, len(stoi))

	fmt.Printf("Stats: N=%d, K=%d\n", brain.N, brain.K)
	fmt.Printf("Target Active: %d (%.1f%%)\n", brain.P.KActive, brain.TargetRate*100)
	fmt.Printf("Vocab size: %d\n", len(stoi))

	losses := make([]float32, 0, printEvery)
	n := len(ids) - 1
	t0 := time.Now()

	for step := 0; step < steps; step++ {
		i := step % n
		x := ids[i]
		y := ids[i+1]

		probs := brain.Forward(x)
		loss := brain.Learn(y, probs)
		losses = append(losses, loss)

		if (step+1)%printEvery == 0 {
			sumL := float32(0)
			for _, l := range losses[len(losses)-printEvery:] {
				sumL += l
			}
			avgLoss := sumL / float32(printEvery)
			dt := time.Since(t0).Seconds()
			fmt.Printf("Step %d: Loss=%.4f | DA=%.3f | AvgTheta=%.3f | |W|=%.3f | Rate~%.3f | %.1fs\n",
				step+1, avgLoss, brain.Dopamine, brain.AvgTheta(), brain.MeanAbsW(), brain.FiringRate(), dt)
			t0 = time.Now()
		}
	}
	return brain, stoi, itos
}

// -----------------------------
// Sampling
// -----------------------------

func applySamplingFilters(probs []float32, temperature float32, topK int, topP float32) []float32 {
	p := make([]float32, len(probs))
	copy(p, probs)

	// Temp
	if temperature != 0 && temperature != 1.0 {
		for i := range p {
			p[i] = float32(math.Pow(float64(p[i]), 1.0/float64(temperature)))
		}
		p = normalizeProbs(p)
	}

	// Helper to sort
	pairs := make([]idxVal, len(p))
	for i, v := range p {
		pairs[i] = idxVal{i, v}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Val > pairs[j].Val
	})

	// Top K
	if topK > 0 && topK < len(p) {
		keep := make(map[int]bool)
		for i := 0; i < topK; i++ {
			keep[pairs[i].Idx] = true
		}
		for i := range p {
			if !keep[i] {
				p[i] = 0
			}
		}
		p = normalizeProbs(p)
	}

	// Top P
	if topP > 0 && topP < 1.0 {
		// Re-sort current p (might have changed)
		for i, v := range p {
			pairs[i] = idxVal{i, v}
		}
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].Val > pairs[j].Val
		})

		csum := float32(0)
		cutIdx := len(pairs) - 1
		for i, pair := range pairs {
			csum += pair.Val
			if csum > topP {
				cutIdx = i
				break
			}
		}
		// Keep up to cutIdx
		keep := make(map[int]bool)
		for i := 0; i <= cutIdx; i++ {
			keep[pairs[i].Idx] = true
		}
		for i := range p {
			if !keep[i] {
				p[i] = 0
			}
		}
		p = normalizeProbs(p)
	}

	return p
}

func sample(brain *TextBrain, stoi map[rune]int, itos map[int]rune, seed string, n int, temp float32, topK int, topP float32) string {
	brain.ResetState()
	seedRunes := []rune(seed)

	// Default start if empty seed or not in vocab
	if len(seedRunes) == 0 {
		for k := range stoi {
			seedRunes = []rune{k}
			break
		}
	}

	for _, r := range seedRunes[:len(seedRunes)-1] {
		if id, ok := stoi[r]; ok {
			brain.Forward(id)
		}
	}

	lastChar := seedRunes[len(seedRunes)-1]
	x := stoi[lastChar]
	out := string(seedRunes)

	for i := 0; i < n; i++ {
		probs := brain.Forward(x)
		p := applySamplingFilters(probs, temp, topK, topP)

		// Weighted choice
		r := brain.Rng.Float32()
		acc := float32(0)
		chosen := 0
		for idx, v := range p {
			acc += v
			if r < acc {
				chosen = idx
				break
			}
		}
		x = chosen
		out += string(itos[x])
	}
	return out
}

// -----------------------------
// Benchmarks
// -----------------------------

func benchmarkSelfRepair(genome, text string) {
	stoi, itos := buildVocab(text)
	runes := []rune(text)
	ids := make([]int, len(runes))
	for i, r := range runes {
		ids[i] = stoi[r]
	}

	brain := NewTextBrain(genome, len(stoi))
	n := len(ids) - 1

	evalLoss := func(steps int) float32 {
		losses := []float32{}
		for step := 0; step < steps; step++ {
			i := step % n
			x := ids[i]
			y := ids[i+1]
			probs := brain.Forward(x)
			p := probs[y]
			losses = append(losses, -float32(math.Log(float64(p+1e-9))))
		}
		sum := float32(0)
		for _, v := range losses {
			sum += v
		}
		return sum / float32(len(losses))
	}

	trainSteps := func(steps int) {
		for step := 0; step < steps; step++ {
			i := step % n
			x := ids[i]
			y := ids[i+1]
			probs := brain.Forward(x)
			brain.Learn(y, probs)
		}
	}

	preTrainSteps := 12000
	evalSteps := 2000
	recoverySteps := 12000
	damageFrac := float32(0.2)
	reportEvery := 2000

	fmt.Println("Pre-training...")
	trainSteps(preTrainSteps)
	preEval := evalLoss(evalSteps)

	fmt.Println("Damaging edges...")
	brain.DamageEdges(damageFrac)
	postDamageEval := evalLoss(evalSteps)

	fmt.Println("Recovering...")
	recoveryCurve := []float32{}
	remaining := recoverySteps
	for remaining > 0 {
		chunk := reportEvery
		if chunk > remaining {
			chunk = remaining
		}
		trainSteps(chunk)
		remaining -= chunk
		recoveryCurve = append(recoveryCurve, evalLoss(evalSteps))
	}

	fmt.Printf("Self-repair(eval): pre=%.4f | post_damage=%.4f | final=%.4f\n", preEval, postDamageEval, recoveryCurve[len(recoveryCurve)-1])

	fmt.Print("Recovery curve: ")
	for i, v := range recoveryCurve {
		fmt.Printf("%d:%.4f | ", (i+1)*reportEvery, v)
	}
	fmt.Println()

	gen := sample(brain, stoi, itos, "To be, or not to be", 300, 0.75, 18, 0.92)
	fmt.Println("------------------------------------------------------------")
	fmt.Println(gen)
	fmt.Println("------------------------------------------------------------")
}

// -----------------------------
// Main
// -----------------------------

func main() {
	text := `
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
    `
	// Repeat text
	var sb strings.Builder
	for i := 0; i < 40; i++ {
		sb.WriteString(text)
	}
	fullText := strings.Join(strings.Fields(sb.String()), " ")

	genome := "30321031103321200112332100123"

	fmt.Printf("Initializing Brain with genome: %s...\n", genome)
	brain, stoi, itos := trainOnText(genome, fullText, 120000, 5000)

	fmt.Println("\nTraining done. Generating text...")
	generated := sample(brain, stoi, itos, "To be, or not to be", 500, 0.75, 18, 0.92)

	fmt.Println("------------------------------------------------------------")
	fmt.Println(generated)
	fmt.Println("------------------------------------------------------------")

	fmt.Println("\nRunning self-repair benchmark...")
	benchmarkSelfRepair(genome, fullText)
}
