# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**MagicBrain** (v0.6.0) — универсальная платформа для гетерогенных и гибридных архитектур нейронных сетей. Ядро — библиотека спайковых нейронных сетей (SNN) с ДНК-подобным кодированием и структурной пластичностью. Часть экосистемы StudyNinja для исследования когнитивных процессов обучения и адаптивных алгоритмов.

**Важно**: MagicBrain — исследовательская библиотека, не production-сервис.

## Installation

```bash
# Core (NumPy only)
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Specific extras
pip install -e ".[dev]"          # pytest, ruff
pip install -e ".[torch]"       # PyTorch + torchvision
pip install -e ".[transformers]" # HuggingFace
pip install -e ".[jax]"         # JAX backend
pip install -e ".[viz]"         # matplotlib + plotly
pip install -e ".[platform]"    # torch + transformers
```

Requires Python >= 3.9. Core dependency: `numpy >= 1.24.0`.

## Common Commands

### CLI

```bash
# Train a new model
magicbrain train --genome "30121033102301230112332100123" --steps 10000 --out model.npz

# Train from custom text file
magicbrain train --text /path/to/text.txt --steps 20000 --out model.npz

# Resume training from existing model
magicbrain train --load model.npz --steps 5000 --out model_v2.npz

# Generate text from trained model
magicbrain sample --model model.npz --seed "To be" --n 500 --temp 0.75

# Run self-repair benchmark
magicbrain repair --genome "30121033102301230112332100123" --damage 0.2

# Evolve genomes via genetic algorithm
magicbrain evolve --genome "30121033102301230112332100123" --generations 10 --population 20 --fitness loss

# Train with live diagnostics monitoring
magicbrain monitor --genome "30121033102301230112332100123" --steps 10000 --metrics metrics.json

# NeuroGenesis Engine: compile dataset into genome
magicbrain neurogenesis compile --text /path/to/text.txt --strategy statistical

# NeuroGenesis: full pipeline (compile → develop → train → reconstruct)
magicbrain neurogenesis run --text /path/to/text.txt --strategy statistical --steps 10000 --out model.npz

# NeuroGenesis: full pipeline with CPPN weight generation
magicbrain neurogenesis run --text /path/to/text.txt --cppn --steps 10000

# NeuroGenesis: benchmark strategies against baselines
magicbrain neurogenesis benchmark --text /path/to/text.txt --steps 5000 --trials 3

# NeuroGenesis: discover attractors in trained model
magicbrain neurogenesis attractors --model model.npz --probes 500
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=magicbrain

# Run specific test file
pytest tests/test_smoke.py

# Run with verbose output
pytest -v

# Run single test function
pytest tests/test_smoke.py::test_brain_init

# Run platform tests only
pytest tests/platform/

# Run hybrid tests only
pytest tests/hybrid/

# Run neurogenesis tests only
pytest tests/neurogenesis/
```

Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`. Tests are collected from `tests/` directory.

### REST API

```bash
# Start the FastAPI server
cd api && python main.py
```

API endpoints: model CRUD, training, inference, diagnostics, evolution. See `api/README_API.md`.

### Quick Python Usage

```python
from magicbrain import TextBrain, decode_genome, sample, save_model, load_model

brain = TextBrain("30121033102301230112332100123", vocab_size=50)
probs = brain.forward(token_id=0)
loss = brain.learn(target_id=1, probs=probs)
```

## Code Organization

```
magicbrain/
├── __init__.py              - Public API: TextBrain, decode_genome, sample, save_model, load_model
├── brain.py                 - Core TextBrain SNN class (forward, learn, plasticity)
├── genome.py                - DNA-like genome decoding to hyperparameters
├── graph.py                 - 3D spatial graph construction with axonal delays
├── sampling.py              - Text generation (temperature, top-k, top-p)
├── io.py                    - Save/load models (.npz format)
├── utils.py                 - Utilities (softmax, sigmoid, sparsify_topm)
├── cli.py                   - CLI: train, sample, repair, evolve, monitor, neurogenesis
│
├── tasks/
│   ├── text_task.py         - Vocab building and training loop
│   ├── self_repair.py       - Self-repair benchmark after edge damage
│   └── neurogenesis_task.py - NeuroGenesis E2E pipeline & benchmark
│
├── architectures/
│   └── hierarchical_brain.py - Multi-layer hierarchical SNN
│
├── backends/
│   ├── backend_interface.py - Abstract computation backend
│   ├── numpy_backend.py     - NumPy implementation (default)
│   └── jax_backend.py       - JAX implementation (optional)
│
├── diagnostics/
│   ├── live_monitor.py      - Real-time training metrics & logging
│   ├── neuronal_dynamics.py - Neural activity analysis
│   ├── plasticity_tracker.py - Synaptic weight evolution tracking
│   └── synaptic_metrics.py  - Synaptic statistics computation
│
├── evolution/
│   ├── fitness_functions.py - Fitness: loss, convergence, stability
│   ├── genome_mutator.py    - Point, crossover, blend mutations
│   └── simple_ga.py         - Population-based genetic algorithm
│
├── learning_rules/
│   ├── stdp.py              - Spike-Timing-Dependent Plasticity
│   └── stdp_brain.py        - TextBrain variant with STDP learning
│
├── hybrid/                  - Hybrid architecture combinations
│   ├── base.py              - HybridArchitecture base class
│   ├── builder.py           - Fluent HybridBuilder API
│   ├── snn_dnn.py           - SNN + DNN combination
│   ├── snn_transformer.py   - SNN + Transformer combination
│   ├── cnn_snn.py           - CNN + SNN combination
│   └── spiking_attention.py - Attention mechanism in spike domain
│
├── platform/                - Core orchestration framework
│   ├── model_interface.py   - ModelInterface, ModelType, OutputType, ModelMetadata
│   ├── orchestrator/
│   │   └── orchestrator.py  - ModelOrchestrator with execution strategies
│   ├── registry/
│   │   └── model_registry.py - Model versioning & metadata registry
│   └── communication/
│       ├── message.py       - Message data structures
│       ├── message_bus.py   - Message routing between models
│       └── converters.py    - Type conversion between model outputs
│
├── models/                  - Platform model adapters
│   ├── snn/text_model.py    - SNNTextModel adapter
│   ├── dnn/pytorch_model.py - PyTorch DNN adapter
│   ├── rnn/recurrent_model.py - LSTM/GRU adapter
│   ├── cnn/vision_model.py  - Torchvision adapter
│   └── transformers/hf_model.py - Hugging Face adapter
│
├── neurogenesis/               - NeuroGenesis Engine (neurogenomic memory)
│   ├── __init__.py             - Public API
│   ├── compiler.py             - GenomeCompiler: dataset → genome (hash/statistical/hybrid)
│   ├── energy.py               - Hopfield energy E(s,W) & gradient computation
│   ├── attractor_dynamics.py   - Continuous dynamics converging to attractors
│   ├── cppn.py                 - CPPN: spatial coordinates → synaptic weights
│   ├── development.py          - Development operator: genome → 3D neural tissue
│   ├── pattern_memory.py       - Hopfield memory with Storkey learning rule
│   ├── reconstruction.py       - Attractor/autoregressive data reconstruction
│   └── genome_v2.py            - Extended genome format (72+ positions)
│
├── integration/
│   ├── knowledgebase_client.py - External KnowledgeBase API client
│   └── neural_digital_twin.py  - Brain state modeling (digital twin)
│
└── zoo/
    └── zoo_manager.py       - Pretrained model catalog & management

api/                         - FastAPI REST microservice
├── main.py                  - Uvicorn entry point
├── app/
│   ├── api/routes/
│   │   ├── models.py        - Model CRUD endpoints
│   │   ├── training.py      - Training endpoints
│   │   ├── inference.py     - Forward pass & sampling
│   │   ├── diagnostics.py   - Brain metrics
│   │   └── evolution.py     - Genome evolution
│   └── core/config.py       - Settings
└── tests/test_api.py        - API tests

tests/                       - Test suite (~150 tests)
├── test_smoke.py            - Core SNN sanity tests
├── test_backends.py         - Backend implementations
├── test_evolution.py        - Genome evolution & GA
├── test_diagnostics.py      - Monitoring tools
├── test_io_metadata.py      - Serialization roundtrips
├── test_integration.py      - End-to-end workflows
├── test_stdp.py             - STDP learning rule
├── test_hierarchical.py     - Hierarchical brain
├── platform/                - Platform framework tests
│   ├── test_orchestrator.py
│   ├── test_registry.py
│   ├── test_async_orchestrator.py
│   ├── test_communication.py
│   └── test_snn_adapter.py
├── hybrid/                  - Hybrid architecture tests
│   ├── test_hybrid_architectures.py
│   └── test_builder.py
├── neurogenesis/            - NeuroGenesis Engine tests (93 tests)
│   ├── test_compiler.py
│   ├── test_energy.py
│   ├── test_cppn.py
│   ├── test_development.py
│   ├── test_pattern_memory.py
│   ├── test_reconstruction.py
│   └── test_genome_v2.py
└── integration/
    └── test_knowledgebase_api.py

examples/
├── quickstart.py
└── platform/
    ├── basic_usage.py
    └── ensemble_example.py
```

## Architecture

### Layer 1: Core SNN Engine

#### Genome System (`genome.py`)

The network is fully determined by a base-4 string (characters '0'-'3'). `decode_genome()` maps it to 17+ hyperparameters: network size N (256-832), connectivity K (8-20), learning rate, decay rates, pruning params, dopamine modulation. One genome = one reproducible architecture.

Default genome: `"30121033102301230112332100123"`

#### Brain (`brain.py`)

**TextBrain** — the core spiking neural network for next-character prediction.

Key state variables:
- `a` — binary spike activation (N,)
- `trace_fast` / `trace_slow` — dual-timescale synaptic traces
- `w_slow` / `w_fast` — dual weight system (long-term consolidation + fast plasticity)
- `theta` — homeostatic adaptive threshold
- Delay buffers (1-5 timesteps) for axonal transmission modeling
- `R`, `b` — linear readout layer for logits

**Forward pass** (`forward(token_id)`): shift delay buffers → compute input signal → sparse top-k selection → update traces → propagate through graph with delays → compose state (`a + alpha*trace_fast + beta*trace_slow`) → normalize → readout logits.

**Learning** (`learn(target_id, probs)`): compute loss → dopamine neuromodulation (advantage-based) → gradient step on R/b → Hebbian recurrent update (`dW = lr * dopamine * advantage * pre * post`) → consolidate w_fast→w_slow → homeostatic threshold adaptation → periodic pruning/rewiring.

**Invariants**:
- Sparse activation: only `k_active` neurons (~5% of N) fire per step
- E/I balance: inhibitory neuron weights are clamped to non-positive after every update
- Dopamine modulation is reward-based (advantage = loss_ema - loss)

#### Graph (`graph.py`)

Neurons positioned in 3D space. K-nearest-neighbor local connectivity plus `p_long` fraction of random long-range connections. Axonal delays proportional to Euclidean distance (1-5 steps).

#### Sampling (`sampling.py`)

Temperature scaling, top-k filtering, top-p (nucleus) sampling for text generation.

### Layer 2: Extended Capabilities

#### Learning Rules (`learning_rules/`)

STDP (Spike-Timing-Dependent Plasticity) — alternative biologically-inspired learning rule. `STDPBrain` is a TextBrain variant using STDP instead of dopamine-modulated Hebbian learning.

#### Hierarchical Architecture (`architectures/hierarchical_brain.py`)

Multi-layer SNN where each layer is a TextBrain. Information flows bottom-up with configurable inter-layer connectivity.

#### Backends (`backends/`)

Abstract computation backend with NumPy (default) and JAX (optional, for GPU acceleration) implementations.

#### Diagnostics (`diagnostics/`)

- `LiveMonitor` — real-time training metrics (loss, dopamine, firing rate, theta)
- `NeuronalDynamics` — spike raster analysis, population activity
- `PlasticityTracker` — weight distribution evolution over time
- `SynapticMetrics` — connectivity statistics, weight magnitude distributions

#### Evolution (`evolution/`)

Genetic algorithm for genome optimization:
- `GenomeMutator` — point mutation, crossover, blend mutations on base-4 genomes
- `FitnessEvaluator` — loss-based, convergence, and stability fitness functions
- `SimpleGA` — tournament selection, elitism, population-based search

### Layer 3: Platform Framework

#### Model Interface (`platform/model_interface.py`)

Universal `ModelInterface` abstract base class. All model types (SNN, DNN, CNN, RNN, Transformer, Hybrid, Ensemble) implement this interface. Key types: `ModelType`, `OutputType`, `ModelMetadata`, `ModelState`.

#### Orchestrator (`platform/orchestrator/`)

`ModelOrchestrator` executes multi-model workflows with 7 strategies:
- SEQUENTIAL, PARALLEL, PIPELINE, HIERARCHICAL, FEEDBACK, CASCADED, MIXTURE_OF_EXPERTS

#### Registry (`platform/registry/`)

`ModelRegistry` — model versioning, metadata storage, dependency tracking.

#### Communication (`platform/communication/`)

`MessageBus` for inter-model messaging. `ConverterRegistry` handles automatic type conversion between different output types (e.g., spikes → dense).

#### Model Adapters (`models/`)

Platform adapters wrapping external frameworks into `ModelInterface`:
- `SNNTextModel` — wraps TextBrain
- `DNNModel` — wraps PyTorch models
- `RNNModel` — wraps LSTM/GRU
- `CNNModel` — wraps torchvision models
- `TransformerModel` — wraps Hugging Face models

### Layer 4: Hybrid Architectures

#### Hybrid Base (`hybrid/base.py`)

`HybridArchitecture` base class with automatic type conversion between components. Predefined combinations: SNN+DNN, SNN+Transformer, CNN+SNN.

#### Builder (`hybrid/builder.py`)

Fluent API for composing arbitrary hybrid architectures:

```python
hybrid = (HybridBuilder()
    .add("snn", snn_model)
    .add("dnn", dnn_model)
    .connect("snn", "dnn")
    .set_output("dnn")
    .build("my_hybrid"))
```

#### Spiking Attention (`hybrid/spiking_attention.py`)

Attention mechanism operating in the spike domain for SNN+Transformer hybrids.

### Layer 5: NeuroGenesis Engine (Neurogenomic Memory)

Core thesis: **data is not stored directly — a compact generative program (genome) is stored that can reproduce data through dynamic neural structure activation.**

Pipeline: `Dataset → Compile → Genome → Develop → Train → Reconstruct`

#### GenomeCompiler (`neurogenesis/compiler.py`)

Compiles datasets into deterministic base-4 genomes. Three strategies:
- **hash**: SHA-256(data) → base-4 (pseudorandom baseline)
- **statistical**: data statistics (entropy, vocab, repetitiveness) → informed genome positions
- **hybrid**: statistics for architecture params (N, K, p_long), hash for the rest

`analyze_dataset()` extracts: size, vocab_size, entropy, repetitiveness, ngram concentration.

#### Energy Function (`neurogenesis/energy.py`)

Hopfield-style energy: `E(s) = -½ sᵀWs - θᵀs + λ||s||₁`

Supports both dense (N×N) and sparse (edge-list) weight formats. Provides energy gradient, local field computation, stability checking, and basin profiling.

#### Attractor Dynamics (`neurogenesis/attractor_dynamics.py`)

Continuous neural dynamics converging to attractor states via energy minimization:
- `step()`: `s_{t+1} = momentum * s_t + (1-momentum) * σ(W·s_t/τ + θ)`
- `converge()`: iterate until `||s_{t+1} - s_t|| < tolerance`
- `find_attractors()`: probe from N random states, cluster results

Key difference from TextBrain's hard top-k: uses continuous sigmoid activation for smooth convergence.

#### CPPN (`neurogenesis/cppn.py`)

Compositional Pattern-Producing Network: generates synaptic weights as a function of neuron spatial coordinates: `w_ij = CPPN(pos_i, pos_j, dist_ij, type_i, type_j)`. Architecture (layers, widths, activation functions) is encoded in the genome. 8 basis functions: sin, cos, gaussian, sigmoid, tanh, abs, identity, step.

#### Development Operator (`neurogenesis/development.py`)

Three-stage "morphogenesis":
1. **Morphogenesis**: 3D positions + KNN connectivity (reuses `graph.py`)
2. **Synaptogenesis**: CPPN generates spatially-patterned weights (or random fallback)
3. **Maturation**: threshold calibration based on incoming weight statistics

#### Pattern Memory (`neurogenesis/pattern_memory.py`)

Hopfield-style associative memory with **Storkey learning rule** (capacity ~0.14N patterns). Uses bipolar (-1/+1) encoding internally, 0/1 externally. `text_to_pattern()` converts token sequences to sparse neural patterns. `recall()` converges from noisy/partial cues to stored patterns via tanh dynamics.

#### Reconstruction Operator (`neurogenesis/reconstruction.py`)

Multiple reconstruction modes:
- **Autoregressive**: standard next-token sampling from trained TextBrain
- **Attractor decoding**: find attractors → decode each via readout matrix R
- **Cue-based recall**: prime with partial text → generate continuation
- **Fidelity metrics**: char_accuracy, bigram/trigram overlap, vocab overlap

#### Extended Genome v2 (`neurogenesis/genome_v2.py`)

Backward-compatible extension: `[Topology 0-23][CPPN 24-55][Attractor 56-71][Patterns 72+]`. Standard genomes (24-28 chars) decode identically. Extended genomes add CPPN architecture params (layers, widths, activations), attractor dynamics config (tau, momentum, tolerance), and pattern seeds.

### Layer 6: Integration & Services

#### REST API (`api/`)

FastAPI microservice with endpoints for model management, training, inference, diagnostics, and evolution. Entry point: `api/main.py`.

#### KnowledgeBase Client (`integration/knowledgebase_client.py`)

Client for external KnowledgeBase API — stores/retrieves neural representations.

#### Neural Digital Twin (`integration/neural_digital_twin.py`)

Models brain state evolution over time for analysis and prediction.

#### Model Zoo (`zoo/zoo_manager.py`)

Pretrained model catalog with download and version management.

## File I/O Format

Model files (`.npz`) contain:
- `genome_str` — source genome string
- `vocab` — JSON with `stoi` (str→int) and `itos` (int→str)
- `w_slow`, `w_fast` — synapse weight arrays
- `R`, `b` — readout layer parameters
- `theta` — homeostatic thresholds
- `meta` — metadata (step, timestamp, N, K)

## Testing Strategy

**Smoke tests** (`test_smoke.py`): genome decoding, brain init shapes, forward pass validity, learning step loss, save/load roundtrip, E/I sign invariant.

**Module tests**: Each major subsystem (backends, evolution, diagnostics, STDP, hierarchical, platform, hybrid, integration) has dedicated test files.

**For new features**: Add tests in `tests/` following existing patterns. Place platform tests in `tests/platform/`, hybrid tests in `tests/hybrid/`, etc.

## Common Development Patterns

### Creating a new task

1. Create `magicbrain/tasks/your_task.py`
2. Follow the pattern from `text_task.py`: build vocab → training loop using `brain.forward()` and `brain.learn()`
3. Wire it into `cli.py` as a new subparser command

### Adding a CLI command

1. Add a subparser in `cli.py` under the `main()` function
2. Define arguments
3. Add handler in the command dispatch (`if args.command == "..."`)
4. Import task modules from `magicbrain.tasks`

### Adding a new model adapter

1. Create `magicbrain/models/<type>/<name>_model.py`
2. Implement `ModelInterface` from `magicbrain.platform.model_interface`
3. Register in the corresponding `__init__.py`
4. Add tests in `tests/platform/`

### Creating a hybrid architecture

1. Use `HybridBuilder` for custom combinations, or
2. Subclass `HybridArchitecture` from `magicbrain.hybrid.base`
3. Add tests in `tests/hybrid/`

### Using the NeuroGenesis Engine

```python
from magicbrain.neurogenesis import GenomeCompiler, DevelopmentOperator, AttractorDynamics
from magicbrain.neurogenesis import ReconstructionOperator, PatternMemory

# 1. Compile dataset into genome
compiler = GenomeCompiler()
genome = compiler.compile("your text data here", strategy="statistical")

# 2. Develop neural tissue with CPPN
dev = DevelopmentOperator()
brain, tissue = dev.develop_and_build_brain(genome, vocab_size=50, use_cppn=True)

# 3. Full pipeline (compile → develop → train → reconstruct → evaluate)
from magicbrain.tasks.neurogenesis_task import run_neurogenesis_pipeline
result = run_neurogenesis_pipeline("your data", strategy="statistical", steps=10000)
print(result.fidelity, result.compression)

# 4. Pattern memory (Hopfield-style)
mem = PatternMemory(N=256, sparsity=0.1)
pattern = mem.text_to_pattern([1, 2, 3], vocab_size=50)
mem.imprint_pattern(pattern)
recall = mem.recall(noisy_cue)

# 5. Attractor analysis of trained model
dynamics = AttractorDynamics(tau=0.3)
attractors = dynamics.find_attractors(brain.N, W_dense, brain.theta, n_probes=500)
```

### Modifying core architecture

**Critical paths**:
- `brain.py` methods `forward()` and `learn()` are the core — change with care
- Ensure `io.py` save/load compatibility after structural changes
- Update `genome.py` decode logic if adding new hyperparameters
- Update `tests/test_smoke.py` for core changes

## Debugging Tips

**Training diagnostics** (print during training):
- `brain.dopamine` — should oscillate around 0.5
- `brain.avg_theta()` — should slowly increase toward positive values
- `brain.firing_rate()` — should stay near `target_rate`
- `brain.mean_abs_w()` — average absolute weight magnitude

**If loss does not decrease**:
- Check learning rate in genome
- Verify `k_active` is not too small
- Ensure dopamine modulation is working (not stuck at 0 or 1)

**If generated text is nonsensical**:
- Ensure sufficient training (>10k steps on simple text)
- Check temperature (0.7-0.9 typically works)
- Verify vocab was correctly loaded

**Use the `monitor` CLI command** for real-time diagnostics during training.

## Performance Considerations

- **Memory**: dominated by graph edges (N*K) and trace arrays (N)
- **Typical sizes**: N=384, K=12, vocab=50 → ~200KB model file
- **Scaling**: for N > 1000, consider sparse matrix operations
- **Optional acceleration**: install JAX backend for GPU support

## Key Parameters (via Genome)

Most impactful:
- `N`, `K` — network capacity and connectivity
- `lr` — learning rate (too high → instability)
- `k_active` — number of active neurons per step (sparse representation)
- `alpha` / `beta` — fast vs. slow memory balance
- `dopamine_gain` / `dopamine_bias` — reward signal sensitivity

Default genome for experiments: `"30121033102301230112332100123"`

## Integration with StudyNinja

MagicBrain is part of the StudyNinja ecosystem for:
1. **Modeling student learning** — spiking networks with plasticity as knowledge formation models
2. **Adaptive algorithms** — self-repair and adaptation mechanisms for educational systems
3. **Neural modeling** — biologically plausible cognitive process simulation
