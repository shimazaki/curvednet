# Curved Hopfield Network

Hopfield network simulations with curved (deformed) dynamics based on [Aguilera et al., *Nature Communications* (2025)](https://doi.org/10.1038/s41467-025-56266-0). The curvature parameter `gamma_0` introduces higher-order interactions that make the effective inverse temperature state-dependent, accelerating or decelerating memory retrieval.

## Quickstart

```bash
uv run python generate_patterns.py   # download & binarize images
uv run python compare_gamma.py       # produce side-by-side comparison GIF
```

## Theory

The standard Hopfield network uses Glauber dynamics with fixed inverse temperature beta. The curved variant replaces it with an effective, state-dependent inverse temperature:

```
beta_eff(x) = beta / [1 - gamma_0 * E(x) / N]+
```

where `E(x) = -0.5 * x^T W x` is the Ising energy and `[...]+ = max(..., eps)` is a positivity clamp.

| gamma_0 | Effect |
|---------|--------|
| `< 0`  | Self-regulating annealing: beta_eff increases as energy drops, accelerating convergence |
| `= 0`  | Standard Hopfield/Ising model (constant beta) |
| `> 0`  | Decelerates dynamics, potentially more robust retrieval |

The local field for neuron `i` is `h_i = sum_j W_ij * s_j`, computed as `W[i] @ state`. The spin update probability follows:

```
P(s_i = +1) = (1 + tanh(beta_eff * h_i)) / 2
```

## Scripts

### `generate_patterns.py`

Downloads public-domain images from Wikimedia Commons and converts them to 128x128 binary {-1, +1} patterns stored as `.npy` files.

- **Sources**: Great Wave (Hokusai), Mona Lisa (da Vinci)
- **Output**: `patterns/*.npy` (binary patterns), `patterns/*.png` (previews)
- **Method**: Grayscale conversion, resize to 128x128, median-threshold binarization

### `hopfield_ising_1simple.py`

Minimal Hopfield network demo with deterministic (zero-temperature) dynamics.

- **Patterns**: Procedurally generated cross and square (32x32)
- **Update rule**: `s_i = sign(h_i)` (hard threshold, no stochasticity)
- **Output**: `fig/hopfield_recall.gif`

### `hopfield_ising_2noisy.py`

Stochastic (finite-temperature) Hopfield network with tanh activation.

- **Patterns**: Same procedural cross and square (32x32)
- **Update rule**: `P(s_i = +1) = (1 + tanh(beta * h_i)) / 2` (Glauber dynamics)
- **Parameters**: `beta=2.0`, `N_STEPS=20000`
- **Output**: `fig/hopfield_recall.gif`

### `hopfield_ising.py`

Stochastic Hopfield network using real image patterns from `patterns/*.npy`.

- **Patterns**: Loaded from `generate_patterns.py` output (128x128)
- **Parameters**: `beta=3.0`, `N_STEPS=80000`, `NOISE_FRAC=0.30`
- **Output**: `fig/hopfield_recall.gif`

### `curved_network.py`

Single-run curved Hopfield network with configurable `gamma_0`.

- **Key addition**: State-dependent `beta_eff` via the curvature formula
- **Energy tracking**: Incremental energy updates for efficiency (`energy -= delta * h`)
- **Parameters**: `beta=3.0`, `gamma_0=-1.2`, `N_STEPS=80000`
- **Output**: `fig/curved_recall.gif`

### `compare_gamma.py`

Side-by-side comparison of recall dynamics across different `gamma_0` values.

- **Comparison**: Runs multiple `gamma_0` values from identical noisy initial states
- **RNG synchronization**: Captures and restores RNG state so all runs share the same initial noise
- **Output**: `fig/compare_gamma.gif` with labeled columns (shows beta and gamma')
- **Parameters** (edit at top of file):
  - `GAMMAS`: list of gamma_0 values to compare (default: `[0.0, -2.0]`)
  - `BETA`: inverse temperature (default: `1.5`)
  - `N_STEPS`: number of single-spin-flip updates (default: `100000`)
  - `NOISE_FRAC`: fraction of spins flipped in initial state (default: `0.30`)

## File structure

```
.
├── generate_patterns.py      # Download & binarize images
├── hopfield_ising_1simple.py # Deterministic Hopfield (32x32, procedural)
├── hopfield_ising_2noisy.py  # Stochastic Hopfield (32x32, procedural)
├── hopfield_ising.py         # Stochastic Hopfield (128x128, real images)
├── curved_network.py         # Curved Hopfield, single gamma_0
├── compare_gamma.py          # Side-by-side gamma_0 comparison
├── patterns/
│   ├── .cache/               # Raw downloaded images
│   ├── great_wave.npy        # Binary pattern
│   ├── great_wave.png        # Preview
│   ├── mona_lisa.npy
│   └── mona_lisa.png
└── pyproject.toml
```

## Dependencies

- Python >= 3.8
- numpy
- Pillow
- matplotlib (unused by core scripts, available for analysis)

Install with `uv sync` or `pip install -e .`.
