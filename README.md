# Curved Hopfield Network

Hopfield network simulations with curved (deformed) dynamics based on [Aguilera et al., *Nature Communications* (2025)](https://doi.org/10.1038/s41467-025-61475-w). The curvature parameter `gamma_0` introduces higher-order interactions that make the effective inverse temperature state-dependent, accelerating or decelerating memory retrieval.

## Quickstart

```bash
uv run python generate_patterns.py   # download & binarize images → data/*.npy
uv run python example.py             # shortest end-to-end recall demo → fig/curved_recall.gif
uv run python compare_gamma.py       # side-by-side gamma_0 comparison  → fig/compare_gamma.gif
uv run python gibbs_moments.py -0.3  # equilibrium moments at gamma_0=-0.3 → results/*.npz
uv run python curvednet_binary.py    # binary {0,1} recall demo           → fig/curved_recall_binary.gif
uv run pytest                        # run the 80 unit tests
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

The local field for neuron `i` is `h_i = sum_j W_ij * s_j`, computed as `W[i] @ state`. Two single-spin activation rules are available in `curvednet.py` and selectable via `run_curved_glauber(..., activation=...)`. The default is the **exact** rule.

- **Exact deformed conditional (default, paper Supp. Note 2 Eq. S2.5):** logistic of the gamma-exponential of the energy change for a flip,

    ```
    P(s_i stays) = 1 / (1 + exp_gamma(-2 * beta_eff * s_i * h_i))
    exp_gamma(z) = [1 + gamma * z]_+^(1/gamma),   gamma = gamma_0 / (N * beta)
    ```

    matches the exact conditional of the curved joint distribution `p_gamma(x) propto [1 - gamma * beta * E(x)]_+^(1/gamma)`. Reduces to the sigmoid below as `gamma_0 -> 0`.

- **Large-N approximation (paper Eq. S2.7), `activation="approx"`:** sigmoid driven by the effective inverse temperature,

    ```
    P(s_i = +1) = (1 + tanh(beta_eff * h_i)) / 2
    ```

    obtained by letting `gamma -> 0` with `gamma' = gamma * N` fixed. Differs from the exact rule only near the support boundary `|gamma_0 * E / N| -> 1`.

## Scripts

### `generate_patterns.py`

Downloads public-domain images from Wikimedia Commons and converts them to 128x128 binary {-1, +1} patterns stored as `.npy` files.

- **Sources**: Great Wave (Hokusai), Mona Lisa (da Vinci)
- **Output**: `data/*.npy` (binary patterns), `data/*.png` (previews)
- **Method**: Grayscale conversion, resize to 128x128, median-threshold binarization

### `example.py`

Shortest end-to-end pipeline (≈60 lines of code) — standalone, no project imports. Loads `data/*.npy`, builds the Hebbian matrix, runs a curved-Glauber Markov chain for each pattern with the exact deformed conditional, and writes `fig/curved_recall.gif`. Byte-identical to `curvednet.py`'s output under the same seed.

### `curvednet.py`

Curved Hopfield module. Importable helpers at module level (no side effects):

| Export | Purpose |
|---|---|
| `load_square_patterns` | glob + sorted-load `data/*.npy`, return `(patterns, N_SIDE, N)` |
| `hebbian_weights`      | `(1/N) sum_k x_k x_k^T` with zero diagonal |
| `snap_to_image`        | render a ±1 state as an upscaled grayscale PIL image |
| `prob_stay_gamma`      | `1 / (1 + exp_gamma(z))` — the γ-logistic primitive |
| `activation_exact`     | exact S2.5 conditional, `P(s_i=+1 \| rest)` |
| `activation_approx`    | approximate S2.7 conditional (large-N sigmoid) |
| `iter_curved_glauber`  | **generator** yielding state snapshots one at a time (O(N) peak memory) |
| `run_curved_glauber`   | thin list wrapper: `list(iter_curved_glauber(...))` |

Running the file directly (`uv run python curvednet.py`) executes the single-run demo with `beta=3.0`, `gamma_0=-1.2`, `N_STEPS=80000` and writes `fig/curved_recall.gif`.

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

### `gibbs_moments.py`

Equilibrium-moment sampler. Drives `curvednet.iter_curved_glauber` as one long Markov chain of `N_SWEEPS` sweeps (each = N single-spin updates) and stream-accumulates

```
eta_i  = <s_i>          (shape (N,))
eta_ij = <s_i s_j>      (shape (N, N), float32)
```

after a burn-in. CLI takes `gamma_0` as the first argument (e.g. `gibbs_moments.py -0.3`).

- **Patterns**: downsampled from `data/*.npy` to `N_SIDE x N_SIDE` (default 32×32) so `eta_ij` stays a few MB
- **Activation**: `ACTIVATION = "exact"` by default; set to `"approx"` for the large-N sigmoid
- **Output**: `results/gibbs_moments_g{+X.XX,-X.XX}.npz` containing `eta_i`, `eta_ij`, plus the config constants
- **Parameters** (top of file): `BETA`, `N_SWEEPS`, `BURN_IN`, `SAMPLE_INTERVAL`, `SEED`

## Binary encoding

`curvednet_binary.py` is a parallel module that uses {0, 1} binary neurons instead of {-1, +1} Ising spins. The energy function is:

```
E(x) = -0.5 x^T W x - b^T x
```

Translation functions convert parameters between the two encodings so both define the **same distribution** (up to a constant):

| Function | Input | Output |
|---|---|---|
| `ising_to_binary(W_s)` | Ising weights W_s | `(W_x, b_x)` where `W_x = 4 W_s`, `b_i = -2 sum_j W_s[i,j]` |
| `binary_to_ising(W_x, b_x)` | Binary weights + bias | Ising weights W_s (raises `ValueError` if b_x is inconsistent) |
| `state_ising_to_binary(s)` | {-1,+1} state | {0,1} state via `(s+1)/2` |
| `state_binary_to_ising(x)` | {0,1} state | {-1,+1} state via `2x-1` |

Running `uv run python curvednet_binary.py` produces `fig/curved_recall_binary.gif`.

## File structure

```
.
├── generate_patterns.py      # Download & binarize images
├── curvednet.py              # Curved Hopfield module ({-1,+1} Ising spins)
├── curvednet_binary.py       # Curved Hopfield module ({0,1} binary) + translation
├── compare_gamma.py          # Side-by-side gamma_0 comparison
├── gibbs_moments.py          # Equilibrium-moment Gibbs sampler
├── example.py                # Standalone end-to-end recall demo
├── test_curvednet.py         # pytest tests for curvednet
├── test_curvednet_binary.py  # pytest tests for curvednet_binary
├── data/                     # Stored binary patterns (.npy, .png)
├── fig/                      # Generated GIFs
├── results/                  # Sampler outputs (.npz, gitignored)
├── ref/                      # Reference paper PDF
├── tmp/                      # Older pedagogical Hopfield scripts
├── pyproject.toml
└── uv.lock
```

## Dependencies

- Python >= 3.8
- numpy
- Pillow
- matplotlib (available for downstream analysis; unused by core scripts)

Install with `uv sync`. Dev tools (`pytest`) live under `[dependency-groups] dev` in `pyproject.toml`; bring them in with `uv sync --dev` and run `uv run pytest`.
