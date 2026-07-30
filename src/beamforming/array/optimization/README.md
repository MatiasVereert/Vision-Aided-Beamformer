# Array geometry optimization (`beamforming.array.optimization`)

Python port of

> Y. Konforti, I. Cohen, B. Berdugo, *"Array Geometry Optimization for
> Region-of-Interest Broadband Beamforming"*, IWAENC 2022.

Finds the placement of `M` microphones on a 1-D aperture that **maximises the
worst-case broadband directivity index over a region-of-interest** around
endfire, subject to a minimum White Noise Gain (robustness) and a minimum
inter-microphone distance. This is **Step 1** of the array-optimization plan:
faithfully replicate the paper before adapting the objective toward
data-dependent MVDR.

## Layout

| file | contents |
|------|----------|
| `farfield.py`      | far-field endfire signal model: grid, steering vectors, diffuse coherence `Γ` (Eq. 2, 8, 20, 25) |
| `metrics.py`       | WNG (Eq. 6), directivity factor (Eq. 7), broadband DI (Eq. 9) |
| `superdirective.py`| robust superdirective / diffuse-MVDR coefficient post-processing with `ε`-bisection (Sec. 4, Eq. 28-31) |
| `geometry_opt.py`  | the mixed-integer SOCP (Eq. 27) — the core optimizer |
| `baselines.py`     | ULA and dense-DMA reference geometries (Sec. 5) |
| `reproduce_paper.py` | runnable script that regenerates Figs. 1-2 |

## Quick start

```python
from beamforming.array.optimization import optimize_geometry

res = optimize_geometry(
    A=0.175, N=40, M=6, d_c=0.005,          # aperture, grid, #mics, min spacing
    fL=2000, fH=6000, theta_H_deg=30,        # band and ROI half-width
    delta_db=-10, Q=15, P=15,                # min WNG and #freq/#angle samples
    solver="MOSEK",                          # or "SCIP" (see below)
)
print(res.positions)          # optimal mic x-coordinates [m]
print(res.worstcase_di_db)    # worst-case broadband DI over the ROI [dB]
```

The MISOCP only outputs the **geometry** (`res.positions`). Per the paper, the
beamformer coefficients used for evaluation are then recomputed with the
superdirective post-processing (`robust_superdirective`), which is also applied
to the ULA/dense baselines for a fair comparison.

## Solver: SCIP vs MOSEK

The problem is a **mixed-integer second-order-cone program** with `N` binary
variables (the grid selection). The number of binaries — not `Q`/`P` — drives
the branch-and-bound difficulty.

* **SCIP** (open source, installed via `pyscipopt`) is the default and is fine
  for **validation / small instances**. It does **not** converge on the paper's
  full `N=40` fine-grid configuration in reasonable time (its conic relaxation
  is weak), returning `optimal_inaccurate` or degenerate layouts.
* **MOSEK** is what the authors use and is required to reproduce the full-scale
  **Fig. 1** geometry. It is not installed and needs a licence:

  ```bash
  pip install mosek
  # then place a licence at ~/mosek/mosek.lic
  # free academic licence: https://www.mosek.com/products/academic-licenses/
  ```

  Once licensed, pass `solver="MOSEK"`.

The paper's fine-grid structure (mics clustered near the edges with a central
gap) only emerges when the grid is fine enough that `d_c` binds
(`min_sep = ceil(d_c/Δx) ≥ 2`), i.e. around `N=40` for `A=17.5 cm`,
`d_c=0.5 cm`. Coarse grids collapse the constraint and change the optimum.

## Reproducing the figures

```bash
# reduced config, SCIP (finishes in minutes, qualitative result):
python -m beamforming.array.optimization.reproduce_paper

# paper's full config (needs MOSEK):
python -m beamforming.array.optimization.reproduce_paper --full --solver MOSEK
```

## Tests

```bash
pytest tests/test_array_optimization.py -q            # fast unit checks
pytest tests/test_array_optimization.py -q -m slow    # + tiny solver instance
```

## Next steps (beyond Step 1)

* Replace the isotropic diffuse coherence `Γ` (uniform angular weight) in
  `farfield.diffuse_coherence` with an **interferer-weighted** coherence
  `Γ_w = ∫ w(θ) d(θ)d(θ)ᴴ dθ` to bias the geometry toward MVDR-style
  null-steering while keeping the MISOCP convex.
* Evaluate the optimized vs ULA geometry with **data-dependent MVDR on ISM
  RIRs** and compare speech-enhancement metrics (PESQ/STOI/SIR).
