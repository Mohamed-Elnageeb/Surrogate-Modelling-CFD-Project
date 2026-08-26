# Surrogate Modelling for CFD — airfoil aerodynamics

[![CI](https://github.com/Mohamed-Elnageeb/Surrogate-Modelling-CFD-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohamed-Elnageeb/Surrogate-Modelling-CFD-Project/actions/workflows/ci.yml)

A 2-D airfoil CFD and surrogate-modelling pipeline built around SU2, validated
against published experimental data.

The project began as a surrogate-assisted design loop. Auditing it against
reference aerodynamics turned up errors large enough that the audit became the
substance of the work, so the repository documents both the working pipeline and
what validating it revealed.

**→ [`RESULTS.md`](RESULTS.md)** — every quantitative result, machine-generated.
**→ [`HANDOFF.md`](HANDOFF.md)** — the unfinished research thread, for a separate repo.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

pytest -m "not slow"                              # 76 tests, ~14 s
python scripts/generate_results.py                # regenerates RESULTS.md
python -m cfdagent.validation.figures             # writes reports/figures/
```

Nothing above needs SU2 or a GPU. Exact versions behind the published numbers are
in `requirements.lock.txt`.

For the CFD paths, SU2 is not on PyPI or apt — it ships prebuilt binaries:

```bash
./scripts/install_su2.sh          # SU2 8.1.0 -> ~/.local/su2
export SU2_RUN="$HOME/.local/su2/bin" PATH="$SU2_RUN:$PATH"
```

SU2 is CPU-bound (MPI/OpenMP); a GPU does not accelerate it. Run CFD on local
cores and reserve GPUs for surrogate training.

---

## What this does

### Geometry
10-parameter camber/thickness airfoil family on a NACA-0012 baseline. Exports in
**Selig order** (`to_selig_order`) for external tools — see finding 1 below for
why that matters.

### Meshing
```bash
python -m cfdagent.cfd.bl_mesher --out mesh.su2 --reynolds 6e6
```
Quad boundary-layer extrusion with a y⁺-sized first cell, geometric growth, wake
refinement, 60-chord farfield, and the `Airfoil`/`Farfield` markers SU2 expects.
NACA0012 at Re = 6×10⁶: 162 458 nodes, first layer 4.46×10⁻⁶ chords. This is the
default path in `run_cfd`; the uniform mesher remains behind
`boundary_layer=False` for inviscid work.

### Solving
`run_cfd` drives SU2, extracts coefficients across SU2 history-format variants,
and validates them — negative drag, NaNs, and unconverged residuals are rejected
rather than absorbed.

### Fast evaluation
`cfdagent/lowfid/` wraps NeuralFoil as a `run_cfd`-compatible evaluator at
~3 ms/design, gated by tests against published NACA0012 data (lift-curve slope
102 % of thin-airfoil theory; Cd 0.00747 and L/D 58.0 at α = 4°, Re = 10⁶).

### Surrogates
Snapshot builder (interpolates unstructured SU2 fields onto a regular grid), a
CNN geometry→(Cl, Cd) model, and a U-Net for field prediction.

> **Caveat, stated plainly.** The included CNN-vs-random-search screening
> comparison uses NeuralFoil as *both* the label source and the "expensive"
> solver, so it measures screening behaviour on this design space — not savings
> against RANS. NeuralFoil is itself a neural surrogate and is faster and more
> accurate than the CNN distilled from it. It is included as a pipeline
> demonstration, not as a result.

---

## What the audit found

Full numbers in [`RESULTS.md`](RESULTS.md); each is reproducible from one command.

**1. Coordinate ordering — 8.9× drag error.** The generator emitted a loop
anchored at the leading edge; aerodynamic tools expect Selig order, anchored at
the trailing edge. Passing the wrong one raises no error — it silently describes
a blunt-nosed body. Maximum geometric deviation from a true NACA0012 was
3.55×10⁻⁹: the shape was exact, only the ordering was wrong.
→ `python -m cfdagent.experiments.convention_error`

**2. Is that hazard widespread? No.** All 2 174 airfoils in the UIUC database as
redistributed with AeroSandbox are Selig-ordered. A negative result, recorded
along with a false positive it initially produced.
→ `python -m cfdagent.experiments.convention_census`

**3. Mesh 406× too coarse for viscous drag.** Wall spacing was 9.55×10⁻³ chords
against the ~2.4×10⁻⁵ needed for y⁺ ≈ 1 at Re = 10⁶. With a resolved mesh, Cl
goes 0.0709 → **0.4419** (published 0.43–0.46) and L/D 3.0 → **39.9**. The
original case also exhausted 2 200 iterations without converging.

**4. Negative drag, silently absorbed.** An inviscid run returned Cd = −0.0097.
Drag cannot be negative; the previous post-processing took `abs()` and logged it
as valid training data.

**5. Prediction accuracy depends on which experiment you choose.** Scored against
NASA-hosted data, the same model appears **3.3× more accurate** against an
untripped experiment than a tripped one — because it models natural transition
and the tripped model does not. Drag error is 9.7× the lift error.
→ `python -m cfdagent.validation.transition_study`

---

## Layout

| Path | Purpose |
|---|---|
| `cfdagent/geometry/` | Airfoil parameterisation, Selig-order export |
| `cfdagent/cfd/bl_mesher.py` | Boundary-layer mesher, y⁺-sized first cell |
| `cfdagent/cfd/run_cfd.py` | SU2 driver, metric extraction, validation |
| `cfdagent/lowfid/` | NeuralFoil evaluator, validated |
| `cfdagent/validation/` | Experimental data, transition study, figures |
| `cfdagent/experiments/` | Standalone reproductions of the audit findings |
| `cfdagent/surrogate/` | Snapshot builder, CNN and U-Net models |
| `scripts/` | SU2 install, sweeps, results generation |
| `TestCases/airfoil_naca0012_validated/` | Converged RANS-SA reference case |

## Limitations

- Everything is **2-D and subsonic**.
- The design loop is a random search with a gradient-boosting seed model. It is
  not a competitive optimiser and is not presented as one.
- Results are seeded and portable but **not bit-reproducible** across BLAS
  versions or hardware; the 10-seed error bars are the meaningful unit.
- The Abbott dataset is digitised from a printed plot (n = 12) and its NASA
  source file warns the digitisation is approximate.
- The SU2 half of the transition comparison is scripted but **not run to
  convergence** — see [`HANDOFF.md`](HANDOFF.md). No number from it is quoted as
  a result anywhere in this repository.

## Data

Experimental data in `cfdagent/data/experimental/` comes from the NASA
[Turbulence Modeling Resource](https://tmbwg.github.io/turbmodels/naca0012_val.html)
(public domain), vendored for offline reproducibility. Sources: Ladson, NASA
TM 4074 (1988); Abbott & von Doenhoff, *Theory of Wing Sections*; Gregory &
O'Reilly (1970).
