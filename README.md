# Surrogate Modelling for CFD — airfoil aerodynamics

A 2-D airfoil CFD and surrogate-modelling pipeline built around SU2, with an
emphasis on **validating predictions against experiment** rather than against the
solver that produced the training data.

The project began as a surrogate-assisted design loop. Auditing it turned up
errors large enough that the audit became the interesting part, so the repository
now documents both the pipeline and what validating it revealed.

---

## Headline result

**A surrogate's apparent accuracy changes by 3.3× depending on which experiment
you compare it against — and standard benchmark metrics do not record the
difference.**

NeuralFoil, evaluated on a NACA0012 at Re = 6×10⁶ against two NASA-hosted
experimental datasets:

| Comparison | n | mean abs. error | bias |
|---|---|---|---|
| Drag vs **Ladson** (tripped / turbulent from LE) | 35 | **39.6 %** | −39.6 % |
| Drag vs **Abbott** (untripped / free transition) | 12 | **11.9 %** | −11.9 % |
| Lift vs Ladson, same range | 24 | **4.1 %** | +3.4 % |

Same model, same airfoil, same Reynolds number. The only difference is whether
the 1988 wind-tunnel model carried a strip of grit.

NeuralFoil inherits XFOIL's transition model: it lets the boundary layer run
laminar over the forward chord, so it predicts less drag than a tripped
experiment can show. **The bias is negative by construction** — the model is
answering a different physical question, and the metric cannot tell.

Note the asymmetry: **drag error is 9.7× the lift error**. Lift is set by
circulation and is largely insensitive to transition, so benchmarks reporting
lift or field-MSE hide the effect entirely. Drag is the discriminating quantity.

```bash
python -m cfdagent.validation.transition_study   # reproduces the table
python -m cfdagent.validation.figures            # writes reports/figures/
```

---

## What the audit found

Each item below is reproducible from a script in this repository.

### 1. Coordinate ordering — 8.9× drag error

`design_to_airfoil_coords` emitted a loop anchored at the **leading edge**.
Aerodynamic tools expect **Selig** order, anchored at the trailing edge. Passing
the wrong one raises no error; it silently describes a blunt-nosed body.

| Ordering | Cl | Cd | L/D |
|---|---|---|---|
| LE-anchored (as emitted) | 0.3627 | 0.06628 | 5.5 |
| Selig (corrected) | 0.4331 | 0.00747 | 58.0 |
| Reference NACA0012 | 0.4330 | 0.00747 | 58.0 |

Maximum geometric deviation from a true NACA0012: **3.55×10⁻⁹**. The shape was
exact; only the ordering was wrong.
→ `python -m cfdagent.experiments.convention_error`

**Is this widespread?** Tested, and **no** — all 2 174 airfoils in the UIUC
database as redistributed with AeroSandbox are Selig-ordered. The hazard is real
and documented upstream, but the curated corpus is clean; this was a defect in
our own generator.
→ `python -m cfdagent.experiments.convention_census`

### 2. Mesh resolution — 406× too coarse for viscous drag

The original mesh had 9.55×10⁻³ chord spacing at the wall. Resolving a boundary
layer at Re = 10⁶ needs roughly 2.4×10⁻⁵ (y⁺ ≈ 1).

| | Original mesh | BL-resolved mesh | Published |
|---|---|---|---|
| Cl | 0.0709 | **0.4419** | 0.43–0.46 |
| Cd | 0.0236 | 0.01108 | — |
| L/D | 3.0 | **39.9** | — |
| Wall spacing | 9.55e-03 c | 2.35e-05 c | — |
| Farfield | 20.5 c | 60 c | 50–100 c |

The original case also never converged: `rms[Rho] = −3.48` against its own −12
target, after exhausting all 2 200 iterations.

### 3. Negative drag, silently absorbed

An inviscid run on the original mesh returned **Cd = −0.0097**. Drag cannot be
negative; it was numerical error where D'Alembert's paradox requires Cd → 0. The
previous post-processing took `abs()` of the coefficients, converting it into a
plausible positive number and logging it as a valid training sample.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Exact versions behind the reported numbers are in `requirements.lock.txt`.

SU2 is not on PyPI or apt — it ships prebuilt binaries:

```bash
./scripts/install_su2.sh          # installs SU2 8.1.0 to ~/.local/su2
export SU2_RUN="$HOME/.local/su2/bin" PATH="$SU2_RUN:$PATH"
```

SU2 is CPU-bound (MPI/OpenMP); a GPU does not accelerate it. Run CFD on local
cores and reserve GPUs for surrogate training.

## Reproducing everything

```bash
./scripts/reproduce.sh            # every number in this README
pytest -m "not slow"              # 71 tests, ~100 s
pytest -m slow                    # adds mesh generation
```

---

## Layout

| Path | Purpose |
|---|---|
| `cfdagent/geometry/` | Airfoil parameterisation; `to_selig_order` for external tools |
| `cfdagent/cfd/bl_mesher.py` | Boundary-layer-resolved mesher, y⁺-sized first cell |
| `cfdagent/cfd/run_cfd.py` | SU2 driver, metric extraction, validation |
| `cfdagent/lowfid/` | NeuralFoil evaluator (~3 ms/design), validated against published data |
| `cfdagent/validation/` | Experimental datasets, transition study, figures |
| `cfdagent/experiments/` | Standalone reproductions of the audit findings |
| `cfdagent/surrogate/` | Snapshot builder, CNN and U-Net surrogates |
| `TestCases/airfoil_naca0012_validated/` | Converged RANS-SA reference case |

### The mesher

```bash
python -m cfdagent.cfd.bl_mesher --out mesh.su2 --reynolds 6e6
```

Quad boundary-layer extrusion with a y⁺-sized first cell, geometric growth, wake
refinement, 60-chord farfield, and the `Airfoil`/`Farfield` markers SU2 expects.
For NACA0012 at Re = 6×10⁶: 162 458 nodes, first layer 4.46×10⁻⁶ chords.

### The low-fidelity evaluator

NeuralFoil-backed, `run_cfd`-compatible, ~3 ms per design. Gated by tests
against published NACA0012 data: lift-curve slope 0.1120/deg vs thin-airfoil
theory 0.1097 (**102 %**), Cl(0°) ≈ 0 for a symmetric section, Cd = 0.00747 and
L/D = 58.0 at α = 4°, Re = 10⁶.

Used to generate 4 000 labelled designs in 11.8 s. A CNN trained on the
rasterised geometry reaches held-out R² of 0.979 (Cl) and 0.840 (Cd); screening
with it reaches L/D 78.5 in 5 solver calls versus 76.3 for random search at 40.

**Caveat, stated plainly:** that comparison uses NeuralFoil as *both* the label
source and the "expensive" solver, so it measures screening behaviour on this
design space — not savings against RANS. NeuralFoil is itself a neural surrogate
and is faster and more accurate than the CNN distilled from it.

---

## Status and limitations

- **The SU2 side of the transition comparison is incomplete.** A sweep at
  Re = 6×10⁶ is scripted but not finished; the preliminary α = 0 point gives
  Cd = 0.00970 against Ladson's 0.00809 (+20 %), versus NeuralFoil's −36 %,
  which is the expected direction for a fully-turbulent closure. Completing it
  needs ~8 runs of roughly an hour each.
- The Abbott dataset is digitised from a printed plot (n = 12) and its source
  file warns the digitisation is approximate. Gregory & O'Reilly (Re = 3×10⁶,
  tripped) is included as an independent check but has lift only.
- The design loop is a random search with a gradient-boosting seed model; it is
  not a competitive optimiser and is not presented as one.
- Results are seeded and portable but not bit-reproducible across BLAS/hardware.
  The 10-seed error bars are the meaningful unit.
- Everything here is 2-D and subsonic.

## Data

Experimental data in `cfdagent/data/experimental/` comes from the NASA
[Turbulence Modeling Resource](https://tmbwg.github.io/turbmodels/naca0012_val.html)
(public domain), vendored for offline reproducibility. Sources: Ladson,
NASA TM 4074 (1988); Abbott & von Doenhoff, *Theory of Wing Sections*;
Gregory & O'Reilly (1970).
