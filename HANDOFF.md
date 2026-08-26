# Handoff — the research thread

This project is **finished as an engineering deliverable**: the pipeline works, the
results are validated and reproducible, CI is green. See `README.md` and
`RESULTS.md`.

This document is for the *other* thing that came out of it — an unfinished
research direction that belongs in a separate repository, not this one. It exists
so a future session can pick it up without re-deriving anything.

---

## The finding worth pursuing

**A surrogate's apparent accuracy changes by 3.3× depending on which experiment
you score it against, and no ML benchmark records the difference.**

Measured, reproducible today (`python -m cfdagent.validation.transition_study`):

| Comparison | n | mean abs. error | bias |
|---|---|---|---|
| NeuralFoil drag vs Ladson (**tripped**) | 35 | 39.6 % | −39.6 % |
| NeuralFoil drag vs Abbott (**free transition**) | 12 | 11.9 % | −11.9 % |
| NeuralFoil lift vs Ladson, same range | 24 | 4.1 % | +3.4 % |

Same model, same airfoil (NACA0012), same Reynolds number (6×10⁶). The only
difference is whether the 1988 wind-tunnel model carried a strip of grit.

The mechanism is not subtle: NeuralFoil inherits XFOIL's transition model, so it
lets the boundary layer run laminar over the forward chord and predicts less drag
than a tripped experiment can possibly show. The bias is negative **by
construction** — it is answering a different physical question, and the metric
cannot tell.

**Drag error is 9.7× the lift error.** Lift is set by circulation and is largely
insensitive to transition, so benchmarks reporting lift or field-MSE hide the
effect entirely. Drag is the discriminating quantity.

## The thesis

> Neural aerodynamic surrogates are ranked by agreement with the solver that
> trained them. AirfRANS baselines report 2.95–8.35 % relative error on C_D —
> against the solver. That is not physical accuracy. Against experiment the same
> class of prediction is off by 12–40 % in drag, and the dominant term is a
> **physics-condition mismatch** (transition state) that benchmark metadata does
> not record.

The sharpest testable version: **does ranking models by solver-agreement rank
them by physical accuracy?** If model A beats model B on AirfRANS but sits
further from experiment, the leaderboard is actively misleading. A demonstrated
rank inversion would be the strongest form of this result.

## The experiment to finish: a 2×2 that should be diagonal

| | Tripped experiment (Ladson) | Untripped experiment (Abbott) |
|---|---|---|
| **NeuralFoil** (models transition) | 39.6 % ❌ *measured* | **11.9 %** ✅ *measured* |
| **SU2 RANS-SA** (fully turbulent) | expect ✅ | expect ❌ |

Two cells are done. The SU2 row needs roughly seven runs of about an hour each on
four cores. The machinery is already in this repo and tested:

```bash
./scripts/run_su2_validation_sweep.sh                      # ~7 h on 4 cores
python -m cfdagent.validation.su2_comparison --runs runs/su2_validation
```

`su2_comparison` refuses to report coefficients from runs below the convergence
target, so an unfinished sweep cannot be mistaken for a result.

**Status of the one point attempted here:** α = 0 reached Cd = 0.00912 at
`rms[Rho] = −4.12`, still falling, against a −11 target and Ladson's 0.00809.
The direction is what a fully-turbulent closure should give, but **this is not a
result** and must not be quoted as one.

If the diagonal is good and the off-diagonal bad, that is the whole paper in one
figure: each method is accurate only against the physics it models, and
benchmarks that omit transition state compare across cells without knowing it.

---

## Literature position (checked August 2026)

Three directions were investigated and abandoned because they are already
occupied. Recording them so they are not re-investigated.

| Direction | Why abandoned |
|---|---|
| Neural airfoil surrogates | Saturated. Neural fields reach <1.7 % coefficient error; MARIO, INFINITY, AB-UPT (140 M cells) |
| Multi-fidelity surrogate screening | arXiv **2603.17057** (Mar 2026) — active multi-fidelity for airfoil shape optimisation with uncertainty-triggered HF allocation. Multi-fidelity **CNN** with transfer learning goes back to Phys. Fluids 33:127121 (2021) |
| Agentic CFD / silent failure | **ChatCFD** (2506.02019) already introduces "execution success vs physical fidelity" (82.1 % vs 68.12 %). Also *Plausible but Wrong* (2604.25345), CFDLLMBench (2509.20374), Judge Agent (2603.25780) |

**What remains open**, and why this thread survived the check:

- Industry has articulated the thesis (Luminary, Jul 2026: surrogates "learn,
  encode, and propagate" CFD bias; R² 0.36–0.76 before experimental calibration,
  0.94–0.97 after) — but as a **blog post about a proprietary product**.
- One academic paper (OSTI 2025) applies ASME V&V 20-2009 to a NACA0012 DNN
  surrogate. Single case, credibility-methodology framing.
- **No public benchmark scores neural aerodynamic surrogates against
  experimental data.** The ML surrogate community reports solver-mimicry; the
  CFD V&V community has had simulation-vs-experiment standards for 30 years.
  These two communities are not talking to each other.

**Honest ceiling:** this is a measurement-and-position paper, not a new method.
Main rejection risk is a V&V reviewer saying "everyone knows tripped ≠
untripped" — which is true. The contribution is that the *ML benchmark
community* does not record it, plus the quantified 3.3×.

## Useful resources found

- **NASA Turbulence Modeling Resource** — https://tmbwg.github.io/turbmodels/naca0012_val.html
  Experimental NACA0012 data, public domain, already vendored into
  `cfdagent/data/experimental/` with typed loaders that carry transition state.
- **UniFoil** (NeurIPS 2025 D&B, arXiv 2505.21124) — 500 000+ RANS airfoil
  samples spanning **transitional and turbulent** regimes, e^N transition coupled
  with SA. Removes the data-generation burden entirely; use it instead of
  generating anything.
- **AirfRANS** (NeurIPS 2022) — 1 000 incompressible fully-turbulent RANS cases,
  the standard benchmark. Its simulations were validated against NASA
  experiment, but every ML metric it reports is MSE against its own RANS fields.

## Suggested first session on the new repo

1. Finish the SU2 row of the 2×2 (script above). This is the gating result.
2. Add Gregory & O'Reilly (Re=3×10⁶, tripped) as an independent check — Abbott is
   digitised from a printed plot, n=12, and its source file warns the
   digitisation is approximate.
3. Extend beyond NACA0012 using UniFoil, which labels transition regime per
   sample.
4. Attempt the rank inversion: train two surrogates on AirfRANS, rank by
   solver-agreement, then re-rank against experiment.
5. Target arXiv first (no acceptance needed, no deadline). ML4PS is a free,
   virtual, non-NeurIPS-affiliated shot; an arXiv endorsement in physics.flu-dyn
   or cs.LG is the one external dependency.

## What to carry over

`cfdagent/validation/` transplants directly — experimental loaders, the
transition study, the SU2 comparison tool, and the figure code. So do
`cfdagent/cfd/bl_mesher.py`, `cfdagent/lowfid/`, and `scripts/install_su2.sh`.

Do **not** carry over the CNN-vs-random-search screening comparison. It uses
NeuralFoil as both label source and "expensive" solver, so it measures screening
behaviour on a toy design space rather than savings against RANS — and NeuralFoil
is itself a neural surrogate, faster and more accurate than the CNN distilled
from it. It stays in this repo, clearly caveated, as pipeline demonstration only.
