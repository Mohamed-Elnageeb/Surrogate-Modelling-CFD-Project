# Surrogate Modelling CFD Project

This repository contains Phase 0 scaffolding for a surrogate-assisted CFD
workflow. The goal is to optimise 2‑D airfoil shapes using SU2 as a reference
solver while a physics-aware U‑Net accelerates design exploration.

## Repository layout

- `TestCases/airfoil_naca0012_opt/`: Minimal SU2 configuration and helper
  scripts for a NACA 0012 optimisation case.
- `cfdagent/geometry/`: Sampling utilities that convert 10‑D design vectors
  into airfoil surface coordinates.
- `cfdagent/cfd/`: Thin wrappers around SU2 execution plus post-processing
  helpers.
- `cfdagent/surrogate/`: Dataset loader, UNet model definitions, and training
  CLI entry points.
- `cfdagent/optimizer/`: A light random-search loop suitable for early-phase
  experimentation.

## Getting started

1. **Install SU2** and ensure the `SU2_CFD` binary is on your `PATH`.
2. **Prepare a base case** by placing a working 2‑D airfoil mesh and
   configuration inside `cfdagent/cfd/base_case` as `mesh.su2` and `config.cfg`.
3. **Set up Python**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   ```
4. **Generate data** (once SU2 cases are available):
   ```bash
   python -m cfdagent.scripts.generate_dataset --n_samples 20 --output data/flowfields
   ```

## Key utilities

- **Airfoil generation**: `cfdagent.geometry.design_to_airfoil_coords` converts
  10 control parameters into smooth, cosine-spaced surface coordinates using a
  NACA‑0012 baseline thickness law.
- **Dataset loading**: `cfdagent.surrogate.dataset.CFDDataset` reads metadata
  from `designs.csv` or scans `flowfields/*.npz`, pairing flowfields with
  optional geometry images for model training.
- **Optimisation loop**: `cfdagent.optimizer.optimize_loop.optimize` performs a
  seeded random search over design space and returns the best design and score
  along with an evaluation history.
- **Post-processing**: `cfdagent.cfd.postprocess.save_flowfield` aggregates
  CSV exports from SU2 into compressed `.npz` archives for consistent loading.

## Running tests

Use `pytest` to run the unit suite after activating your environment:

```bash
pytest
```

## Notes on SU2 compatibility

The supplied `config.cfg` in `TestCases/airfoil_naca0012_opt` avoids deprecated
keywords by using `MARKER_HEATFLUX` for adiabatic walls and `MARKER_PLOTTING`
for surface exports. If SU2 reports unknown options, compare against your local
`config_template.cfg` and adjust the boundary/output sections accordingly.
