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
   The repository already includes a baseline NACA 0012 setup (copied from
   `TestCases/airfoil_naca0012_opt`) so you can run the dataset script out of
   the box.
3. **Set up Python**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   ```
4. **Generate data** (once SU2 cases are available). The dataset script logs
   design vectors and metrics to `data/designs.csv`. Add `--snapshot-dir` and
   related flags to persist per-run UNet snapshots directly from SU2 outputs
   (requires ASCII volume/surface tables):
   ```bash
   python -m cfdagent.scripts.generate_dataset \
       --n_samples 20 \
       --snapshot-dir data/snapshots \
       --snapshot-grid-shape 128 256 \
       --snapshot-input-fields MACH PRESSURE TEMPERATURE \
       --snapshot-target-fields MACH PRESSURE TEMPERATURE \
       --snapshot-cl-column CL --snapshot-cd-column CD
   ```

## Training the UNet surrogate

1. **Convert SU2 outputs into snapshots.** The UNet expects `.npz` files with
   `input`, `target_fields`, `cl`, and `cd` arrays. Use
   `cfdagent.surrogate.snapshot_builder.su2_to_unet_snapshot` to transform a
   volume solution and matching surface-force file into that format:
   ```bash
   python - <<'PY'
   from pathlib import Path
   from cfdagent.surrogate.snapshot_builder import SnapshotConfig, su2_to_unet_snapshot

   # Adjust field names and grid_shape to match your SU2 tables
   cfg = SnapshotConfig(
       input_fields=["Mach", "p", "T"],
       target_fields=["u", "v", "p"],
       grid_shape=(128, 256),  # (ny, nx) such that ny * nx == number of rows in the solution table
       cl_name="CL",
       cd_name="CD",
   )

   volume_table = Path("path/to/volume_solution.dat")
   surface_forces = Path("path/to/forces_breakdown.dat")
   snapshot_out = Path("data/snapshots/case01.npz")
   snapshot_out.parent.mkdir(parents=True, exist_ok=True)

   su2_to_unet_snapshot(volume_table, surface_forces, snapshot_out, cfg)
   PY
   ```

2. **Train the model.** Point the training CLI at the directory containing your
   snapshots. Use `--device cuda` to train on GPU if available (it will fall
   back to CPU otherwise):
   ```bash
   python -m cfdagent.surrogate.train_unet_cli \
       data/snapshots \
       --device cuda \
       --epochs 50 \
       --batch-size 8 \
       --base-channels 32 \
       --in-channels 3 --out-channels 3 \
       --model-out models/cfd_unet.pt
   ```

3. **Evaluate a checkpoint.** After training, you can compute mean-squared
   errors over a snapshot directory using the evaluation CLI:
   ```bash
   python -m cfdagent.surrogate.eval_unet_cli \
       data/snapshots \
       --model-path models/cfd_unet.pt \
       --device cuda \
       --batch-size 4
   ```

The training and evaluation scripts use the same channel counts you specify, so
ensure they match the shapes produced when building snapshots.

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
