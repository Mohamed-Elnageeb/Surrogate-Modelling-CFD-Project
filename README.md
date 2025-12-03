# Surrogate Modelling CFD Project: Phase 0 Setup

Goal: 2-D airfoil shape optimisation using SU2 and a physics-aware U-Net surrogate. The CFD solver will generate reference simulations while a surrogate model accelerates design exploration.

Solver: SU2, with a base 2-D airfoil case in `cfdagent/cfd/base_case`.

Agent framework: LangGraph for later phases (just mentioned, not used yet).

Hardware assumption: strong workstation (≈ i9 CPU, 64 GB RAM, RTX 2080 Super 8 GB).

Phase 0 decisions:
- 10 design variables controlling camber and thickness.
- Flow conditions: Re = 5e5, AoA = 4°, low Mach (≈0.05), incompressible laminar / simple RANS.
- Initial dataset size: 50–80 CFD runs.

## Getting started
1. Install SU2 and ensure `SU2_CFD` is on PATH.
2. Copy a working 2-D airfoil SU2 case into `cfdagent/cfd/base_case` as `config.cfg` + `mesh.su2`.
3. Create and activate a Python venv and install requirements (to be added later).
4. Run `python -m cfdagent.scripts.generate_dataset --n_samples 20` once the stubs are implemented.
