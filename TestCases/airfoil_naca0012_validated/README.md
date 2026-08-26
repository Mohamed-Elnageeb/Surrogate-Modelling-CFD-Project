# NACA0012 validated RANS case

Reference case used to validate the solver setup against published NACA0012 data.
Unlike `../airfoil_naca0012_opt`, this case is converged and mesh-resolved.

## Conditions
Re = 1e6, M = 0.15, AoA = 4 deg, RANS with Spalart-Allmaras (fully turbulent).

## Mesh
Generate with the boundary-layer mesher (the mesh is not committed; it is ~40 MB):

```bash
python -m cfdagent.cfd.bl_mesher --out TestCases/airfoil_naca0012_validated/mesh.su2
```

Produces 154,632 nodes / 278,360 elements, first cell 2.353e-05 chords (y+ ~ 1),
60-chord farfield.

## Run
```bash
SU2_CFD -t 4 config.cfg
```

## Validation result

| Quantity | This case | Published NACA0012 |
|---|---|---|
| Cl | 0.4419 | 0.43 - 0.46 |
| Cd | 0.01108 | see note |
| L/D | 39.9 | see note |

**Note on drag.** This is a *fully turbulent* SA computation: the boundary layer
is turbulent from the leading edge. XFOIL/NeuralFoil at the same conditions
report Cd ~ 0.0075 because they model natural transition, leaving roughly the
first 40-50% of chord laminar. The ~50% drag difference is a physical modelling
difference, not numerical error. Any multi-fidelity model that mixes these two
sources without accounting for transition will be fitting that offset rather
than a discretisation correction.

To compare like with like, either enable a transition model here
(`KIND_TRANS_MODEL= LM`) or force the low-fidelity tool fully turbulent.

## Contrast with the original case

The committed `airfoil_naca0012_opt` case returns Cl 0.0709, Cd 0.0236,
L/D 3.0 at rms[Rho] = -3.48 against its own -12 target, i.e. unconverged on a
mesh whose near-wall spacing (9.55e-03 chords) is ~400x too coarse for viscous
drag. An inviscid run on that mesh returns Cd = -0.0097, a negative drag that
the previous `abs()` post-processing silently converted into a valid sample.
