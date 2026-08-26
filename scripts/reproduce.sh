#!/usr/bin/env bash
# Regenerate every quantitative result reported for this project.
#
# The low-fidelity path (steps 1-4) runs in a few minutes on a laptop CPU.
# The SU2 validation (step 5) needs SU2 installed and takes ~1 h on 4 cores.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1. Validate the low-fidelity evaluator against published NACA0012 =="
python -m pytest tests/test_neuralfoil_eval.py -q

echo "== 2. Geometry: coordinate-convention effect on drag =="
python -m cfdagent.experiments.convention_error

echo "== 3. Generate the low-fidelity dataset (4000 designs) =="
python -m cfdagent.scripts.generate_lowfid_dataset --n-samples 4000 --seed 0

echo "== 4. Train the CNN surrogate and run the optimisation study =="
python -m cfdagent.surrogate.train_cnn_surrogate --epochs 40
python -m cfdagent.scripts.mf_optimization_study --seeds 10

echo "== 5. SU2 validation (requires SU2_CFD on PATH) =="
if command -v SU2_CFD >/dev/null 2>&1; then
  python -m cfdagent.cfd.bl_mesher --out TestCases/airfoil_naca0012_validated/mesh.su2
  ( cd TestCases/airfoil_naca0012_validated && SU2_CFD -t "${SU2_THREADS:-4}" config.cfg )
else
  echo "SU2_CFD not found; skipping. Install with scripts/install_su2.sh"
fi
