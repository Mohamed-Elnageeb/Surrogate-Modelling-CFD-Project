#!/usr/bin/env bash
# Run the SU2 half of the transition-condition study.
#
# Sweeps angle of attack on a boundary-layer-resolved NACA0012 mesh at Re=6e6
# with fully-turbulent RANS (Spalart-Allmaras), matching the condition of the
# TRIPPED experiments (Ladson, Gregory). Results feed
# cfdagent.validation.su2_comparison.
#
# Budget roughly one hour per angle on 4 cores.
set -euo pipefail
cd "$(dirname "$0")/.."

ALPHAS="${ALPHAS:--4 0 2 4 6 8 10}"
THREADS="${SU2_THREADS:-4}"
OUT="${OUT_DIR:-runs/su2_validation}"
MESH="$OUT/mesh.su2"
CFG="TestCases/airfoil_naca0012_validated/config.cfg"

command -v SU2_CFD >/dev/null || { echo "SU2_CFD not on PATH (see scripts/install_su2.sh)"; exit 1; }

mkdir -p "$OUT"
if [ ! -f "$MESH" ]; then
  echo "== generating Re=6e6 boundary-layer mesh =="
  python -m cfdagent.cfd.bl_mesher --out "$MESH" --reynolds 6e6 --farfield 60 --layers 40
fi

for A in $ALPHAS; do
  d="$OUT/alpha_$A"; mkdir -p "$d"; cp "$MESH" "$d/mesh.su2"
  sed -e "s/^AOA= .*/AOA= $A.0/" \
      -e "s/^REYNOLDS_NUMBER= .*/REYNOLDS_NUMBER= 6.0E6/" "$CFG" > "$d/rans.cfg"
  echo "== alpha = $A =="
  ( cd "$d" && SU2_CFD -t "$THREADS" rans.cfg > su2.log 2>&1 ) || echo "   alpha $A failed, continuing"
done

echo
echo "Done. Summarise with:"
echo "  python -m cfdagent.validation.su2_comparison --runs $OUT"
