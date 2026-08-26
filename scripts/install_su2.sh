#!/usr/bin/env bash
# Install the SU2 v8.1.0 prebuilt Linux binaries.
#
# SU2 is not on PyPI or in apt. The official GitHub release ships ready-to-run
# binaries, so no compilation is needed.
set -euo pipefail

VERSION="${SU2_VERSION:-8.1.0}"
PREFIX="${SU2_PREFIX:-$HOME/.local/su2}"
URL="https://github.com/su2code/SU2/releases/download/v${VERSION}/SU2-v${VERSION}-linux64.zip"

echo "Installing SU2 v${VERSION} into ${PREFIX}"
mkdir -p "${PREFIX}"
tmp="$(mktemp -d)"
curl -sSL -o "${tmp}/su2.zip" "${URL}"
unzip -q -o "${tmp}/su2.zip" -d "${PREFIX}"
rm -rf "${tmp}"
chmod +x "${PREFIX}"/bin/SU2_* 2>/dev/null || true

cat <<MSG

Done. Add to your shell profile:

  export SU2_RUN="${PREFIX}/bin"
  export SU2_HOME="${PREFIX}"
  export PATH="\$SU2_RUN:\$PATH"

Verify with:  SU2_CFD --help

SU2 is CPU-bound (MPI/OpenMP); a GPU does not accelerate it. Run CFD on your
local cores and reserve Colab/Kaggle GPUs for surrogate training.
MSG
