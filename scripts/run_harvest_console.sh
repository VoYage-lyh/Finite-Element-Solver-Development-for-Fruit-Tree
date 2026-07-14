#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the integrated UI.  The solver stays in the
# orchard-fenicsx environment; only packages missing there (timm/torchvision)
# are discovered at the end of sys.path from orchard-ml, so nothing is copied or
# downloaded and the FEniCSx Python stack keeps precedence.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${CONDA_EXE:-}" ]]; then
    conda_base="$("$CONDA_EXE" info --base)"
elif command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    conda_base="$HOME/miniforge3"
else
    echo "Could not find conda/miniforge. Set CONDA_EXE or add conda to PATH." >&2
    exit 1
fi

fenics_python="$conda_base/envs/orchard-fenicsx/bin/python"
ml_python="$conda_base/envs/orchard-ml/bin/python"
if [[ ! -x "$fenics_python" ]]; then
    echo "Missing environment: $conda_base/envs/orchard-fenicsx" >&2
    exit 1
fi
if [[ ! -x "$ml_python" ]]; then
    echo "Missing environment: $conda_base/envs/orchard-ml" >&2
    exit 1
fi

ml_site="$($ml_python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export ORCHARD_ML_SITE="$ml_site"
cd "$repo_root"

if [[ "${1:-}" == "--check" ]]; then
    exec "$fenics_python" -c '
import os
import sys

sys.path.append(os.environ["ORCHARD_ML_SITE"])
import dolfinx
import timm
import torch
import torchvision

print(f"dolfinx={dolfinx.__version__} ({dolfinx.__file__})")
print(f"torch={torch.__version__} ({torch.__file__})")
print(f"torchvision={torchvision.__version__} ({torchvision.__file__})")
print(f"timm={timm.__version__} ({timm.__file__})")
'
fi

exec "$fenics_python" -c '
import os
import runpy
import sys

sys.path.append(os.environ["ORCHARD_ML_SITE"])
runpy.run_module("orchard_fem.actuator.harvest_console", run_name="__main__")
'
