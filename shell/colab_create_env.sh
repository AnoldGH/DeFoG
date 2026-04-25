#!/usr/bin/env bash
# Replicate the working WSL defog environment on Colab.
# Reference: conda_out.txt (WSL package list, verified to run experiments).
#
# Key choices, learned from a long debugging session:
#
#   1. PyTorch from conda-forge (NOT pip wheel).
#      WSL has `pytorch 2.4.0 cuda120_py39 conda-forge`. This eliminates the
#      libgomp shadowing that pip's torch wheel causes (its bundled libgomp
#      is older than what graph_tool needs from conda-forge).
#
#   2. numpy pinned to 1.23.* in the conda solve.
#      pytorch-lightning 2.0.4 uses `np.Inf`, which was removed in numpy 2.0.
#      Letting conda-forge pick "latest numpy" silently gives 2.x and breaks
#      training. This pin also forces graph-tool/pytorch to pick builds
#      whose compiled .so files are ABI-compatible with numpy 1.23.
#
#   3. Pip layer matches WSL versions exactly.
#      DeFoG was tested against this exact set; deviating risks API drift.
#
#   4. Always rebuild from scratch.
#      No tarball caching. ~8-12 min/session. Eliminates conda-pack relocation
#      bugs and Drive sync issues.

set -euo pipefail

CONDA=/content/miniconda3/bin/conda
MAMBA=/content/miniconda3/bin/mamba
PYTHON=/content/miniconda3/envs/defog/bin/python

if [ -x "$MAMBA" ]; then
    SOLVER="$MAMBA"
    echo "Using mamba (fast solver)."
else
    SOLVER="$CONDA"
    echo "mamba not found, using conda."
fi

# ── Step 1: Create env (always fresh) ─────────────────────────────────────────
echo ""
echo "=== Step 1: Creating conda environment (Python 3.9) ==="
if [ -d /content/miniconda3/envs/defog ]; then
    echo "Removing existing defog env for clean rebuild..."
    rm -rf /content/miniconda3/envs/defog
fi
"$SOLVER" create -n defog python=3.9 -y

# ── Step 2: conda-forge — graph-tool, rdkit, pytorch with CUDA ────────────────
# Pinning numpy=1.23.* in this solve forces conda to pick package builds
# whose compiled extensions are ABI-compatible with numpy 1.23. Without it,
# the solver picks newer builds requiring numpy 2.0+, which breaks
# pytorch-lightning 2.0.4 (uses removed `np.Inf`).
#
# pytorch=2.4.0=cuda* matches any CUDA build of 2.4.0 (e.g., cuda120_py39_*).
# Colab's CUDA 13.0 driver is backward-compatible with CUDA 12.0 binaries.
echo ""
echo "=== Step 2: conda-forge install (graph-tool, rdkit, pytorch+CUDA) ==="
echo "    (~5-8 min: solve + download CUDA stack + graph-tool C extensions.)"
"$SOLVER" install -n defog -c conda-forge --override-channels \
    "graph-tool=2.45" \
    "rdkit=2023.03.2" \
    "pytorch=2.4.0=cuda*" \
    "numpy=1.23.*" \
    pyyaml \
    pillow \
    pip \
    -y

# ── Step 3: pip — exact WSL versions for pure-Python deps ────────────────────
# DO NOT add numpy, pyyaml, or pillow here — conda owns them (Step 2).
# Re-pinning via pip would create the broken-hybrid bug we already fought
# (Python files from one version, .so files from another).
echo ""
echo "=== Step 3: pip install (pure-Python deps, WSL-pinned) ==="
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install \
    "scipy==1.11.0" \
    "matplotlib==3.7.1" \
    "networkx==2.8.7" \
    "pandas==1.4.0" \
    "scikit-learn==1.6.1" \
    "seaborn==0.13.2" \
    "imageio==2.31.1" \
    "tqdm==4.65.0" \
    "hydra-core==1.3.2" \
    "omegaconf==2.3.0" \
    "wandb==0.15.4" \
    "torch_geometric==2.3.1" \
    "pytorch-lightning==2.0.4" \
    "torchmetrics==0.11.4" \
    "pyemd==1.0.0" \
    "PyGSP==0.5.1" \
    "gpustat==0.6.0"

# ── Step 4: Smoke test ────────────────────────────────────────────────────────
# MPLBACKEND=Agg overrides Jupyter's inline backend (doesn't exist in conda).
# LD_PRELOAD is defense-in-depth: with conda-forge pytorch the libgomp
# shadowing is gone, but keeping the preload makes the env robust against
# any stray pip torch install.
echo ""
echo "=== Step 4: Smoke test ==="
MPLBACKEND=Agg \
LD_PRELOAD=/content/miniconda3/envs/defog/lib/libgomp.so.1 \
$PYTHON -c "
import numpy as np
assert np.__version__.startswith('1.23'), f'numpy must be 1.23.x, got {np.__version__}'
assert hasattr(np, 'Inf'), 'np.Inf missing — pytorch-lightning will crash'

import torch, graph_tool, rdkit, scipy, matplotlib, networkx, pandas, imageio
import torch_geometric, pytorch_lightning, hydra, wandb, torchmetrics
print('torch            ', torch.__version__)
print('graph_tool       ', graph_tool.__version__)
print('rdkit            ', rdkit.__version__)
print('numpy            ', np.__version__)
print('scipy            ', scipy.__version__)
print('matplotlib       ', matplotlib.__version__)
print('networkx         ', networkx.__version__)
print('pandas           ', pandas.__version__)
print('imageio          ', imageio.__version__)
print('torch_geometric  ', torch_geometric.__version__)
print('pytorch_lightning', pytorch_lightning.__version__)
print('torchmetrics     ', torchmetrics.__version__)
print('CUDA available   :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU              :', torch.cuda.get_device_name(0))
print('np.Inf accessible:', np.Inf)
print('All imports OK.')
"
