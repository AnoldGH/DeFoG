#!/usr/bin/env bash
# Build the defog conda environment from scratch on a Colab GPU instance.
# Called by 01_env_setup.ipynb after Miniforge is installed.
#
# Install order is intentional and must not be changed:
#   1. conda: graph-tool, rdkit, numpy, scipy, conda-pack
#      graph-tool and rdkit require the conda-forge C-extension solver.
#      numpy and scipy MUST be installed here — conda-forge pins them to
#      versions compatible with its own compiled packages. If pip installs
#      older versions of numpy/scipy afterwards it will create a broken
#      hybrid (conda's compiled extensions expect newer ABI).
#   2. pip: PyTorch CUDA wheel
#      Uses Colab's live CUDA driver version. conda-forge's pytorch brings
#      its own full CUDA stack, making the tarball 3-5 GB larger for no
#      benefit — so pip wheel is preferred here.
#   3. pip: all remaining Python deps (must NOT re-pin numpy/scipy)

set -euo pipefail

CONDA=/content/miniconda3/bin/conda
PIP=/content/miniconda3/envs/defog/bin/python
# NOTE: call pip as "python -m pip" throughout to avoid broken shebang lines
# that can appear after conda-unpack relocates the environment.

# ── Detect Colab's CUDA version ───────────────────────────────────────────────
# Prefer nvcc (exact toolkit version); fall back to nvidia-smi driver report.
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+' | head -1)
else
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' | head -1 || echo "12.1")
fi
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
echo "Detected CUDA ${CUDA_MAJOR}.${CUDA_MINOR}"

# PyTorch 2.4.0 official CUDA wheel variants: cu118, cu121, cu124.
# CUDA 12.2 / 12.3 drivers are backward-compatible with the cu121 wheel.
if   [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [ "$CUDA_MAJOR" -ge 12 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
fi
echo "PyTorch wheel index: $TORCH_INDEX"

# ── Step 1: Create the conda environment ──────────────────────────────────────
echo ""
echo "=== Step 1: Creating conda environment (Python 3.9) ==="
$CONDA create -n defog python=3.9 -y

# ── Step 2: conda-forge packages ──────────────────────────────────────────────
# numpy and scipy are installed here so the conda-forge solver determines
# versions compatible with graph-tool's compiled C extensions.
# Pinning them later via pip would silently downgrade them and break imports.
#
# numpy: >=1.25 required for numpy._utils (used by graph_tool ≥ 2.45)
# scipy: >=1.12 required for scipy._lib._array_api (used by torch_geometric)
echo ""
echo "=== Step 2: Installing conda-forge packages (graph-tool, rdkit, numpy, scipy) ==="
echo "    (This is the slowest step — ~10 min to solve + download.)"
$CONDA install -n defog -c conda-forge \
    "graph-tool=2.45" \
    "rdkit=2023.03.2" \
    "numpy>=1.25,<2.0" \
    "scipy>=1.12" \
    "conda-pack" \
    -y

# ── Step 3: PyTorch via pip ───────────────────────────────────────────────────
echo ""
echo "=== Step 3: Installing PyTorch 2.4.0 (CUDA wheel) ==="
$PIP -m pip install torch==2.4.0 torchvision --index-url "$TORCH_INDEX"

# ── Step 4: Remaining pip dependencies ───────────────────────────────────────
# DO NOT add numpy or scipy here — they are managed by conda (see Step 2).
# Adding them would overwrite conda's carefully solved versions.
#
# Version notes:
#   torch_geometric 2.3.1 — matches DeFoG's original environment
#   pytorch-lightning 2.0.4 — stable with torch 2.4
#   pandas 2.0.3 — minimum version that is ABI-safe with numpy >= 1.25
#   scikit-learn 1.3.2 — compatible with numpy >= 1.25, scipy >= 1.12
#   torchmetrics 0.11.4 — last version compatible with lightning 2.0.x
echo ""
echo "=== Step 4: Installing remaining pip dependencies ==="
$PIP -m pip install \
    "torch_geometric==2.3.1" \
    "pytorch-lightning==2.0.4" \
    "hydra-core==1.3.2" \
    "omegaconf==2.3.0" \
    "wandb==0.15.4" \
    "scikit-learn==1.3.2" \
    "matplotlib==3.7.1" \
    "pandas==2.0.3" \
    "networkx==2.8.7" \
    "tqdm==4.65.0" \
    "torchmetrics==0.11.4" \
    "pyemd==1.0.0" \
    "PyGSP==0.5.1" \
    "seaborn==0.13.2" \
    "Pillow" \
    "PyYAML==6.0.2" \
    "gpustat==0.6.0"

echo ""
echo "=== defog environment created successfully ==="
$PIP -m pip show torch | grep -E "^(Name|Version)"
/content/miniconda3/envs/defog/bin/python -c "
import graph_tool, torch, scipy, numpy
print('graph_tool', graph_tool.__version__)
print('torch     ', torch.__version__)
print('numpy     ', numpy.__version__)
print('scipy     ', scipy.__version__)
"
