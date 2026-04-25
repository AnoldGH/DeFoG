#!/usr/bin/env bash
# Build the defog conda environment from scratch on a Colab GPU instance.
# Called by 01_env_setup.ipynb after Miniconda is installed.
#
# Install order is intentional and must not be changed:
#   1. conda: graph-tool + rdkit  (C extensions; conda-forge solver must run
#      before pip touches the environment)
#   2. pip:   PyTorch CUDA wheel  (pip wheel matches Colab's live CUDA driver;
#      conda-forge's PyTorch carries its own full CUDA stack, making the
#      conda-pack tarball 3-5 GB larger for no benefit)
#   3. pip:   all remaining Python deps

set -euo pipefail

CONDA=/content/miniconda3/bin/conda
PIP=/content/miniconda3/envs/defog/bin/pip

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
# graph-tool and rdkit are not pip-installable; they must live in the conda
# layer so the conda-forge solver can satisfy their system-library constraints.
# conda-pack is installed here so it is available for the packing step.
echo ""
echo "=== Step 2: Installing graph-tool, rdkit, conda-pack (conda-forge) ==="
echo "    (This is the slowest step — ~10 min to solve + download.)"
$CONDA install -n defog -c conda-forge \
    "graph-tool=2.45" \
    "rdkit=2023.03.2" \
    "conda-pack" \
    -y

# ── Step 3: PyTorch via pip ───────────────────────────────────────────────────
echo ""
echo "=== Step 3: Installing PyTorch 2.4.0 (CUDA wheel) ==="
$PIP install torch==2.4.0 torchvision --index-url "$TORCH_INDEX"

# ── Step 4: Remaining pip dependencies ───────────────────────────────────────
# Versions are pinned to match the WSL defog environment exactly.
echo ""
echo "=== Step 4: Installing remaining pip dependencies ==="
$PIP install \
    "torch_geometric==2.3.1" \
    "pytorch-lightning==2.0.4" \
    "hydra-core==1.3.2" \
    "omegaconf==2.3.0" \
    "wandb==0.15.4" \
    "numpy==1.23.0" \
    "scipy==1.11.0" \
    "scikit-learn==1.6.1" \
    "matplotlib==3.7.1" \
    "pandas==1.4.0" \
    "networkx==2.8.7" \
    "tqdm==4.65.0" \
    "torchmetrics==0.11.4" \
    "pyemd==1.0.0" \
    "PyGSP==0.5.1" \
    "seaborn==0.13.2" \
    "Pillow" \
    "PyYAML==6.0.2" \
    "pyemd==1.0.0" \
    "gpustat==0.6.0"

echo ""
echo "=== defog environment created successfully ==="
$PIP show torch | grep -E "^(Name|Version)"
/content/miniconda3/envs/defog/bin/python -c "import graph_tool; print('graph_tool', graph_tool.__version__)"
